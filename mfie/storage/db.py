"""Engine/session management, with optional TimescaleDB promotion.

``DATABASE_URL`` decides the backend:

* ``sqlite:///data/mfie.db``  (default) — zero setup, good to a few million rows.
* ``postgresql+psycopg2://...`` — production. If the TimescaleDB extension is
  available, the time-series tables are converted to hypertables and given
  compression policies automatically.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from mfie.config import Settings, get_settings
from mfie.core.utils import get_logger
from mfie.storage.models import HYPERTABLES, Base

log = get_logger(__name__)


class Database:
    def __init__(self, url: str | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.url = url or self.settings.database_url
        connect_args = {"check_same_thread": False} if self.url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(
            self.url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)

    @property
    def is_postgres(self) -> bool:
        return self.engine.dialect.name in ("postgresql",)

    @property
    def is_sqlite(self) -> bool:
        return self.engine.dialect.name == "sqlite"

    # ------------------------------------------------------------------ schema
    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.is_sqlite:
            self._tune_sqlite()
        elif self.is_postgres:
            self._promote_hypertables()

    def _tune_sqlite(self) -> None:
        """WAL + relaxed sync: this database is a cache of public market data,
        so durability matters far less than write throughput."""
        with self.engine.begin() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))

    def _promote_hypertables(self) -> None:
        with self.engine.begin() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            except SQLAlchemyError as exc:
                log.info("TimescaleDB extension unavailable, using plain PostgreSQL: %s", exc)
                return
            for table, time_column in HYPERTABLES.items():
                try:
                    conn.execute(
                        text(
                            "SELECT create_hypertable(:t, :c, "
                            "if_not_exists => TRUE, migrate_data => TRUE)"
                        ),
                        {"t": table, "c": time_column},
                    )
                    conn.execute(
                        text(
                            f"ALTER TABLE {table} SET ("
                            "timescaledb.compress, timescaledb.compress_orderby = 'ts DESC')"
                        )
                    )
                    conn.execute(
                        text("SELECT add_compression_policy(:t, INTERVAL '30 days')"),
                        {"t": table},
                    )
                except SQLAlchemyError as exc:
                    log.debug("Hypertable setup skipped for %s: %s", table, exc)

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)

    # ---------------------------------------------------------------- sessions
    @contextmanager
    def session(self) -> Iterator[Session]:
        sess = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def healthcheck(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as exc:
            log.error("Database healthcheck failed: %s", exc)
            return False


@functools.lru_cache(maxsize=1)
def get_database() -> Database:
    db = Database()
    db.create_all()
    return db


def init_db(url: str | None = None) -> Database:
    """Create a database and its schema explicitly (used by the CLI and tests)."""
    db = Database(url)
    db.create_all()
    return db
