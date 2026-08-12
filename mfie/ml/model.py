"""Supervised direction model (gradient boosting / random forest).

Scope discipline: this predicts the probability that the next ``horizon`` bars
close higher. That probability is one input among many — it is fed into the
same macro filter chain as every other strategy, never traded on its own.

Validation is walk-forward with purging. A model that looks excellent under
random k-fold cross-validation and mediocre under walk-forward is not a good
model; it is a leak.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mfie.config import DATA_DIR
from mfie.core.utils import clamp, get_logger
from mfie.ml.features import FEATURE_COLUMNS, align_xy, build_feature_matrix, make_labels
from mfie.technical.indicators import IndicatorSet

log = get_logger(__name__)

MODEL_DIR = DATA_DIR / "models"


@dataclass
class ModelReport:
    accuracy: float
    auc: float
    n_train: int
    n_test: int
    folds: int
    feature_importance: dict[str, float] = field(default_factory=dict)
    baseline: float = 0.5

    @property
    def edge(self) -> float:
        """Accuracy above the majority-class baseline. Near zero means no signal."""
        return self.accuracy - self.baseline


class DirectionModel:
    """Wraps a scikit-learn classifier with the project's train/predict contract."""

    def __init__(
        self,
        horizon: int = 12,
        threshold: float = 0.0,
        algorithm: str = "gradient_boosting",
        random_state: int = 7,
    ) -> None:
        self.horizon = horizon
        self.threshold = threshold
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.report: ModelReport | None = None
        self.feature_columns = list(FEATURE_COLUMNS)

    # ------------------------------------------------------------------ build
    def _make_estimator(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.svm import SVC

        if self.algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=6,          # shallow: financial features are noisy
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        if self.algorithm == "svm":
            return SVC(C=1.0, kernel="rbf", probability=True, random_state=self.random_state)
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=self.random_state,
        )

    def _pipeline(self):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", self._make_estimator()),
            ]
        )

    # ------------------------------------------------------------------ train
    def fit(self, indicators: IndicatorSet, validate: bool = True) -> ModelReport:
        features = build_feature_matrix(indicators)
        labels = make_labels(indicators.df["close"], self.horizon, self.threshold)
        x, y = align_xy(features, labels)

        if len(x) < 200 or y.nunique() < 2:
            raise ValueError(
                f"Not enough usable training data: {len(x)} rows, {y.nunique()} classes"
            )

        report = (
            walk_forward_score(x, y, self.horizon, self._pipeline)
            if validate
            else ModelReport(0.0, 0.0, len(x), 0, 0)
        )

        self.model = self._pipeline()
        self.model.fit(x, y)
        self.feature_columns = list(x.columns)

        try:
            clf = self.model.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                report.feature_importance = {
                    col: float(imp)
                    for col, imp in sorted(
                        zip(x.columns, clf.feature_importances_, strict=True),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:15]
                }
        except (AttributeError, KeyError):
            pass

        self.report = report
        return report

    # ---------------------------------------------------------------- predict
    def predict_proba(self, indicators: IndicatorSet) -> float | None:
        """Probability that the next ``horizon`` bars close higher, for the latest bar."""
        if self.model is None:
            return None
        features = build_feature_matrix(indicators)
        row = features.tail(1)
        if row.isna().all(axis=1).iloc[0]:
            return None
        try:
            proba = self.model.predict_proba(row)[0]
        except Exception as exc:  # pragma: no cover - shape/NaN edge cases
            log.debug("Model prediction failed: %s", exc)
            return None
        classes = list(getattr(self.model.named_steps["clf"], "classes_", [0.0, 1.0]))
        if 1.0 in classes:
            return float(proba[classes.index(1.0)])
        return float(proba[-1])

    # ------------------------------------------------------------- persistence
    def save(self, path: Path | None = None) -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = path or MODEL_DIR / f"direction_{self.algorithm}_h{self.horizon}.pkl"
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "horizon": self.horizon,
                    "threshold": self.threshold,
                    "algorithm": self.algorithm,
                    "feature_columns": self.feature_columns,
                    "report": self.report,
                },
                fh,
            )
        return path

    @classmethod
    def load(cls, path: Path) -> DirectionModel:
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        instance = cls(
            horizon=blob["horizon"],
            threshold=blob["threshold"],
            algorithm=blob["algorithm"],
        )
        instance.model = blob["model"]
        instance.feature_columns = blob["feature_columns"]
        instance.report = blob.get("report")
        return instance


def walk_forward_score(
    x: pd.DataFrame,
    y: pd.Series,
    horizon: int,
    pipeline_factory,
    folds: int = 4,
) -> ModelReport:
    """Expanding-window walk-forward validation with a purge gap.

    Each fold trains on everything before the test window (minus ``horizon``
    bars of purge) and tests on the window that follows. This is the only
    validation scheme that answers the question you actually care about: would
    this model have worked on data it had never seen, in the order it arrived?
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    n = len(x)
    if n < 300:
        folds = 2

    fold_size = n // (folds + 1)
    accuracies: list[float] = []
    aucs: list[float] = []
    n_train_total = 0
    n_test_total = 0

    for fold in range(1, folds + 1):
        split = fold_size * fold
        train_end = max(split - horizon, 10)
        test_end = min(split + fold_size, n)

        x_train, y_train = x.iloc[:train_end], y.iloc[:train_end]
        x_test, y_test = x.iloc[split:test_end], y.iloc[split:test_end]

        if len(x_train) < 100 or len(x_test) < 20 or y_train.nunique() < 2:
            continue

        model = pipeline_factory()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracies.append(float(accuracy_score(y_test, predictions)))

        if y_test.nunique() > 1:
            try:
                proba = model.predict_proba(x_test)[:, 1]
                aucs.append(float(roc_auc_score(y_test, proba)))
            except (ValueError, IndexError):
                pass

        n_train_total += len(x_train)
        n_test_total += len(x_test)

    # Majority-class baseline: beating 50% means nothing if 60% of bars are up.
    baseline = float(max(y.mean(), 1.0 - y.mean()))

    return ModelReport(
        accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
        auc=float(np.mean(aucs)) if aucs else 0.5,
        n_train=n_train_total,
        n_test=n_test_total,
        folds=len(accuracies),
        baseline=baseline,
    )


def probability_to_strength(probability: float, deadband: float = 0.05) -> tuple[str, float]:
    """Map a class probability to ``(direction, strength)``.

    The deadband around 0.5 is deliberate: a 51% model is not a signal, it is
    rounding error, and trading it just pays spread.
    """
    if probability is None:
        return "flat", 0.0
    edge = probability - 0.5
    if abs(edge) < deadband:
        return "flat", 0.0
    direction = "long" if edge > 0 else "short"
    return direction, float(clamp(abs(edge) * 2.0, 0.0, 1.0))
