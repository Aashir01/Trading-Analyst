"""Telegram alert bot.

Two modes:

* **Push loop** (``python -m mfie bot``) — scans on an interval and pushes only
  *new* actionable signals, plus a macro brief when the regime changes.
* **Command polling** — if ``python-telegram-bot`` is installed, responds to
  ``/analyze EURUSD``, ``/macro``, ``/watch`` and ``/status``.

Only the push loop is required; it uses the plain HTTP API, so the optional
dependency is genuinely optional.

Deduplication matters more than it looks: without it, a 15-minute scan loop will
re-send the same trend signal every 15 minutes for as long as the trend lasts,
and you will mute the bot within a day.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import requests

from mfie.config import get_settings
from mfie.core.types import ScoredSignal
from mfie.core.utils import configure_console, get_logger
from mfie.interfaces.report import DISCLAIMER, format_macro_brief, signal_to_telegram

log = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


@dataclass
class BotState:
    """What has already been sent, so the same setup is not re-alerted."""

    sent: dict[str, datetime] = field(default_factory=dict)
    last_macro_regime: str | None = None
    last_liquidity_regime: str | None = None
    cooldown: timedelta = timedelta(hours=4)

    def key(self, signal: ScoredSignal) -> str:
        return f"{signal.instrument.symbol}|{signal.raw.strategy}|{signal.direction.value}"

    def should_send(self, signal: ScoredSignal, now: datetime) -> bool:
        last = self.sent.get(self.key(signal))
        return last is None or (now - last) >= self.cooldown

    def mark_sent(self, signal: ScoredSignal, now: datetime) -> None:
        self.sent[self.key(signal)] = now


def send_message(text: str, token: str | None = None, chat_id: str | None = None,
                 parse_mode: str = "Markdown") -> bool:
    settings = get_settings()
    token = token or settings.telegram_bot_token
    chat_id = chat_id or settings.telegram_chat_id
    if not token or not chat_id:
        log.error("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set to send messages")
        return False

    try:
        response = requests.post(
            TELEGRAM_API.format(token=token, method="sendMessage"),
            json={
                "chat_id": chat_id,
                "text": text[:4096],  # Telegram's hard message limit
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=settings.http_timeout,
        )
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        log.error("Telegram send failed: %s", exc)
        return False


def scan_once(
    symbols: list[str] | None,
    timeframe: str,
    state: BotState,
    min_confidence: float = 0.5,
    send: bool = True,
) -> list[str]:
    """Run the pipeline and return the messages that were (or would be) sent."""
    from mfie.pipeline.engine import AnalysisEngine

    result = AnalysisEngine().run(symbols=symbols, timeframe=timeframe)
    now = datetime.now(timezone.utc)
    messages: list[str] = []

    # Regime changes are the highest-value alert: they change which strategies
    # should be running at all, not just one trade.
    macro = result.macro
    if state.last_macro_regime and macro.macro_regime.value != state.last_macro_regime:
        messages.append(
            f"⚠️ *Macro regime change*: {state.last_macro_regime} → "
            f"{macro.macro_regime.value}\n\n```\n{format_macro_brief(macro)}\n```"
        )
    if state.last_liquidity_regime and macro.gli_regime.value != state.last_liquidity_regime:
        messages.append(
            f"💧 *Liquidity regime change*: {state.last_liquidity_regime} → "
            f"{macro.gli_regime.value} (ΔGLI {macro.gli_delta:+.2%})"
        )
    state.last_macro_regime = macro.macro_regime.value
    state.last_liquidity_regime = macro.gli_regime.value

    for signal in result.tradable:
        if signal.confidence < min_confidence:
            continue
        if not state.should_send(signal, now):
            continue
        messages.append(signal_to_telegram(signal, macro))
        state.mark_sent(signal, now)

    if send:
        for message in messages:
            send_message(message)

    return messages


def run_bot(
    symbols: list[str] | None = None,
    interval: int = 900,
    timeframe: str = "1h",
    min_confidence: float = 0.5,
    once: bool = False,
) -> None:
    """Scan on a loop and push new actionable signals to Telegram."""
    configure_console()
    settings = get_settings()
    state = BotState()

    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        log.warning(
            "Telegram credentials missing — running in dry-run mode, messages print to stdout."
        )
        dry_run = True
    else:
        dry_run = False
        send_message(
            f"📊 *MFIE started*\nWatching {', '.join(symbols) if symbols else 'full universe'} "
            f"on {timeframe}, scanning every {interval // 60} min.\n\n_{DISCLAIMER}_"
        )

    while True:
        try:
            messages = scan_once(symbols, timeframe, state, min_confidence, send=not dry_run)
            if dry_run:
                for message in messages:
                    print(message)
                    print("-" * 60)
            log.info("Scan complete: %d alert(s)", len(messages))
        except Exception as exc:
            log.exception("Scan failed: %s", exc)
            if not dry_run:
                send_message(f"❌ Scan failed: `{exc}`")

        if once:
            return
        time.sleep(max(interval, 60))


# --------------------------------------------------------------------------- #
# Optional: interactive command handling
# --------------------------------------------------------------------------- #
def run_command_bot(symbols: list[str] | None = None) -> None:  # pragma: no cover
    """Interactive bot. Requires ``pip install python-telegram-bot``."""
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, ContextTypes
    except ImportError as exc:
        raise RuntimeError(
            "Interactive mode needs python-telegram-bot: pip install python-telegram-bot"
        ) from exc

    settings = get_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    from mfie.interfaces.report import format_watchlist
    from mfie.pipeline.engine import AnalysisEngine

    async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        requested = context.args or symbols or ["BTCUSDT"]
        result = AnalysisEngine().run(symbols=requested)
        if not result.tradable:
            await update.message.reply_text("No actionable signal right now.")
            return
        for signal in result.tradable[:3]:
            await update.message.reply_markdown(signal_to_telegram(signal, result.macro))

    async def cmd_macro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        engine = AnalysisEngine()
        from mfie.core.universe import UNIVERSE

        macro = engine.build_macro_context(list(UNIVERSE.values()))
        await update.message.reply_text(f"```\n{format_macro_brief(macro, True)}\n```",
                                        parse_mode="Markdown")

    async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        result = AnalysisEngine().run(symbols=context.args or symbols)
        await update.message.reply_text(f"```\n{format_watchlist(result)}\n```",
                                        parse_mode="Markdown")

    async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        from mfie.data.hub import get_hub

        lines = [
            f"{name}: {'live' if ok else 'simulated'}"
            for name, ok in get_hub().provider_status().items()
        ]
        await update.message.reply_text("\n".join(lines))

    app = Application.builder().token(settings.telegram_bot_token).build()
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("macro", cmd_macro))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("status", cmd_status))
    log.info("Telegram command bot polling...")
    app.run_polling()
