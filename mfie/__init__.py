"""
MFIE - Macro-Informed Financial Intelligence Engine.

A Forex + Crypto analysis pipeline:

    Ingestion -> Storage -> Technical signal -> Econometric filter chain
              -> Risk sizing -> Interface (CLI / Streamlit / Telegram)

Nothing in this package places orders. It produces *analysis*: a signal with a
confidence score, a position size suggestion, and an audit trail explaining
which macroeconomic rules amplified or suppressed it.
"""

__version__ = "0.1.0"

from mfie.config import Settings, get_params, get_settings  # noqa: E402,F401

__all__ = ["Settings", "get_settings", "get_params", "__version__"]
