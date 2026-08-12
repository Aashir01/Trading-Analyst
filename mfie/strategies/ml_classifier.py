"""Machine-learning direction strategy.

Wraps ``mfie.ml.model.DirectionModel``. The model is trained on the instrument's
own history the first time it is asked for a signal and cached per symbol and
timeframe, so a dashboard refresh does not retrain.

The strategy refuses to emit anything when walk-forward validation says the
model has no edge over the majority-class baseline. A model that cannot beat
"always guess up" should not be allowed to size a position.
"""

from __future__ import annotations

from mfie.core.types import Direction, RawSignal
from mfie.core.utils import clamp, get_logger
from mfie.ml.model import DirectionModel, probability_to_strength
from mfie.strategies.base import Strategy, StrategyContext, build_signal, scale_strength

log = get_logger(__name__)

# symbol|timeframe|horizon -> fitted model
_MODEL_CACHE: dict[str, DirectionModel] = {}


class MLClassifierStrategy(Strategy):
    name = "ml_classifier"
    family = "ml"
    description = "Gradient-boosted classifier on stationary technical features, walk-forward validated"
    min_bars = 400

    horizon = 12
    min_edge = 0.02        # accuracy must beat the baseline by this much
    min_auc = 0.53
    deadband = 0.06        # probability must clear 0.5 ± deadband

    def __init__(self, algorithm: str = "gradient_boosting", retrain: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.retrain = retrain

    def _cache_key(self, ctx: StrategyContext) -> str:
        return f"{ctx.instrument.symbol}|{ctx.timeframe}|{self.horizon}|{self.algorithm}"

    def _get_model(self, ctx: StrategyContext) -> DirectionModel | None:
        key = self._cache_key(ctx)
        if not self.retrain and key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        model = DirectionModel(horizon=self.horizon, algorithm=self.algorithm)
        try:
            # Threshold at half the typical bar range so the label means
            # "a move worth trading" rather than "any tick up".
            atr_pct = ctx.indicators.last("atr_pct", 0.0)
            model.threshold = float(max(atr_pct * 0.5, 0.0))
            model.fit(ctx.indicators)
        except Exception as exc:
            log.debug("ML model training failed for %s: %s", ctx.instrument.symbol, exc)
            return None

        _MODEL_CACHE[key] = model
        return model

    def generate(self, ctx: StrategyContext) -> RawSignal | None:
        model = self._get_model(ctx)
        if model is None or model.report is None:
            return None

        report = model.report
        if report.edge < self.min_edge or report.auc < self.min_auc:
            log.debug(
                "%s: ML model rejected (edge %.3f, auc %.3f)",
                ctx.instrument.symbol, report.edge, report.auc,
            )
            return None

        probability = model.predict_proba(ctx.indicators)
        if probability is None:
            return None

        label, raw_strength = probability_to_strength(probability, self.deadband)
        if label == "flat":
            return None

        direction = Direction.LONG if label == "long" else Direction.SHORT

        # Scale conviction by validated quality, not just by the probability.
        quality = clamp((report.auc - 0.5) / 0.15, 0.0, 1.0)
        strength = 0.30 + 0.35 * raw_strength + 0.20 * quality
        strength *= ctx.regime.strategy_weight("momentum" if raw_strength > 0.5 else "mean_reversion")

        top_features = list(report.feature_importance.items())[:4]
        rationale = [
            f"Model probability of an up move: {probability:.1%} (deadband ±{self.deadband:.0%})",
            f"Walk-forward accuracy {report.accuracy:.1%} vs baseline {report.baseline:.1%} "
            f"(edge {report.edge:+.1%})",
            f"ROC-AUC {report.auc:.3f} over {report.folds} folds, {report.n_test} test bars",
        ]
        if top_features:
            rationale.append(
                "Top features: "
                + ", ".join(f"{name} ({imp:.2f})" for name, imp in top_features)
            )

        return build_signal(
            ctx,
            direction,
            self.name,
            scale_strength(strength),
            rationale,
            features={
                "probability": probability,
                "model_edge": report.edge,
                "model_auc": report.auc,
                "horizon_bars": float(self.horizon),
            },
        )


def clear_model_cache() -> None:
    """Drop cached models — used by the CLI's ``--retrain`` flag and by tests."""
    _MODEL_CACHE.clear()
