"""A small task-conditioned failure discriminator for WAM recovery routing.

The router is intentionally independent of OpenWAM and LIBERO.  It consumes
low-dimensional online features and predicts the probability that each
recovery action will lead to task completion.  The benchmark script uses it
with a fixed geometric local-repair controller; this module only chooses the
route.  The implementation uses NumPy so that the protocol can be inspected
and reproduced without a deep-learning runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ACTIONS = ("continue", "local_repair", "wam_replan", "contact_recovery", "safe_stop")
SOURCES = ("nominal_or_nuisance", "execution_drift", "task_relevant_world_change",
           "wam_plan_failure", "contact_failure", "unsafe_state")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


@dataclass
class RouteDecision:
    action: str
    probabilities: dict[str, float]
    source_probabilities: dict[str, float]
    repair_benefit: float
    uncertainty: float
    margin: float

    @property
    def choice(self) -> str:
        """Compact Jev-like field name for an online decision envelope."""
        return self.action

    @property
    def score(self) -> float:
        """A bounded 0--10 decisiveness score, not a calibrated probability."""
        decisiveness = 0.5 * np.clip(self.margin / 0.20, 0.0, 1.0)
        decisiveness += 0.5 * (1.0 - np.clip(self.uncertainty, 0.0, 1.0))
        return float(10.0 * np.clip(decisiveness, 0.0, 1.0))

    @property
    def noul(self) -> bool:
        """Whether the fast path should defer to another observation/model."""
        return bool(self.uncertainty > 0.42 or self.margin < 0.03)

    def as_fast_json(self) -> dict[str, object]:
        """Return a small machine-readable envelope for a control loop."""
        return {"choice": self.choice, "score": self.score, "noul": self.noul,
                "repair_benefit": self.repair_benefit, "margin": self.margin,
                "action_probabilities": self.probabilities,
                "source_probabilities": self.source_probabilities}


class FailureRouter:
    """Two-layer multi-head failure discriminator.

    Each output head estimates ``P(success | feature, action)``.  The caller
    subtracts a fixed intervention cost before choosing an action.  This makes
    the model a router rather than a generic anomaly detector: a large visual
    residual is not sufficient to trigger a repair when replanning has higher
    expected task success.
    """

    def __init__(self, hidden: int = 64, seed: int = 7) -> None:
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: np.ndarray | None = None
        self.wo: np.ndarray | None = None
        self.bo: np.ndarray | None = None
        self.n_features: int | None = None

    @property
    def parameter_count(self) -> int:
        if self.n_features is None:
            return 0
        return int((self.n_features + 1) * self.hidden
                   + (self.hidden + 1) * self.hidden
                   + (self.hidden + 1) * (len(ACTIONS) + len(SOURCES)))

    def _check_ready(self) -> None:
        if any(value is None for value in
               (self.mean, self.scale, self.w1, self.b1, self.w2, self.b2,
                self.wo, self.bo)):
            raise RuntimeError("FailureRouter has not been fitted")

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
        self._check_ready()
        assert self.mean is not None and self.scale is not None
        assert self.w1 is not None and self.b1 is not None
        assert self.w2 is not None and self.b2 is not None
        assert self.wo is not None and self.bo is not None
        z = (x - self.mean) / self.scale
        h1_pre = z @ self.w1 + self.b1
        h1 = np.maximum(h1_pre, 0.0)
        h2_pre = h1 @ self.w2 + self.b2
        h2 = np.maximum(h2_pre, 0.0)
        logits = h2 @ self.wo + self.bo
        action_probs = _sigmoid(logits[:, :len(ACTIONS)])
        source_logits = logits[:, len(ACTIONS):]
        source_logits = source_logits - np.max(source_logits, axis=1, keepdims=True)
        source_exp = np.exp(source_logits)
        source_probs = source_exp / np.sum(source_exp, axis=1, keepdims=True)
        return action_probs, source_probs, (z, h1_pre, h1, h2_pre, h2, logits)

    def fit(self, features: np.ndarray, labels: np.ndarray, source_labels: np.ndarray, *, epochs: int = 260,
            learning_rate: float = 0.025, weight_decay: float = 2e-4,
            batch_size: int = 256, source_loss_weight: float = 0.65,
            verbose: bool = False) -> dict[str, float]:
        """Fit the shared MLP on counterfactual action-success labels.

        The split is performed by the benchmark before calling this method;
        this function never sees development or held-out rows.
        """
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        source_y = np.asarray(source_labels, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("features and labels must be two-dimensional with equal rows")
        if y.shape[1] != len(ACTIONS):
            raise ValueError(f"labels must have {len(ACTIONS)} action columns")
        if source_y.shape != (x.shape[0], len(SOURCES)):
            raise ValueError(f"source_labels must have shape (N, {len(SOURCES)})")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("non-finite training data")
        self.n_features = int(x.shape[1])
        self.mean = np.mean(x, axis=0)
        self.scale = np.std(x, axis=0)
        self.scale[self.scale < 1e-6] = 1.0
        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / self.n_features),
                             (self.n_features, self.hidden))
        self.b1 = np.zeros(self.hidden, dtype=np.float64)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden),
                             (self.hidden, self.hidden))
        self.b2 = np.zeros(self.hidden, dtype=np.float64)
        self.wo = rng.normal(0.0, np.sqrt(2.0 / self.hidden),
                             (self.hidden, len(ACTIONS) + len(SOURCES)))
        self.bo = np.zeros(len(ACTIONS) + len(SOURCES), dtype=np.float64)

        order = np.arange(x.shape[0])
        eps = 1e-8
        final_loss = float("nan")
        for epoch in range(int(epochs)):
            rng.shuffle(order)
            for start in range(0, len(order), int(batch_size)):
                ids = order[start:start + int(batch_size)]
                xb, yb = x[ids], y[ids]
                probs, source_probs, cache = self._forward(xb)
                z, h1_pre, h1, h2_pre, h2, _ = cache
                n = float(len(ids))
                source_batch = source_y[ids]
                dlogits = np.zeros((len(ids), len(ACTIONS) + len(SOURCES)), dtype=np.float64)
                dlogits[:, :len(ACTIONS)] = (probs - yb) / n
                dlogits[:, len(ACTIONS):] = source_loss_weight * (source_probs - source_batch) / n
                assert self.wo is not None and self.bo is not None
                assert self.w2 is not None and self.b2 is not None
                assert self.w1 is not None and self.b1 is not None
                dwo = h2.T @ dlogits + weight_decay * self.wo
                dbo = np.sum(dlogits, axis=0)
                dh2 = dlogits @ self.wo.T
                dh2[h2_pre <= 0.0] = 0.0
                dw2 = h1.T @ dh2 + weight_decay * self.w2
                db2 = np.sum(dh2, axis=0)
                dh1 = dh2 @ self.w2.T
                dh1[h1_pre <= 0.0] = 0.0
                dw1 = z.T @ dh1 + weight_decay * self.w1
                db1 = np.sum(dh1, axis=0)
                self.wo -= learning_rate * dwo
                self.bo -= learning_rate * dbo
                self.w2 -= learning_rate * dw2
                self.b2 -= learning_rate * db2
                self.w1 -= learning_rate * dw1
                self.b1 -= learning_rate * db1
            probs, source_probs, _ = self._forward(x)
            action_loss = -np.mean(y * np.log(probs + eps)
                                   + (1.0 - y) * np.log(1.0 - probs + eps))
            source_loss = -np.mean(np.sum(source_y * np.log(source_probs + eps), axis=1))
            final_loss = float(action_loss + source_loss_weight * source_loss)
            if verbose and (epoch == 0 or (epoch + 1) % 50 == 0):
                print(f"epoch={epoch + 1} loss={final_loss:.5f}", flush=True)
        return {"train_bce": final_loss, "parameters": float(self.parameter_count)}

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        probs, _, _ = self._forward(x)
        return probs

    def predict_sources(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        _, probs, _ = self._forward(x)
        return probs

    def decide(self, features: Iterable[float] | np.ndarray, *,
               costs: dict[str, float] | None = None,
               uncertainty_margin: float = 0.06) -> RouteDecision:
        values = np.asarray(features, dtype=np.float64).reshape(1, -1)
        probs = self.predict_proba(values)[0]
        source_probs = self.predict_sources(values)[0]
        costs = costs or {}
        utilities = np.asarray([float(probs[i]) - float(costs.get(action, 0.0))
                                for i, action in enumerate(ACTIONS)])
        order = np.argsort(utilities)[::-1]
        best, second = int(order[0]), int(order[1])
        margin = float(utilities[best] - utilities[second])
        entropy = -probs * np.log(probs + 1e-8) - (1.0 - probs) * np.log(1.0 - probs + 1e-8)
        uncertainty = float(np.mean(entropy) / np.log(2.0))
        # A close decision is represented explicitly.  The caller may observe
        # another frame or use a slower WAM check; we keep the fast path safe by
        # preferring continuation only when no intervention has a clear edge.
        action = ACTIONS[best]
        if margin < float(uncertainty_margin) and action in ("local_repair", "wam_replan"):
            action = "continue"
        return RouteDecision(action=action,
                             probabilities={name: float(probs[i])
                                            for i, name in enumerate(ACTIONS)},
                             source_probabilities={name: float(source_probs[i])
                                                   for i, name in enumerate(SOURCES)},
                             repair_benefit=float(probs[ACTIONS.index("local_repair")]
                                                  - probs[ACTIONS.index("continue")]),
                             uncertainty=uncertainty, margin=margin)
