"""Evaluate inventory-aware forecast components independently in original demand units."""

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F

from src.analysis.inventory_framework import COMPONENTS

LOG2 = math.log(2.0)


class InventoryAwareEvaluator:
    """Compute nine separate evaluation metrics using training-only historical scales.

    Forecasts and targets use (batch, horizon, SKU) axes by default. Training
    history always uses (batch, history, SKU). Evaluation returns a dictionary
    of scalar tensors without gradients; components are never combined.
    """

    def __init__(
        self,
        base_kind: str = "mae",
        huber_delta: float = 1.0,
        rel_eps: float = 1e-6,
        sum_relative: bool = True,
        share_divergence: str = "js",
        share_mode: str = "relu",
        share_eps: float = 1e-8,
        alloc_relative: bool = False,
        quantile_tau: float = 0.7,
        delta_kind: str = "mse",
        cap_mode: str = "factor_history_max",
        cap_value: Optional[Union[float, torch.Tensor]] = None,
        cap_factor: float = 2.0,
        components_axis: Optional[int] = -1,
    ):
        self.base_kind = base_kind.lower()
        self.huber_delta = float(huber_delta)
        self.rel_eps = float(rel_eps)
        self.sum_relative = bool(sum_relative)
        self.share_divergence = share_divergence.lower()
        self.share_mode = share_mode.lower()
        self.share_eps = float(share_eps)
        self.alloc_relative = bool(alloc_relative)
        self.quantile_tau = float(quantile_tau)
        self.delta_kind = delta_kind.lower()
        self.cap_mode = cap_mode.lower()
        self.cap_value = cap_value
        self.cap_factor = float(cap_factor)
        self.components_axis = components_axis

    @torch.no_grad()
    def evaluate(
        self, y_hat: torch.Tensor, y: torch.Tensor, history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return each component separately, without optimization or aggregation."""
        yh, yt = self._to_btd(y_hat, y)
        values = {
            "item": self._term_item(yh, yt),
            "sum": self._term_sum(yh, yt),
            "share": self._term_share(yh, yt),
            "alloc": self._term_alloc(yh, yt),
            "scaled": self._term_scaled(yh, yt, history),
            "cap": self._term_cap(yh, yt, history),
            "asym": self._term_asym(yh, yt),
            "delta": self._term_delta(yh, yt) if yh.shape[1] >= 2 else yh.new_zeros(()),
            "zero": self._term_zero(yh, yt, history),
        }
        return {name: values[name] for name in COMPONENTS}

    def _to_btd(self, y_hat, y):
        if y_hat.ndim != 3 or y.ndim != 3:
            raise ValueError(f"Expected 3-D tensors, got {y_hat.shape}, {y.shape}")
        if y_hat.shape != y.shape:
            raise ValueError(f"Shape mismatch: {y_hat.shape} vs {y.shape}")
        if self.components_axis in (-1, 2):
            return y_hat, y
        if self.components_axis == 1:
            return y_hat.transpose(1, 2), y.transpose(1, 2)
        if self.components_axis is None:
            return y_hat, y
        raise ValueError("components_axis must be -1/2 for (B,T,D), 1 for (B,D,T), or None")

    def _elemwise(self, pred, true, kind):
        e = pred - true
        if kind == "mse":
            return e * e
        if kind == "mae":
            return e.abs()
        if kind == "huber":
            return F.huber_loss(pred, true, delta=self.huber_delta, reduction="none")
        if kind == "logcosh":
            ax = e.abs()
            return ax + torch.log1p(torch.exp(-2.0 * ax)) - LOG2
        raise ValueError(kind)

    def _term_item(self, yh, y):
        return self._elemwise(yh, y, self.base_kind).mean()

    def _history_btd(self, history, reference):
        """Require finite nonnegative training demand aligned to batch and SKU axes."""
        if history is None:
            raise ValueError("Training history is required for MASE, Zero and historical Cap.")
        history = torch.as_tensor(history, device=reference.device, dtype=reference.dtype)
        if history.ndim != 3 or history.shape[0] != reference.shape[0] or history.shape[2] != reference.shape[2]:
            raise ValueError("Training history must have shape (B, T_history, D), aligned to forecasts.")
        if history.shape[1] < 2 or not torch.isfinite(history).all() or (history < 0).any():
            raise ValueError("Training history requires at least two finite nonnegative observations per SKU.")
        return history

    def _term_scaled(self, yh, y, history):
        """MASE with lag-one training scale; exclude constant-history SKUs exactly."""
        history = self._history_btd(history, yh)
        scale = history.diff(dim=1).abs().mean(dim=1, keepdim=True)
        eligible = (scale > 0).expand_as(yh)
        if not eligible.any():
            return yh.new_full((), float("nan"))
        return ((yh - y).abs()[eligible] / scale.expand_as(yh)[eligible]).mean()

    def _term_zero(self, yh, y, history):
        """Mean signed forecast / training mean over eligible zero-demand observations."""
        history = self._history_btd(history, yh)
        mean = history.mean(dim=1, keepdim=True)
        eligible = (y == 0) & (mean > 0)
        if not eligible.any():
            return yh.new_full((), float("nan"))
        return (yh[eligible] / mean.expand_as(yh)[eligible]).mean()

    def _term_sum(self, yh, y):
        sh, s = yh.sum(-1), y.sum(-1)
        if self.sum_relative:
            denom = torch.maximum(s.abs(), torch.as_tensor(self.rel_eps, device=y.device, dtype=y.dtype))
            return (((sh - s) / denom) ** 2).mean()
        return self._elemwise(sh, s, self.base_kind).mean()

    def _term_alloc(self, yh, y):
        e = yh - y
        ea = e - e.mean(-1, keepdim=True)
        if self.alloc_relative:
            ea = ea / (y.sum(-1, keepdim=True).abs() + self.rel_eps)
        return (ea ** 2).mean()

    def _shares(self, x):
        if self.share_mode == "relu":
            x = F.relu(x)
        elif self.share_mode == "softplus":
            x = F.softplus(x)
        else:
            x = torch.clamp(x, min=0.0)
        s = x.sum(-1, keepdim=True) + self.share_eps
        p = torch.clamp(x / s, min=self.share_eps)
        return p / p.sum(-1, keepdim=True)

    def _term_share(self, yh, y):
        target_total = y.clamp(min=0.0).sum(-1)
        active = target_total > self.share_eps
        if not torch.any(active):
            return torch.zeros((), device=yh.device, dtype=yh.dtype)

        ph, p = self._shares(yh), self._shares(y)
        if self.share_divergence == "js":
            m = 0.5 * (ph + p)
            values = 0.5 * (ph * (ph / m).log()).sum(-1) + 0.5 * (p * (p / m).log()).sum(-1)
            return values[active].mean()
        if self.share_divergence == "kl":
            values = (p * (p / ph).log()).sum(-1)
            return values[active].mean()
        values = -(p * ph.log()).sum(-1)
        return values[active].mean()

    def _compute_cap(self, yh, y, history=None):
        if self.cap_mode == "none":
            return None
        B, T, D = yh.shape
        if self.cap_mode == "fixed":
            cv = self.cap_value
            if cv is None:
                raise ValueError("cap_mode='fixed' requires cap_value")
            cv = cv if isinstance(cv, torch.Tensor) else torch.tensor(cv, device=yh.device, dtype=yh.dtype)
            cv = cv.to(device=yh.device, dtype=yh.dtype)
            if cv.ndim == 0:
                return cv.view(1, 1, 1).expand(B, T, D)
            return cv.view(1, 1, D).expand(B, T, D)
        if self.cap_mode == "factor_true_max":
            return (self.cap_factor * y.max(1, keepdim=True).values).expand(B, T, D)
        if self.cap_mode == "factor_true_mean":
            return (self.cap_factor * y.mean(1, keepdim=True)).expand(B, T, D)
        if self.cap_mode == "factor_history_max":
            history = self._history_btd(history, yh)
            return (self.cap_factor * history.max(1, keepdim=True).values).expand(B, T, D)
        raise ValueError(self.cap_mode)

    def _term_cap(self, yh, y, history=None):
        c = self._compute_cap(yh, y, history)
        if c is None:
            return torch.zeros((), device=yh.device, dtype=yh.dtype)
        return (F.relu(yh - c) ** 2).mean()

    def _term_asym(self, yh, y):
        e = y - yh
        return torch.maximum(self.quantile_tau * e, (self.quantile_tau - 1) * e).mean()

    def _term_delta(self, yh, y):
        return self._elemwise(
            yh[:, 1:] - yh[:, :-1], y[:, 1:] - y[:, :-1], self.delta_kind
        ).mean()
