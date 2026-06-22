import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

LOG2 = math.log(2.0)

class InventoryDemandLoss(nn.Module):
    def __init__(
        self,
        lambda_item: float = 1.0,
        lambda_sum: float = 0.0,
        lambda_share: float = 0.0,
        lambda_alloc: float = 0.0,
        lambda_weighted: float = 0.0,
        lambda_rel: float = 0.0,
        lambda_neg: float = 0.0,
        lambda_cap: float = 0.0,
        lambda_asym: float = 0.0,
        lambda_sparse: float = 0.0,
        lambda_delta: float = 0.0,
        lambda_tv: float = 0.0,
        lambda_robust: float = 0.0,
        lambda_int: float = 0.0,
        base_kind: str = "mse",
        huber_delta: float = 1.0,
        rel_eps: float = 1e-6,
        item_weights: Optional[torch.Tensor] = None,
        sum_relative: bool = True,
        share_divergence: str = "js",
        share_mode: str = "relu",
        share_eps: float = 1e-8,
        alloc_relative: bool = False,
        quantile_tau: float = 0.7,
        sparse_pos_weight: float = 2.0,
        sparse_spike_weight: float = 1.0,
        spike_temp: float = 10.0,
        delta_kind: str = "mse",
        tv_kind: str = "l1",
        robust_clip: float = 10.0,
        cap_mode: str = "none",
        cap_value: Optional[Union[float, torch.Tensor]] = None,
        cap_factor: float = 2.0,
        int_mode: str = "cos",
        components_axis: Optional[int] = -1,
        auto_normalize: bool = False,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_item = float(lambda_item)
        self.lambda_sum = float(lambda_sum)
        self.lambda_share = float(lambda_share)
        self.lambda_alloc = float(lambda_alloc)
        self.lambda_weighted = float(lambda_weighted)
        self.lambda_rel = float(lambda_rel)
        self.lambda_neg = float(lambda_neg)
        self.lambda_cap = float(lambda_cap)
        self.lambda_asym = float(lambda_asym)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_delta = float(lambda_delta)
        self.lambda_tv = float(lambda_tv)
        self.lambda_robust = float(lambda_robust)
        self.lambda_int = float(lambda_int)
        self.base_kind = base_kind.lower()
        self.huber_delta = float(huber_delta)
        self.rel_eps = float(rel_eps)
        if item_weights is not None:
            if not isinstance(item_weights, torch.Tensor):
                item_weights = torch.tensor(item_weights, dtype=torch.float32)
            self.register_buffer("item_weights", item_weights.detach().clone())
        else:
            self.item_weights = None
        self.sum_relative = bool(sum_relative)
        self.share_divergence = share_divergence.lower()
        self.share_mode = share_mode.lower()
        self.share_eps = float(share_eps)
        self.alloc_relative = bool(alloc_relative)
        self.quantile_tau = float(quantile_tau)
        self.sparse_pos_weight = float(sparse_pos_weight)
        self.sparse_spike_weight = float(sparse_spike_weight)
        self.spike_temp = float(spike_temp)
        self.delta_kind = delta_kind.lower()
        self.tv_kind = tv_kind.lower()
        self.robust_clip = float(robust_clip)
        self.cap_mode = cap_mode.lower()
        self.cap_value = cap_value
        self.cap_factor = float(cap_factor)
        self.int_mode = int_mode.lower()
        self.components_axis = components_axis
        self.auto_normalize = bool(auto_normalize)
        self.norm_eps = float(norm_eps)
        self.last_terms: Dict[str, torch.Tensor] = {}
        # Para armazenar stats de normalização
        self.register_buffer("_scale", torch.tensor(1.0))
        self.register_buffer("_shift", torch.tensor(0.0))

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

    def _term_weighted_item(self, yh, y):
        B, T, D = yh.shape
        base = self._elemwise(yh, y, self.base_kind)
        if self.item_weights is not None:
            w = self.item_weights.to(device=yh.device, dtype=yh.dtype).view(1, 1, D)
        else:
            s = y.abs().mean(dim=(0, 1)) + self.rel_eps
            w = (1.0 / s).view(1, 1, D)
            w = w / (w.mean() + self.rel_eps)
        return (base * w).mean()

    def _term_relative(self, yh, y):
        denom = torch.maximum(y.abs(), torch.as_tensor(self.rel_eps, device=y.device, dtype=y.dtype))
        return ((yh - y) / denom).pow(2).mean()

    def _term_sum(self, yh, y):
        # Soma ao longo da última dimensão (itens/features)
        sh, s = yh.sum(-1), y.sum(-1)

        if self.sum_relative:
            # Erro relativo (já normalizado)
            denom = torch.maximum(s.abs(), torch.as_tensor(self.rel_eps, device=y.device, dtype=y.dtype))
            return (((sh - s) / denom) ** 2).mean()

        # Para auto_normalize: dividir pelo número de timesteps para manter escala consistente
        # independente de output_chunk_length
        loss_val = self._elemwise(sh, s, self.base_kind).mean()

        # Se auto_normalize está ativo, normalizar pela dimensão temporal
        if self.auto_normalize:
            T = yh.shape[1]  # Número de timesteps
            loss_val = loss_val / T

        return loss_val

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

    def _term_neg(self, yh):
        return (F.relu(-yh) ** 2).mean()

    def _compute_cap(self, yh, y):
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
        raise ValueError(self.cap_mode)

    def _term_cap(self, yh, y):
        c = self._compute_cap(yh, y)
        if c is None:
            return torch.zeros((), device=yh.device, dtype=yh.dtype)
        return (F.relu(yh - c) ** 2).mean()

    def _term_asym(self, yh, y):
        e = y - yh
        return torch.maximum(self.quantile_tau * e, (self.quantile_tau - 1) * e).mean()

    def _term_sparse(self, yh, y):
        base = self._elemwise(yh, y, self.base_kind)
        ip = (y > 0).to(yh.dtype)
        pp = torch.sigmoid(self.spike_temp * yh)
        iz = (y.abs() <= self.rel_eps).to(yh.dtype)
        w = 1 + (self.sparse_pos_weight - 1) * ip + (self.sparse_spike_weight - 1) * (iz * pp)
        return (w * base).mean()

    def _term_delta(self, yh, y):
        return self._elemwise(
            yh[:, 1:] - yh[:, :-1], y[:, 1:] - y[:, :-1], self.delta_kind
        ).mean()

    def _term_tv(self, yh):
        d = yh[:, 1:] - yh[:, :-1]
        return d.abs().mean() if self.tv_kind == "l1" else (d * d).mean()

    def _term_robust(self, yh, y):
        return (torch.clamp(yh - y, -self.robust_clip, self.robust_clip) ** 2).mean()

    def _term_int(self, yh):
        return (1.0 - torch.cos(2.0 * math.pi * yh)).mean()

    def _normalize(self, yh, y):
        """Normaliza pred e target pela escala do target (média absoluta)."""
        # Calcula escala baseada no target (batch)
        scale = y.abs().mean() + self.norm_eps

        # Armazena para debug
        self._scale = scale.detach()

        # Normaliza
        yh_norm = yh / scale
        y_norm = y / scale

        return yh_norm, y_norm, scale

    def forward(self, y_hat, y):
        yh, y = self._to_btd(y_hat, y)

        # Auto-normalização se habilitada
        if self.auto_normalize:
            yh, y, scale = self._normalize(yh, y)

        loss = torch.zeros((), device=yh.device, dtype=yh.dtype)
        t = {}
        _pairs = [
            ("lambda_item", lambda: self._term_item(yh, y)),
            ("lambda_sum", lambda: self._term_sum(yh, y)),
            ("lambda_share", lambda: self._term_share(yh, y)),
            ("lambda_alloc", lambda: self._term_alloc(yh, y)),
            ("lambda_weighted", lambda: self._term_weighted_item(yh, y)),
            ("lambda_rel", lambda: self._term_relative(yh, y)),
            ("lambda_neg", lambda: self._term_neg(yh)),
            ("lambda_cap", lambda: self._term_cap(yh, y)),
            ("lambda_asym", lambda: self._term_asym(yh, y)),
            ("lambda_sparse", lambda: self._term_sparse(yh, y)),
            ("lambda_delta", lambda: self._term_delta(yh, y) if yh.shape[1] >= 2 else torch.zeros((), device=yh.device, dtype=yh.dtype)),
            ("lambda_tv", lambda: self._term_tv(yh) if yh.shape[1] >= 2 else torch.zeros((), device=yh.device, dtype=yh.dtype)),
            ("lambda_robust", lambda: self._term_robust(yh, y)),
            ("lambda_int", lambda: self._term_int(yh)),
        ]
        _nm = {
            "lambda_item": "item", "lambda_sum": "sum", "lambda_share": "share",
            "lambda_alloc": "alloc", "lambda_weighted": "weighted_item", "lambda_rel": "rel",
            "lambda_neg": "neg", "lambda_cap": "cap", "lambda_asym": "asym",
            "lambda_sparse": "sparse", "lambda_delta": "delta", "lambda_tv": "tv",
            "lambda_robust": "robust", "lambda_int": "int",
        }
        for lk, fn in _pairs:
            lv = getattr(self, lk, 0.0)
            if lv != 0.0:
                v = fn()
                loss = loss + lv * v
                t[_nm[lk]] = v.detach()
        self.last_terms = t
        return loss
