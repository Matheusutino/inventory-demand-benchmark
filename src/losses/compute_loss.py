import torch
from einops import rearrange
from typing import Optional, Union
from src.losses.loss import InventoryDemandLoss

def custom_compute_loss(
    self,
    quantile_preds,         # (B, num_quantiles, T_out)
    future_target,          # (B, T_pred)
    future_target_mask,
    patched_future_covariates_mask,
    loc_scale,
    num_output_patches,
    # Loss weights
    lambda_item: float = 1.0,
    lambda_sum: float = 0.0,
    lambda_share: float = 0.0,
    lambda_alloc: float = 0.0,
    lambda_weighted: float = 0.0,
    lambda_rel: float = 0.0,
    lambda_neg: float = 0.1,
    lambda_cap: float = 0.0,
    lambda_asym: float = 0.5,
    lambda_sparse: float = 0.3,
    lambda_delta: float = 0.1,
    lambda_tv: float = 0.0,
    lambda_robust: float = 0.0,
    lambda_int: float = 0.0,
    # Loss parameters
    base_kind: str = "huber",
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
    components_axis: Optional[int] = None,
):
    batch_size = future_target.shape[0]
    output_patch_size = self.chronos_config.output_patch_size

    # --- 1. Normaliza o target (igual ao original) ---
    future_target, _ = self.instance_norm(future_target, loc_scale)
    future_target = future_target.unsqueeze(1).to(self.device)  # (B, 1, T)
    future_target_mask = (
        future_target_mask.unsqueeze(1).to(self.device)
        if future_target_mask is not None
        else ~torch.isnan(future_target)
    )
    future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

    # --- 2. Padding se necessário (igual ao original) ---
    if quantile_preds.shape[-1] > future_target.shape[-1]:
        pad_shape = (*future_target.shape[:-1], quantile_preds.shape[-1] - future_target.shape[-1])
        future_target = torch.cat([future_target, torch.zeros(pad_shape).to(future_target)], dim=-1)
        future_target_mask = torch.cat(
            [future_target_mask, torch.zeros(pad_shape).to(future_target_mask)], dim=-1
        )

    # --- 3. Monta a máscara combinada (igual ao original) ---
    inv_future_covariate_mask = 1 - rearrange(
        patched_future_covariates_mask,
        "b n p -> b 1 (n p)",
        b=batch_size,
        n=num_output_patches,
        p=output_patch_size,
    )
    loss_mask = (future_target_mask.float() * inv_future_covariate_mask)  # (B, 1, T)

    # --- 4. Extrai a mediana como point forecast ---
    median_idx = len(self.chronos_config.quantiles) // 2
    point_pred = quantile_preds[:, median_idx:median_idx+1, :]  # (B, 1, T)

    # Aplica máscara (zera timesteps inválidos)
    point_pred = point_pred * loss_mask
    future_target = future_target * loss_mask

    # --- 5. Reshape para (B, T, D) esperado pelo InventoryDemandLoss ---
    # D = 1 para univariado; para multivariado ajuste conforme necessário
    y_hat = point_pred.permute(0, 2, 1)   # (B, T, 1)
    y     = future_target.permute(0, 2, 1) # (B, T, 1)

    # --- 6. Computa a loss customizada ---
    loss_fn = InventoryDemandLoss(
        lambda_item=lambda_item,
        lambda_sum=lambda_sum,
        lambda_share=lambda_share,
        lambda_alloc=lambda_alloc,
        lambda_weighted=lambda_weighted,
        lambda_rel=lambda_rel,
        lambda_neg=lambda_neg,
        lambda_cap=lambda_cap,
        lambda_asym=lambda_asym,
        lambda_sparse=lambda_sparse,
        lambda_delta=lambda_delta,
        lambda_tv=lambda_tv,
        lambda_robust=lambda_robust,
        lambda_int=lambda_int,
        base_kind=base_kind,
        huber_delta=huber_delta,
        rel_eps=rel_eps,
        item_weights=item_weights,
        sum_relative=sum_relative,
        share_divergence=share_divergence,
        share_mode=share_mode,
        share_eps=share_eps,
        alloc_relative=alloc_relative,
        quantile_tau=quantile_tau,
        sparse_pos_weight=sparse_pos_weight,
        sparse_spike_weight=sparse_spike_weight,
        spike_temp=spike_temp,
        delta_kind=delta_kind,
        tv_kind=tv_kind,
        robust_clip=robust_clip,
        cap_mode=cap_mode,
        cap_value=cap_value,
        cap_factor=cap_factor,
        int_mode=int_mode,
        components_axis=components_axis,
    )
    loss = loss_fn(y_hat, y)

    return loss