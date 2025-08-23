from typing import Optional

import torch

from .fused_linear_cross_entropy_function import LigerFusedLinearCrossEntropyFunction
from .fused_linear_cross_entropy_function import LigerFusedLinearCrossEntropyKLDivFunction

class LigerFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.0, #if this is very large, then new model will be almost the same with original model
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.beta = beta
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss

    def forward(self, lin_weight, _input, target, ref_hidden, ref_weight, bias=None):
        loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            ref_hidden,     #added by luke
            ref_weight,     #added by luke
            bias,
            self.beta,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss


class LigerFusedLinearCrossEntropyKLDivLoss(torch.nn.Module):
    def __init__(
        self,
        alpha: float = 0.0, #if this is very large, then new model will be almost the same with original model
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none",
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.alpha = alpha
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss


    #lin_weight: lm_head,
    # _input: shift_hidden
    #target: label
    #ref_hidden: shift_ref_hidden
    #ref_weight: ref lm_head
    def forward(self, lin_weight, _input, target, ref_hidden, ref_weight, bias=None):
        loss, z_loss = LigerFusedLinearCrossEntropyKLDivFunction.apply(
            _input,
            lin_weight,
            target,
            ref_hidden,     #added by luke
            ref_weight,     #added by luke
            bias,
            self.alpha,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss
