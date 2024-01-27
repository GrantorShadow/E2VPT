#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils import logging
logger = logging.get_logger("visual_prompt")


class SigmoidLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def is_local(self):
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        loss = F.cross_entropy(logits, targets, weight, reduction="none")

        return torch.sum(loss) / targets.shape[0]

# Todo - 2024
# make new negative softmax function
# check if we need to apply any regularization or any kinda limiter since loss can go haywire

class NegativeCrossEntropyLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(NegativeCrossEntropyLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        negative_loss = -(F.cross_entropy(logits, targets, weight, reduction="none")) # negate the loss

        # TODO check if we need to apply any regularization or any kinda limiter since loss can go haywire
        return negative_loss

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        return self.loss(pred_logits, targets, per_cls_weights)

    # Next TODO
    # figure out a way to switch between losses
    # also steps might need to be figured out

# adversarial_IVC loss Shaunak Code
def NegativeCrossEntropyWithThresholdLoss(SigmoidLoss):
    def __init__(self, threshold, cfg=None):
        super(NegativeCrossEntropyWithThresholdLoss, self).__init__()
        self.threshold = threshold

    def loss(self, logits, targets, per_cls_weights):
        # Compute the standard cross-entropy loss
        weight = torch.tensor(per_cls_weights, device=logits.device)
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')

        # Negate the cross-entropy loss
        neg_ce_loss = -ce_loss

        # Apply the threshold to the negative cross-entropy loss
        threshold_tensor = torch.full_like(neg_ce_loss, fill_value=self.threshold)
        loss_with_threshold = torch.max(neg_ce_loss, threshold_tensor)

        # Return the mean of the loss with threshold
        return loss_with_threshold.mean()

    def forward(self, pred_logits, targets, per_cls_weights):
        # Forward pass to compute the loss
        return self.loss(pred_logits, targets, per_cls_weights)


LOSS = {
    "softmax": SoftmaxLoss,
    "negative_ce_with_threshold": NegativeCrossEntropyWithThresholdLoss,
    "negative_ce": NegativeCrossEntropyLoss
}


def build_loss(cfg, mode=None):

    # if mode == "adversarial_IVC":
    #     loss_fn = LOSS["negative_ce_with_threshold"]
    #     return loss_fn(cfg)
    # elif mode == "defense":
    #     loss_fn = LOSS["softmax"]
    #     return loss_fn(cfg)

    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    loss_fn = LOSS[loss_name]
    if not loss_fn:
        return None
    else:
        return loss_fn(cfg)

# def adv_build_loss(cfg, mode):
#
#         assert loss_name in LOSS, \
#             f'loss name {loss_name} is not supported'
#     loss_fn = LOSS[loss_name]
#     if not loss_fn:
#         return None
#     else:
#         return loss_fn(cfg)
