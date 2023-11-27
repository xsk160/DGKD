import torch
import torch.nn as nn
import torch.nn.functional as F

def dgkd_loss(logits_mlp, logits_gnn, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_mlp, target)
    other_mask = _get_other_mask(logits_gnn, target)
    pred_mlp = F.softmax(logits_mlp / temperature, dim=1)
    pred_gnn = F.softmax(logits_gnn / temperature, dim=1)
    pred_mlp = cat_mask(pred_mlp, gt_mask, other_mask)
    pred_gnn = cat_mask(pred_gnn, gt_mask, other_mask)
    log_pred_mlp = torch.log(pred_mlp)
    tcgd_loss = (
        F.kl_div(log_pred_mlp, pred_gnn, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_gnn_part2 = F.softmax(
        logits_gnn / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_mlp_part2 = F.log_softmax(
        logits_mlp / temperature - 1000.0 * gt_mask, dim=1
    )
    ncgd_loss = (
        F.kl_div(log_pred_mlp_part2, pred_gnn_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tcgd_loss + beta * ncgd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
