import torch
import torch.nn.functional as F


def _flatten_pair(predictions, targets):
    pred = predictions.view(-1).float()
    tgt = targets.view(-1).float()
    valid = ~torch.isnan(pred) & ~torch.isnan(tgt)
    return pred[valid], tgt[valid]


def compute_mae(predictions, targets):
    pred, tgt = _flatten_pair(predictions, targets)
    return torch.mean(torch.abs(pred - tgt))


def compute_mse(predictions, targets):
    pred, tgt = _flatten_pair(predictions, targets)
    return torch.mean((pred - tgt) ** 2)


def compute_rmse(predictions, targets):
    return torch.sqrt(compute_mse(predictions, targets))


def compute_mape(predictions, targets, eps: float = 1e-8):
    pred, tgt = _flatten_pair(predictions, targets)
    denom = torch.clamp(torch.abs(tgt), min=eps)
    return torch.mean(torch.abs((tgt - pred) / denom)) * 100.0


def compute_r(predictions, targets, eps: float = 1e-8):
    pred, tgt = _flatten_pair(predictions, targets)
    pred_centered = pred - torch.mean(pred)
    tgt_centered = tgt - torch.mean(tgt)
    numerator = torch.sum(pred_centered * tgt_centered)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(tgt_centered ** 2) + eps)
    return numerator / denominator


def compute_r2(predictions, targets):
    return compute_r(predictions, targets) ** 2


def compute_nse(predictions, targets, eps: float = 1e-8):
    pred, tgt = _flatten_pair(predictions, targets)
    sse = torch.sum((pred - tgt) ** 2)
    sst = torch.sum((tgt - torch.mean(tgt)) ** 2)
    return 1.0 - sse / torch.clamp(sst, min=eps)


def compute_kge(predictions, targets, eps: float = 1e-8):
    pred, tgt = _flatten_pair(predictions, targets)
    pred_mean = torch.mean(pred)
    tgt_mean = torch.mean(tgt)
    pred_std = torch.std(pred, unbiased=False)
    tgt_std = torch.std(tgt, unbiased=False)

    r = compute_r(pred, tgt, eps=eps)
    alpha = pred_std / (tgt_std + eps)
    beta = (pred_mean + eps) / (tgt_mean + eps)
    return 1.0 - torch.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def huber_loss(predictions, targets, delta: float = 1.0):
    """Huber 损失，兼容不同版本的 PyTorch。"""
    pred, tgt = _flatten_pair(predictions, targets)

    if hasattr(torch.nn, "HuberLoss"):
        return torch.nn.HuberLoss(delta=delta)(pred, tgt)

    try:
        return F.smooth_l1_loss(pred, tgt, beta=delta, reduction="mean")
    except TypeError:
        diff = pred - tgt
        abs_diff = diff.abs()
        quadratic = torch.clamp(abs_diff, max=delta)
        linear = abs_diff - quadratic
        return (0.5 * quadratic ** 2 + delta * linear).mean()
