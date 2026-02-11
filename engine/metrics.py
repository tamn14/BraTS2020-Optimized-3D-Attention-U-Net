import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric


"""
    Metrics for BraTS segmentation evaluation.
    Includes Dice scores for WT, TC, ET regions and HD95 computation
"""
hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean_batch")

def make_metrics():
    dice_wt = DiceMetric(include_background=False, reduction="mean_batch")
    dice_tc = DiceMetric(include_background=False, reduction="mean_batch")
    dice_et = DiceMetric(include_background=False, reduction="mean_batch")
    return (dice_wt, dice_tc, dice_et)

def agg_safe(metric_obj):
    try:
        val = metric_obj.aggregate()
        if isinstance(val, torch.Tensor):
            return float(val.item())
        elif isinstance(val, (tuple, list, np.ndarray)):
            return float(val[0])
        else:
            return float(val)
    except Exception:
        return float("nan")

def safe_dice_np(pred, gt):
    dices = {}
    for c in [1, 2, 3]:
        p = (pred == c)
        g = (gt == c)
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        denom = 2*tp + fp + fn
        if g.sum() == 0:
            dices[c] = np.nan
        else:
            dices[c] = (2*tp)/denom if denom > 0 else 0.0
    return dices

def calc_region_metrics(pred_idx, gt_idx):
    out = {}
    regions = {
        "WT": ([1, 2, 3], [1, 2, 3]),
        "TC": ([1, 3], [1, 3]),
        "ET": ([3], [3]),
    }
    for r, (p_set, g_set) in regions.items():
        p_mask = np.isin(pred_idx, p_set)
        g_mask = np.isin(gt_idx, g_set)
        tp = np.logical_and(p_mask, g_mask).sum()
        fp = np.logical_and(p_mask, ~g_mask).sum()
        fn = np.logical_and(~p_mask, g_mask).sum()
        tn = np.logical_and(~p_mask, ~g_mask).sum()
        
        if g_mask.sum() == 0:
            dice = iou = sens = spec = prec = np.nan
        else:
            dice = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
            iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            
        out[f"dice_{r}"] = float(dice)
        out[f"iou_{r}"] = float(iou)
        out[f"sens_{r}"] = float(sens)
        out[f"spec_{r}"] = float(spec)
        out[f"prec_{r}"] = float(prec)
    return out

def brats_map_from_index(idx_map):
    m = np.zeros_like(idx_map, dtype=np.uint8)
    m[idx_map == 1] = 1
    m[idx_map == 2] = 2
    m[idx_map == 3] = 4 # Convert label
    return m

def compute_hd95_from_onehot(pred_onehot, true_onehot, class_indices):
    try:
        mask_pred = torch.zeros_like(pred_onehot[0:1, ...], dtype=torch.bool)
        mask_true = torch.zeros_like(true_onehot[0:1, ...], dtype=torch.bool)
        for c in class_indices:
            mask_pred = mask_pred | pred_onehot[c:c+1, ...].bool()
            mask_true = mask_true | true_onehot[c:c+1, ...].bool()
        
        if not torch.any(mask_true): return np.nan

        ypred = mask_pred.unsqueeze(0).bool()
        ytrue = mask_true.unsqueeze(0).bool()

        hd_metric.reset()
        hd_metric(ypred, ytrue)
        agg = hd_metric.aggregate()
        hd_metric.reset()

        v = agg.detach().cpu().numpy() if isinstance(agg, torch.Tensor) else np.array(agg)
        return float(v.flatten()[0]) if not (np.isnan(v).any() or np.isinf(v).any()) else np.nan
    except Exception:
        return np.nan