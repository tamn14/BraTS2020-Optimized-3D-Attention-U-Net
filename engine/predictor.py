import os
import torch
import numpy as np
import pandas as pd
import traceback
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from .metrics import (
    calc_region_metrics, 
    safe_dice_np, 
    compute_hd95_from_onehot, 
    brats_map_from_index
)

class BraTSPredictor:
    """
        Predictor for BraTS segmentation evaluation.
        Runs inference on validation/test dataset, computes metrics and aggregates results across folds.
        
    """
    def __init__(self, device, config, post_pred, post_lab):
        self.device = device
        self.cfg = config
        self.post_pred = post_pred
        self.post_lab = post_lab
        self.cm_labels = [0, 1, 2, 4]

    def run_inference(self, model, loader, fold_num, visualizer=None):
        model.eval()
        fold_results = []
        total_cm = np.zeros((4, 4), dtype=np.int64)
        
        from sklearn.metrics import confusion_matrix

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Predicting Fold {fold_num}"):
                case_id = batch.get("case_id", ["unknown"])[0]
                try:
                    imgs = batch["image"].to(self.device)
                    label = batch["label"]

                    logits = sliding_window_inference(
                        imgs, 
                        self.cfg.VAL_ROI_SIZE, 
                        self.cfg.SW_BATCH_INFER, 
                        lambda x: model(x),
                        overlap=0.5, 
                        mode="gaussian"
                    )
                    
                    outs = decollate_batch(logits, detach=True)
                    labs = decollate_batch(label, detach=True)

                    pred_onehot = self.post_pred(outs[0].to("cpu"))
                    true_onehot = self.post_lab(labs[0].to("cpu"))

                    pred_idx = torch.argmax(pred_onehot, dim=0).numpy().astype(np.uint8)
                    true_idx = torch.argmax(true_onehot, dim=0).numpy().astype(np.uint8)

                    region_stats = calc_region_metrics(pred_idx, true_idx)
                    d_np = safe_dice_np(pred_idx, true_idx)
                    
                    hd95_wt = compute_hd95_from_onehot(pred_onehot, true_onehot, [1, 2, 3])
                    hd95_tc = compute_hd95_from_onehot(pred_onehot, true_onehot, [1, 3])
                    hd95_et = compute_hd95_from_onehot(pred_onehot, true_onehot, [3])

                    brats_pred = brats_map_from_index(pred_idx).flatten()
                    brats_true = brats_map_from_index(true_idx).flatten()
                    total_cm += confusion_matrix(brats_true, brats_pred, labels=self.cm_labels)

                    rec = {
                        "case_id": case_id,
                        "fold": fold_num,
                        **region_stats,  # includes dice_WT, dice_TC, dice_ET, etc.
                        "hd95_WT": hd95_wt,
                        "hd95_TC": hd95_tc,
                        "hd95_ET": hd95_et,
                        "dice_NCR": d_np[1],
                        "dice_ED": d_np[2],
                        "dice_ET_channel": d_np[3]
                    }
                    fold_results.append(rec)
                    
                    if visualizer is not None:
                        flair_vol = imgs[0, 0].cpu().numpy() 
                        slice_idx = visualizer.pick_best_slice(true_idx)
                        
                        f_slice = flair_vol[:, :, slice_idx]
                        gt_slice = true_idx[:, :, slice_idx]
                        pr_slice = pred_idx[:, :, slice_idx]
                        
                        dice_scores = (rec["dice_WT"], rec["dice_TC"], rec["dice_ET"])
                        
                        visualizer.save_mri_prediction(
                            f_slice, gt_slice, pr_slice, 
                            case_id, fold_num, dice_scores
                        )

                except Exception as e:
                    print(f"Error evaluating case {case_id} (fold {fold_num}): {e}")
                    traceback.print_exc()
                    fold_results.append({"case_id": case_id, "fold": fold_num, "dice_WT": np.nan})

        return fold_results, total_cm

    def save_summary_stats(self, df_all, output_path):
        primary_cols = ["dice_WT", "dice_TC", "dice_ET", "hd95_WT", "hd95_TC", "hd95_ET"]
        out = []
        for c in primary_cols:
            if c not in df_all.columns: continue
            arr = df_all[c].dropna().values
            n = len(arr)
            if n == 0: continue
            
            mean = np.nanmean(arr)
            std = np.nanstd(arr, ddof=1) if n > 1 else 0.0
            se = std / np.sqrt(n) if n > 1 else 0.0
            
            out.append({
                "metric": c, 
                "n": n, 
                "mean": mean, 
                "std": std,
                "median": np.nanmedian(arr),
                "ci95_low": mean - 1.96 * se,
                "ci95_high": mean + 1.96 * se
            })
        
        summary_df = pd.DataFrame(out)
        summary_df.to_csv(output_path, index=False)
        return summary_df