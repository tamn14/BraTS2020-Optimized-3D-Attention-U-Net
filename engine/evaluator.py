import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from .metrics import make_metrics, agg_safe


class BraTSEvaluator:
    def __init__(self, device, val_roi_size, sw_batch_size, out_channels):
        self.device = device
        self.val_roi_size = val_roi_size
        self.sw_batch_size = sw_batch_size
        self.out_channels = out_channels

    def validate(self, model, loader):
        model.eval()

        dice_wt_metric, dice_tc_metric, dice_et_metric = make_metrics()
        dice_wt_metric.reset()
        dice_tc_metric.reset()
        dice_et_metric.reset()

        with torch.no_grad():
            for val_batch in tqdm(loader, desc="Validation"):
                val_images = val_batch["image"].to(self.device)
                val_labels = val_batch["label"].to(self.device).long()

                # === sliding window inference ===
                val_logits = sliding_window_inference(
                    val_images,
                    self.val_roi_size,
                    self.sw_batch_size,
                    model,
                    overlap=0.5,
                    mode="gaussian"
                )

                # === argmax ===
                val_pred = torch.argmax(val_logits, dim=1, keepdim=True)

                # === manual one-hot===
                val_pred = torch.nn.functional.one_hot(
                    val_pred.squeeze(1),
                    num_classes=self.out_channels
                ).permute(0, 4, 1, 2, 3).float()

                val_label = torch.nn.functional.one_hot(
                    val_labels.squeeze(1),
                    num_classes=self.out_channels
                ).permute(0, 4, 1, 2, 3).float()

                # === WT (1,2,3) ===
                y_pred_wt = val_pred[:, 1:4].sum(1, keepdim=True)
                y_true_wt = val_label[:, 1:4].sum(1, keepdim=True)
                dice_wt_metric(y_pred=y_pred_wt, y=y_true_wt)

                # === TC (1,3) ===
                y_pred_tc = val_pred[:, [1, 3]].sum(1, keepdim=True)
                y_true_tc = val_label[:, [1, 3]].sum(1, keepdim=True)
                dice_tc_metric(y_pred=y_pred_tc, y=y_true_tc)

                # === ET (3) ===
                y_pred_et = val_pred[:, 3:4]
                y_true_et = val_label[:, 3:4]
                dice_et_metric(y_pred=y_pred_et, y=y_true_et)

        # === aggregate ===
        dice_wt = agg_safe(dice_wt_metric)
        dice_tc = agg_safe(dice_tc_metric)
        dice_et = agg_safe(dice_et_metric)

        dices = [d for d in (dice_wt, dice_tc, dice_et) if not np.isnan(d)]
        mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0

        return {
            "mean_dice": mean_dice,
            "dice_wt": dice_wt,
            "dice_tc": dice_tc,
            "dice_et": dice_et,
        }
