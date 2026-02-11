import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

"""
Visualizer for BraTS outputs.
Minimal fixes:
 - consistent attributes (self.output_dir, self.plots_dir, self.vis_dir)
 - correct color mapping (1=NCR red, 2=ED yellow, 3=ET teal)
 - pick_best_slice expects a volume; save_mri_prediction expects a slice index argument
 - save paths use plots_dir / vis_dir consistently
"""

class BraTSVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        self.vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def smooth_curve(self, values, weight=0.9):
        smoothed = []
        if len(values) == 0:
            return values
        last = float(values[0])
        for v in values:
            last = last * weight + (1 - weight) * float(v)
            smoothed.append(last)
        return smoothed

    def plot_training_results(self, log_file_paths):
        summary_rows = []

        for fold_num, file_path in enumerate(log_file_paths):
            fold_label = f"Fold {fold_num + 1}"

            if not os.path.exists(file_path):
                print(f"[Warning] Log not found for {fold_label}: {file_path}")
                continue

            df = pd.read_csv(file_path)
            if df.empty:
                print(f"[Warning] Empty log at {fold_label}. Skipped.")
                continue

            epochs = df['Epoch']

            if "TrainLoss" in df.columns:
                plt.figure(figsize=(12, 6), dpi=150)
                y_raw = df["TrainLoss"].values
                y_smooth = self.smooth_curve(y_raw)
                plt.plot(epochs, y_smooth, linewidth=2.5, label="Train Loss")
                plt.title(f"{fold_label} – Training Loss", fontsize=16)
                plt.xlabel("Epoch", fontsize=14); plt.ylabel("Loss", fontsize=14)
                plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f"fold_{fold_num+1}_train_loss.png"))
                plt.close()
                print(f" Saved Train Loss plot for {fold_label}")

            # Dice plots (MeanDice or Dice_WT/...)
            dice_cols = []
            if "MeanDice" in df.columns:
                dice_cols.append("MeanDice")
            for c in ["Dice_WT", "Dice_TC", "Dice_ET"]:
                if c in df.columns:
                    dice_cols.append(c)

            if dice_cols:
                plt.figure(figsize=(12, 6), dpi=150)
                for col in dice_cols:
                    y = df[col].values
                    y_smooth = self.smooth_curve(y, weight=0.85)
                    plt.plot(epochs, y_smooth, linewidth=2.3, label=col)
                plt.title(f"{fold_label} – Validation Dice Scores", fontsize=16)
                plt.xlabel("Epoch", fontsize=14); plt.ylabel("Dice", fontsize=14)
                plt.ylim(0, 1.0); plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f"fold_{fold_num+1}_dice_scores.png"))
                plt.close()
                print(f"Saved Dice plots for {fold_label}")

            if "MeanIoU" in df.columns or "MeanSens" in df.columns:
                plt.figure(figsize=(12, 6), dpi=150)
                if "MeanIoU" in df.columns:
                    plt.plot(epochs, self.smooth_curve(df["MeanIoU"].values), linewidth=2.2, label="IoU")
                if "MeanSens" in df.columns:
                    plt.plot(epochs, self.smooth_curve(df["MeanSens"].values), linewidth=2.2, linestyle="--", label="Sensitivity")
                plt.title(f"{fold_label} – IoU / Sensitivity", fontsize=16)
                plt.xlabel("Epoch", fontsize=14); plt.ylabel("Value", fontsize=14)
                plt.grid(True, linestyle="--", alpha=0.5); plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, f"fold_{fold_num+1}_iou_sens.png"))
                plt.close()

            best_epoch = int(df['MeanDice'].idxmax()) if 'MeanDice' in df else "NA"
            best_dice = df['MeanDice'].max() if 'MeanDice' in df else "NA"
            summary_rows.append({
                "Fold": fold_num + 1,
                "BestEpoch": best_epoch,
                "BestMeanDice": best_dice
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(self.plots_dir, "training_summary.csv"), index=False)
        return summary_df

    def plot_confusion_matrix(self, total_cm, n_cases):
        cm_norm = total_cm.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        plt.figure(figsize=(6, 5), dpi=150)
        sns.heatmap(cm_norm, annot=total_cm, fmt='', cmap='Blues',
                    xticklabels=["BG(0)","NCR(1)","ED(2)","ET(4)"],
                    yticklabels=["BG(0)","NCR(1)","ED(2)","ET(4)"])
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title(f"Confusion Matrix (Total Cases: {n_cases})")
        plt.tight_layout()
        path = os.path.join(self.plots_dir, "confusion_matrix_final.png")
        plt.savefig(path); plt.close()
        print(f"Saved confusion matrix: {path}")

    def plot_dice_boxplot(self, df_all):
        dice_plot_df = df_all.melt(id_vars=["case_id"],
                                   value_vars=[c for c in ["dice_WT","dice_TC","dice_ET"] if c in df_all.columns],
                                   var_name="Region", value_name="Dice")
        plt.figure(figsize=(8, 6), dpi=150)
        sns.boxplot(x="Region", y="Dice", data=dice_plot_df)
        plt.title("Dice Distribution Across All Folds")
        plt.tight_layout()
        path = os.path.join(self.plots_dir, "boxplot_final_dice.png")
        plt.savefig(path); plt.close()
        print(f"Saved boxplot: {path}")

    def colorize_label(self, mask_2d):
        """Correct color mapping consistent with project:
           1 => NCR (red), 2 => ED (yellow), 3 => ET (teal)"""
        h, w = mask_2d.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[mask_2d == 1] = [1.0, 0.0, 0.0]   # NCR/NET: Red
        rgb[mask_2d == 2] = [1.0, 1.0, 0.0]   # ED: Yellow
        rgb[mask_2d == 3] = [0.0, 0.7, 1.0]   # ET: Teal
        return rgb

    def pick_best_slice(self, label_vol):
        """Find slice index with largest ET area, else WT area. Expects 3D label_vol (H,W,D)."""
        if label_vol.ndim != 3:
            raise ValueError("pick_best_slice expects 3D label volume (H,W,D)")
        et_areas = (label_vol == 3).sum(axis=(0, 1))
        if et_areas.sum() > 20:
            return int(np.argmax(et_areas))
        wt_areas = (label_vol > 0).sum(axis=(0, 1))
        return int(np.argmax(wt_areas))

    def save_mri_prediction(self, flair_slice, gt_slice, pr_slice, case_id, fold_num, dice_scores, alpha=0.5, out_fname=None):
        """Save overlay for a given 2D slice. flair_slice, gt_slice, pr_slice are 2D arrays."""
        gt_col = self.colorize_label(gt_slice)
        pr_col = self.colorize_label(pr_slice)
        d_wt, d_tc, d_et = dice_scores

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        axs[0].imshow(np.rot90(flair_slice), cmap="gray")
        axs[0].set_title(f"FLAIR - Case: {case_id}")
        axs[0].axis("off")

        axs[1].imshow(np.rot90(flair_slice), cmap="gray")
        axs[1].imshow(np.rot90(gt_col), alpha=alpha)
        axs[1].set_title("Ground Truth (NCR / ED / ET)")
        axs[1].axis("off")

        axs[2].imshow(np.rot90(flair_slice), cmap="gray")
        axs[2].imshow(np.rot90(pr_col), alpha=alpha)
        axs[2].set_title(f"Prediction\nWT:{d_wt:.3f} TC:{d_tc:.3f} ET:{d_et:.3f}")
        axs[2].axis("off")

        p_ncr = mpatches.Patch(color=[1.0, 0.0, 0.0], label="NCR/NET")
        p_ed  = mpatches.Patch(color=[1.0, 1.0, 0.0], label="ED")
        p_et  = mpatches.Patch(color=[0.0, 0.7, 1.0], label="ET")
        fig.legend(handles=[p_ncr, p_ed, p_et], loc="lower center", ncol=3)

        info = f"Fold {fold_num} | Case {case_id}"
        fig.text(0.01, 0.01, info, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        if out_fname is None:
            out_fname = f"fold{fold_num}_{case_id}.png"
        out_path = os.path.join(self.vis_dir, out_fname)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", out_path)
