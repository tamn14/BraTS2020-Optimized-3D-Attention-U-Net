import csv
import os


"""
    Logger for BraTS segmentation training and evaluation metrics.
    Logs epoch number, learning rate, training loss, mean Dice score,
    individual Dice scores for WT, TC, ET, and HD95 scores for WT, TC, ET.
    
"""
class BraTSLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "LR", "TrainLoss", "MeanDice", "Dice_WT", "Dice_TC", "Dice_ET"])

    def log_epoch(self, epoch, lr, loss, mean_dice, dice_scores):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{lr:.6e}", f"{loss:.6f}", f"{mean_dice:.4f}",
                f"{dice_scores[0]:.4f}", f"{dice_scores[1]:.4f}", f"{dice_scores[2]:.4f}"
            ])