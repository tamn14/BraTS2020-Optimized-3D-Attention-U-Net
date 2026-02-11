import argparse
import os
import copy
import gc  
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.transforms import AsDiscrete
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import DiceCELoss

import config.config as cfg
from core.dirs import create_directories
from core.runtime import init_runtime, get_device
from data.brats_index import build_brats_index
from data.splits import make_kfold_splits
from data.transforms import get_fine_transforms
from models.factory import build_model

from engine.trainer import BraTSTrainer
from engine.evaluator import BraTSEvaluator
from engine.predictor import BraTSPredictor
from engine.visualizer import BraTSVisualizer
from engine.logger import BraTSLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized 3D Attention U-Net")
    parser.add_argument("--model_name", type=str, default=cfg.MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=cfg.FINE_EPOCHS)
    parser.add_argument("--kfold", type=int, default=cfg.N_SPLITS)
    parser.add_argument("--skip_train", action="store_true", help="Skip training and run inference only")
    return parser.parse_args()

def run_training_fold(fold_num, train_files, val_files, device):
    print(f"\n{'='*10} TRAINING FOLD {fold_num} {'='*10}\n")
    
    # 1. Data Setup
    train_tf, val_tf, _ = get_fine_transforms(patch_size=cfg.PATCH_SIZE)
    train_ds = CacheDataset(train_files, train_tf, cache_rate=0.1, num_workers=4)
    val_ds = CacheDataset(val_files, val_tf, cache_rate=0.1, num_workers=2)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.FINE_BATCH_SIZE, shuffle=True, 
                              num_workers=4, collate_fn=list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=2, collate_fn=list_data_collate)

    # 2. Model & Optimizer Setup
    model = build_model(cfg).to(device)
    ema_model = copy.deepcopy(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.OPT_LR, weight_decay=1e-5)
    
    t_max = max(1, cfg.FINE_EPOCHS - cfg.WARMUP_EPOCHS)
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=5e-6)

    class_weights = torch.tensor([0.1, 1.0, 1.0, 3.0]).to(device)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True,
                           weight=class_weights, lambda_dice=1.0, lambda_ce=0.2).to(device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP)

    # 3. Engines
    logger = BraTSLogger(Path(cfg.LOG_DIR) / f"log_fold_{fold_num}.csv")
    
    trainer = BraTSTrainer(model, ema_model, optimizer, criterion, scaler, device,
                           cfg.ACCUMULATION_STEPS, cfg.EMA_DECAY, cfg.CLIP_GRAD_NORM,
                           scheduler, cfg.WARMUP_EPOCHS, cfg.OPT_LR)
    
    evaluator = BraTSEvaluator(device, cfg.VAL_ROI_SIZE, cfg.SW_BATCH_INFER, cfg.OUT_CHANNELS_3D)
    
    best_dice = -1.0
    best_model_path = Path(cfg.MODEL_DIR) / f"best_model_fold_{fold_num}.pth"

    # 4. Loop
    for epoch in range(cfg.FINE_EPOCHS):
        # Train
        avg_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        ema_model.eval()
        with torch.no_grad():
            metrics = evaluator.validate(ema_model, val_loader)
        mean_dice = metrics["mean_dice"]
        
        # Log & Save
        current_lr = optimizer.param_groups[0]["lr"]
        logger.log_epoch(epoch + 1, optimizer.param_groups[0]["lr"], avg_loss, mean_dice, 
                         [metrics["dice_wt"], metrics["dice_tc"], metrics["dice_et"]])
        
        print(f"Fold {fold_num} | Ep {epoch+1:03d} | Loss {avg_loss:.4f} | MeanDice {mean_dice:.4f} | dice_wt {metrics["dice_wt"]:.4f} | dice_tc {metrics["dice_tc"]:.4f} | dice_et {metrics["dice_et"]:.4f}  ")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(ema_model.state_dict(), best_model_path)
            print(f" >> Saved New Best EMA Model (Dice: {best_dice:.4f})")
    
    del model, ema_model, optimizer, scheduler, train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

def main():
    args = parse_args()
    cfg.MODEL_NAME = args.model_name
    cfg.FINE_EPOCHS = args.epochs
    cfg.N_SPLITS = args.kfold

    init_runtime(cfg.RANDOM_STATE)
    device = get_device()
    create_directories(cfg.OUTPUT_DIR, cfg.MODEL_DIR, cfg.LOG_DIR, cfg.PRED_DIR, cfg.VIS_DIR)

    all_files_list = build_brats_index(cfg.DATA_PATH)
    fold_splits = make_kfold_splits(len(all_files_list), cfg.N_SPLITS, cfg.RANDOM_STATE)

    post_pred = AsDiscrete(argmax=True, to_onehot=cfg.OUT_CHANNELS_3D)
    post_lab = AsDiscrete(to_onehot=cfg.OUT_CHANNELS_3D)

    # === Stage 1: TRAINING ===
    if not args.skip_train:
        for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
            train_files = [all_files_list[i] for i in train_idx]
            val_files = [all_files_list[i] for i in val_idx]
            run_training_fold(fold_idx + 1, train_files, val_files, device)
    else:
        print("\n >>> SKIP_TRAIN: ON. Loading existing models for inference...")

    # === Stage 2: GLOBAL EVALUATION ===
    all_fold_results = []
    total_cm_global = np.zeros((cfg.OUT_CHANNELS_3D, cfg.OUT_CHANNELS_3D), dtype=np.int64)
    output_dir = Path(cfg.OUTPUT_DIR)
    visualizer = BraTSVisualizer(output_dir) 
    
    predictor = BraTSPredictor(device, cfg, post_pred, post_lab)

    print("\n --- STARTING GLOBAL INFERENCE ---")
    for fold_idx, (_, val_idx) in enumerate(fold_splits):
        fold_num = fold_idx + 1
        val_files = [all_files_list[i] for i in val_idx]
        
        # Loader for Validation (No Cache)
        _, val_tf, _ = get_fine_transforms(patch_size=cfg.PATCH_SIZE)
        val_ds = CacheDataset(val_files, val_tf, cache_rate=0.0) 
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

        # Load Model
        model_path = Path(cfg.MODEL_DIR) / f"best_model_fold_{fold_num}.pth"
        if not model_path.exists():
            print(f"Warning: Model file for fold {fold_num} not found. Skipping.")
            continue
            
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Predict
        with torch.no_grad():
            fold_res, fold_cm = predictor.run_inference(model, val_loader, fold_num, visualizer)
            all_fold_results.extend(fold_res)
            total_cm_global += fold_cm

        del model, val_loader
        gc.collect() 
        torch.cuda.empty_cache()

    # === Stage 3: AGGREGATION & VISUALIZATION ===
    if not all_fold_results:
        print("Error: No inference results gathered.")
        return

    df_all = pd.DataFrame(all_fold_results)
    output_dir = Path(cfg.OUTPUT_DIR)
    df_all.to_csv(output_dir / "final_kfold_all_case_metrics.csv", index=False)

    predictor.save_summary_stats(df_all, output_dir / "final_kfold_summary_stats.csv")

    log_files = [Path(cfg.LOG_DIR) / f"log_fold_{i+1}.csv" for i in range(cfg.N_SPLITS)]
    
    if all(f.exists() for f in log_files):
        visualizer.plot_training_results(log_files)
        
    visualizer.plot_confusion_matrix(total_cm_global, len(df_all))
    visualizer.plot_dice_boxplot(df_all)
    

    print(f"\n Pipeline Finished. Output: {output_dir}")

if __name__ == "__main__":
    main()