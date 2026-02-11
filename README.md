# 🧠 Optimized 3D Attention U-Net for Glioma Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Project-blue?style=for-the-badge)](https://monai.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-ffd43b?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Note:** This repository focuses on documenting the **model architecture** and the overall **processing/evaluation pipeline**. It is designed as a reference implementation rather than a complete training framework. Researchers are encouraged to adapt the training loops to their specific requirements.

---

## 📖 1. Introduction

This project presents an **optimized implementation of 3D Attention U-Net** for multi-class glioma segmentation on the **BraTS 2020** dataset.

Rather than introducing unnecessary architectural complexity, we focus on improving optimization behavior and training stability. Our ablation experiments (3-fold, 50 epochs) demonstrate that these refinements yield smoother convergence dynamics, particularly for the challenging Tumor Core (TC) region.

### 🚀 Key Refinements

We introduce three strategic modifications to the standard architecture:

1.  **Instance Normalization:** Replaced Batch Normalization to ensure stable feature scaling under micro-batch ($N=1$) 3D training.
2.  **LeakyReLU Activation:** Adopted to maintain gradient flow in low-activation regimes, which are common in hypointense MRI regions.
3.  **Calibrated Attention Gates:** Implemented a positive bias shift ($\beta \in [1.5, 2.0]$) to improve early feature transmission and mitigate premature gate saturation.

---

## 🛠️ 2. Method: The "Open-Gate" Strategy

The core innovation lies in our handling of the **AttentionBlock**. This block filters skip-connection features, allowing the decoder to focus on task-relevant information.

### The Problem: Cold-Start

Standard attention gates often suffer from a "cold-start" problem where encoder features are prematurely suppressed early in training due to low initial sigmoid activations.

### The Solution: Weighted Bias Initialization

To improve early training stability, we apply **Weighted Bias Initialization** in the `InstanceNorm` layer of the attention transformation:

- **Configuration:** We set the affine bias **$\beta = 2.0$** (default is 0).
- **Effect:** This shifts the sigmoid output to an initially high attention value (~0.88), keeping the gate in a "near-open" state at the start.
- **Benefit:** This allows stronger error signals to backpropagate to earlier layers immediately, mitigating the vanishing gradient problem in the attention mechanism.

---

## 📊 3. Performance

### Quantitative Results (5-Fold Cross-Validation)

The model achieves high segmentation accuracy across all tumor sub-regions, with the Whole Tumor (WT) class exceeding **0.90 DSC**.

| Metric               | Class  |      Mean ± Std      |   Median   |
| :------------------- | :----: | :------------------: | :--------: |
| **DSC (Dice Score)** | **WT** | **0.9009 ± 0.0842**  | **0.9255** |
|                      |   TC   |   0.8505 ± 0.1580    |   0.9178   |
|                      |   ET   |   0.8103 ± 0.1791    |   0.8718   |
| **HD95 (mm)**        | **WT** | **8.1907 ± 13.5499** | **3.7416** |
|                      |   TC   |   7.2373 ± 12.8598   |   3.0000   |
|                      |   ET   |   5.6587 ± 11.2767   |   1.7320   |

### 🖼️ Visualization

Representative slices from validation folds showcasing the segmentation quality. The model exhibits consistent morphology capture across different patients.

<p align="center">
  <img src="kfold_results/visualizations/fold1_BraTS20_Training_001.png" width="30%" alt="Fold 1 Result"/>
  <img src="kfold_results/visualizations/fold2_BraTS20_Training_007.png" width="30%" alt="Fold 2 Result"/>
  <img src="kfold_results/visualizations/fold5_BraTS20_Training_002.png" width="30%" alt="Fold 5 Result"/>
</p>

---

## ⚙️ 4. Installation & Setup

### 1. Dependencies

Ensure you have the required libraries installed.

```bash
pip install --upgrade monai[all] torch nibabel numpy pandas matplotlib seaborn scikit-learn tqdm
2. Dataset SetupDownload: Get the BraTS 2020 dataset from Kaggle:👉 Download HereConfigure: Update the path in config/config.py.Open config/config.py and modify DATA_PATH:Python# Old Path
DATA_PATH = Path("/root/Seg/MICCAI_BraTS2020_TrainingData")

# New Path (Example)
DATA_PATH = Path("/your/local/path/MICCAI_BraTS2020_TrainingData")
🏃‍♂️ 5. UsageThe training script supports various configurations via command-line arguments.Available Modelsoptimized_unet3d: The proposed model with stability improvements.attention_unet3d: The baseline attention U-Net model.Training CommandRun the training loop with cross-validation: train.py --model_name optimized_unet3d --epochs 100 --kfold 5
Inference OnlySkip the training phase and run inference using pre-trained weights: train.py --model_name optimized_unet3d --skip_train
Arguments ReferenceArgumentDescription--model_nameArchitecture to use (optimized_unet3d or attention_unet3d)--epochsNumber of training epochs--kfoldNumber of cross-validation folds--skip_trainFlag to skip training and run inference only
```
