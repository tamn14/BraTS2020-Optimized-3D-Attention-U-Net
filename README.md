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

### The Problem: Suboptimal Gradient Flow

In standard **Attention Gates (AGs)**, zero-bias initialization often leads to low initial sigmoid activations. This can cause:

- **Feature Suppression:** Important encoder features are prematurely attenuated before the model learns to identify relevant spatial regions.
- **Convergence Instability:** The initial "closed-gate" state restricts the flow of gradients, making it difficult for the model to exit local minima in early epochs, particularly in high-variance regions like the Tumor Core.

---

### The Solution: Weighted Bias Initialization

To ensure more stable convergence dynamics, we shift the initial state toward a more **permissive configuration**:

- **Mechanism:** By setting the affine bias $\beta \in [1.5, 2.0]$, the initial gate activation is shifted to $\alpha \approx 0.82 \text{--} 0.88$.
- **Result:** This "near-open" state ensures that encoder features pass through almost freely during the initial phase.
- **Optimization:** Instead of struggling to "open" the gate, the model focuses on refining the attention map, leading to a much smoother loss trajectory and sustained performance gains in later epochs.

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

### 4.1. Dependencies

Install the required libraries using pip:

```bash
pip install --upgrade monai[all] torch nibabel numpy pandas matplotlib seaborn scikit-learn tqdm
```

### 4.2. Dataset Setup

1. **Download:** Get the **BraTS 2020** dataset from [Kaggle](https://www.kaggle.com/code/samenimran/brats2020/input).
2. **Configuration:** Update the dataset directory path in `config/config.py`.

Open:

config/config.py

and modify the following line:

DATA_PATH = Path("/root/Seg/MICCAI_BraTS2020_TrainingData")

Replace it with the path to your extracted dataset directory. For example:

DATA_PATH = Path("/your/local/path/MICCAI_BraTS2020_TrainingData")

### 4.2. Usage

Available Models

- **`optimized_unet3d`**: The proposed model featuring enhanced stability and optimized gradient flow.
- **`attention_unet3d`**: The standard 3D Attention U-Net baseline for comparison.

### 4.3 Train

```bash
python main.py --model_name optimized_unet3d
```

```bash
python main.py --model_name attention_unet3d
```

Specify epochs and k-fold:

```bash
python main.py --model_name optimized_unet3d --epochs 100 --kfold 5
```

### Inference Only

```bash
python main.py --model_name optimized_unet3d --skip_train
```

### Arguments

- --model_name : Model architecture (optimized_unet3d or
  attention_unet3d)
- --epochs : Number of training epochs
- --kfold : Number of cross-validation folds
- --skip_train : Skip training and run inference only

---
