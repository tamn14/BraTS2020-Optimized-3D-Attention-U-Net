
from pathlib import Path

"""
    Configuration file for training 3D U-Net on BraTS2020 dataset. 
"""


"""
    We use Path(__file__).resolve().parent.parent to get the project root directory,
    assuming this config.py file is located in the config/ subdirectory.
    for example, if the project structure is:
                                                project/
                                            ├── config/
                                            │   └── config.py
                                            ├── train.py
                                            └── ...
    => __file__ refers to config.py, we have path as project/config/config.py
    => Path(__file__).resolve() gives the absolute path to config.py, we have path as /absolute/path/to/project/config/config.py
    => .parent gives the parent directory, we have path as /absolute/path/to/project/config
    => .parent again gives the project root directory, we have path as /absolute/path/to/project
"""
PROJECT_PATH = Path(__file__).resolve().parent.parent    # Project root directory

DATA_PATH = Path("/root/Seg/MICCAI_BraTS2020_TrainingData")  # Path to BraTS2020 training data

OUTPUT_DIR = PROJECT_PATH / "kfold_results"   # Directory to save outputs
MODEL_DIR = OUTPUT_DIR / "models"             # Directory to save trained models
LOG_DIR = OUTPUT_DIR / "logs"                 # Directory to save training logs
PRED_DIR = OUTPUT_DIR / "predictions"         # Directory to save model predictions
VIS_DIR = OUTPUT_DIR / "visualizations"       # Directory to save visualizations

N_SPLITS = 5                                 # Number of folds for cross-validation
RANDOM_STATE = 42                             # Random seed for reproducibility

IN_CHANNELS_3D = 4                     # Number of input channels for 3D U-Net                      
OUT_CHANNELS_3D = 4                    # Number of output channels for 3D U-Net
FEATURES = (16, 32, 64, 128, 256)      # Feature map sizes for each layer in U-Net

FINE_EPOCHS = 100                      # Number of epochs for fine-tuning
WARMUP_EPOCHS = 2                      # Number of warm-up epochs
OPT_LR = 3e-4                         # Learning rate for fine-tuning
FINE_BATCH_SIZE = 1                    # Batch size for fine-tuning
ACCUMULATION_STEPS = 4                 # Gradient accumulation steps
FINE_EARLY_STOPPING_PATIENCE = 20      # Early stopping patience for fine-tuning
USE_AMP = True   

WEIGHT_DECAY = 1e-5
EMA_DECAY = 0.997          
CLIP_GRAD_NORM = 3.0
SW_BATCH_INFER = 2
CACHE_RATE_TRAIN = 0.2
CACHE_RATE_VAL = 0.2 

PATCH_SIZE = (128, 128, 128)           # Input patch size for training
VAL_ROI_SIZE = (128, 128, 128)         # ROI size for validation 
    
MODEL_NAME = "optimized_unet3d"        # Model architecture name
