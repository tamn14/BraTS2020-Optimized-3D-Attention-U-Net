
from models.optimized_unet3D.optimized_unet3D import OptimizedUNet3D
from models.attention_unet_biseline.attention_unet import Standard_AttentionUNet3D

def build_model(cfg):
    """
    Factory function to build model from config.

    Args:
        cfg: config module or dict with model settings

    Returns:
        torch.nn.Module
    """
    model_name = cfg.MODEL_NAME.lower()

    if model_name == "optimized_unet3d":
        return OptimizedUNet3D(
            in_channels=cfg.IN_CHANNELS_3D,
            out_channels=cfg.OUT_CHANNELS_3D,
            features=cfg.FEATURES,
            use_norm=True
        )
    
    
    elif model_name == "attention_unet3d":
        return Standard_AttentionUNet3D(
            in_channels=cfg.IN_CHANNELS_3D,
            out_channels=cfg.OUT_CHANNELS_3D,
            features=cfg.FEATURES
        )
        
        
       

    else:
        raise ValueError(f"Unknown model: {cfg.MODEL_NAME}")