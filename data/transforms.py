

from typing import Tuple, Sequence

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    MapLabelValued,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    ConcatItemsd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    Rand3DElasticd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
    ToTensord
)


def get_fine_transforms(
    image_keys: Sequence[str] = ("image_flair", "image_t1", "image_t1ce", "image_t2"),
    label_key: str = "label",
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    pixdim: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    # elastic_prob: float = 0.1,
    affine_prob: float = 0.05,
    flip_prob: float = 0.5,
    gaussian_noise_prob: float = 0.1,
) -> Tuple[Compose, Compose, Compose]:

    all_keys = list(image_keys) + [label_key]

    train_transforms = Compose([
        LoadImaged(keys=all_keys, image_only=False),
        EnsureChannelFirstd(keys=all_keys),
        MapLabelValued(keys=[label_key], orig_labels=[1, 2, 4], target_labels=[1, 2, 3]),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=list(image_keys) + [label_key], pixdim=pixdim,
                 mode=("bilinear",) * len(image_keys) + ("nearest",)),

        ScaleIntensityRangePercentilesd(keys=list(image_keys), lower=lower_percentile, upper=upper_percentile,
                                        b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=list(image_keys), nonzero=True, channel_wise=True),

        ConcatItemsd(keys=list(image_keys), name="image"),

        CropForegroundd(keys=["image", label_key], source_key="image"),
        SpatialPadd(keys=["image", label_key], spatial_size=patch_size, method="end"),

        RandCropByPosNegLabeld(
            keys=["image", label_key],
            label_key=label_key,
            spatial_size=patch_size,
            pos=2,  
            neg=1,
            num_samples=2,
        ),

        RandFlipd(keys=["image", label_key], prob=flip_prob, spatial_axis=[0]),
        RandFlipd(keys=["image", label_key], prob=flip_prob, spatial_axis=[1]),
        RandFlipd(keys=["image", label_key], prob=flip_prob, spatial_axis=[2]),
        RandRotate90d(keys=["image", label_key], prob=0.5, max_k=3),
        RandAffined(keys=["image", label_key], prob=affine_prob,
                    rotate_range=(0.05, 0.05, 0.05), scale_range=(0.05, 0.05, 0.05), mode=("bilinear", "nearest")),
        
        # Rand3DElasticd(keys=["image", label_key], sigma_range=(5.0, 8.0), magnitude_range=(50.0, 150.0), prob=elastic_prob,
        #                 mode=("bilinear", "nearest")),

        RandGaussianNoised(keys=["image"], prob=gaussian_noise_prob, mean=0.0, std=0.01),
        RandAdjustContrastd(keys=["image"], prob=0.1, gamma=(0.85, 1.15)),

        EnsureTyped(keys=["image", label_key]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=all_keys, image_only=False),
        EnsureChannelFirstd(keys=all_keys),
        MapLabelValued(keys=[label_key], orig_labels=[1, 2, 4], target_labels=[1, 2, 3]),
        Orientationd(keys=all_keys, axcodes="RAS"),
        Spacingd(keys=list(image_keys) + [label_key], pixdim=pixdim,
                 mode=("bilinear",) * len(image_keys) + ("nearest",)),
        ScaleIntensityRangePercentilesd(keys=list(image_keys), lower=lower_percentile, upper=upper_percentile,
                                        b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=list(image_keys), nonzero=True, channel_wise=True),
        ConcatItemsd(keys=list(image_keys), name="image"),
        CropForegroundd(keys=["image", label_key], source_key="image"),
        SpatialPadd(keys=["image", label_key], spatial_size=patch_size, method="end"),
        EnsureTyped(keys=["image", label_key]),
    ])

    test_transforms = Compose([
        LoadImaged(keys=list(image_keys), image_only=True),
        EnsureChannelFirstd(keys=list(image_keys)),
        Orientationd(keys=list(image_keys), axcodes="RAS"),
        Spacingd(keys=list(image_keys), pixdim=pixdim, mode=("bilinear",) * len(image_keys)),
        ScaleIntensityRangePercentilesd(keys=list(image_keys), lower=lower_percentile, upper=upper_percentile,
                                        b_min=0.0, b_max=1.0, clip=True),
        NormalizeIntensityd(keys=list(image_keys), nonzero=True, channel_wise=True),
        ConcatItemsd(keys=list(image_keys), name="image"),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=patch_size, method="end"),
        EnsureTyped(keys=["image"]),
    ])

    return train_transforms, val_transforms, test_transforms