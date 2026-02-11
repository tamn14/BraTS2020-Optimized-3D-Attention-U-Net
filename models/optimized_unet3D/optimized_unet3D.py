

import torch
import torch.nn as nn

from .blocks import ConvBlock3D, UpConv3D, AttentionBlock


class OptimizedUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, features, use_norm=True):
        super().__init__()
        f1, f2, f3, f4, f5 = features

        self.enc1 = ConvBlock3D(in_channels, f1, use_norm)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(f1, f2, use_norm)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(f2, f3, use_norm)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = ConvBlock3D(f3, f4, use_norm)
        self.pool4 = nn.MaxPool3d(2)

        self.center = ConvBlock3D(f4, f5, use_norm)

        self.att4 = AttentionBlock(f4, f4, f4 // 2)
        self.up4 = UpConv3D(f5, f4)
        self.dec4 = ConvBlock3D(f4 * 2, f4, use_norm)

        self.att3 = AttentionBlock(f3, f3, f3 // 2)
        self.up3 = UpConv3D(f4, f3)
        self.dec3 = ConvBlock3D(f3 * 2, f3, use_norm)

        self.att2 = AttentionBlock(f2, f2, f2 // 2)
        self.up2 = UpConv3D(f3, f2)
        self.dec2 = ConvBlock3D(f2 * 2, f2, use_norm)

        self.att1 = AttentionBlock(f1, f1, f1 // 2)
        self.up1 = UpConv3D(f2, f1)
        self.dec1 = ConvBlock3D(f1 * 2, f1, use_norm)

        self.out_conv = nn.Conv3d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        center = self.center(self.pool4(e4))

        d4 = self.up4(center)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1))

        return self.out_conv(d1)
