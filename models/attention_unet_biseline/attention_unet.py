import torch
import torch.nn as nn

from .block import ConvBlock3D, AttentionBlock

"""
    Attention U-Net architecture for 3D medical image segmentation.
    Reference: "Attention U-Net: Learning Where to Look for the Pancreas" by Oktay et al.
"""

class Standard_AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, features=[16, 32, 64, 128, 256]):
        super().__init__()
        f1, f2, f3, f4, f5 = features

        self.enc1 = ConvBlock3D(in_channels, f1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock3D(f1, f2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock3D(f2, f3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock3D(f3, f4)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.center = ConvBlock3D(f4, f5)

        self.up4 = nn.ConvTranspose3d(f5, f4, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=f4, F_l=f4, F_int=f4 // 2)
        self.dec4 = ConvBlock3D(f4 + f4, f4)

        self.up3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=f3, F_l=f3, F_int=f3 // 2)
        self.dec3 = ConvBlock3D(f3 + f3, f3)

        self.up2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=f2, F_l=f2, F_int=f2 // 2)
        self.dec2 = ConvBlock3D(f2 + f2, f2)

        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=f1, F_l=f1, F_int=f1 // 2)
        self.dec1 = ConvBlock3D(f1 + f1, f1)

        self.out_conv = nn.Conv3d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        center = self.center(self.pool4(e4))

        d4 = self.up4(center)
        x4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat((x4, d4), dim=1))

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat((x3, d3), dim=1))

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat((x2, d2), dim=1))

        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat((x1, d1), dim=1))

        return self.out_conv(d1)


