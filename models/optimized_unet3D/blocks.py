
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
   
    def __init__(self, in_ch, out_ch, use_norm=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        layers.append(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_norm)
        )
        
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.1, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionBlock(nn.Module):
    
    def __init__(self, F_g, F_l, F_int, init_bias=2.0):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, bias=False),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=False),
            nn.InstanceNorm3d(1, affine=True)
        )
        self.relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.psi[1].weight, 1.0)  
        nn.init.constant_(self.psi[1].bias, init_bias)  

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        S = self.relu(g1 + x1)
        psi_hat = self.psi(S)
        alpha = torch.sigmoid(psi_hat)
        return x * alpha


class UpConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride=stride)

    def forward(self, x):
        return self.up(x)
