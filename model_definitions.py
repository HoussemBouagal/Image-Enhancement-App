import torch
import torch.nn as nn
import torch.nn.functional as F

# ======== ESPCN Model ========
class ESPCN(nn.Module):
    def __init__(self, upscale_factor=2):
        super(ESPCN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 3 * (upscale_factor ** 2), kernel_size=3, padding=1)
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.features(x)
        x = self.pixel_shuffle(x)
        return x

# ======== U-Net Components ========
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# ======== Conditional U-Net ========
class ConditionalUNet(nn.Module):
    def __init__(self, cond_channels=3):
        super().__init__()
        self.down1 = DoubleConv(3 + cond_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up1 = DoubleConv(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x, cond):
        cond_up = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, cond_up], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        u2 = self.up2(torch.cat([F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False), d2], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False), d1], dim=1))
        return self.final(u1)

# ======== Fonction de bruit (noise) ========
def add_noise(img, noise_level=0.1):
    noise = torch.randn_like(img) * noise_level
    return torch.clamp(img + noise, 0., 1.)
