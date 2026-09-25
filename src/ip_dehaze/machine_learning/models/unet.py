"""U-Net dehazing backbones.

`UNet` is the plain (blind) model. With `gated=True` it also predicts a
positive scalar density `d` per image and returns the convex blend
`J = d * f(P) + (1 - d) * P`, so correction strength is a learned,
self-supervised function of the input — no density labels needed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Graded capacity for the paper-style routed branches.
# heavy is width-48/depth-4 (17.5M params): base 48 at depth 5 measured
# 70M params and 3.12 GB at full-res validation, which would have dominated
# the training budget on a 6 GB GPU.
SIZE_PRESETS = {
    "tiny": {"base": 16, "depth": 3},
    "light": {"base": 24, "depth": 3},   # 1.1M params
    "medium": {"base": 32, "depth": 4},  # 7.8M params
    "heavy": {"base": 48, "depth": 4},   # 17.5M params
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32, depth: int = 4, gated: bool = False,
                 frozen_gate: bool = False) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        channels = [base * 2**level for level in range(depth)]

        self.encoders = nn.ModuleList()
        prev = in_ch
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch))
            prev = ch

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        bottleneck_ch = channels[-1] * 2

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(bottleneck_ch, ch, 2, stride=2))
            self.decoders.append(ConvBlock(ch * 2, ch))
            bottleneck_ch = ch

        self.head = nn.Conv2d(channels[0], 3, 1)
        self.gated = gated
        # frozen_gate: keep the density head in place but force d=1, i.e. the
        # output is the raw restoration. This is the d==1 ablation used to test
        # whether the learned gate itself earns any gain.
        self.frozen_gate = frozen_gate
        self.density_head = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1] * 2, 1)
            )
            if gated
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        skips = []
        features = x
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)

        features = self.bottleneck(features)
        if not self.gated:
            density = None
        elif self.frozen_gate:
            density = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
        else:
            density = torch.sigmoid(self.density_head(features))

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            features = upconv(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(features, size=skip.shape[-2:], mode="nearest")
            features = decoder(torch.cat([features, skip], dim=1))

        output = torch.sigmoid(self.head(features))
        if density is None:
            return output, None
        return density.view(-1, 1, 1, 1) * output + (1 - density.view(-1, 1, 1, 1)) * x, density


def build_unet(size: str = "medium", gated: bool = False, frozen_gate: bool = False) -> UNet:
    return UNet(gated=gated, frozen_gate=frozen_gate, **SIZE_PRESETS[size])
