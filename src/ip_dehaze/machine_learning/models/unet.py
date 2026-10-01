"""U-Net dehazing backbones.

`UNet` is the plain (blind) model. With `gated=True` it also predicts a
positive scalar density `d` per image and returns the convex blend
`J = d * f(P) + (1 - d) * P`, so correction strength is a learned,
self-supervised function of the input — no density labels needed.
"""

from __future__ import annotations

import math

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
                 frozen_gate: bool = False, gate_init: float | None = None,
                 gate_spatial: bool = False, gate_context: bool = False,
                 dcp_t: bool = False) -> None:
        super().__init__()
        if dcp_t:
            # The 4th input channel is DCP's per-image transmission estimate —
            # the physical cue that transfers to real fog (P3.7 step 2). The
            # channel is prepended by the caller: x = cat([hazy, dcp_t]).
            in_ch = in_ch + 1
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
        # gate_spatial: one gate value per pixel (fog is spatially varying —
        # `t(x)` in the scattering model) instead of a single scalar per image.
        self.gate_spatial = gate_spatial
        self.density_head = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels[-1] * 2, 1)
            )
            if gated and not gate_spatial and not gate_context
            else None
        )
        self.spatial_head = nn.Conv2d(channels[0], 1, 1) if gated and gate_spatial else None
        # gate_context: the gate reads a whole downscaled scene instead of the
        # restoration path's field of view, so train and eval agree on scale.
        self.context_stem = (
            nn.Sequential(
                nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 1),
            )
            if gated and gate_context
            else None
        )
        # gate_init: a default init puts the gate at d~0.4, so the very first
        # output is mostly the hazy image and the restoration head only sees a
        # d-scaled gradient. Starting at d->1 keeps the blend a no-op and leaves
        # the gate with nothing to learn but the deviations from full strength.
        if gated and gate_init is not None:
            if not 0.0 < gate_init < 1.0:
                raise ValueError("gate_init must be in (0, 1)")
            with torch.no_grad():
                for head in (self.density_head, self.spatial_head, self.context_stem):
                    if head is None:
                        continue
                    last = head[-1] if isinstance(head, nn.Sequential) else head
                    last.weight.zero_()
                    last.bias.fill_(math.log(gate_init / (1 - gate_init)))

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The gated blend falls back to the HAZY IMAGE, not the raw input: with
        # the DCP-transmission channel prepended, x is 4-channel while output
        # is 3-channel, and blending against x would broadcast-fail (or, worse,
        # silently broadcast). Slice off the channel after the encoders read it.
        hazy = x[:, :3]
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
        elif self.context_stem is not None:
            if context is None:
                raise ValueError("this model gates on a whole-image context; pass context=")
            density = torch.sigmoid(self.context_stem(context))
        elif self.spatial_head is not None:
            density = None  # per-pixel gate is computed after the decoder, at input resolution
        else:
            density = torch.sigmoid(self.density_head(features))

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            features = upconv(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(features, size=skip.shape[-2:], mode="nearest")
            features = decoder(torch.cat([features, skip], dim=1))

        if self.spatial_head is not None and not self.frozen_gate:
            density = torch.sigmoid(self.spatial_head(features))

        output = torch.sigmoid(self.head(features))
        if density is None:
            return output, None
        gate = density if density.dim() > 2 else density.view(-1, 1, 1, 1)
        return gate * output + (1 - gate) * hazy, density


def build_unet(size: str = "medium", gated: bool = False, frozen_gate: bool = False,
               gate_init: float | None = None, gate_spatial: bool = False,
               gate_context: bool = False, dcp_t: bool = False) -> UNet:
    return UNet(gated=gated, frozen_gate=frozen_gate, gate_init=gate_init,
                gate_spatial=gate_spatial, gate_context=gate_context, dcp_t=dcp_t,
                **SIZE_PRESETS[size])
