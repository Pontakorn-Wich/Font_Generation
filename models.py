"""Few-shot font style transfer model.

Architecture (re-designed after the previous AdaIN-only model collapsed):

  Content path:  U-Net encoder with skip connections at every scale, so the
                 raw spatial structure of the requested glyph is carried all
                 the way to the output. The bottleneck no longer has to be
                 a complete representation of the glyph.

  Style path:    Per-reference CNN encoder followed by self-attention over
                 the K reference images and a learned aggregation token.
                 Produces a single style vector regardless of K.

  Decoder:       U-Net decoder. At every scale we apply AdaIN modulated by
                 the style vector, upsample, concatenate the matching
                 encoder skip, and run a residual block.

  Discriminator: PatchGAN trunk with spectral normalisation, plus two
                 auxiliary heads (font classifier, character classifier)
                 that share the trunk. The aux heads force D to learn
                 features that disentangle style and content, which is a
                 much stronger signal than real/fake alone.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv2d + (optional norm) + activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: str = "instance",
        act: str = "lrelu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        ]
        if norm == "instance":
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        elif norm == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        if act == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == "relu":
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    """Standard residual block with InstanceNorm + LeakyReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        h = self.norm2(self.conv2(h))
        return F.leaky_relu(x + h, 0.2)


class AdaIN(nn.Module):
    """Adaptive instance normalisation conditioned on a style vector."""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.fc = nn.Linear(style_dim, channels * 2)
        nn.init.zeros_(self.fc.bias)
        # Initialise so the modulation starts close to identity.
        self.fc.bias.data[:channels] = 1.0  # gamma ≈ 1
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        params = self.fc(style)
        gamma, beta = params.chunk(2, dim=1)
        return gamma.unsqueeze(-1).unsqueeze(-1) * h + beta.unsqueeze(-1).unsqueeze(-1)


class AdaINResBlock(nn.Module):
    """Residual block with AdaIN replacing both norm layers."""

    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.adain1 = AdaIN(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adain2 = AdaIN(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.adain1(x, style), 0.2))
        h = self.conv2(F.leaky_relu(self.adain2(h, style), 0.2))
        return x + h


# ---------------------------------------------------------------------------
# Content encoder (U-Net encoder that returns features at every scale)
# ---------------------------------------------------------------------------


class ContentEncoder(nn.Module):
    """U-Net encoder. Returns a list of feature maps from shallow to deep."""

    def __init__(self, in_ch: int = 1, base: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.stem = ConvBlock(in_ch, base, 7, 1, 3)
        self.downs = nn.ModuleList()
        ch = base
        self.channels: List[int] = [base]
        for _ in range(depth):
            next_ch = min(ch * 2, 512)
            self.downs.append(
                nn.Sequential(
                    ConvBlock(ch, next_ch, 4, 2, 1),
                    ResBlock(next_ch),
                )
            )
            ch = next_ch
            self.channels.append(ch)
        self.depth = depth

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = [self.stem(x)]                  # (B, base, 128, 128)
        for down in self.downs:
            feats.append(down(feats[-1]))       # halves spatial each time
        return feats                            # length == depth + 1


# ---------------------------------------------------------------------------
# Style encoder (per-ref CNN + transformer attention over K refs)
# ---------------------------------------------------------------------------


class StyleEncoder(nn.Module):
    """Encode K reference images into a single style vector via attention."""

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        n_down: int = 5,
        max_ch: int = 256,
        style_dim: int = 256,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvBlock(in_ch, base, 7, 1, 3, norm="instance")]
        ch = base
        for _ in range(n_down):
            next_ch = min(ch * 2, max_ch)
            layers.append(ConvBlock(ch, next_ch, 4, 2, 1, norm="instance"))
            ch = next_ch
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(ch, style_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=style_dim,
            num_heads=attn_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(style_dim)
        # Learned aggregation token (CLS-style)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, style_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.style_dim = style_dim

    def encode_one(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one image (B, 1, H, W) to a vector (B, style_dim)."""
        h = self.conv(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.proj(h)

    def forward(self, refs: torch.Tensor) -> torch.Tensor:
        """refs: (B, K, 1, H, W) or (B, 1, H, W). Returns (B, style_dim)."""
        if refs.dim() == 4:
            return self.encode_one(refs)

        B, K, C, H, W = refs.shape
        flat = refs.view(B * K, C, H, W)
        tokens = self.encode_one(flat).view(B, K, -1)        # (B, K, D)

        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)                # (B, K+1, D)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        attn_out = self.norm(seq + attn_out)
        return attn_out[:, 0]                                # CLS token


# ---------------------------------------------------------------------------
# Decoder (U-Net decoder with AdaIN modulation + skip concatenation)
# ---------------------------------------------------------------------------


class UpBlock(nn.Module):
    """AdaIN -> upsample -> concat with skip -> conv -> AdaIN -> conv."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        style_dim: int,
    ) -> None:
        super().__init__()
        self.adain_in = AdaIN(in_ch, style_dim)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1)
        self.adain_mid = AdaIN(out_ch, style_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        # Residual projection if channel count changes
        self.res_proj = nn.Conv2d(in_ch + skip_ch, out_ch, 1) if (in_ch + skip_ch) != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.adain_in(x, style), 0.2)
        h = self.up(h)
        h = torch.cat([h, skip], dim=1)
        residual = self.res_proj(h)
        h = self.conv1(h)
        h = F.leaky_relu(self.adain_mid(h, style), 0.2)
        h = self.conv2(h)
        return h + residual


class Decoder(nn.Module):
    """U-Net decoder. Takes a list of encoder features and a style vector."""

    def __init__(
        self,
        encoder_channels: List[int],
        out_ch: int = 1,
        style_dim: int = 256,
        n_bottleneck_res: int = 3,
    ) -> None:
        super().__init__()
        # encoder_channels e.g. [64, 128, 256, 512, 512] from shallow→deep
        rev = list(reversed(encoder_channels))           # [512, 512, 256, 128, 64]
        bottleneck_ch = rev[0]
        self.bottleneck = nn.ModuleList(
            [AdaINResBlock(bottleneck_ch, style_dim) for _ in range(n_bottleneck_res)]
        )

        self.ups = nn.ModuleList()
        cur = bottleneck_ch
        for skip_ch, target_ch in zip(rev[1:], rev[1:]):
            self.ups.append(UpBlock(cur, skip_ch, target_ch, style_dim))
            cur = target_ch
        self.final_adain = AdaIN(cur, style_dim)
        self.final = nn.Sequential(
            nn.Conv2d(cur, cur, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cur, out_ch, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, feats: List[torch.Tensor], style: torch.Tensor) -> torch.Tensor:
        rev = list(reversed(feats))                # deep→shallow
        h = rev[0]
        for blk in self.bottleneck:
            h = blk(h, style)
        for up, skip in zip(self.ups, rev[1:]):
            h = up(h, skip, style)
        h = F.leaky_relu(self.final_adain(h, style), 0.2)
        return self.final(h)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    def __init__(
        self,
        image_channels: int = 1,
        base: int = 64,
        depth: int = 4,
        style_dim: int = 256,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.content_encoder = ContentEncoder(
            in_ch=image_channels, base=base, depth=depth
        )
        self.style_encoder = StyleEncoder(
            in_ch=image_channels,
            base=base,
            n_down=5,
            max_ch=256,
            style_dim=style_dim,
            attn_heads=attn_heads,
        )
        self.decoder = Decoder(
            encoder_channels=self.content_encoder.channels,
            out_ch=image_channels,
            style_dim=style_dim,
        )
        self.style_dim = style_dim

    def encode_content(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.content_encoder(x)

    def encode_style(self, refs: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(refs)

    def decode(self, feats: List[torch.Tensor], style: torch.Tensor) -> torch.Tensor:
        return self.decoder(feats, style)

    def forward(self, content: torch.Tensor, style_refs: torch.Tensor) -> torch.Tensor:
        feats = self.encode_content(content)
        style = self.encode_style(style_refs)
        return self.decode(feats, style)


# ---------------------------------------------------------------------------
# Discriminator (PatchGAN + aux classification heads, all with spectral norm)
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    """Spectral-norm PatchGAN trunk shared by 3 heads."""

    def __init__(
        self,
        in_ch: int = 1,
        n_fonts: int = 0,
        n_chars: int = 0,
        base: int = 64,
        n_down: int = 4,
        max_ch: int = 512,
    ) -> None:
        super().__init__()
        self.n_fonts = n_fonts
        self.n_chars = n_chars

        trunk: list[nn.Module] = [
            spectral_norm(nn.Conv2d(in_ch, base, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base
        for _ in range(n_down - 1):
            next_ch = min(ch * 2, max_ch)
            trunk.append(spectral_norm(nn.Conv2d(ch, next_ch, 4, 2, 1)))
            trunk.append(nn.LeakyReLU(0.2, inplace=True))
            ch = next_ch
        self.trunk = nn.Sequential(*trunk)
        self.trunk_out_ch = ch

        # Patch head
        self.patch_head = spectral_norm(nn.Conv2d(ch, 1, 4, 1, 1))

        # Aux heads
        self.font_head = nn.Linear(ch, n_fonts) if n_fonts > 0 else None
        self.char_head = nn.Linear(ch, n_chars) if n_chars > 0 else None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        h = self.trunk(x)
        patch = self.patch_head(h)
        pooled = F.adaptive_avg_pool2d(h, 1).flatten(1)
        font_logits = self.font_head(pooled) if self.font_head is not None else None
        char_logits = self.char_head(pooled) if self.char_head is not None else None
        return patch, font_logits, char_logits


# ---------------------------------------------------------------------------
# VGG perceptual loss (instantiated lazily by the trainer)
# ---------------------------------------------------------------------------


class VGGPerceptual(nn.Module):
    """Perceptual loss using ImageNet VGG16 features.

    Outputs L1 distance averaged over a handful of layers. Grayscale inputs
    in [-1, 1] are converted to 3-channel ImageNet-normalised tensors.
    """

    def __init__(self, layer_indices: tuple[int, ...] = (3, 8, 15, 22)) -> None:
        super().__init__()
        from torchvision import models

        weights = models.VGG16_Weights.IMAGENET1K_V1
        vgg = models.vgg16(weights=weights).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.layer_indices = layer_indices
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        # x is grayscale [-1, 1]; convert to RGB and ImageNet-normalize
        x = (x + 1.0) / 2.0
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self._prepare(x)
        y = self._prepare(y)
        loss = x.new_tensor(0.0)
        max_idx = max(self.layer_indices)
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layer_indices:
                loss = loss + F.l1_loss(x, y)
            if i >= max_idx:
                break
        return loss / len(self.layer_indices)


__all__ = [
    "Generator",
    "Discriminator",
    "VGGPerceptual",
]
