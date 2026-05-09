from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _norm(channels: int, kind: str = "instance") -> nn.Module:
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(kind)


class ConvBlock(nn.Module):
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
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            _norm(out_ch, norm),
        ]
        if act == "lrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "tanh":
            layers.append(nn.Tanh())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "instance") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = _norm(channels, norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = _norm(channels, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h)


class AdaIN(nn.Module):
    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.fc(style).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * h + beta


class AdaINResBlock(nn.Module):
    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.adain1 = AdaIN(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.adain2 = AdaIN(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.adain1(x, style)))
        h = self.conv2(F.relu(self.adain2(h, style)))
        return x + h


class ContentEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 64, n_down: int = 3, n_res: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvBlock(in_ch, base, 7, 1, 3)]
        ch = base
        for _ in range(n_down):
            layers.append(ConvBlock(ch, ch * 2, 4, 2, 1))
            ch *= 2
        for _ in range(n_res):
            layers.append(ResBlock(ch))
        self.net = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        n_down: int = 4,
        max_ch: int = 256,
        style_dim: int = 128,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [ConvBlock(in_ch, base, 7, 1, 3, norm="none")]
        ch = base
        for _ in range(n_down):
            next_ch = min(ch * 2, max_ch)
            layers.append(ConvBlock(ch, next_ch, 4, 2, 1, norm="none"))
            ch = next_ch
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(ch, style_dim)
        self.style_dim = style_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            features = self.conv(x.view(B * K, C, H, W))
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            features = self.fc(features).view(B, K, -1)
            return features.mean(dim=1)
        features = self.conv(x)
        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.fc(features)


class Decoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 512,
        out_ch: int = 1,
        n_up: int = 3,
        n_res: int = 4,
        style_dim: int = 128,
    ) -> None:
        super().__init__()
        self.adain_res = nn.ModuleList(
            [AdaINResBlock(in_ch, style_dim) for _ in range(n_res)]
        )
        up_layers: list[nn.Module] = []
        ch = in_ch
        for _ in range(n_up):
            up_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            up_layers.append(ConvBlock(ch, ch // 2, 5, 1, 2))
            ch //= 2
        up_layers.append(nn.Conv2d(ch, out_ch, 7, 1, 3))
        up_layers.append(nn.Tanh())
        self.up = nn.Sequential(*up_layers)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        for blk in self.adain_res:
            x = blk(x, style)
        return self.up(x)


class Generator(nn.Module):
    def __init__(self, image_channels: int = 1, style_dim: int = 128) -> None:
        super().__init__()
        self.content_encoder = ContentEncoder(in_ch=image_channels, base=64, n_down=3, n_res=2)
        self.style_encoder = StyleEncoder(
            in_ch=image_channels, base=64, n_down=4, max_ch=256, style_dim=style_dim
        )
        self.decoder = Decoder(
            in_ch=self.content_encoder.out_channels,
            out_ch=image_channels,
            n_up=3,
            n_res=4,
            style_dim=style_dim,
        )

    def encode_style(self, style_images: torch.Tensor) -> torch.Tensor:
        return self.style_encoder(style_images)

    def decode(self, content_feat: torch.Tensor, style_code: torch.Tensor) -> torch.Tensor:
        return self.decoder(content_feat, style_code)

    def forward(
        self, content_image: torch.Tensor, style_images: torch.Tensor
    ) -> torch.Tensor:
        content_feat = self.content_encoder(content_image)
        style_code = self.style_encoder(style_images)
        return self.decoder(content_feat, style_code)


class Discriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization."""

    def __init__(self, in_ch: int = 1, base: int = 64, n_down: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            spectral_norm(nn.Conv2d(in_ch, base, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base
        for _ in range(n_down - 1):
            next_ch = min(ch * 2, 512)
            layers.append(spectral_norm(nn.Conv2d(ch, next_ch, 4, 2, 1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = next_ch
        layers.append(spectral_norm(nn.Conv2d(ch, 1, 4, 1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
