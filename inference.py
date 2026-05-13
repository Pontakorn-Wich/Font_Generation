from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from models import Generator


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def load_image(path: Path, image_size: int) -> torch.Tensor:
    return build_transform(image_size)(Image.open(path).convert("L"))


def load_generator(checkpoint: Path, device: str, style_dim: int = 128) -> Generator:
    G = Generator(image_channels=1, style_dim=style_dim).to(device)
    state = torch.load(checkpoint, map_location=device)
    G.load_state_dict(state["G"])
    G.eval()
    return G


@torch.no_grad()
def transfer(
    G: Generator,
    content_images: Sequence[torch.Tensor],
    style_images: Sequence[torch.Tensor],
    device: str,
) -> List[torch.Tensor]:
    style_tensor = torch.stack(list(style_images), dim=0).unsqueeze(0).to(device)
    style_code = G.style_encoder(style_tensor)
    outputs: List[torch.Tensor] = []
    for content in content_images:
        content_tensor = content.unsqueeze(0).to(device)
        content_feat = G.content_encoder(content_tensor)
        fake = G.decode(content_feat, style_code)
        outputs.append(fake.squeeze(0).cpu())
    return outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate font images for new characters in a target style")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--content-dir",
        type=str,
        required=True,
        help="Directory of content images (one PNG per character to generate)",
    )
    p.add_argument(
        "--style-dir",
        type=str,
        required=True,
        help="Directory of K reference style images for the target font",
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--style-dim", type=int, default=128)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content_paths = sorted(Path(args.content_dir).glob("*.png"))
    style_paths = sorted(Path(args.style_dir).glob("*.png"))
    if not content_paths or not style_paths:
        raise RuntimeError("Need at least one content image and one style image")

    G = load_generator(Path(args.checkpoint), args.device, style_dim=args.style_dim)

    content_tensors = [load_image(p, args.image_size) for p in content_paths]
    style_tensors = [load_image(p, args.image_size) for p in style_paths]

    fakes = transfer(G, content_tensors, style_tensors, args.device)
    for path, fake in zip(content_paths, fakes):
        save_image((fake + 1) / 2, output_dir / path.name)

    print(f"Wrote {len(fakes)} images to {output_dir}")


if __name__ == "__main__":
    main()
