from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import FontPairDataset
from models import Discriminator, Generator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train few-shot font style transfer model")
    p.add_argument("--labels-csv", type=str, default="data/labels.csv")
    p.add_argument("--out-dir", type=str, default="runs/exp1")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--k-style", type=int, default=4)
    p.add_argument("--style-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--lambda-rec", type=float, default=10.0)
    p.add_argument("--lambda-style", type=float, default=1.0)
    p.add_argument("--lambda-content", type=float, default=1.0)
    p.add_argument("--lambda-identity", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--limit-samples", type=int, default=0,
                   help="Use only the first N samples for fast iteration (0 = full)")
    p.add_argument("--resume", type=str, default="",
                   help="Path to checkpoint to resume from")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    dataset = FontPairDataset(
        labels_csv=args.labels_csv,
        image_size=args.image_size,
        k_style=args.k_style,
    )
    if args.limit_samples > 0:
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(args.device == "cuda"),
    )

    G = Generator(image_channels=1, style_dim=args.style_dim).to(args.device)
    D = Discriminator(in_ch=1).to(args.device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device)
        G.load_state_dict(state["G"])
        D.load_state_dict(state["D"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        start_epoch = state.get("epoch", 0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            content_image = batch["content_image"].to(args.device, non_blocking=True)
            style_images = batch["style_images"].to(args.device, non_blocking=True)
            target_image = batch["target_image"].to(args.device, non_blocking=True)

            # ===== Train D =====
            with torch.no_grad():
                fake = G(content_image, style_images)
            d_real = D(target_image)
            d_fake = D(fake.detach())
            d_loss = hinge_d_loss(d_real, d_fake)

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # ===== Train G =====
            content_feat = G.content_encoder(content_image)
            style_code = G.style_encoder(style_images)
            fake = G.decode(content_feat, style_code)

            g_adv = hinge_g_loss(D(fake))
            g_rec = F.l1_loss(fake, target_image)

            # style consistency: re-encoding the fake image should match the style
            style_fake = G.style_encoder(fake.unsqueeze(1))
            g_style = F.l1_loss(style_fake, style_code.detach())

            # content consistency: content features must be preserved
            content_fake = G.content_encoder(fake)
            g_content = F.l1_loss(content_fake, content_feat.detach())

            # identity: feeding the target style as content should reconstruct itself
            style_first = style_images[:, 0]  # (B, 1, H, W)
            content_self = G.content_encoder(style_first)
            recon_self = G.decode(content_self, style_code)
            g_identity = F.l1_loss(recon_self, style_first)

            g_loss = (
                g_adv
                + args.lambda_rec * g_rec
                + args.lambda_style * g_style
                + args.lambda_content * g_content
                + args.lambda_identity * g_identity
            )

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()

            pbar.set_postfix(
                d=f"{d_loss.item():.3f}",
                g=f"{g_adv.item():.3f}",
                rec=f"{g_rec.item():.3f}",
                sty=f"{g_style.item():.3f}",
                ctn=f"{g_content.item():.3f}",
            )

        if (epoch + 1) % args.save_every == 0:
            G.eval()
            with torch.no_grad():
                vis_n = min(8, content_image.size(0))
                fake_vis = G(content_image[:vis_n], style_images[:vis_n])
            grid = torch.cat(
                [
                    content_image[:vis_n].cpu(),
                    style_images[:vis_n, 0].cpu(),
                    fake_vis.cpu(),
                    target_image[:vis_n].cpu(),
                ],
                dim=0,
            )
            save_image(
                (grid + 1) / 2,
                out_dir / "samples" / f"epoch_{epoch + 1:03d}.png",
                nrow=vis_n,
            )
            ckpt_path = out_dir / "ckpt" / f"epoch_{epoch + 1:03d}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )
            latest = out_dir / "ckpt" / "latest.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G": G.state_dict(),
                    "D": D.state_dict(),
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(),
                    "args": vars(args),
                },
                latest,
            )
            G.train()


if __name__ == "__main__":
    main()
