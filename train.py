"""Train the few-shot font style transfer model.

Loss design (deliberately rebalanced from the previous attempt that collapsed):
    - Adversarial: hinge GAN on real/fake patch logits
    - Aux classification: cross-entropy on font and character labels — pushes
      D to learn features that disentangle style and content
    - VGG perceptual: L1 on VGG16 features (replaces dominant pixel L1)
    - L1 reconstruction: small weight, just to anchor pixel intensity
    - Style/content consistency: keep encoder representations aligned

Training tricks:
    - EMA generator (decay 0.999) — sample images and serve.py use the EMA copy
    - TTUR: discriminator learning rate is higher than generator's
    - R1 gradient penalty every R1_INTERVAL steps for D stability
    - bfloat16 autocast on MPS / CUDA for speed and lower memory
    - Adam betas (0, 0.99) following StyleGAN — pairs well with R1
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import FontPairDataset
from models import Discriminator, Generator, VGGPerceptual

R1_INTERVAL = 16  # apply R1 penalty every N D steps (lazy regularisation)
EMA_WARMUP_STEPS = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train font style transfer model")
    p.add_argument("--labels-csv", type=str, default="data/labels_train.csv")
    p.add_argument("--out-dir", type=str, default="runs/v2")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--k-style", type=int, default=4)
    p.add_argument("--style-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--g-lr", type=float, default=1e-4)
    p.add_argument("--d-lr", type=float, default=4e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--ema-decay", type=float, default=0.999)
    # loss weights — perceptual is the workhorse, pixel L1 is just a hint
    p.add_argument("--lambda-adv", type=float, default=1.0)
    p.add_argument("--lambda-perceptual", type=float, default=5.0)
    p.add_argument("--lambda-rec", type=float, default=0.5)
    p.add_argument("--lambda-content", type=float, default=2.0)
    p.add_argument("--lambda-style", type=float, default=1.0)
    p.add_argument("--lambda-font-cls", type=float, default=1.0)
    p.add_argument("--lambda-char-cls", type=float, default=1.0)
    p.add_argument("--lambda-r1", type=float, default=10.0)
    p.add_argument("--save-every", type=int, default=2)
    p.add_argument("--limit-samples", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def hinge_d_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - real).mean() + F.relu(1 + fake).mean()


def hinge_g_loss(fake: torch.Tensor) -> torch.Tensor:
    return -fake.mean()


def r1_gradient_penalty(d_real: torch.Tensor, real_images: torch.Tensor) -> torch.Tensor:
    """0.5 * ||grad_x D(x)||^2 averaged over batch."""
    grad = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.reshape(grad.size(0), -1)
    return 0.5 * grad.pow(2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        # During warmup, just copy
        if step < EMA_WARMUP_STEPS:
            for p_ema, p in zip(self.ema.parameters(), model.parameters()):
                p_ema.data.copy_(p.data)
            for b_ema, b in zip(self.ema.buffers(), model.buffers()):
                b_ema.data.copy_(b.data)
            return
        d = self.decay
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1 - d)
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.data.copy_(b.data)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    with (out_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    use_bf16 = (not args.no_bf16) and args.device in {"mps", "cuda"}
    autocast_device = "cuda" if args.device == "cuda" else "cpu" if args.device == "cpu" else "mps"
    autocast_kwargs = {"device_type": autocast_device, "dtype": torch.bfloat16, "enabled": use_bf16}

    dataset = FontPairDataset(
        labels_csv=args.labels_csv,
        image_size=args.image_size,
        k_style=args.k_style,
        augment=True,
    )
    full_n_fonts = dataset.n_fonts
    full_n_chars = dataset.n_chars
    if args.limit_samples > 0:
        dataset = Subset(dataset, range(min(args.limit_samples, len(dataset))))
    print(f"Dataset size: {len(dataset)}  fonts={full_n_fonts}  chars={full_n_chars}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(args.device == "cuda"),
    )

    G = Generator(image_channels=1, style_dim=args.style_dim).to(device)
    D = Discriminator(in_ch=1, n_fonts=full_n_fonts, n_chars=full_n_chars).to(device)
    vgg = VGGPerceptual().to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    ema = EMA(G, decay=args.ema_decay)

    start_epoch = 0
    global_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        G.load_state_dict(state["G"])
        D.load_state_dict(state["D"])
        opt_g.load_state_dict(state["opt_g"])
        opt_d.load_state_dict(state["opt_d"])
        ema.ema.load_state_dict(state.get("G_ema", state["G"]))
        start_epoch = state.get("epoch", 0)
        global_step = state.get("global_step", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            content = batch["content_image"].to(device, non_blocking=True)
            style_refs = batch["style_images"].to(device, non_blocking=True)
            target = batch["target_image"].to(device, non_blocking=True)
            target_font_id = batch["target_font_id"].to(device, non_blocking=True)
            char_id = batch["char_id"].to(device, non_blocking=True)

            # ============================================================
            #  D step
            # ============================================================
            do_r1 = (global_step % R1_INTERVAL == 0)
            real_for_d = target.detach().requires_grad_(do_r1)

            with torch.autocast(**autocast_kwargs):
                with torch.no_grad():
                    fake = G(content, style_refs)
                d_real_patch, d_real_font, d_real_char = D(real_for_d)
                d_fake_patch, _, _ = D(fake.detach())
                d_adv = hinge_d_loss(d_real_patch, d_fake_patch)
                d_font_cls = F.cross_entropy(d_real_font, target_font_id)
                d_char_cls = F.cross_entropy(d_real_char, char_id)
                d_loss = (
                    args.lambda_adv * d_adv
                    + args.lambda_font_cls * d_font_cls
                    + args.lambda_char_cls * d_char_cls
                )

            if do_r1:
                # R1 must be computed in fp32 for stable gradients
                d_real_patch_fp = D(real_for_d)[0]
                r1 = r1_gradient_penalty(d_real_patch_fp, real_for_d)
                d_total = d_loss + (args.lambda_r1 * R1_INTERVAL) * r1
            else:
                d_total = d_loss

            opt_d.zero_grad(set_to_none=True)
            d_total.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
            opt_d.step()

            # ============================================================
            #  G step
            # ============================================================
            with torch.autocast(**autocast_kwargs):
                feats = G.encode_content(content)
                style_code = G.encode_style(style_refs)
                fake = G.decode(feats, style_code)

                g_fake_patch, g_fake_font, g_fake_char = D(fake)
                g_adv = hinge_g_loss(g_fake_patch)

                g_font_cls = F.cross_entropy(g_fake_font, target_font_id)
                g_char_cls = F.cross_entropy(g_fake_char, char_id)

                g_perceptual = vgg(fake.float(), target.float())  # vgg in fp32
                g_rec = F.l1_loss(fake, target)

                # Consistency: re-encode fake → features should match
                content_fake = G.encode_content(fake)
                g_content_consistency = sum(
                    F.l1_loss(cf, c.detach()) for cf, c in zip(content_fake, feats)
                ) / len(feats)

                style_fake = G.encode_style(fake.unsqueeze(1))
                g_style_consistency = F.l1_loss(style_fake, style_code.detach())

                g_loss = (
                    args.lambda_adv * g_adv
                    + args.lambda_perceptual * g_perceptual
                    + args.lambda_rec * g_rec
                    + args.lambda_content * g_content_consistency
                    + args.lambda_style * g_style_consistency
                    + args.lambda_font_cls * g_font_cls
                    + args.lambda_char_cls * g_char_cls
                )

            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_g.step()

            ema.update(G, global_step)
            global_step += 1

            pbar.set_postfix(
                adv=f"{g_adv.item():.2f}",
                vgg=f"{g_perceptual.item():.2f}",
                rec=f"{g_rec.item():.3f}",
                con=f"{g_content_consistency.item():.3f}",
                d=f"{d_adv.item():.2f}",
            )

        # =================================================================
        #  Save checkpoint + samples
        # =================================================================
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ema.ema.eval()
            with torch.no_grad():
                vis_n = min(8, content.size(0))
                with torch.autocast(**autocast_kwargs):
                    fake_vis = ema.ema(content[:vis_n], style_refs[:vis_n]).float()
            grid = torch.cat(
                [
                    content[:vis_n].cpu(),
                    style_refs[:vis_n, 0].cpu(),
                    fake_vis.cpu(),
                    target[:vis_n].cpu(),
                ],
                dim=0,
            )
            save_image(
                (grid + 1) / 2,
                out_dir / "samples" / f"epoch_{epoch + 1:03d}.png",
                nrow=vis_n,
            )
            state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "G": G.state_dict(),
                "G_ema": ema.ema.state_dict(),
                "D": D.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "args": vars(args),
                "vocab": {
                    "n_fonts": full_n_fonts,
                    "n_chars": full_n_chars,
                },
            }
            torch.save(state, out_dir / "ckpt" / f"epoch_{epoch + 1:03d}.pt")
            torch.save(state, out_dir / "ckpt" / "latest.pt")


if __name__ == "__main__":
    main()
