"""Evaluate the EMA generator on all validation splits.

Saves:
  results/metrics.json              — per-split mean metrics
  results/per_sample_metrics.csv    — full per-sample table
  results/grids/{split}.png         — visual grid (content|style|fake|target)
  results/plots/metrics_bar.png     — metric comparison across splits
  results/plots/score_hist.png      — L1 / SSIM / VGG histograms per split
  results/plots/per_font_ssim.png   — per-font SSIM box plot (top-N fonts)
  results/plots/l1_vs_ssim.png      — scatter L1 vs SSIM coloured by split
  results/images/{split}/           — generated PNGs (optional, --save-images)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import FontPairDataset
from models import Generator, VGGPerceptual


# ---------------------------------------------------------------------------
# SSIM (torch, no extra deps)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return kernel / kernel.sum()


def batch_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Per-image SSIM. x, y: (B, 1, H, W) in [-1, 1]. Returns (B,)."""
    # MPS/bfloat16 safe: conv2d on MPS requires float32
    x = x.float()
    y = y.float()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    kernel = _gaussian_kernel(11, 1.5).to(device=x.device, dtype=torch.float32)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,11,11)

    def _blur(t: torch.Tensor) -> torch.Tensor:
        B, C, H, W = t.shape
        return F.conv2d(t, kernel.expand(C, -1, -1, -1), padding=5, groups=C)

    mu_x = _blur(x)
    mu_y = _blur(y)
    mu_xx = _blur(x * x)
    mu_yy = _blur(y * y)
    mu_xy = _blur(x * y)

    var_x = mu_xx - mu_x * mu_x
    var_y = mu_yy - mu_y * mu_y
    cov_xy = mu_xy - mu_x * mu_y

    ssim_map = (
        (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
        / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    )
    return ssim_map.mean(dim=(1, 2, 3))  # (B,)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_generator(checkpoint: Path, device: torch.device) -> Generator:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    G = Generator(image_channels=1, style_dim=state.get("args", {}).get("style_dim", 256)).to(device)
    G.load_state_dict(state.get("G_ema", state["G"]))
    G.eval()
    return G


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(
    G: Generator,
    vgg: VGGPerceptual,
    csv_path: Path,
    device: torch.device,
    image_size: int,
    k_style: int,
    batch_size: int,
    save_images: bool,
    images_dir: Path,
) -> List[Dict]:
    dataset = FontPairDataset(
        labels_csv=csv_path,
        image_size=image_size,
        k_style=k_style,
        min_chars_per_font=k_style,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    img_idx = 0

    for batch in tqdm(loader, desc=f"  {csv_path.stem}", leave=False):
        content = batch["content_image"].to(device)
        style_refs = batch["style_images"].to(device)
        target = batch["target_image"].to(device)

        fake = G(content, style_refs).float()  # float32 for MPS metric stability

        l1 = F.l1_loss(fake, target.float(), reduction="none").mean(dim=(1, 2, 3))
        ssim_scores = batch_ssim(fake, target.float())
        vgg_loss = vgg(fake, target.float())  # VGGPerceptual already handles fp32 prep
        # per-image vgg: re-run per sample would be slow; store batch mean replicated
        vgg_val = vgg_loss.item()

        for i in range(content.size(0)):
            rec = {
                "content_char": batch["content_char"][i],
                "target_font": batch["target_font"][i],
                "content_font": batch["content_font"][i],
                "l1": l1[i].item(),
                "ssim": ssim_scores[i].item(),
                "vgg": vgg_val,
            }
            records.append(rec)

            if save_images:
                img = (fake[i].cpu() + 1.0) / 2.0
                fname = f"{img_idx:05d}_{batch['target_font'][i]}_{batch['content_char'][i]}.png"
                save_image(img, images_dir / fname)
            img_idx += 1

    return records


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

SPLIT_COLORS = {
    "labels_val_unseen_font": "#4C72B0",
    "labels_val_unseen_char": "#DD8452",
    "labels_val_unseen_both": "#55A868",
}
SPLIT_LABELS = {
    "labels_val_unseen_font": "Unseen Font",
    "labels_val_unseen_char": "Unseen Char",
    "labels_val_unseen_both": "Unseen Both",
}


def plot_metrics_bar(summary: Dict, out_path: Path) -> None:
    splits = list(summary.keys())
    metrics = ["l1", "ssim", "vgg"]
    metric_labels = ["L1 ↓", "SSIM ↑", "VGG Perceptual ↓"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m, ml in zip(axes, metrics, metric_labels):
        vals = [summary[s][m] for s in splits]
        colors = [SPLIT_COLORS.get(s, "#888") for s in splits]
        labels = [SPLIT_LABELS.get(s, s) for s in splits]
        bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(ml, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("Validation Metrics by Split", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_histograms(all_records: Dict[str, List[Dict]], out_path: Path) -> None:
    metrics = ["l1", "ssim", "vgg"]
    metric_labels = ["L1", "SSIM", "VGG Perceptual"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    for ax, m, ml in zip(axes, metrics, metric_labels):
        for split, recs in all_records.items():
            vals = [r[m] for r in recs]
            color = SPLIT_COLORS.get(split, "#888")
            label = SPLIT_LABELS.get(split, split)
            ax.hist(vals, bins=50, alpha=0.6, color=color, label=label, density=True)
        ax.set_xlabel(ml, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Score Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_font_ssim(all_records: Dict[str, List[Dict]], out_path: Path, top_n: int = 20) -> None:
    # Combine all splits
    combined: Dict[str, List[float]] = {}
    for recs in all_records.values():
        for r in recs:
            combined.setdefault(r["target_font"], []).append(r["ssim"])

    # Sort by median, take top_n most-sampled fonts
    font_counts = {f: len(v) for f, v in combined.items()}
    selected = sorted(font_counts, key=lambda f: -font_counts[f])[:top_n]
    selected = sorted(selected, key=lambda f: np.median(combined[f]))

    fig, ax = plt.subplots(figsize=(14, 6))
    data = [combined[f] for f in selected]
    short_names = [f.split("_")[0][:18] for f in selected]

    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(selected) + 1))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("SSIM ↑", fontsize=11)
    ax.set_title(f"Per-Font SSIM Distribution (Top {top_n} fonts by sample count)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(np.concatenate(list(combined.values())).mean(), color="red",
               linestyle="--", linewidth=1, label="Global mean")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_l1_vs_ssim(all_records: Dict[str, List[Dict]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for split, recs in all_records.items():
        # subsample for readability
        step = max(1, len(recs) // 2000)
        xs = [r["l1"] for r in recs[::step]]
        ys = [r["ssim"] for r in recs[::step]]
        color = SPLIT_COLORS.get(split, "#888")
        label = SPLIT_LABELS.get(split, split)
        ax.scatter(xs, ys, c=color, alpha=0.3, s=8, label=label, rasterized=True)

    ax.set_xlabel("L1 ↓", fontsize=11)
    ax.set_ylabel("SSIM ↑", fontsize=11)
    ax.set_title("L1 vs SSIM (subsampled)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_visual_grid(
    G: Generator,
    csv_path: Path,
    device: torch.device,
    image_size: int,
    k_style: int,
    out_path: Path,
    n_samples: int = 16,
) -> None:
    dataset = FontPairDataset(
        labels_csv=csv_path,
        image_size=image_size,
        k_style=k_style,
        min_chars_per_font=k_style,
        augment=False,
    )
    if len(dataset) == 0:
        return
    # Fixed indices spread across the dataset
    indices = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset)), dtype=int).tolist()

    contents, styles, fakes, targets, chars, fonts = [], [], [], [], [], []
    for idx in tqdm(indices, desc="  grid samples", leave=False):
        sample = dataset[idx]
        content = sample["content_image"].unsqueeze(0).to(device)
        style_refs = sample["style_images"].unsqueeze(0).to(device)
        target = sample["target_image"]
        with torch.no_grad():
            fake = G(content, style_refs).squeeze(0).cpu()
        contents.append(content.squeeze(0).cpu())
        styles.append(style_refs[0, 0].cpu())  # first style ref
        fakes.append(fake)
        targets.append(target)
        chars.append(sample["content_char"])
        fonts.append(sample["target_font"].split("_")[0][:12])

    def to_np(t: torch.Tensor) -> np.ndarray:
        return ((t.squeeze().numpy() + 1) / 2).clip(0, 1)

    ncols = n_samples
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.4))
    row_labels = ["Content", "Style Ref", "Generated", "Target (GT)"]

    for row_idx, (row_data, row_label) in enumerate(zip(
        [contents, styles, fakes, targets], row_labels
    )):
        for col_idx, (img, char, font) in enumerate(zip(row_data, chars, fonts)):
            ax = axes[row_idx, col_idx]
            ax.imshow(to_np(img), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"{char}\n{font}", fontsize=6, pad=2)
        axes[row_idx, 0].set_ylabel(row_label, fontsize=9, rotation=90, labelpad=4)

    split_label = SPLIT_LABELS.get(csv_path.stem, csv_path.stem)
    fig.suptitle(f"Visual Results — {split_label}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate font style transfer model")
    p.add_argument("--checkpoint", type=str, default="latest.pt")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--k-style", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--save-images", action="store_true", help="Save all generated images to results/images/")
    p.add_argument("--grid-samples", type=int, default=16, help="Samples per visual grid")
    p.add_argument("--per-font-top-n", type=int, default=20, help="Fonts shown in box plot")
    p.add_argument(
        "--device",
        type=str,
        default=(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["labels_val_unseen_font", "labels_val_unseen_char", "labels_val_unseen_both"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    plots_dir = out_dir / "plots"
    grids_dir = out_dir / "grids"
    plots_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    G = load_generator(Path(args.checkpoint), device)
    vgg = VGGPerceptual().to(device).eval()
    for p in vgg.parameters():
        p.requires_grad_(False)

    all_records: Dict[str, List[Dict]] = {}
    summary: Dict[str, Dict] = {}

    splits_pbar = tqdm(args.splits, desc="Splits", unit="split")
    for split_name in splits_pbar:
        splits_pbar.set_postfix(split=split_name)
        csv_path = data_dir / f"{split_name}.csv"
        if not csv_path.exists():
            tqdm.write(f"  Skip {csv_path} (not found)")
            continue

        tqdm.write(f"\n[{split_name}]")
        images_dir = out_dir / "images" / split_name

        records = evaluate_split(
            G=G,
            vgg=vgg,
            csv_path=csv_path,
            device=device,
            image_size=args.image_size,
            k_style=args.k_style,
            batch_size=args.batch_size,
            save_images=args.save_images,
            images_dir=images_dir,
        )
        all_records[split_name] = records

        if not records:
            tqdm.write(f"  n=0  (no samples passed filters, skipping)")
            continue
        mean_l1 = float(np.mean([r["l1"] for r in records]))
        mean_ssim = float(np.mean([r["ssim"] for r in records]))
        mean_vgg = float(np.mean([r["vgg"] for r in records]))
        summary[split_name] = {"l1": mean_l1, "ssim": mean_ssim, "vgg": mean_vgg, "n": len(records)}

        tqdm.write(f"  n={len(records)}  L1={mean_l1:.4f}  SSIM={mean_ssim:.4f}  VGG={mean_vgg:.4f}")

        tqdm.write(f"  Generating visual grid...")
        plot_visual_grid(
            G=G,
            csv_path=csv_path,
            device=device,
            image_size=args.image_size,
            k_style=args.k_style,
            out_path=grids_dir / f"{split_name}.png",
            n_samples=args.grid_samples,
        )

    # Save metrics JSON
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics.json")

    # Save per-sample CSV
    csv_out = out_dir / "per_sample_metrics.csv"
    all_rows = []
    for split, recs in all_records.items():
        for r in recs:
            all_rows.append({"split": split, **r})
    if all_rows:
        with csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"Saved per_sample_metrics.csv  ({len(all_rows)} rows)")

    # Plots
    plot_tasks = []
    if len(summary) > 0:
        plot_tasks.append(("metrics_bar.png", lambda: plot_metrics_bar(summary, plots_dir / "metrics_bar.png")))
    if all_records:
        plot_tasks += [
            ("score_hist.png",    lambda: plot_score_histograms(all_records, plots_dir / "score_hist.png")),
            ("per_font_ssim.png", lambda: plot_per_font_ssim(all_records, plots_dir / "per_font_ssim.png", top_n=args.per_font_top_n)),
            ("l1_vs_ssim.png",    lambda: plot_l1_vs_ssim(all_records, plots_dir / "l1_vs_ssim.png")),
        ]
    for name, fn in tqdm(plot_tasks, desc="Plots", unit="plot"):
        tqdm.write(f"  {name}")
        fn()

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
