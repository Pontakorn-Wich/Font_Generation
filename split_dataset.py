"""Split labels.csv into train + 3 validation sets for evaluating generalization.

Produces:
    labels_train.csv             train fonts X train chars  (training data)
    labels_val_unseen_font.csv   held-out fonts X train chars  (style generalization)
    labels_val_unseen_char.csv   train fonts X held-out chars  (content generalization)
    labels_val_unseen_both.csv   held-out X held-out  (hardest test)
    split_meta.json              held-out font/char lists + counts + seed
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


FIELDNAMES = ["label_character", "font_name", "picture_path"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=str, default="data/labels.csv")
    p.add_argument("--out-dir", type=str, default="",
                   help="Directory for output splits (default: same as labels-csv)")
    p.add_argument("--val-fonts", type=int, default=50,
                   help="Number of held-out fonts (random per --seed)")
    p.add_argument("--val-chars", type=str, default="KQXjz",
                   help="String of held-out characters")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels_csv = Path(args.labels_csv)
    out_dir = Path(args.out_dir) if args.out_dir else labels_csv.parent

    rows = load_rows(labels_csv)
    if not rows:
        raise RuntimeError(f"{labels_csv} is empty")

    all_fonts: Set[str] = {r["font_name"] for r in rows}
    all_chars: Set[str] = {r["label_character"] for r in rows}

    rng = random.Random(args.seed)

    val_fonts_count = min(args.val_fonts, max(0, len(all_fonts) - 1))
    val_fonts: Set[str] = set(rng.sample(sorted(all_fonts), val_fonts_count))

    requested_val_chars = set(args.val_chars)
    val_chars: Set[str] = requested_val_chars & all_chars
    missing_chars = requested_val_chars - all_chars
    if missing_chars:
        print(f"warning: --val-chars contains chars not in dataset, skipping: {sorted(missing_chars)}")

    buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        font_held = r["font_name"] in val_fonts
        char_held = r["label_character"] in val_chars
        if not font_held and not char_held:
            buckets["train"].append(r)
        elif font_held and not char_held:
            buckets["val_unseen_font"].append(r)
        elif not font_held and char_held:
            buckets["val_unseen_char"].append(r)
        else:
            buckets["val_unseen_both"].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": out_dir / "labels_train.csv",
        "val_unseen_font": out_dir / "labels_val_unseen_font.csv",
        "val_unseen_char": out_dir / "labels_val_unseen_char.csv",
        "val_unseen_both": out_dir / "labels_val_unseen_both.csv",
    }
    for key, path in paths.items():
        write_rows(path, buckets.get(key, []))

    meta = {
        "source": str(labels_csv),
        "seed": args.seed,
        "total_rows": len(rows),
        "total_fonts": len(all_fonts),
        "total_chars": len(all_chars),
        "val_fonts": sorted(val_fonts),
        "val_chars": sorted(val_chars),
        "split_counts": {k: len(buckets.get(k, [])) for k in paths},
        "split_files": {k: str(p) for k, p in paths.items()},
    }
    meta_path = out_dir / "split_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Wrote splits to {out_dir}/")
    print(f"  total fonts={meta['total_fonts']}, total chars={meta['total_chars']}, total rows={meta['total_rows']}")
    print(f"  held-out fonts: {len(val_fonts)} | held-out chars: {sorted(val_chars)}")
    for k, count in meta["split_counts"].items():
        print(f"  {k:<22} {count:>8}  -> {paths[k].name}")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()
