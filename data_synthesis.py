from __future__ import annotations

import argparse
import csv
import itertools
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

GOOGLE_FONTS_TREE_API = "https://api.github.com/repos/google/fonts/git/trees/main?recursive=1"
GOOGLE_FONTS_RAW_BASE = "https://raw.githubusercontent.com/google/fonts/main/"


def safe_name(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in "-_":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def github_headers() -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def list_google_font_paths(session: requests.Session) -> List[str]:
    response = session.get(GOOGLE_FONTS_TREE_API, timeout=60)
    response.raise_for_status()
    tree = response.json().get("tree", [])

    out: List[str] = []
    for item in tree:
        if item.get("type") != "blob":
            continue
        path = item.get("path", "")
        if not (path.endswith(".ttf") or path.endswith(".otf")):
            continue
        if "apache/" in path or "ofl/" in path or "ufl/" in path:
            out.append(path)
    return out


def download_google_fonts(font_dir: Path, max_fonts: int = 0) -> int:
    font_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    with requests.Session() as session:
        session.headers.update(github_headers())
        paths = list_google_font_paths(session)
        if max_fonts > 0:
            paths = paths[:max_fonts]
        for path in tqdm(paths, desc="Downloading fonts"):
            output_path = font_dir / path.replace("/", "_")
            if output_path.exists():
                continue

            url = GOOGLE_FONTS_RAW_BASE + path
            response = session.get(url, timeout=60)
            if response.status_code == 200:
                output_path.write_bytes(response.content)
                downloaded += 1
    return downloaded


def glyph_exists(font: ImageFont.FreeTypeFont, ch: str) -> bool:
    try:
        mask = font.getmask(ch)
        return mask.getbbox() is not None
    except Exception:
        return False


def render_character_images(
    font_dir: Path,
    image_dir: Path,
    labels_csv_path: Path,
    mapping_csv_path: Path,
    chars: str,
    image_size: int = 128,
    font_size: int = 96,
    cleanup: bool = True,
    min_chars_per_font: Optional[int] = None,
) -> Dict[str, object]:
    """Render every (font, char) pair.

    With ``cleanup=True``:
      A) fonts that PIL cannot load           → .ttf deleted
      B+C) fonts rendering fewer than          → .ttf + image folder deleted
           ``min_chars_per_font`` glyphs
           (defaults to len(chars) — strict)

    Returns rich stats including which fonts were removed.
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    if min_chars_per_font is None:
        min_chars_per_font = len(chars)

    rows: List[Dict[str, str]] = []
    index_by_char: Dict[str, List[Dict[str, str]]] = {}
    failed_pil_load: List[Dict[str, str]] = []      # case A: PIL truetype raised
    deleted_ttf_pil_fail: List[str] = []            # case A: ttf removed
    deleted_incomplete: List[Dict[str, object]] = []  # case B+C: ttf + dir removed
    glyphs_per_font: Dict[str, int] = {}
    font_to_path: Dict[str, Path] = {}

    font_files = sorted(list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf")))

    for font_path in tqdm(font_files, desc="Rendering"):
        font_name = safe_name(font_path.stem)
        font_to_path[font_name] = font_path
        try:
            font = ImageFont.truetype(str(font_path), font_size)
        except Exception as e:
            failed_pil_load.append({"file": font_path.name, "error": str(e)[:80]})
            if cleanup:
                try:
                    font_path.unlink()
                    deleted_ttf_pil_fail.append(font_path.name)
                except OSError:
                    pass
            continue

        font_out_dir = image_dir / font_name
        font_out_dir.mkdir(parents=True, exist_ok=True)
        rendered_here = 0

        for ch in chars:
            if not glyph_exists(font, ch):
                continue

            img = Image.new("L", (image_size, image_size), color=255)
            draw = ImageDraw.Draw(img)

            bbox = draw.textbbox((0, 0), ch, font=font)
            if bbox is None:
                continue

            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            x = (image_size - width) // 2 - bbox[0]
            y = (image_size - height) // 2 - bbox[1]

            draw.text((x, y), ch, font=font, fill=0)

            code = f"U+{ord(ch):04X}"
            image_path = font_out_dir / f"{code}.png"
            img.save(image_path)
            rendered_here += 1

            rel_path = image_path.as_posix()
            row = {
                "label_character": ch,
                "font_name": font_name,
                "picture_path": rel_path,
            }
            rows.append(row)
            index_by_char.setdefault(ch, []).append(row)

        glyphs_per_font[font_name] = rendered_here

    # === Strict cleanup: remove fonts that rendered fewer than min_chars_per_font ===
    if cleanup:
        bad = set()
        for font_name, count in list(glyphs_per_font.items()):
            if count < min_chars_per_font:
                out_dir = image_dir / font_name
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                ttf = font_to_path.get(font_name)
                if ttf is not None and ttf.exists():
                    try:
                        ttf.unlink()
                    except OSError:
                        pass
                deleted_incomplete.append({
                    "font": font_name,
                    "rendered": count,
                    "required": min_chars_per_font,
                })
                bad.add(font_name)
        if bad:
            rows = [r for r in rows if r["font_name"] not in bad]
            for ch in list(index_by_char.keys()):
                index_by_char[ch] = [r for r in index_by_char[ch] if r["font_name"] not in bad]
                if not index_by_char[ch]:
                    del index_by_char[ch]
            for name in bad:
                glyphs_per_font.pop(name, None)

    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label_character", "font_name", "picture_path"])
        writer.writeheader()
        writer.writerows(rows)

    mapping_rows: List[Dict[str, str]] = []
    for ch, items in index_by_char.items():
        for source, target in itertools.combinations(items, 2):
            mapping_rows.append(
                {
                    "label_character": ch,
                    "source_font": source["font_name"],
                    "source_picture_path": source["picture_path"],
                    "target_font": target["font_name"],
                    "target_picture_path": target["picture_path"],
                }
            )

    with mapping_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label_character",
                "source_font",
                "source_picture_path",
                "target_font",
                "target_picture_path",
            ],
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    return {
        "font_files_scanned": len(font_files),
        "complete_fonts": len(glyphs_per_font),
        "labels_rows": len(rows),
        "mapping_rows": len(mapping_rows),
        "required_glyphs": min_chars_per_font,
        "failed_pil_load": failed_pil_load,
        "deleted_ttf_pil_fail": deleted_ttf_pil_fail,
        "deleted_incomplete": deleted_incomplete,
        "glyphs_per_font": glyphs_per_font,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download fonts and synthesize character image dataset")
    parser.add_argument("--root", type=str, default=".", help="Project root path")
    parser.add_argument(
        "--chars",
        type=str,
        default="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        help="Characters to render",
    )
    parser.add_argument("--image-size", type=int, default=128, help="Output image size")
    parser.add_argument("--font-size", type=int, default=96, help="Text drawing font size")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading fonts and only render using existing font files",
    )
    parser.add_argument(
        "--max-fonts",
        type=int,
        default=0,
        help="Limit number of fonts to download (0 = all). Useful for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    font_dir = root / "data" / "fonts"
    image_dir = root / "data" / "images"
    labels_csv = root / "data" / "labels.csv"
    mapping_csv = root / "data" / "char_mapping.csv"

    if not args.skip_download:
        downloaded = download_google_fonts(font_dir, max_fonts=args.max_fonts)
        print(f"Downloaded {downloaded} new font files into {font_dir}")

    stats = render_character_images(
        font_dir=font_dir,
        image_dir=image_dir,
        labels_csv_path=labels_csv,
        mapping_csv_path=mapping_csv,
        chars=args.chars,
        image_size=args.image_size,
        font_size=args.font_size,
    )

    print("Synthesis completed.")
    print(f"Font files scanned   : {stats['font_files_scanned']}")
    print(f"Complete fonts kept  : {stats['complete_fonts']}  (required {stats['required_glyphs']} glyphs each)")
    print(f"Rows in labels.csv   : {stats['labels_rows']}")
    print(f"Rows in mapping.csv  : {stats['mapping_rows']}")
    if stats["deleted_ttf_pil_fail"]:
        print(f"\nDeleted {len(stats['deleted_ttf_pil_fail'])} font(s) PIL could not load:")
        for name in stats["deleted_ttf_pil_fail"]:
            print(f"  - {name}")
    if stats["deleted_incomplete"]:
        print(f"\nDeleted {len(stats['deleted_incomplete'])} incomplete font(s) "
              f"(< {stats['required_glyphs']} glyphs):")
        for x in sorted(stats["deleted_incomplete"], key=lambda d: d["rendered"]):
            print(f"  - {x['font']}  ({x['rendered']}/{x['required']} glyphs)")


if __name__ == "__main__":
    main()
