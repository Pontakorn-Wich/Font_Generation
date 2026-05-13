from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_index(labels_csv: Path) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Set[str]]]:
    """Build (font -> char -> path) and (char -> {fonts}) indices from labels.csv."""
    font_to_char_to_path: Dict[str, Dict[str, str]] = defaultdict(dict)
    char_to_fonts: Dict[str, Set[str]] = defaultdict(set)
    with Path(labels_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            font = row["font_name"]
            ch = row["label_character"]
            path = row["picture_path"]
            font_to_char_to_path[font][ch] = path
            char_to_fonts[ch].add(font)
    return font_to_char_to_path, char_to_fonts


class FontPairDataset(Dataset):
    """Few-shot font style transfer dataset.

    Each item yields:
      content_image   target character rendered in some other (content) font
      style_images    K reference characters rendered in the target font
      target_image    target character rendered in the target font (ground truth)
    """

    def __init__(
        self,
        labels_csv: str | Path,
        image_size: int = 128,
        k_style: int = 4,
        min_chars_per_font: int = 5,
        seed: int | None = None,
    ) -> None:
        self.labels_csv = Path(labels_csv)
        self.image_size = image_size
        self.k_style = k_style

        font_to_char_to_path, char_to_fonts = load_index(self.labels_csv)

        self.font_to_char_to_path = {
            f: chars
            for f, chars in font_to_char_to_path.items()
            if len(chars) >= min_chars_per_font + 1
        }
        self.char_to_fonts = {
            ch: {f for f in fonts if f in self.font_to_char_to_path}
            for ch, fonts in char_to_fonts.items()
        }

        self.samples: List[Tuple[str, str]] = [
            (font, ch)
            for font, chars in self.font_to_char_to_path.items()
            for ch in chars
            if len(self.char_to_fonts.get(ch, ())) >= 2
        ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> torch.Tensor:
        return self.transform(Image.open(path).convert("L"))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        target_font, content_char = self.samples[idx]

        candidate_content_fonts = [
            f for f in self.char_to_fonts[content_char] if f != target_font
        ]
        content_font = (
            self._rng.choice(candidate_content_fonts)
            if candidate_content_fonts
            else target_font
        )

        target_chars = list(self.font_to_char_to_path[target_font].keys())
        style_pool = [c for c in target_chars if c != content_char]
        if len(style_pool) >= self.k_style:
            style_chars = self._rng.sample(style_pool, self.k_style)
        else:
            style_chars = self._rng.choices(style_pool or target_chars, k=self.k_style)

        content_path = self.font_to_char_to_path[content_font][content_char]
        target_path = self.font_to_char_to_path[target_font][content_char]
        style_paths = [self.font_to_char_to_path[target_font][c] for c in style_chars]

        return {
            "content_image": self._load(content_path),
            "style_images": torch.stack([self._load(p) for p in style_paths], dim=0),
            "target_image": self._load(target_path),
            "content_char": content_char,
            "target_font": target_font,
            "content_font": content_font,
        }
