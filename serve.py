"""FastAPI service exposing the trained font style transfer model.

Designed to be the inference backend for the Next.js UI in
https://github.com/AmaDeuSZodiacXz/gen-ai

Endpoints
    GET  /health                         — service status
    POST /api/transfer                   — multipart upload of K style refs +
                                           list of characters to generate.
                                           Returns base64 PNGs.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models import Generator

CHECKPOINT_PATH = Path(os.getenv("CHECKPOINT_PATH", "runs/exp1/ckpt/latest.pt"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "128"))
FONT_SIZE = int(os.getenv("FONT_SIZE", "96"))
STYLE_DIM = int(os.getenv("STYLE_DIM", "128"))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# A pre-rendered "neutral" font used as content source when the client
# requests characters by unicode codepoint. The UI uploads only style refs,
# and we synthesize content on the server side from this font.
NEUTRAL_FONT_FILE = Path(os.getenv("NEUTRAL_FONT_FILE", ""))
NEUTRAL_IMAGE_DIR = Path(os.getenv("NEUTRAL_IMAGE_DIR", ""))

app = FastAPI(title="Font Style Transfer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

_generator: Optional[Generator] = None
_neutral_pil_font: Optional[ImageFont.FreeTypeFont] = None


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        if not CHECKPOINT_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Checkpoint not found at {CHECKPOINT_PATH}. Train first.",
            )
        g = Generator(image_channels=1, style_dim=STYLE_DIM).to(DEVICE)
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        g.load_state_dict(state["G"])
        g.eval()
        _generator = g
    return _generator


def _autodetect_neutral_font() -> Optional[Path]:
    fonts_dir = Path("data/fonts")
    if not fonts_dir.exists():
        return None
    preferred = ["NotoSans-Regular", "Roboto-Regular", "OpenSans-Regular", "DejaVuSans"]
    candidates = sorted(list(fonts_dir.glob("*.ttf")) + list(fonts_dir.glob("*.otf")))
    for keyword in preferred:
        for c in candidates:
            if keyword.lower() in c.name.lower():
                return c
    for c in candidates:
        try:
            ImageFont.truetype(str(c), FONT_SIZE)
            return c
        except OSError:
            continue
    return None


def render_neutral_glyph(ch: str) -> Image.Image:
    """Render `ch` in the neutral content font, returning a 128x128 grayscale PIL."""
    global _neutral_pil_font
    if _neutral_pil_font is None:
        font_file = NEUTRAL_FONT_FILE if NEUTRAL_FONT_FILE.is_file() else _autodetect_neutral_font()
        if font_file is None:
            raise HTTPException(
                status_code=503,
                detail="No neutral content font available. Set NEUTRAL_FONT_FILE env var.",
            )
        try:
            _neutral_pil_font = ImageFont.truetype(str(font_file), FONT_SIZE)
        except OSError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot open neutral font {font_file}: {e}",
            )

    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=_neutral_pil_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (IMAGE_SIZE - width) // 2 - bbox[0]
    y = (IMAGE_SIZE - height) // 2 - bbox[1]
    draw.text((x, y), ch, font=_neutral_pil_font, fill=0)
    return img


def _decode_upload(file_bytes: bytes) -> torch.Tensor:
    pil = Image.open(io.BytesIO(file_bytes)).convert("L")
    return _transform(pil)


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    return _transform(pil)


def _encode_png(tensor: torch.Tensor) -> str:
    arr = ((tensor.clamp(-1, 1) + 1) / 2 * 255).round().byte().squeeze().cpu().numpy()
    pil = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": DEVICE,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "image_size": IMAGE_SIZE,
    }


@app.post("/api/transfer")
async def transfer(
    style_files: List[UploadFile] = File(..., description="K reference images of the target font style"),
    characters: str = Form("", description="String of characters to generate (e.g. 'Hello123')"),
    content_files: Optional[List[UploadFile]] = File(None, description="Optional explicit content images"),
) -> dict:
    if not style_files:
        raise HTTPException(status_code=400, detail="At least one style image is required")
    if not characters and not content_files:
        raise HTTPException(
            status_code=400,
            detail="Provide either `characters` (string) or `content_files` (uploads)",
        )

    G = get_generator()

    style_tensors = [_decode_upload(await f.read()) for f in style_files]
    style_batch = torch.stack(style_tensors, dim=0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        style_code = G.style_encoder(style_batch)

    results: List[dict] = []

    if content_files:
        for f in content_files:
            content_tensor = _decode_upload(await f.read()).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                fake = G.decode(G.content_encoder(content_tensor), style_code)
            results.append(
                {
                    "label": f.filename,
                    "image_base64": _encode_png(fake[0]),
                }
            )

    for ch in characters:
        if ch.isspace():
            continue
        content_pil = render_neutral_glyph(ch)
        content_tensor = _pil_to_tensor(content_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake = G.decode(G.content_encoder(content_tensor), style_code)
        results.append(
            {
                "label": ch,
                "codepoint": f"U+{ord(ch):04X}",
                "image_base64": _encode_png(fake[0]),
            }
        )

    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
