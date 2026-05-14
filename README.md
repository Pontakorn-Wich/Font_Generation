# Font_Generation

Few-shot font style transfer. Given a handful of reference glyphs in some
style, the model generates any other character in that same style.

The implementation here is **the second iteration**. The first design
(plain content encoder → AdaIN decoder → PatchGAN) collapsed during
training and produced the same blob shape regardless of input. This
version uses a U-Net generator with skip connections, a transformer style
encoder, a multi-task discriminator, and VGG perceptual loss — fixing the
collapse and producing recognisable glyphs.

This README explains every architectural and training choice in enough
depth that the reasoning, not just the recipe, is documented.

---

## Table of contents

1. [Problem definition](#1-problem-definition)
2. [Architecture, in depth](#2-architecture-in-depth)
3. [Loss design](#3-loss-design)
4. [Training tricks](#4-training-tricks)
5. [Data pipeline](#5-data-pipeline)
6. [Step-by-step runbook](#6-step-by-step-runbook)
7. [Hyperparameter reference](#7-hyperparameter-reference)
8. [Wall-time estimates](#8-wall-time-estimates)
9. [Repository layout](#9-repository-layout)
10. [Colab notebook](#10-colab-notebook)
11. [Inference and serving](#11-inference-and-serving)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Problem definition

Given a target font that the model has never seen at training time and only
a few (K = 4) reference glyphs of that font, generate the same font's
rendering of an arbitrary other character.

Inputs:

```
content_image   (B, 1, 128, 128)         single glyph (e.g. "H" rendered
                                          in some neutral font) — tells the
                                          model what character to draw
style_images    (B, K, 1, 128, 128)      K example glyphs from the target
                                          font (any K characters except the
                                          one being requested) — tells the
                                          model what style to draw in
```

Output:

```
fake            (B, 1, 128, 128)         the requested character rendered
                                          in the target style
```

The training objective is to learn an encoder that disentangles **content**
("which character is this glyph") from **style** ("what does this font
look like") so that at test time we can mix any content with any style.

---

## 2. Architecture, in depth

```
                                      ┌──────────────────────────────────┐
content_image  ──► ContentEncoder ──► │ feature pyramid (5 scales)       │ ─┐
                  (U-Net encoder)     │ 128×128×64 → … → 8×8×512         │  │
                                      └──────────────────────────────────┘  │
                                                                            ▼
                                                                     Decoder
                                      ┌──────────────────────────────────┐  ▲
K style refs   ──► StyleEncoder    ──►│ style vector  (B, 256)           │ ─┘
                  (CNN + transformer) │ aggregated via attention + CLS    │
                                      └──────────────────────────────────┘
                                                                            ▼
                                                                          fake
```

### 2.1 ContentEncoder — a U-Net encoder

Implementation: `models.py: ContentEncoder`.

- 5 levels, each halving the spatial resolution: `128 → 64 → 32 → 16 → 8`
- Channel widths: `64 → 128 → 256 → 512 → 512` (capped at 512)
- Each downsampling block is `ConvBlock(stride=2)` + `ResBlock`
- Returns a **list** of feature maps from every level, not just the
  bottleneck

The reason features at every level are kept and exposed is for U-Net skip
connections in the decoder. In the previous (collapsed) design the entire
content signal had to squeeze through the 8×8 bottleneck and was lost; the
decoder then had to "guess" the glyph shape from style alone, found that
producing one universal blob was a local minimum of the L1 loss, and
collapsed. With skip connections, the spatial structure of the input glyph
is *carried verbatim* into the decoder — collapse becomes geometrically
impossible.

### 2.2 StyleEncoder — CNN + transformer aggregation

Implementation: `models.py: StyleEncoder`.

For each reference image:
1. A six-layer convolutional trunk reduces it to a flat feature.
2. A linear projection maps to a 256-D style token.

Across the K reference tokens:
3. A learnable **CLS token** is prepended.
4. One multi-head self-attention layer mixes information between tokens.
5. The CLS token's output is returned as the final style vector `(B, 256)`.

Why attention instead of `mean()`:

- Different reference glyphs carry different amounts of style information.
  The serif of an `R` tells you more about a typeface than the dot of an
  `i`. Mean pooling weights them equally; attention learns to weight them
  by informativeness.
- The CLS-token / transformer aggregator is invariant to the number K, so
  the model works at inference with any number of reference glyphs the
  user uploads, not just K=4.

### 2.3 Decoder — U-Net decoder with AdaIN modulation

Implementation: `models.py: Decoder`, `UpBlock`, `AdaINResBlock`, `AdaIN`.

Starts from the bottleneck (8×8×512), applies several AdaIN residual
blocks at full depth, then upsamples four times. At every upsample step
the block does:

1. **AdaIN** modulation conditioned on the style vector
2. **Upsample 2×** (nearest)
3. **Concat** with the corresponding encoder feature map (skip connection)
4. **Conv** to mix the merged channels
5. **AdaIN** again
6. **Conv** with residual

AdaIN is the same modulation operator StyleGAN uses: per-channel
instance-normalise the activations, then re-scale and shift with
`(gamma, beta) = MLP(style_vector)`. The decoder's *shape* comes from the
skip connections; the decoder's *texture, weight, slant* comes from
AdaIN's per-channel rescaling. This is a clean separation that maps
directly onto the content/style decomposition the model is trying to
learn.

The final layer is a `Conv → tanh`, so outputs are in `[-1, 1]`. Images
are stored on disk as `[0, 1]` after the trainer transforms them back.

### 2.4 Discriminator — PatchGAN + auxiliary classifiers

Implementation: `models.py: Discriminator`.

- Spectral-normalised convolutional trunk (5 layers)
- Three heads share the trunk:
  1. **Patch head**: 7×7 grid of real/fake logits (PatchGAN)
  2. **Font classifier**: predicts which font the image came from
  3. **Character classifier**: predicts which character is shown

Why three heads?

- The patch head alone tells G "is this realistic somewhere?" but
  doesn't say whether it's realistic *for this font* or *for this
  character*. With only the patch signal it is easy for D to converge to
  a feature space that ignores style, which lets G ignore style too.
- The auxiliary classifiers force D's trunk to produce features that
  separate fonts from each other and characters from each other. When G
  is then asked to fool these classifiers, it has to produce outputs
  that the trunk *recognises as the right font and the right
  character*. That is a much sharper training signal than a binary
  real/fake.
- Spectral normalisation Lipschitz-bounds D, which combined with the R1
  penalty (see §3.5) is the most reliable D regularisation we know of.

### 2.5 VGG perceptual loss

Implementation: `models.py: VGGPerceptual`.

A frozen ImageNet VGG16 is wrapped so that grayscale inputs are converted
to 3-channel and renormalised. L1 distance is computed at four
intermediate layers (`relu1_2, relu2_2, relu3_3, relu4_3`) and averaged.

Why perceptual loss is the workhorse:

- Pixel L1 has a degenerate global minimum: the *mean* of all targets. A
  small G that always outputs the same blurry blob can drive pixel L1 to
  a low (but useless) value. This is exactly what happened in the
  previous design.
- VGG features are sensitive to *structure*: edges, junctions, stroke
  widths. The L1 distance in VGG feature space is small only when the
  outputs look the same to a network trained on natural images. There is
  no "average glyph" that simultaneously minimises distance to every
  glyph in feature space — the perceptual minimum is genuinely close to
  the target.

### Parameter counts

```
Generator:      50.0M parameters (39.5M G + 10.5M attention)
Discriminator:   2.9M parameters
VGG (frozen):   14.7M parameters (no gradient)
```

---

## 3. Loss design

The total generator loss is a weighted sum:

```
L_G = λ_adv    · L_adv_G         (1.0)   adversarial signal from D
    + λ_vgg    · L_perceptual    (5.0)   VGG feature distance — workhorse
    + λ_rec    · L_rec_L1        (0.5)   pixel L1 — small, just anchors intensity
    + λ_con    · L_content       (2.0)   E_c(fake) ≈ E_c(content)
    + λ_sty    · L_style         (1.0)   E_s(fake) ≈ E_s(refs)
    + λ_font   · L_font_cls      (1.0)   D's font classifier on fake
    + λ_char   · L_char_cls      (1.0)   D's char classifier on fake
```

The discriminator loss is:

```
L_D = λ_adv  · L_adv_D           hinge real/fake on the patch head
    + λ_font · CE(font_real)     font classification on real images
    + λ_char · CE(char_real)     char classification on real images
    + λ_r1   · R1(D, real)       gradient penalty, every 16 steps
```

### 3.1 Adversarial: hinge GAN

```
L_adv_D = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))
L_adv_G = -mean(D(fake))
```

The hinge formulation avoids the vanishing-gradient pathology of the
original Goodfellow `log(1-D(fake))` formulation when D becomes confident.
It is the de-facto choice for modern image GANs.

### 3.2 VGG perceptual

Already explained in §2.5. Weight 5.0 puts it dominant over pixel L1.

### 3.3 Pixel L1 (small weight)

Kept at 0.5 — just enough to nudge mean intensity towards correct, not
enough to dominate. In the old run this was at 10.0 and was the direct
cause of collapse.

### 3.4 Consistency: content and style

These are "cycle-like" regularisers:

- **Content consistency**: `L1(E_c(fake), E_c(content).detach())`
  averaged over every encoder scale. Forces the content path to be
  preserved through G — if the fake glyph isn't "the same character",
  the content encoder will disagree.
- **Style consistency**: `L1(E_s(fake.unsqueeze(1)), E_s(refs).detach())`.
  Forces the style code re-extracted from the generated image to match
  the style code used to generate it.

### 3.5 R1 gradient penalty (lazy)

```
R1 = 0.5 · mean(‖∂D(real)/∂real‖²)
```

This penalises D for having large gradients on real images, which keeps D
Lipschitz-smooth and prevents the discriminator from overfitting to
spurious local features. Applied **lazily** (every 16 steps) for speed
following the StyleGAN convention. The weight is multiplied by 16 so the
effective penalty matches what it would be if applied every step.

### 3.6 Auxiliary classification

Cross-entropy loss on:
- D's font head over real images (D learns to identify fonts)
- D's char head over real images (D learns to identify characters)
- G must fool both heads on its fakes

This is the strongest signal in the loss bundle for style/content
disentanglement. A G that produces wrong-font or wrong-character outputs
gets penalised twice (adversarial + aux), not once.

---

## 4. Training tricks

### 4.1 EMA generator

A copy of the generator's weights is kept and updated each step as

```
W_ema ← decay · W_ema + (1 - decay) · W_current        (decay = 0.999)
```

This averaged generator is what `save_image` writes for visualisation and
what `inference.py` and `serve.py` load at deploy time. EMA filters out
high-frequency noise in the training trajectory and produces noticeably
cleaner outputs than the raw generator at the same step. There is a 1k
step warm-up during which the EMA just *copies* the current weights (so
it doesn't lag uselessly from random init).

### 4.2 TTUR (Two-timescale update rule)

D learns at `4e-4`, G learns at `1e-4`. The 4× faster D matches the
StyleGAN-2-ADA convention. With aux classifiers and R1 we want D to be
strong; G in turn benefits from a stronger D's gradients.

### 4.3 bf16 autocast

Forward passes are wrapped in `torch.autocast(device_type=..., dtype=torch.bfloat16)`.
bfloat16 has fp32's dynamic range but half the memory bandwidth, which
gives roughly a 2× speedup on T4/A100 GPUs and on Apple Silicon (MPS).
The VGG perceptual call is forced back to fp32 because the pretrained
weights expect it.

### 4.4 Adam betas (0, 0.99)

StyleGAN-style optimizer settings. β1 = 0 (no momentum on first moment)
pairs well with R1 — momentum on the discriminator's adversarial gradient
can destabilise training in the presence of gradient penalties.

### 4.5 Gradient clipping

Both G and D are gradient-clipped to L2-norm 1.0. Cheap insurance against
the occasional pathological batch.

### 4.6 Light augmentation on content only

`transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.93, 1.07))`
is applied to the content image only, with white fill.

The model only ever sees synthetically-rendered, perfectly-centred glyphs
during training, but at inference the user will upload imperfect images —
glyphs that are slightly off-centre, slightly rotated, slightly scaled.
Without augmentation the content encoder grows brittle to those
real-world perturbations. We augment **only the content image** because
target and style refs come from the same canonical dataset and should
not drift.

---

## 5. Data pipeline

### 5.1 Synthesis with strict cleanup

`data_synthesis.py` downloads .ttf/.otf files from the `google/fonts`
GitHub repo (apache/ofl/ufl licensed) and rasterises each character into
a 128×128 grayscale PNG centred on the canvas.

**Strict cleanup policy.** After rendering, every font that did not
render **every** requested character is deleted entirely — both the .ttf
file and any partial image folder. The training set therefore contains
only fonts with uniform glyph coverage. Three failure modes are detected:

```
A) PIL ImageFont.truetype raises                → .ttf deleted
B) PIL loads, font has 0 Latin glyphs           → .ttf + empty dir deleted
C) PIL loads, font renders < len(CHARS) glyphs  → .ttf + partial dir deleted
```

A JSON report at `data/synthesis_report.json` lists every cleanup
action for forensics.

You can relax the policy by passing
`min_chars_per_font=<N>` to `render_character_images` (kept if it
renders at least N glyphs). The default (`None`) requires all
`len(CHARS)` glyphs — strictest possible.

### 5.2 Train/val splits

`split_dataset.py` slices `labels.csv` into four files:

```
labels_train.csv             train fonts × train chars   — training data
labels_val_unseen_font.csv   held-out fonts × train chars — tests style generalization
labels_val_unseen_char.csv   train fonts × held-out chars — tests content generalization
labels_val_unseen_both.csv   held-out × held-out         — hardest combined test
```

A `data/split_meta.json` records the seed, the chosen held-out font list,
and the held-out chars so the split is fully reproducible.

The split has two knobs:

```
--val-fonts 50      number of fonts random-sampled for validation
--val-chars KQXjz   characters to hold out
--seed 42           RNG seed for reproducibility
```

If you want **all** characters available at training time (at the cost
of losing the unseen-char generalisation metric), pass `--val-chars ""`.

### 5.3 Why both held-out fonts AND held-out chars

Real-world inference always involves a font the model has *never* seen —
that is the entire point of few-shot style transfer. `val_unseen_font`
directly measures performance in that scenario.

`val_unseen_char` instead tests whether the model has learned a *general*
notion of character structure or has just memorised the specific glyph
shapes in training. A model that scores well on `val_unseen_font` but
poorly on `val_unseen_char` is overfitting to the character vocabulary —
it knows what an `R` looks like in many fonts but can't draw an `R`
robustly when asked for a *novel* character.

---

## 6. Step-by-step runbook

This is the local Apple-Silicon workflow. For Colab use the notebook —
see §10.

### 6.1 Environment setup (once per machine)

```bash
git clone https://github.com/Pontakorn-Wich/Font_Generation.git
cd Font_Generation
git checkout feat/font-style-transfer-model
pip install -r requirements.txt
```

Verify GPU/MPS:

```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available(), '| cuda:', torch.cuda.is_available())"
```

(Optional) GitHub PAT to skip the 60-req/hour anonymous rate limit:

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 6.2 Data synthesis

```bash
python data_synthesis.py --max-fonts 500
```

- `--max-fonts 100` — quick smoke (~3 min)
- `--max-fonts 500` — recommended baseline (~10–15 min)
- omit / `0`     — all Google Fonts (~25 min)

The script will print, at the end:

```
Synthesis completed.
Font files scanned   : 500
Complete fonts kept  : 478  (required 62 glyphs each)
Rows in labels.csv   : 29636
Rows in mapping.csv  : ...
Deleted 2 font(s) PIL could not load:
  - ...
Deleted 20 incomplete font(s) (< 62 glyphs):
  -  0/62  apache_iconfont_X
  - 30/62  ofl_partialfont_Y
  ...
```

### 6.3 Train/val split

```bash
python split_dataset.py --val-fonts 50 --val-chars KQXjz --seed 42
```

### 6.4 Training (long, often overnight)

```bash
tmux new -s train

python train.py \
  --labels-csv data/labels_train.csv \
  --out-dir runs/v2_unet \
  --device mps \
  --batch-size 8 \
  --num-workers 0 \
  --k-style 4 \
  --epochs 60 \
  --save-every 2 \
  2>&1 | tee runs/v2_unet/train.log
```

(Inside tmux: detach with `Ctrl+B`, then `D`. Reattach later with
`tmux attach -t train`.)

CUDA variant:

```bash
python train.py --labels-csv data/labels_train.csv --out-dir runs/v2_unet \
  --device cuda --batch-size 32 --num-workers 4 --epochs 60 --save-every 2
```

### 6.5 Monitor (second terminal)

```bash
open runs/v2_unet/samples/$(ls runs/v2_unet/samples/ | tail -1)
```

Each saved grid has four rows top → bottom:

1. **content** — the requested character in a source font
2. **style ref** — one reference glyph from the target font
3. **generated** ★ — what the model produced
4. **ground truth** — the correct answer

A healthy model has row 3 ≈ row 4 by ~epoch 15.

### 6.6 Resume from a crash

```bash
python train.py \
  --labels-csv data/labels_train.csv \
  --out-dir runs/v2_unet \
  --device mps --batch-size 8 --num-workers 0 \
  --epochs 60 --save-every 2 \
  --resume runs/v2_unet/ckpt/latest.pt
```

### 6.7 Smoke test inference

```bash
mkdir -p /tmp/style_refs /tmp/content_chars
cp data/images/<some_handwritten_font>/U+0042.png /tmp/style_refs/  # B
cp data/images/<some_handwritten_font>/U+0043.png /tmp/style_refs/  # C
cp data/images/<some_handwritten_font>/U+0044.png /tmp/style_refs/  # D
cp data/images/<some_handwritten_font>/U+0045.png /tmp/style_refs/  # E
cp data/images/<some_other_font>/U+0048.png /tmp/content_chars/     # H
cp data/images/<some_other_font>/U+0069.png /tmp/content_chars/     # i

python inference.py \
  --checkpoint runs/v2_unet/ckpt/latest.pt \
  --content-dir /tmp/content_chars \
  --style-dir   /tmp/style_refs \
  --output-dir  /tmp/output

open /tmp/output/U+0048.png    # "H" rendered in the chosen handwriting style
```

---

## 7. Hyperparameter reference

| Flag | Default | Notes |
| --- | --- | --- |
| `--labels-csv` | `data/labels_train.csv` | use the train split, not the full labels |
| `--out-dir` | `runs/v2` | checkpoints + sample grids go here |
| `--device` | auto (cuda → mps → cpu) | force with `--device cuda` |
| `--batch-size` | 8 | 8 on MPS, 32 on a 4090, 64 on A100 |
| `--num-workers` | 0 | keep 0 on macOS (MPS fork issues) |
| `--k-style` | 4 | references per sample |
| `--style-dim` | 256 | style code dimension |
| `--epochs` | 60 | total epochs |
| `--save-every` | 2 | save & visualise every N epochs |
| `--g-lr` | 1e-4 | generator learning rate |
| `--d-lr` | 4e-4 | discriminator learning rate (TTUR) |
| `--beta1` | 0.0 | Adam β1 (StyleGAN convention) |
| `--beta2` | 0.99 | Adam β2 |
| `--ema-decay` | 0.999 | EMA generator decay |
| `--lambda-adv` | 1.0 | adversarial weight |
| `--lambda-perceptual` | 5.0 | VGG perceptual (workhorse) |
| `--lambda-rec` | 0.5 | pixel L1 (small intentionally) |
| `--lambda-content` | 2.0 | content consistency |
| `--lambda-style` | 1.0 | style consistency |
| `--lambda-font-cls` | 1.0 | font aux classification |
| `--lambda-char-cls` | 1.0 | char aux classification |
| `--lambda-r1` | 10.0 | R1 gradient penalty (multiplied by 16 for lazy reg) |
| `--limit-samples` | 0 | use only the first N samples (smoke testing) |
| `--no-bf16` | off | disable bf16 autocast |
| `--resume` | "" | path to a checkpoint to continue from |

---

## 8. Wall-time estimates

500 fonts ≈ 30K training samples, batch=16 ≈ 1.9K batches/epoch.

| Hardware | Batch | sec/iter | min/epoch | 60 epochs |
| --- | --- | --- | --- | --- |
| Apple M1/M2 MPS | 8 | ~0.35 | ~6 | **~6 h** |
| Apple M3/M4 Max MPS | 8 | ~0.22 | ~4 | ~4 h |
| RTX 4090 | 32 | ~0.10 | ~0.5 | ~30 min |
| A100 80GB | 64 | ~0.04 | ~0.2 | ~12 min |

Each checkpoint is ~500 MB (G + G_ema + D + 2 optimizer states). Plan for
3–5 GB of `runs/v2_unet/` over a 60-epoch run.

---

## 9. Repository layout

```
data_synthesis.py    Download Google Fonts and render character PNGs.
                     Strict cleanup deletes incomplete fonts.
split_dataset.py     Split labels.csv into 1 train + 3 validation CSVs.
dataset.py           PyTorch Dataset. Returns (content, K-style refs,
                     target, font_id, char_id). Light affine aug on content.
models.py            Generator (U-Net + transformer style + AdaIN decoder),
                     Discriminator (PatchGAN + aux heads, spectral norm),
                     VGGPerceptual.
train.py             Training loop: hinge GAN + VGG + L1 + content / style
                     consistency + font / char aux + R1, with EMA, TTUR,
                     bf16, gradient clipping.
inference.py         CLI: load EMA checkpoint, generate from content + refs.
serve.py             FastAPI server for the web UI.
requirements.txt     torch, torchvision, pillow, tqdm, requests, fastapi,
                     uvicorn, python-multipart.

font_style_transfer_colab.ipynb
                     End-to-end notebook for Google Colab: synthesis,
                     split, model, training, eval, inference all in one.

data/                (gitignored) fonts, rendered images, csv labels,
                     synthesis_report.json, split_meta.json.
runs/                (gitignored) per-experiment checkpoints + samples.
```

---

## 10. Colab notebook

`font_style_transfer_colab.ipynb` contains the whole pipeline as a single
notebook so you can train on a free T4 or paid A100 without setting up
anything local. It has nine sections:

```
0. Setup                       install + GPU check + optional Drive mount
1. Data synthesis              configure MAX_FONTS, run download + render
1b. Synthesis diagnostic       table of what got deleted and why
2. Train/val split             carve out three validation sets
3. Model architecture          U-Net + style transformer + multi-task D
4. Dataset class               same as dataset.py but inlined
5. Training                    config + 60-epoch loop with inline preview
6. Visualize                   open any saved sample grid
7. Evaluation                  L1 + VGG on train + 3 val splits
8. Inference demo              upload refs, type text, see grid of results
```

To use:

1. Open the .ipynb in Colab.
2. **Runtime → Change runtime type → T4 GPU** (or A100).
3. Optionally add a Colab Secret `GITHUB_TOKEN` to skip the rate limit.
4. **Runtime → Run all**, or run cells one by one.

On a T4 with the default `MAX_FONTS = 500`, expect **~3 hours** for the
full 60-epoch training run.

---

## 11. Inference and serving

### 11.1 CLI

```bash
python inference.py \
  --checkpoint runs/v2_unet/ckpt/latest.pt \
  --content-dir path/to/content_pngs \
  --style-dir path/to/style_pngs \
  --output-dir out/
```

`inference.py` loads `G_ema` from the checkpoint (the EMA weights, not
the raw generator), encodes the K style images once, and then decodes
each content image against that fixed style code.

### 11.2 FastAPI server

```bash
CHECKPOINT_PATH=runs/v2_unet/ckpt/latest.pt DEVICE=mps python serve.py
```

Endpoints:

- `GET /health` — basic status, device, checkpoint path
- `POST /api/transfer` — multipart upload of `style_files[]` + a
  `characters` string. The server renders each requested character with
  a neutral content font (auto-detected from `data/fonts/`), runs the
  model, and returns base64 PNGs.

The intended frontend is the Next.js UI at
[gen-ai](https://github.com/AmaDeuSZodiacXz/gen-ai). It has a Route
Handler at `app/api/transfer/route.ts` that proxies multipart requests
to this server (set via `BACKEND_URL` in `.env.local`).

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `MPS backend out of memory` | batch too large | `--batch-size 4` |
| Loss diverges to NaN | LR too high | `--g-lr 5e-5 --d-lr 2e-4` |
| `d` adversarial collapses to ~0 | D dominates G | raise `--lambda-rec` or lower `--d-lr` |
| Generated rows look identical (collapse) | pixel L1 dominates | check that `--lambda-rec` is small (0.5) and `--lambda-perceptual` is large (5.0); inspect samples around epoch 10 |
| `Cannot open neutral font` in `serve.py` | `data/fonts/` empty | run synthesis first, or `export NEUTRAL_FONT_FILE=path/to/regular.ttf` |
| GitHub `403 rate limit exceeded` | anonymous API | `export GITHUB_TOKEN=…` |
| `rec` stuck high (~0.5) for many epochs | bad data paths in CSV | inspect `data/labels_train.csv` |
| Training works but inference is blurry | reading raw G instead of EMA | `inference.py`/`serve.py` already load `G_ema` — make sure checkpoint has it |

If something else fails, inspect `data/synthesis_report.json` and the
in-flight progress bar metrics — the loss components are individually
readable (`adv`, `vgg`, `rec`, `con`, `d`).
