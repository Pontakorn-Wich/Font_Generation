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
2. [Review of existing methods](#2-review-of-existing-methods)
3. [Architecture, in depth](#3-architecture-in-depth)
4. [Loss design](#4-loss-design)
5. [Training tricks](#5-training-tricks)
6. [Data pipeline](#6-data-pipeline)
7. [Step-by-step runbook](#7-step-by-step-runbook)
8. [Hyperparameter reference](#8-hyperparameter-reference)
9. [Wall-time estimates](#9-wall-time-estimates)
10. [Repository layout](#10-repository-layout)
11. [Colab notebook](#11-colab-notebook)
12. [Inference and serving](#12-inference-and-serving)
13. [Troubleshooting](#13-troubleshooting)
14. [References](#14-references)

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

## 2. Review of existing methods

Five categories of prior work are relevant: neural style transfer, unconditional GANs, general few-shot image-to-image translation, font-specific GANs, and diffusion-based font generation. Papers in **bold** are direct font generation methods from the reference list (`ref.bib`).

### 2.1 Overview table

| Method | Year | Category | Few-shot | Pros | Cons |
|--------|------|----------|:--------:|------|------|
| Neural Style Transfer (Gatys et al.) | 2016 | Style transfer | ✗ | No training required; works on arbitrary image pairs | Per-image optimisation (~minutes); transfers texture not stroke structure; no character-level control |
| Vanilla GAN (unconditional) | 2014+ | GAN | ✗ | Simple; well-studied | Cannot condition on content or style; one model per font; cannot request a specific character |
| FUNIT (Liu et al.) | 2019 | Few-shot i2i | ✓ | Established content + style enc + AdaIN decoder pattern; general few-shot framework | No skip connections → blob collapse on fine glyph detail; mean-pool style aggregation loses per-glyph weighting |
| **DG-Font** (Xie et al.) | 2021 | Font GAN | ✓ | Deformable content features preserve stroke geometry explicitly | Deformation field network adds complexity and auxiliary supervision requirements; style encoder uses mean pooling |
| **MX-Font** (Park et al.) | 2021 | Font GAN | ✓ | Multiple localized experts capture stroke-level style cues; better than one encoder | Expert count is a hard hyperparameter; per-expert mean pooling; requires component region annotations |
| **CG-GAN** (Kong et al.) | 2022 | Font GAN | ✓ | Component-based discriminator head → part-level supervision → sharper serifs and counters | Needs stroke/radical component labels; less applicable to Latin scripts without a component vocabulary |
| **FontDiffuser** (Yang et al.) | 2024 | Diffusion | ✓ | State-of-the-art visual quality; stable training (no GAN mode collapse); multi-scale content aggregation | 100+ denoising steps per image → seconds to minutes; unsuitable for real-time inference |
| **Ours** | 2024 | Font GAN | ✓ | U-Net skips prevent content collapse; transformer aggregation weights informative refs; multi-task D without component labels; single-pass inference (~5 ms) | GAN training less stable than diffusion; quality ceiling below FontDiffuser at small data scale |

### 2.2 Detailed notes

**vs. Neural Style Transfer.**
Gatys et al. (2016) and Johnson et al. (2016) optimise a single image to match VGG gram matrices of a style reference. No training is needed. The limitation for font generation is that gram matrices encode *texture statistics* — they have no concept of stroke width, counter shape, or serif geometry. Applied to glyphs, output matches the visual texture of the reference font but loses glyph identity entirely. Inference is also per-image (an iterative optimisation), not a single forward pass.

**vs. Vanilla GAN.**
An unconditional GAN (DCGAN, StyleGAN without conditioning) can model the distribution of one font and sample from it, but has no mechanism for style transfer at inference time. A separate model must be trained per font, and there is no way to request a specific character in a previously unseen style — the entire goal of this project.

**vs. FUNIT.**
FUNIT (Liu et al., ICCV 2019) is the direct architectural ancestor of this pipeline. It establishes the content encoder → style encoder → AdaIN decoder pattern and demonstrates few-shot generalisation to unseen classes at inference time. The core limitation for fonts is missing skip connections: all content signal must pass through the 8×8 bottleneck, making the pixel-L1 loss degenerate — the mean glyph shape becomes a local minimum and the model collapses. This is exactly the failure mode encountered in our first design iteration (described in the introduction) and the primary motivation for U-Net skip connections.

**vs. DG-Font.**
DG-Font (Xie et al., CVPR 2021) attacks content collapse by predicting spatial deformation offsets that warp content feature maps before AdaIN injection, explicitly preserving stroke geometry. Our approach avoids deformation entirely: U-Net skip connections carry spatial content information directly to the decoder at every resolution, which is parameter-efficient and requires no auxiliary deformation supervision.

**vs. MX-Font.**
MX-Font (Park et al., ICCV 2021) uses multiple localized expert encoders, each responsible for a different stroke region. The underlying insight — that different parts of a reference glyph carry different amounts of style information — is the same as our attention aggregation. The difference is that MX-Font applies it within a fixed set of predefined regions, while our transformer encoder learns which reference glyphs to up-weight end-to-end without manual region assignment.

**vs. CG-GAN.**
CG-GAN (Kong et al., CVPR 2022) augments the discriminator with a component-level head that supervises individual strokes or radicals. Our multi-task discriminator uses the same two-head principle (font classifier + character classifier) but operates at the whole-glyph level, avoiding stroke-level annotations. For Latin-alphabet fonts where a standard stroke decomposition vocabulary does not exist, this makes our approach straightforward to apply.

**vs. FontDiffuser.**
FontDiffuser (Yang et al., AAAI 2024) is the current state of the art for font generation quality. Diffusion avoids GAN mode collapse entirely and produces higher-fidelity outputs. The trade-off is inference speed: diffusion requires 100–1000 denoising steps (seconds to minutes per image), whereas our generator is a single forward pass (~5 ms on GPU). This project's FastAPI serve endpoint (see §12) requires near-real-time latency, which makes a GAN the appropriate choice even at the cost of some visual quality.

---

## 3. Architecture, in depth

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

### 3.1 ContentEncoder — a U-Net encoder

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

### 3.2 StyleEncoder — CNN + transformer aggregation

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

### 3.3 Decoder — U-Net decoder with AdaIN modulation

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

### 3.4 Discriminator — PatchGAN + auxiliary classifiers

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

### 3.5 VGG perceptual loss

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

## 4. Loss design

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

### 4.1 Adversarial: hinge GAN

```
L_adv_D = mean(relu(1 - D(real))) + mean(relu(1 + D(fake)))
L_adv_G = -mean(D(fake))
```

The hinge formulation avoids the vanishing-gradient pathology of the
original Goodfellow `log(1-D(fake))` formulation when D becomes confident.
It is the de-facto choice for modern image GANs.

### 4.2 VGG perceptual

Already explained in §3.5. Weight 5.0 puts it dominant over pixel L1.

### 4.3 Pixel L1 (small weight)

Kept at 0.5 — just enough to nudge mean intensity towards correct, not
enough to dominate. In the old run this was at 10.0 and was the direct
cause of collapse.

### 4.4 Consistency: content and style

These are "cycle-like" regularisers:

- **Content consistency**: `L1(E_c(fake), E_c(content).detach())`
  averaged over every encoder scale. Forces the content path to be
  preserved through G — if the fake glyph isn't "the same character",
  the content encoder will disagree.
- **Style consistency**: `L1(E_s(fake.unsqueeze(1)), E_s(refs).detach())`.
  Forces the style code re-extracted from the generated image to match
  the style code used to generate it.

### 4.5 R1 gradient penalty (lazy)

```
R1 = 0.5 · mean(‖∂D(real)/∂real‖²)
```

This penalises D for having large gradients on real images, which keeps D
Lipschitz-smooth and prevents the discriminator from overfitting to
spurious local features. Applied **lazily** (every 16 steps) for speed
following the StyleGAN convention. The weight is multiplied by 16 so the
effective penalty matches what it would be if applied every step.

### 4.6 Auxiliary classification

Cross-entropy loss on:
- D's font head over real images (D learns to identify fonts)
- D's char head over real images (D learns to identify characters)
- G must fool both heads on its fakes

This is the strongest signal in the loss bundle for style/content
disentanglement. A G that produces wrong-font or wrong-character outputs
gets penalised twice (adversarial + aux), not once.

---

## 5. Training tricks

### 5.1 EMA generator

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

### 5.2 TTUR (Two-timescale update rule)

D learns at `4e-4`, G learns at `1e-4`. The 4× faster D matches the
StyleGAN-2-ADA convention. With aux classifiers and R1 we want D to be
strong; G in turn benefits from a stronger D's gradients.

### 5.3 bf16 autocast

Forward passes are wrapped in `torch.autocast(device_type=..., dtype=torch.bfloat16)`.
bfloat16 has fp32's dynamic range but half the memory bandwidth, which
gives roughly a 2× speedup on T4/A100 GPUs and on Apple Silicon (MPS).
The VGG perceptual call is forced back to fp32 because the pretrained
weights expect it.

### 5.4 Adam betas (0, 0.99)

StyleGAN-style optimizer settings. β1 = 0 (no momentum on first moment)
pairs well with R1 — momentum on the discriminator's adversarial gradient
can destabilise training in the presence of gradient penalties.

### 5.5 Gradient clipping

Both G and D are gradient-clipped to L2-norm 1.0. Cheap insurance against
the occasional pathological batch.

### 5.6 Light augmentation on content only

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

## 6. Data pipeline

### 6.1 Synthesis with strict cleanup

`data_synthesis.py` downloads .ttf/.otf files from the `google/fonts`
GitHub repo (apache/ofl/ufl licensed) and rasterises each character into
a 128×128 grayscale PNG centred on the canvas.

```mermaid
flowchart TD
    A["GitHub API\ngoogle/fonts tree"] -->|"list apache/ofl/ufl\n.ttf/.otf paths"| B["Download fonts\ndata/fonts/*.ttf  (up to --max-fonts)"]
    B --> C{"For each font file"}

    C --> D{"PIL ImageFont\n.truetype()"}
    D -->|"FAIL — Case A"| E["Delete .ttf\nlog → synthesis_report.json"]
    E --> C

    D -->|"OK"| F{"For each char in CHARS\nA–Z a–z 0–9 = 62 chars"}
    F -->|"glyph_exists = False"| G["Skip char"]
    G --> F
    F -->|"glyph_exists = True"| H["Render 128×128 grayscale PNG\ncentred on canvas\ndata/images/{font_name}/U+XXXX.png"]
    H --> I["Append row to buffer\n(label_character, font_name, picture_path)"]
    I --> F

    F -->|"All chars done"| J{"rendered_count < 62?\nCase B / C"}
    J -->|"YES — incomplete"| K["Delete .ttf + image folder\nremove rows from buffer\nlog → synthesis_report.json"]
    K --> C
    J -->|"NO — all 62 glyphs OK"| L["Keep font rows"]
    L --> C

    C -->|"All fonts done"| M["Write data/labels.csv\n(label_character, font_name, picture_path)"]
    C -->|"All fonts done"| N["Write data/char_mapping.csv\nall same-char cross-font pairs"]
    C -->|"All fonts done"| O["Write data/synthesis_report.json\nfull deletion log"]
```

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

#### Rendering algorithm — abstract diagram

```mermaid
flowchart TD
    A["font_dir/*.ttf | *.otf"] --> B["PIL ImageFont.truetype\nfont_size=96"]
    B -->|"load fails"| FA["Case A\ndelete .ttf"]
    B -->|"load ok"| C["For each char in CHARS\n(A–Z a–z 0–9, 62 glyphs)"]

    C --> D{"glyph_exists?\nfont.getmask(ch).getbbox()"}
    D -->|"None → skip"| C
    D -->|"exists"| E["Create canvas\n128×128 px, grayscale, white=255"]

    E --> F["draw.textbbox((0,0), ch, font)\n→ (x0,y0,x1,y1)"]
    F --> G["Center offset\nx = (128−w)//2 − x0\ny = (128−h)//2 − y0"]
    G --> H["draw.text((x,y), ch\nfill=0 (black))"]
    H --> I["Save PNG\ndata/images/{font_name}/U+XXXX.png"]
    I --> J["Append row to labels.csv\n(char, font_name, path)"]
    J --> C

    C -->|"all chars done"| K{"rendered ≥ min_chars_per_font?"}
    K -->|"yes"| L["Keep font\nemit mapping pairs"]
    K -->|"no"| FB["Case B/C\ndelete .ttf + image folder\nomit rows from CSV"]
```

### 6.2 Train/val splits

`split_dataset.py` slices `labels.csv` into four files:

```
labels_train.csv             train fonts × train chars   — training data
labels_val_unseen_font.csv   held-out fonts × train chars — tests style generalization
labels_val_unseen_char.csv   train fonts × held-out chars — tests content generalization
labels_val_unseen_both.csv   held-out × held-out         — hardest combined test
```

```mermaid
flowchart TD
    A["data/labels.csv\n(label_character, font_name, picture_path)"] --> B["Load all rows"]

    B --> C["Random sample 50 fonts → val_fonts\nseed=42  (--val-fonts, --seed)"]
    B --> D["val_chars = { K, Q, X, j, z }\n(--val-chars argument)"]

    C & D --> E{"For each row\nfont_held? char_held?"}

    E -->|"font NOT held\nchar NOT held"| F["labels_train.csv\ntrain fonts × train chars\n— used by train.py"]
    E -->|"font HELD\nchar NOT held"| G["labels_val_unseen_font.csv\nheld-out fonts × train chars\n— tests style generalisation"]
    E -->|"font NOT held\nchar HELD"| H["labels_val_unseen_char.csv\ntrain fonts × held-out chars\n— tests content generalisation"]
    E -->|"font HELD\nchar HELD"| I["labels_val_unseen_both.csv\nheld-out fonts × held-out chars\n— hardest combined test"]

    F & G & H & I --> J["Write data/split_meta.json\nseed, val_fonts list, val_chars list, row counts per split"]
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

### 6.3 Why both held-out fonts AND held-out chars

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

## 7. Step-by-step runbook

This is the local Apple-Silicon workflow. For Colab use the notebook —
see §11.

### 7.1 Environment setup (once per machine)

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

### 7.2 Data synthesis

```bash
python data_synthesis.py --max-fonts 500
```

**Options:**

| Flag | Default | Effect |
| --- | --- | --- |
| `--max-fonts N` | 0 (all) | Limit fonts downloaded from Google Fonts |
| `--skip-download` | off | Skip download, render from existing `data/fonts/` |
| `--chars` | `A-Za-z0-9` (62) | Characters to render per font |
| `--image-size` | 128 | Output PNG resolution (px) |
| `--font-size` | 96 | PIL drawing size inside the canvas |
| `--root` | `.` | Project root; all paths resolve relative to this |

**Speed presets:**

- `--max-fonts 100` — quick smoke (~3 min)
- `--max-fonts 500` — recommended baseline (~10–15 min)
- omit / `0`     — all Google Fonts (~25 min)

**Artifacts produced** (all under `data/`):

```
data/
├── fonts/                      downloaded .ttf / .otf files (one per font)
├── images/
│   └── {font_name}/
│       └── U+XXXX.png          128×128 grayscale PNG, one per (font, char)
├── labels.csv                  every kept (char, font, path) row — input to split_dataset.py
├── char_mapping.csv            all same-char cross-font pairs (source → target)
└── synthesis_report.json       deletion log: which fonts failed and why
```

Fonts that rendered fewer than all 62 glyphs are **deleted** (both `.ttf` and image folder).
Only fonts with full coverage survive into `labels.csv`.

The script prints a summary at the end:

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

### 7.3 Train/val split

```bash
python split_dataset.py --val-fonts X --val-chars KQXjz --seed 42
```

**Options:**

| Flag | Default | Effect |
| --- | --- | --- |
| `--labels-csv` | `data/labels.csv` | Source CSV from step 6.2 |
| `--val-fonts N` | 50 | Fonts randomly held out for validation |
| `--val-chars` | `KQXjz` | Characters held out for validation (pass `""` to hold out none) |
| `--seed` | 42 | RNG seed — fix this for reproducible splits |
| `--out-dir` | same dir as labels CSV | Where split files are written |

**Artifacts produced** (all under `data/`):

```
data/
├── labels_train.csv             train fonts × train chars   → used by train.py
├── labels_val_unseen_font.csv   held-out fonts × train chars  (tests style generalisation)
├── labels_val_unseen_char.csv   train fonts × held-out chars  (tests content generalisation)
├── labels_val_unseen_both.csv   held-out fonts × held-out chars  (hardest combined test)
└── split_meta.json              seed, held-out font list, held-out chars, row counts per split
```

`split_meta.json` lets you fully reproduce the split later — rerun with the same `--seed` and `--val-fonts`/`--val-chars` to get identical files.

### 7.4 Training (long, often overnight)

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

### 7.5 Monitor (second terminal)

```bash
open runs/v2_unet/samples/$(ls runs/v2_unet/samples/ | tail -1)
```

Each saved grid has four rows top → bottom:

1. **content** — the requested character in a source font
2. **style ref** — one reference glyph from the target font
3. **generated** ★ — what the model produced
4. **ground truth** — the correct answer

A healthy model has row 3 ≈ row 4 by ~epoch 15.

### 7.6 Resume from a crash

```bash
python train.py \
  --labels-csv data/labels_train.csv \
  --out-dir runs/v2_unet \
  --device mps --batch-size 8 --num-workers 0 \
  --epochs 60 --save-every 2 \
  --resume runs/v2_unet/ckpt/latest.pt
```

### 7.7 Smoke test inference

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

### 7.8 Evaluation

Run `evaluate.py` against all three validation splits using the EMA generator from a checkpoint.

```bash
python evaluate.py \
  --checkpoint latest.pt \
  --data-dir   data \
  --out-dir    results \
  --device     mps \
  --batch-size 16
```

CUDA variant:

```bash
python evaluate.py \
  --checkpoint runs/v2_unet/ckpt/latest.pt \
  --data-dir   data \
  --out-dir    results \
  --device     cuda \
  --batch-size 64
```

**Options:**

| Flag | Default | Effect |
| --- | --- | --- |
| `--checkpoint` | `latest.pt` | Checkpoint file (must contain `G_ema` key) |
| `--data-dir` | `data` | Directory with `labels_val_*.csv` split files |
| `--out-dir` | `results` | Root output directory |
| `--device` | auto-detect | `mps` / `cuda` / `cpu` |
| `--batch-size` | 32 | Images per forward pass |
| `--k-style` | 4 | Number of style reference images (must match training) |
| `--grid-samples` | 16 | Columns in each visual grid |
| `--per-font-top-n` | 20 | Fonts shown in per-font SSIM box plot |
| `--save-images` | off | Write every generated PNG to `results/images/{split}/` |
| `--splits` | all three | Override which splits to evaluate (space-separated stems) |

**Artifacts produced** (all under `results/`):

```
results/
├── metrics.json                  mean L1 / SSIM / VGG per split
├── per_sample_metrics.csv        per-image L1, SSIM, VGG, font, char
├── grids/
│   ├── labels_val_unseen_font.png   visual grid: content | style ref | generated | GT
│   ├── labels_val_unseen_char.png
│   └── labels_val_unseen_both.png
└── plots/
    ├── metrics_bar.png            bar chart: all three metrics × all three splits
    ├── score_hist.png             L1 / SSIM / VGG score distributions per split
    ├── per_font_ssim.png          box plot of SSIM for the top-N fonts by sample count
    └── l1_vs_ssim.png             scatter L1 vs SSIM coloured by split
```

**Interpreting results:**

The three splits test different generalization axes:

| Split | What is unseen | Measures |
| --- | --- | --- |
| `unseen_font` | Font (style) | Style generalization — can the model apply a new font style? |
| `unseen_char` | Character (content) | Content generalization — can the model generate unseen glyphs? |
| `unseen_both` | Font + character | Hardest combined test |

`unseen_font` will score best (easy to interpolate seen characters in an unseen style).
`unseen_both` will score worst — expected, not a model defect.

---

#### Results — `latest.pt`

Evaluated on the shared split (`split_meta.json`, seed 42). Metrics are L1 pixel error, SSIM (structural similarity), and VGG perceptual distance — all computed on the EMA generator output vs ground truth at 128×128.

**Summary:**

| Split | n | L1 ↓ | SSIM ↑ | VGG ↓ |
| --- | --- | --- | --- | --- |
| Unseen Font | 1140 | 0.1137 | 0.8314 | 0.3712 |
| Unseen Char | 740  | 0.1422 | 0.8123 | 0.4206 |
| Unseen Both | 100  | 0.1536 | 0.7976 | 0.4392 |

![Metrics bar chart](results/plots/metrics_bar.png)

---

**Score distributions:**

| Split | SSIM std | SSIM min | SSIM max |
| --- | --- | --- | --- |
| Unseen Font | 0.063 | 0.549 | 0.977 |
| Unseen Char | 0.060 | 0.418 | 0.925 |
| Unseen Both | 0.060 | 0.587 | 0.914 |

Across all 1980 samples: **3.7% score below SSIM 0.70** (hard failures), **7.5% score at or above 0.90** (near-perfect).

![Score histograms](results/plots/score_hist.png)

![L1 vs SSIM](results/plots/l1_vs_ssim.png)

The L1–SSIM scatter shows the two metrics correlate tightly. The `unseen_font` cloud (blue) sits higher-right than the other two, confirming style generalization is easier than content generalization.

---

**SSIM degradation vs unseen_font baseline:**

```
unseen_font  →  unseen_char  :  −0.019   (content generalization cost)
unseen_font  →  unseen_both  :  −0.034   (style + content combined)
unseen_char  →  unseen_both  :  −0.015   (marginal cost of also unseen font)
```

The content gap (−0.019) is larger than the marginal font gap (−0.015), meaning unseen characters are the harder axis. The model transfers a seen style to a new character less reliably than it transfers a new style to a seen character.

---

**Per-character SSIM (held-out chars: K Q X j z):**

| Char | SSIM | Notes |
| --- | --- | --- |
| `j` | 0.851 | Simple descender — shape consistent across fonts |
| `X` | 0.837 | Symmetric diagonal strokes — predictable |
| `z` | 0.820 | Mid-complexity |
| `K` | 0.798 | Asymmetric branching arm — harder |
| `Q` | 0.755 | Hardest — distinctive tail varies wildly between fonts |

`Q` is the worst held-out character. Its tail/flourish is the most font-specific feature in the held-out set; the model produces a plausible `O`-like form but misses the tail in many fonts.

![Unseen char grid](results/grids/labels_val_unseen_char.png)

---

**Per-font SSIM (top 20 fonts by sample count):**

![Per-font SSIM](results/plots/per_font_ssim.png)

Worst fonts are expressive handwriting or novelty display faces with irregular stroke widths:

| Font | SSIM |
| --- | --- |
| RockSalt-Regular | 0.613 |
| jsMath-cmex10 | 0.656 |
| Ultra-Regular | 0.738 |
| HomemadeApple-Regular | 0.739 |
| CherryCreamSoda-Regular | 0.740 |

Best fonts are clean geometric or condensed sans-serifs:

| Font | SSIM |
| --- | --- |
| Agdasima-Regular | 0.890 |
| AlegreyaSansSC-Thin | 0.883 |
| AdventPro (variable) | 0.882 |
| Abel-Regular | 0.880 |
| OpenSansHebrewCondensed-Light | 0.872 |

---

**Visual grids — content \| style ref \| generated \| ground truth:**

*Unseen Font* — model applies a never-seen font style to known characters. Quality is highest here.

![Unseen font grid](results/grids/labels_val_unseen_font.png)

*Unseen Both* — both font and character are unseen. Hardest setting; some characters lose fine details.

![Unseen both grid](results/grids/labels_val_unseen_both.png)

---

**Takeaway:** Style generalization (unseen font) is strong at SSIM 0.831. Content generalization (unseen char) costs −0.019 SSIM. The main failure mode is decorative/handwriting fonts with irregular strokes — collecting more samples from those font families is the highest-leverage data improvement.

### 7.9 Diagram of training process

#### Data loading (per batch)

```mermaid
sequenceDiagram
    participant L as DataLoader
    participant D as FontPairDataset
    participant FS as Filesystem

    L->>D: __getitem__(idx)
    D->>D: pick target_font, content_char from samples[]
    D->>D: pick content_font ≠ target_font (random)
    D->>D: pick K=4 style_chars from target_font ≠ content_char (random)
    D->>FS: load content image  (content_font / content_char)
    D->>FS: load target image   (target_font / content_char)
    D->>FS: load K style images (target_font / style_chars)
    FS-->>D: PIL images
    D->>D: apply augment to content only (RandomAffine)
    D->>D: normalize all to [-1, 1]
    D-->>L: {content_image, style_images[K], target_image, font_id, char_id}
    L->>L: collate batch of B samples
```

#### Training loop (per step)

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant G as Generator
    participant D as Discriminator
    participant V as VGG (frozen)
    participant E as EMA

    DL-->>G: content (B,1,128,128)
    DL-->>G: style_refs (B,K,1,128,128)
    DL-->>D: target (B,1,128,128) — real images

    Note over G,D: D step
    G->>G: forward(content, style_refs) → fake  [no_grad]
    D->>D: D(real) → patch_logits, font_logits, char_logits
    D->>D: D(fake.detach()) → patch_logits
    D->>D: L_D = hinge(real,fake) + CE(font) + CE(char)
    alt every 16 steps
        D->>D: R1 penalty on real (fp32)
    end
    D->>D: backward + clip_grad(1.0) + opt_D.step()

    Note over G,D: G step
    G->>G: encode_content(content) → feature pyramid [5 scales]
    G->>G: encode_style(style_refs) → style_code (B,256)
    G->>G: decode(features, style_code) → fake
    D->>D: D(fake) → patch, font, char logits
    G->>V: perceptual_loss(fake, target)  [fp32]
    G->>G: L1(fake, target)
    G->>G: encode_content(fake) → consistency vs content feats
    G->>G: encode_style(fake) → consistency vs style_code
    G->>G: L_G = adv + perceptual + L1 + content_cons + style_cons + font_cls + char_cls
    G->>G: backward + clip_grad(1.0) + opt_G.step()

    G->>E: EMA.update(G, step)

    Note over G,E: every save_every epochs
    E->>E: G_ema.forward(content, style_refs) → vis grid
    E->>E: save samples/epoch_NNN.png
    E->>E: save ckpt/epoch_NNN.pt + ckpt/latest.pt
```

#### What each checkpoint contains

```
ckpt/latest.pt
├── G          raw generator weights
├── G_ema      EMA generator weights  ← used by inference.py and serve.py
├── D          discriminator weights
├── opt_g      Adam state for G
├── opt_d      Adam state for D
├── epoch      last completed epoch
├── global_step
├── args       full CLI args snapshot
└── vocab      {n_fonts, n_chars} needed to rebuild D heads
```

---

## 8. Hyperparameter reference

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

## 9. Wall-time estimates

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

## 10. Repository layout

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

## 11. Colab notebook

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

## 12. Inference and serving

### 12.1 CLI

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

### 12.2 FastAPI server

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

## 13. Troubleshooting

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

---

## 14. References

### Few-shot font / image style transfer

[liu2019funit] Liu, M.-Y., Huang, X., Mallya, A., Karras, T., Laine, S., Lehtinen, J., & Kautz, J. (2019). **Few-Shot Unsupervised Image-to-Image Translation.** ICCV 2019.
*Used for: main architecture — content encoder + style encoder + AdaIN decoder; few-shot generalization framework.*

[xie2021dgfont] Xie, Y., Yao, X., Sun, J., & Lian, Z. (2021). **DG-Font: Deformable Generative Networks for Unsupervised Font Generation.** CVPR 2021.
*Used for: content/style disentanglement for font generation; AdaIN modulation.*

[park2021mxfont] Park, S.-E., Chun, S., Oh, J. H., Jang, B., & Han, J. (2021). **Multiple Heads Are Better than One: Few-shot Font Generation with Multiple Localized Experts.** ICCV 2021.
*Used for: style encoder with multiple localized experts; attention aggregation over reference characters.*

[kong2022cgggan] Kong, L. & Lian, Z. (2022). **Look Closer to Supervise Better: One-Shot Font Generation via Component-Based Discriminator.** CVPR 2022.
*Used for: auxiliary component classifier in discriminator (component-aware supervision).*

[yang2024fontdiffuser] Yang, Z., Xu, D., & Lian, Z. (2024). **FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning.** AAAI 2024.
*Used for: SOTA diffusion-based font generation — cited for comparison.*

### U-Net / skip connections

[ronneberger2015unet] Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** MICCAI 2015.
*Used for: encoder-decoder with skip connections at every scale; prevents content collapse in generator bottleneck.*

[isola2017pix2pix] Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). **Image-to-Image Translation with Conditional Adversarial Networks.** CVPR 2017.
*Used for: U-Net applied to image-to-image translation; PatchGAN discriminator trunk.*

### Adaptive Instance Normalisation (AdaIN)

[huang2017adain] Huang, X. & Belongie, S. (2017). **Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization.** ICCV 2017.
*Used for: AdaIN operator — instance-normalize activations then affine-transform via style code (γ, β).*

[huang2018munit] Huang, X., Liu, M.-Y., Belongie, S., & Kautz, J. (2018). **Multimodal Unsupervised Image-to-Image Translation.** ECCV 2018.
*Used for: AdaIN in image-to-image translation; content/style latent space decomposition.*

[karras2019stylegan] Karras, T., Laine, S., & Aila, T. (2019). **A Style-Based Generator Architecture for Generative Adversarial Networks.** CVPR 2019.
*Used for: AdaIN modulation injected at every resolution; style mixing regularization.*

### Style encoder — CLS token + self-attention

[dosovitskiy2021vit] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.** ICLR 2021.
*Used for: [CLS] token + self-attention as aggregate pooling for the reference-set style encoder.*

[lee2019settransformer] Lee, J., Lee, Y., Kim, J., Kosiorek, A. R., Choi, S., & Teh, Y. W. (2019). **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.** ICML 2019.
*Used for: permutation-invariant aggregation of K reference images via attention.*

[devlin2019bert] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** NAACL-HLT 2019.
*Used for: [CLS] token as sequence aggregate — origin of the pattern used in the style encoder.*

### Multi-task discriminator / auxiliary classifiers

[odena2017acgan] Odena, A., Olah, C., & Shlens, J. (2017). **Conditional Image Synthesis with Auxiliary Classifier GANs.** ICML 2017.
*Used for: auxiliary classifier head in discriminator.*

[choi2018stargan] Choi, Y., Choi, M., Kim, M., Ha, J.-W., Kim, S., & Choo, J. (2018). **StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation.** CVPR 2018.
*Used for: multi-domain classifier in discriminator.*

[miyato2018spectralnorm] Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). **Spectral Normalization for Generative Adversarial Networks.** ICLR 2018.
*Used for: Lipschitz regularization of discriminator via spectral norm.*

### Perceptual loss

[gatys2016styletransfer] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). **Image Style Transfer Using Convolutional Neural Networks.** CVPR 2016.
*Used for: VGG features for style/content matching — conceptual basis for perceptual loss.*

[johnson2016perceptual] Johnson, J., Alahi, A., & Fei-Fei, L. (2016). **Perceptual Losses for Real-Time Style Transfer and Super-Resolution.** ECCV 2016.
*Used for: L1 on VGG feature maps as training loss — direct implementation used in this project.*

[zhang2018lpips] Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.** CVPR 2018.
*Used for: LPIPS — validates VGG perceptual loss as human-aligned metric.*

### Training stability

[lim2017geometricgan] Lim, J. H. & Ye, J. C. (2017). **Geometric GAN.** arXiv:1705.02894.
*Used for: hinge adversarial loss.*

[mescheder2018r1] Mescheder, L., Geiger, A., & Nowozin, S. (2018). **Which Training Methods for GANs do actually Converge?** ICML 2018.
*Used for: R1 gradient penalty on real samples — gradient norm regularization for D.*

[karras2020stylegan2] Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). **Analyzing and Improving the Image Quality of StyleGAN.** CVPR 2020.
*Used for: lazy R1 regularization (every 16 steps); EMA generator weights; Adam β=(0, 0.99).*

[heusel2017ttur] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). **GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.** NeurIPS 2017.
*Used for: TTUR — discriminator learning rate higher than generator learning rate.*

[tarvainen2017meanteacher] Tarvainen, A. & Valpola, H. (2017). **Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results.** NeurIPS 2017.
*Used for: EMA weight averaging concept applied to generator.*

[micikevicius2018mixedprecision] Micikevicius, P., Narang, S., Alben, J., et al. (2018). **Mixed Precision Training.** ICLR 2018.
*Used for: bf16 autocast for training efficiency.*
