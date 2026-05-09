# Font_Generation

Few-shot font style transfer — given a handful of reference glyphs in some
style, generate any other character in that same style.

The pipeline downloads Google Fonts, renders each character to PNG, trains a
content/style-disentangled GAN (AdaIN decoder + PatchGAN discriminator with
spectral norm), and exposes the trained model through a FastAPI service that
the [gen-ai](https://github.com/AmaDeuSZodiacXz/gen-ai) Next.js UI can call.

## Repository layout

```
data_synthesis.py    Download Google Fonts and render character PNGs
split_dataset.py     Split labels.csv into train + 3 validation sets
dataset.py           PyTorch Dataset (samples content/style/target triplets)
models.py            Generator (Content + Style encoders + AdaIN decoder) + D
train.py             Training loop (hinge GAN + L1 + style/content/identity)
inference.py         CLI generate characters in a target style
serve.py             FastAPI server for the web UI
requirements.txt     Python dependencies
```

---

## Step 0 — Environment setup (once per machine)

### 0.1 Check Python and git

```bash
python --version    # 3.10+
git --version
```

### 0.2 Clone and check out the model branch

```bash
git clone https://github.com/Pontakorn-Wich/Font_Generation.git
cd Font_Generation
git checkout feat/font-style-transfer-model
```

### 0.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 0.4 Verify accelerator availability

```bash
python -c "import torch; print('torch', torch.__version__, '| mps:', torch.backends.mps.is_available(), '| cuda:', torch.cuda.is_available())"
```

Expected on Apple Silicon: `mps: True`. If both are `False`, training still
works on CPU but is 5–10× slower — pass `--device cpu` later.

### 0.5 (Recommended) GitHub token to bypass the anonymous rate limit

Without a token, the Google Fonts download is throttled to ~60 requests/hour.

```bash
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Step 1 — Data preparation

Downloads Google Fonts and renders 52 Latin characters per font into
128×128 grayscale PNGs.

### 1.1 Run

```bash
python data_synthesis.py --max-fonts 1000
```

| `--max-fonts` | Use case | Wall time (with token) |
| --- | --- | --- |
| `300` | Quick iteration | ~3 min |
| `1000` | **Recommended baseline** | ~10–15 min |
| `0` (omit) | Full Google Fonts (~1.5–1.7K) | ~25 min |

### 1.2 What you should see

```
Downloading fonts: 100%|████████| 1000/1000 [10:23<00:00,  1.6it/s]
Downloaded 1000 new font files into data/fonts
Rendering: 100%|████████| 1000/1000 [00:18<00:00, 54.1it/s]
Synthesis completed.
Font files scanned: 1000
Rows in labels.csv: 51896
Rows in char_mapping.csv: 1342596
```

### 1.3 Verify

```bash
ls data/                              # fonts/ images/ labels.csv char_mapping.csv
wc -l data/labels.csv                 # ~52000
ls data/fonts/ | wc -l                # ~1000
ls data/images/ | wc -l               # ~1000
ls data/images/ofl_caveat_Caveat_wght/ | head    # U+0041.png ... U+007A.png
```

> Re-running this step is safe: already-downloaded fonts are skipped.

### 1.4 Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `403 rate limit exceeded` | Anonymous GitHub | Set `GITHUB_TOKEN` (Step 0.5) |
| Very slow downloads | API throttling | Set token, lower `--max-fonts` |
| `Rows in labels.csv: 0` | All fonts failed to load | Check that `data/fonts/` actually contains files |

---

## Step 2 — Split dataset

Carves out three validation sets so we can measure generalization to
unseen styles, unseen characters, and the combination.

### 2.1 Run

```bash
python split_dataset.py --val-fonts 50 --val-chars KQXjz --seed 42
```

Flags:
- `--val-fonts 50` — random hold-out fonts (style generalization test)
- `--val-chars KQXjz` — held-out characters (content generalization test)
- `--seed 42` — fixed seed for reproducibility

### 2.2 What you should see

```
Wrote splits to data/
  total fonts=1000, total chars=52, total rows=51896
  held-out fonts: 50 | held-out chars: ['K', 'Q', 'X', 'j', 'z']
  train                     43615  -> labels_train.csv
  val_unseen_font            2350  -> labels_val_unseen_font.csv
  val_unseen_char            4750  -> labels_val_unseen_char.csv
  val_unseen_both             250  -> labels_val_unseen_both.csv
  meta: data/split_meta.json
```

### 2.3 Verify

```bash
ls data/labels_*.csv data/split_meta.json
cat data/split_meta.json | head -25
```

The four split row counts must sum to roughly `total_rows` (off-by-a-few is
expected when some characters are missing from some fonts).

---

## Step 3 — Training

This is the long-running step. Plan to leave it overnight on Apple Silicon
or for ~1 hour on a 4090.

### 3.1 Pre-flight checks

```bash
# Make sure runs/ is gitignored (checkpoints are ~480MB each, exceed
# the 100MB GitHub limit)
grep -E "^(runs|data|__pycache__)/" .gitignore

# Free disk space (~3–5 GB needed for 30 epochs of checkpoints + samples)
df -h .
```

### 3.2 Launch training (Apple Silicon / MPS)

```bash
python train.py \
  --labels-csv data/labels_train.csv \
  --out-dir runs/v1_baseline \
  --device mps \
  --batch-size 8 \
  --num-workers 0 \
  --k-style 4 \
  --epochs 30 \
  --save-every 2
```

NVIDIA CUDA variant:

```bash
python train.py \
  --labels-csv data/labels_train.csv \
  --out-dir runs/v1_baseline \
  --device cuda --batch-size 32 --num-workers 4 \
  --k-style 4 --epochs 50 --save-every 2
```

Run inside `tmux` so you can close the terminal:

```bash
tmux new -s train
# paste the train.py command above
# detach: Ctrl+B then D
# reattach: tmux attach -t train
```

### 3.3 Flag reference

| Flag | Meaning |
| --- | --- |
| `--labels-csv` | Use `labels_train.csv` (not `labels.csv`) so val splits never leak |
| `--out-dir` | Where checkpoints (`ckpt/`) and sample images (`samples/`) are written |
| `--device` | `mps` / `cuda` / `cpu` |
| `--batch-size` | 8 on MPS, 32+ on CUDA |
| `--num-workers` | 0 on Mac (avoid fork issues with MPS) |
| `--k-style` | Number of style reference images per training step |
| `--epochs` | Total epochs |
| `--save-every` | Save a checkpoint and sample grid every N epochs |
| `--limit-samples` | (optional) Use only the first N rows — useful for smoke tests |
| `--resume <ckpt>` | Resume from a checkpoint after a crash |

### 3.4 Reading the progress bar

```
Epoch 1/30:   2%|▏  | 100/5451 [00:24<22:01, 4.05it/s, ctn=0.34, d=1.71, g=0.05, rec=0.21, sty=0.78]
```

Healthy ranges:
- `rec` (L1 reconstruction): drops from ~0.9 to <0.1 in ~5 epochs, <0.05 by epoch 15
- `d` (discriminator): 0.5–2.5 — collapse to 0 means D is winning too hard
- `g` (G adversarial): hovers around 0
- `sty`, `ctn`: drift downward over time, occasional spikes are fine

### 3.5 Wall-time estimates

| Hardware | Batch | sec/iter | min/epoch | 30 epochs |
| --- | --- | --- | --- | --- |
| Apple M1/M2 (MPS) | 8 | ~0.25 | ~22 | **~11 h** |
| Apple M3/M4 Max | 8 | ~0.18 | ~16 | ~8 h |
| RTX 4090 | 32 | ~0.08 | ~1.8 | ~1 h |
| A100 | 64 | ~0.04 | ~0.5 | ~15 min |

### 3.6 Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `MPS backend out of memory` | Batch too large | `--batch-size 4` |
| Loss → `NaN` | LR too high | `--lr 1e-4` and `--lambda-rec 5` |
| `d` collapses to ~0 | D dominates G | Raise `--lambda-rec` to 20, slow D |
| `rec` stuck at ~0.5 | Bad data paths | Inspect `data/labels_train.csv` |
| Process killed | OOM kill | Reduce batch, then resume (Step 3.7) |

### 3.7 Stop / resume

Stop early: `Ctrl+C` in the training terminal. The latest checkpoint is at
`runs/v1_baseline/ckpt/latest.pt`.

Resume:

```bash
python train.py \
  --labels-csv data/labels_train.csv \
  --out-dir runs/v1_baseline \
  --device mps --batch-size 8 --num-workers 0 \
  --epochs 30 --save-every 2 \
  --resume runs/v1_baseline/ckpt/latest.pt
```

---

## Step 4 — Monitor training (parallel terminal)

In a second terminal while training runs:

### 4.1 Inspect the latest sample grid

```bash
ls -lt runs/v1_baseline/samples/ | head -5
open runs/v1_baseline/samples/epoch_010.png      # macOS
```

The 4-row grid is read top-to-bottom:

```
Row 1: content     — character we asked for, in the source font
Row 2: style ref   — example glyph from the target font
Row 3: generated   — the model's output ★
Row 4: ground truth— the correct answer
```

Quality milestones:
- Epoch 2: blurry blobs
- Epoch 5: vague glyph silhouettes
- Epoch 10: glyphs are "readable" but style may be off
- Epoch 15+: style starts matching Row 2

### 4.2 Disk usage

```bash
du -sh runs/v1_baseline/
ls -lh runs/v1_baseline/ckpt/
```

### 4.3 Capture training logs to a file

If you forgot to `tee`, you can also restart the process inside `tmux` with:

```bash
python train.py ... 2>&1 | tee runs/v1_baseline/train.log
```

### 4.4 System resources

```bash
top -pid $(pgrep -f "python train.py")
sudo powermetrics --samplers gpu_power -n 1 -i 1000     # Apple Silicon GPU
df -h .
```

---

## Step 5 — Verify training succeeded

### 5.1 Artifacts

```bash
ls runs/v1_baseline/                         # args.json  ckpt/  samples/
ls runs/v1_baseline/ckpt/                    # epoch_002.pt ... epoch_030.pt latest.pt
ls runs/v1_baseline/samples/ | tail -5
```

### 5.2 Visual inspection of the final epoch

```bash
open runs/v1_baseline/samples/epoch_030.png
```

Row 3 (generated) should look very close to Row 4 (ground truth). If not,
keep training (Step 3.7) or revisit data quality.

### 5.3 End-to-end sanity test with `inference.py`

```bash
mkdir -p /tmp/style_refs /tmp/content_chars
cp data/images/ofl_caveat_Caveat_wght/U+0042.png /tmp/style_refs/
cp data/images/ofl_caveat_Caveat_wght/U+0043.png /tmp/style_refs/
cp data/images/ofl_caveat_Caveat_wght/U+0044.png /tmp/style_refs/
cp data/images/ofl_caveat_Caveat_wght/U+0045.png /tmp/style_refs/
cp data/images/ofl_robotomono_RobotoMono_wght/U+0048.png /tmp/content_chars/   # H
cp data/images/ofl_robotomono_RobotoMono_wght/U+0069.png /tmp/content_chars/   # i

python inference.py \
  --checkpoint runs/v1_baseline/ckpt/latest.pt \
  --content-dir /tmp/content_chars \
  --style-dir /tmp/style_refs \
  --output-dir /tmp/output

open /tmp/output/U+0048.png    # "H" rendered in the Caveat handwriting style
```

If the output is a recognisable handwritten "H", training succeeded.

---

## Cheat-sheet — full pipeline in one block

```bash
# === Once per machine ===
cd /Users/skb/Documents/Font_Generation/Font_Generation
git checkout feat/font-style-transfer-model
git pull
pip install -r requirements.txt
export GITHUB_TOKEN=ghp_xxx              # optional but recommended

# === Step 1: Data ===
python data_synthesis.py --max-fonts 1000

# === Step 2: Split ===
python split_dataset.py --val-fonts 50 --val-chars KQXjz --seed 42

# === Step 3: Train (overnight on MPS) ===
tmux new -s train
python train.py \
  --labels-csv data/labels_train.csv \
  --out-dir runs/v1_baseline \
  --device mps --batch-size 8 --num-workers 0 \
  --k-style 4 --epochs 30 --save-every 2 \
  2>&1 | tee runs/v1_baseline/train.log
# Ctrl+B then D to detach

# === Step 4: Monitor (second terminal) ===
watch -n 60 "ls -lt runs/v1_baseline/samples/ | head -3"
open runs/v1_baseline/samples/epoch_010.png

# === Step 5: Verify ===
open runs/v1_baseline/samples/$(ls runs/v1_baseline/samples/ | tail -1)
```

---

## What comes after training

1. **Serve the model** — `CHECKPOINT_PATH=runs/v1_baseline/ckpt/latest.pt DEVICE=mps python serve.py` (FastAPI on `:8000`).
2. **Plug into the UI** — clone [gen-ai](https://github.com/AmaDeuSZodiacXz/gen-ai), `npm install && npm run dev`, browse to `http://localhost:3000`.
3. **(Optional) Improve the model** — perceptual VGG loss, EMA generator, R1 gradient penalty, mixed-precision (bf16), and a held-out evaluator (`evaluate.py`) are the high-ROI next steps.
