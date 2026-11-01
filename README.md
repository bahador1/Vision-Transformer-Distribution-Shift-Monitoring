# Vision-Transformer-Distribution-Shift-Monitoring

Implemented DINOv2-based distribution shift detection with deep kernel MMD permutation tests for a loading dock monitoring system; developed a reproducible CIFAR-10/CIFAR-10-C demonstration.

## Research Question

At the same false-alarm rate, does MMD on frozen DINOv2 features detect image corruptions more reliably than simpler methods, and do its alarms correspond to a loss in classification accuracy?

## Approach

The project is a **comparative evaluation of shift detectors**:

- **Detectors:** MMD on frozen DINOv2 features, compared against two simpler baselines: **Centroid** (distance between the mean feature vectors of the two groups) and **Entropy** (based on the classifier's predicted probabilities). Each detector produces a permutation-test p-value.
- **Fair comparison:** all detectors are calibrated to the same false-alarm rate (5% by default, `--alpha`) and scored on the same batches of clean and corrupted images.
- **Detectable vs. harmful shift:** classification accuracy is measured separately, to tell whether a detected change actually hurts the model.

If MMD catches mild corruptions more often, that shows when its extra complexity pays off. If the detectors perform equally, the simpler method is enough.

**Status:** the synthetic demonstration has been run (see `demo-results/`). The full DINOv2 / CIFAR-10-C study has not been run yet, so which detector performs better is still an open question. See `Report.md` for assumptions and limitations.

## Setup

Use Python 3.10 or 3.11 in a fresh environment, and run commands from the project directory. The program refuses to write into a nonempty output directory, so use a new `--out` for each run.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 1. Synthetic demonstration (CPU, no downloads)

```bash
python monitor.py demo --out runs/demo --trials 100
```

This runs the full statistical pipeline on synthetic feature vectors. `demo-results/` contains the output of a previous run of this command. Only the `cifar` mode uses DINOv2 and real images.

## 2. Image dependencies

Install in the same environment. For GPU, pick the torch/torchvision wheels that match your CUDA driver from <https://pytorch.org/get-started/previous-versions/>. Example for CUDA 12.4 on Linux/Windows:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.49.0 pillow
python -c "import torch,torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

For CPU on Linux/Windows, use `https://download.pytorch.org/whl/cpu` as the index URL. On macOS, omit `--index-url`. CPU feature extraction can be slow; `--encode-batch 8` reduces GPU memory use.

## 3. CIFAR-10-C data

Download `CIFAR-10-C.tar` (2.9 GB) from <https://zenodo.org/records/2535967> (official MD5: `56bf5dcef84df0e2308c6dcbcbbd8499`) and extract it into `data/`:

```bash
mkdir -p data
curl -fL --retry 3 https://zenodo.org/records/2535967/files/CIFAR-10-C.tar -o data/CIFAR-10-C.tar
md5sum data/CIFAR-10-C.tar
tar -xf data/CIFAR-10-C.tar -C data
```

- `--corruption-dir` must directly contain `labels.npy`, `gaussian_noise.npy`, `motion_blur.npy`, etc.
- `--data` should point to the parent of `cifar-10-batches-py/` if clean CIFAR-10 is already downloaded; otherwise torchvision downloads it.
- The first real run downloads the public `facebook/dinov2-small` checkpoint. To use a local snapshot, pass `--model /path/to/snapshot`.

## 4. Real experiment

Smoke test (one corruption, one severity):

```bash
python monitor.py cifar --data ./data --corruption-dir ./data/CIFAR-10-C \
  --fit 500 --pool 128 --windows 32 --trials 10 --permutations 99 \
  --corruptions gaussian_noise --severities 3 --out runs/cifar-smoke
```

Full study (3 corruption families × 5 severities × 3 window sizes, 200 trials per cell):

```bash
python monitor.py cifar --data ./data --corruption-dir ./data/CIFAR-10-C \
  --windows 32 64 128 --severities 1 2 3 4 5 \
  --corruptions gaussian_noise motion_blur brightness \
  --trials 200 --permutations 199 --out runs/cifar-study
```

Key options (see `--help` for all):

- `--fit`: number of clean training images used to fit the classification probe.
- `--pool`: number of unique clean images in each test pool.
- `--encode-batch`: GPU inference batch size, independent of `--windows`.
- `--revision`: pin the model to the commit recorded in `run.json`.
- `--seed` / `--out`: use distinct values to check sensitivity to the data split.

Embeddings are reused within a run but not cached on disk between runs.

## Outputs

- `summary.csv`: alarm rate per condition, method and window size; clean vs. corrupted accuracy on the whole target pool; median scoring time (ms).
- `trials.csv`: per trial and detector: score, permutation p-value, alarm flag, window accuracy and scoring time.
- `run.json`: arguments, package versions, bandwidth, feature dimension, and for real runs the model revision, processor config, device and total encoding time.
- `image_ids.npz` (real runs only): fit, reference and target image IDs. Fit IDs refer to the training set; reference and target IDs are disjoint subsets of the clean test set.
- One PNG per corruption family: severity 0 shows the clean false-alarm rate, higher severities show detection power. Windows are resampled from fixed pools, so the plot is descriptive.

Ground-truth labels are used only for fitting the probe, archive consistency checks and offline accuracy analysis. Detector scores use only features or predicted probabilities. Reported timing covers the statistical test only, excluding feature extraction, probe inference and file I/O.
