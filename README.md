# [ICLR2026] Free Lunch for Stabilizing Rectified Flow Inversion

Official implementation of **"Free Lunch for Stabilizing Rectified Flow Inversion"** (ICLR 2026).
Chenru Wang, Beier Zhu, Chi Zhang. AGI Lab, Westlake University.

[📄 Paper (OpenReview)](https://openreview.net/forum?id=QIEPzPzkaO) &nbsp;|&nbsp; [📦 PIE-Bench dataset](https://huggingface.co/datasets/Yuanshi/PIE-Bench_v1) &nbsp;

## TL;DR

Rectified-Flow (RF) inversion accumulates approximation errors across timesteps, drifting away from the prior and degrading reconstruction / editing quality. We propose two **training-free, plug-and-play** velocity-correction modules:

- **PMI (Proximal-Mean Inversion)** — at each inversion step, pulls the predicted velocity toward a *running average* of past velocities via a single closed-form gradient step. The step radius is derived from the spherical Gaussian high-density region in the instability analysis, so the corrected trajectory provably stays out of low-density tails. **No extra NFE.**
- **mimic-CFG** — a lightweight editing-time velocity corrector. Projects the current velocity onto the running-average direction and linearly interpolates with weight `w` (default `0.94`). The form mirrors classifier-free guidance, hence the name.

Both modules drop into existing solvers (Euler, Heun, RF-Solver, FireFlow). On **PIE-Bench**, adding PMI / mimic-CFG to every baseline improves reconstruction (e.g. RF-Solver PSNR 29.17 → 30.72; unconditional PSNR 27.81 → 29.87) and editing (better PSNR/SSIM/CLIP at the same or fewer NFE).

## Installation

```bash
conda create -n pmi python=3.10 -y
conda activate pmi
pip install -r requirements.txt
```

The code targets **FLUX.1-dev** as the base RF model. Weights are pulled via `huggingface_hub` on first run, or you can point to a local cache:

```bash
export HF_HOME=/path/to/hf_cache       # use an existing HF cache
# or set direct paths to safetensors:
export FLUX_DEV=/path/to/flux1-dev.safetensors
export AE=/path/to/ae.safetensors
```

T5 (`google/t5-v1_1-xxl`), CLIP (`openai/clip-vit-large-patch14`), and an NSFW classifier (`Falconsai/nsfw_image_detection`) are also fetched on first run.

## Reconstruction (PMI)

`recon.py` runs inversion → re-denoising with the *same* prompt to measure inversion fidelity.

```bash
python recon.py \
    --source_img_dir   examples/source/cat.jpg \
    --source_prompt    "a cat sitting on grass" \
    --target_prompt    "a cat sitting on grass" \
    --sampling_strategy fireflow_m \
    --num_steps 10 --guidance 1 \
    --output_dir examples/edit-result/recon/cat
```

Setting `--sampling_strategy fireflow_m` (or `reflow_m`) activates PMI on top of the corresponding base solver. Use `fireflow` / `reflow` / `rf_solver` / `rf_midpoint` for the vanilla baselines.

## Editing (PMI + mimic-CFG)

`edit.py` runs inversion under `--source_prompt` and re-denoises under `--target_prompt`, with attention-feature sharing across the two passes.

```bash
python edit.py \
    --source_img_dir   examples/source/cat.jpg \
    --source_prompt    "a cat sitting on grass" \
    --target_prompt    "a tiger sitting on grass" \
    --sampling_strategy fireflow_m \
    --num_steps 12 --guidance 2 --w 0.94 \
    --inject 2 --start_layer_index 0 --end_layer_index 37 \
    --output_dir examples/edit-result/cmp
```

`--w` is the mimic-CFG interpolation weight (Eq. 17 in the paper). Sec. 5.4 shows `w=0.94` gives the best PSNR/SSIM/CLIP trade-off on PIE-Bench; lowering `w` strengthens the correction (more background preservation, less editability), raising it weakens it.

## Key arguments

| Flag | Default | Meaning |
|---|---|---|
| `--name` | `flux-dev` | Base FLUX model (`flux-dev` or `flux-schnell`). |
| `--sampling_strategy` | `fireflow_m` | One of `reflow`, `reflow_m`, `rf_solver`, `rf_midpoint`, `fireflow`, `fireflow_m`. The `_m` variants enable PMI / mimic-CFG. |
| `--num_steps` | `10`–`12` | Steps for inversion *and* denoising (NFE depends on solver). |
| `--guidance` | `1` (recon) / `2` (edit) | Distilled FLUX guidance. |
| `--w` | `0.94` | mimic-CFG interpolation weight (edit only). |
| `--inject` | `2` | Number of timesteps over which attention features are shared between inversion and denoising passes. |
| `--start_layer_index`, `--end_layer_index` | `0`, `37` | Range of FLUX single-stream blocks (0–37) used for feature sharing. |
| `--reuse_v` | `1` | Cache / inject the V tensor across passes. |
| `--editing_strategy` | `replace_v` | Which of Q/K/V is replaced during editing. |
| `--qkv_ratio` | `1.0,1.0,1.0` | Per-channel scaling on cached Q,K,V. |
| `--feature_path` | `feature` | Directory for the in-RAM attention-feature dict (created if missing). |
| `--offload` | off | Move T5/CLIP/AE/model between CPU and GPU to fit smaller VRAM. |
| `--seed` | `1234` / `0` | Seed; values `<= 0` randomise. |

Output filenames are auto-numbered as `{prefix}_inject_{N}_start_layer_index_{S}_end_layer_index_{E}_img_{idx}.jpg`.

## Method at a glance

**PMI (Algorithm 1).** Maintain a running average of past velocities `v̄_tk`. At each step:

```
v_tk  ← vθ(z_tk, t_k)                    # baseline velocity
v̂_tk  ← v_tk - r_tk · ∇F(v_tk)/‖∇F(v_tk)‖₂
ẑ_{t_{k+1}} ← ẑ_tk + (t_{k+1}-t_k) · v̂_tk
```
where `F(v) = ‖v − v_{tk-1}‖₁ + (1/2λ)‖v − v̄_tk‖₂²` and the radius
`r_i = √(2n + 3√(2n)) · Δt_i / T` (Proposition 1) keeps the corrected sample inside the high-density spherical Gaussian.

**mimic-CFG (Algorithm 2).** During editing, project the predicted velocity onto the running-average direction and interpolate:
```
v̄_proj = (vᵀ v̄_edit / ‖v̄_edit‖²) · v̄_edit
v̂      = (1 − w) · v̄_proj + w · v
```

Both algorithms add **zero NFE** on top of the underlying solver.

## Repository layout

```
edit.py        # PMI + mimic-CFG editing entry point
recon.py       # PMI reconstruction entry point
flux/
  sampling.py  # denoise_{reflow, reflow_m, rf_solver, fireflow, fireflow_m, midpoint}
  model.py     # FLUX transformer (returns (img, info) for feature sharing)
  modules/     # autoencoder, conditioner, layers (attention feature cache)
  util.py      # FLUX / AE / T5 / CLIP loaders, watermark
examples/source/  # demo input images
```

See `CLAUDE.md` for an architecture-oriented walkthrough (the `info` side-channel, sampler internals, gotchas).

## Evaluation data

We evaluate on **PIE-Bench** (700 images across 10 editing categories). Download from the original release:

- Project page / repo: <https://github.com/cure-lab/PnPInversion>
- HuggingFace mirror: <https://huggingface.co/datasets/Yuanshi/PIE-Bench_v1>

## Citation

```bibtex
@inproceedings{wang2026free,
  title     = {Free Lunch for Stabilizing Rectified Flow Inversion},
  author    = {Chenru Wang and Beier Zhu and Chi Zhang},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=QIEPzPzkaO}
}
```

## Acknowledgements

Built on top of [FLUX](https://github.com/black-forest-labs/flux), [RF-Solver](https://github.com/wangjiangshan0725/RF-Solver-Edit), and [FireFlow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing). Evaluation uses [PIE-Bench](https://github.com/cure-lab/PnPInversion).
