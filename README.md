# TweezeEdit

**[AAAI 2026] Official Implementation**

**TweezeEdit: Consistent and Efficient Image Editing with Path Regularization**

---

## 🔍 Overview

TweezeEdit is a training-free image editing framework that improves editing consistency by regularizing source–target denoising paths instead of relying on inversion anchors (the inversion noise of source images) or intrusive model modifications.

---

## 📄 Paper

* **ArXiv**: [https://arxiv.org/abs/2508.10498](https://arxiv.org/abs/2508.10498)

---

## 🛠 Installation

```bash
conda env create -f environment.yml
conda activate TweezeEdit
```

---

## 🚀 Running the Code

The main entry point is `main.py`. The framework supports **multiple models** with **shared and model-specific arguments**.

### Basic Command

```bash
python main.py \
  --model lcm \
  --dataset_config ./dataset.json \
  --base_image_path ./ \
  --out_dir ./results
```

---

## ⚙️ Supported Models

### 1️⃣ LCM (Latent Consistency Model) Model

* Default resolution: **512 × 512**
* Default inference steps: **15**

Key arguments:

* `--is_p2p` : Enable P2P editing
* `--self_replace_steps`: P2P editing param
* `--cross_replace_steps`: P2P editing param
* `--reg_mode`

  * `1`: Approximate but **precise strength control** (recommended)
  * `2`: Original formulation with less precise control

---

### 2️⃣ FLUX Model

* Default resolution: **1024 × 1024**
* Default inference steps: **28**
* Requires a HuggingFace access token (`--hf_token` or `.env`)

Example:

```bash
python main.py \
  --model flux \
  --hf_token YOUR_HF_TOKEN
```

---

## 🧩 Gamma Scheduler

The **Gamma Scheduler** controls the strength of path regularization at each diffusion step.
Proper scheduling enables a balanced trade-off between editing and consistency.

> ⚠️ Editing performance is ultimately bounded by the **generation capability of the base model**.

---

### Default Gamma Scheduling (LCM)

```python
gamma_step = GammaScheduler.generate(
    length=15,
    mode="step",
    value=1.0,
    k=10
)
```

---

### Custom Tail Scheduling (LCM)

For some examples (e.g., `examples/0002.jpg`), a custom tail schedule yields better consistency–editing trade-offs:

```python
custom_list = [0.0, 0.8, 0.8, 0.8, 0.0, 0.0]

gamma_step = GammaScheduler.generate(
    length=15,
    mode="custom_tail",
    value=1.0,
    k=9,
    custom_list=custom_list
)
```
---

## 📌 Notes

* `--skip_step`: Skips diffusion steps where `gamma = 1.0` to reduce computation (performance impact)
* Output images are automatically resized back to original resolution
* For edits that **fail or collapse**, consider:
  * Adjusting gamma schedule
  * Using a stronger base model

---

## 📎 Citation

If you find our work helpful, please cite:

```bibtex
@article{mao2025tweezeedit,
  title={Tweezeedit: Consistent and efficient image editing with path regularization},
  author={Mao, Jianda and Wang, Kaibo and Xiang, Yang and Chen, Kani},
  journal={arXiv preprint arXiv:2508.10498},
  year={2025}
}
```
