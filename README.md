# MedRecon — Physics-Informed MRI Reconstruction

An interactive demo of compressed sensing and learned reconstruction for accelerated MRI,
built to show how physics-informed inverse problem solvers transfer across imaging modalities.

**[▶ Live Demo](https://xjzhang2365-medrecon.streamlit.app)**  
**Related work:** [3D TEM Reconstruction Pipeline](https://github.com/xjzhang2365/3D-Reconstruction-Low-Dose-Imaging) · [arXiv preprint](https://arxiv.org/abs/2604.07271)

---

## The inverse problem

MRI scanners measure **k-space** — the 2D Fourier transform of the image. Accelerated MRI
skips most k-space lines to reduce scan time, creating an ill-posed inverse problem:

```
y = F_u · x + noise
```

where `y` is undersampled k-space, `F_u` is the partial Fourier operator, and `x` is the
unknown image. Recovering `x` requires physics-informed regularisation.

This is mathematically identical to the author's TEM reconstruction work — same framework,
different forward model:

| | MRI reconstruction | TEM reconstruction |
|---|---|---|
| Forward model | Partial Fourier `F_u` | Contrast transfer function |
| Measurement | Undersampled k-space | Single low-dose 2D projection |
| Regulariser | Wavelet sparsity | MD + Tersoff potential |
| Optimiser | FISTA / unrolled gradient | Simulated Annealing |
| Accuracy | SSIM 0.55 at R=4 | RMSD ~0.45 Å (z-direction) |

---

## Algorithms

### FISTA — compressed sensing

Minimises a regularised objective with wavelet-domain sparsity prior:

```
min_x  0.5 · ‖F_u x − y‖²  +  λ · ‖Wx‖₁
```

Solved with accelerated proximal gradient descent (Beck & Teboulle 2009).
Pure NumPy — no deep learning required.

### Learned ISTA — algorithm unrolling

Takes the ISTA iteration and replaces fixed hyperparameters with learnable scalars,
trained end-to-end via gradient descent on the data-consistency loss. A minimal
implementation of the algorithm-unrolling paradigm behind MoDL, E2E-VarNet, and
similar architectures.

### U-Net — deep learning post-processor

A lightweight U-Net (3 encoder/decoder levels, ~180k parameters) deployed via
[ONNX Runtime](https://onnxruntime.ai/) for dependency-free inference. The network
is structured as a residual learner: it predicts the aliasing artifact, which is
subtracted from the zero-filled reconstruction — a standard design in clinical MRI
reconstruction systems (fastMRI baseline architecture).

> **Note:** The current ONNX model uses random weights and serves as a structural
> demonstration of the learned reconstruction pathway. The FISTA and Learned ISTA
> paths produce quantitatively meaningful results.

---

## Results (Shepp-Logan phantom, R=4, random undersampling)

| Method | SSIM | PSNR (dB) |
|---|---|---|
| Zero-filled baseline | 0.38 | 22.5 |
| FISTA (60 iterations) | 0.55 | 32.5 |
| Learned ISTA (8 unrolls) | 0.55 | 32.5 |

---

## Quickstart

```bash
git clone https://github.com/xjzhang2365/medrecon
cd medrecon
pip install -r requirements.txt
streamlit run app.py
```

### Requirements

```
streamlit
numpy
scipy
scikit-image
matplotlib
Pillow
onnxruntime
```

No PyTorch required — inference runs via ONNX Runtime.

---

## App features

- Built-in Shepp-Logan phantom or upload your own grayscale image
- Adjustable acceleration factor R (2–8×)
- Random or equispaced k-space sampling patterns
- Three reconstruction algorithms: FISTA, Learned ISTA, U-Net
- Side-by-side comparison of reconstruction methods
- Error maps and SSIM / PSNR / runtime metrics
- Inline explainer connecting the math to physical intuition

---

## Repository structure

```
medrecon/
├── app.py            # Streamlit UI
├── recon.py          # FISTA and Learned ISTA (pure NumPy)
├── unet.py           # U-Net inference via ONNX Runtime
├── unet_mri.onnx     # Exported model (~180k params)
├── export_onnx.py    # Script to regenerate ONNX from PyTorch weights
├── assets/           # Static assets
└── requirements.txt
```

---

## Author

**Xiaojun Zhang** — Computational Imaging Scientist  
PhD, Computational Science, City University of Hong Kong  
[GitHub](https://github.com/xjzhang2365) · [LinkedIn](https://www.linkedin.com/in/xiaojun-zhang-9b8495321) · [arXiv](https://arxiv.org/abs/2604.07271)

*This project is part of a broader research programme on physics-informed inverse problems
for scientific imaging, spanning electron microscopy and medical imaging.*
