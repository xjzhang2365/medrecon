import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize
from PIL import Image
import io

from recon import (
    load_phantom, simulate_kspace, zero_filled,
    recon_fista, recon_learned_ista, compute_metrics
)

try:
    from unet import UNet, recon_unet
    UNET_AVAILABLE = True
except Exception:
    UNET_AVAILABLE = False

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MedRecon — MRI Reconstruction Demo",
    page_icon="🧠",
    layout="wide",
)

# ── header ────────────────────────────────────────────────────────────────────

st.title("🧠 MedRecon")
st.markdown(
    """
    **Physics-informed MRI reconstruction from undersampled k-space.**  
    Demonstrates the same inverse-problem framework as atomic-scale TEM reconstruction —
    same math, different forward model.  
    *FISTA (compressed sensing) vs Learned ISTA (unrolled network).*
    """
)
st.divider()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Parameters")

    image_source = st.radio(
        "Image source",
        ["Built-in Shepp-Logan phantom", "Upload your own image"],
        index=0,
    )

    uploaded = None
    if image_source == "Upload your own image":
        uploaded = st.file_uploader("Upload PNG / JPG (grayscale)", type=["png", "jpg", "jpeg"])

    st.subheader("Acquisition")
    acceleration = st.slider(
        "Acceleration factor R", min_value=2, max_value=8, value=4, step=1,
        help="How many fewer k-space lines are acquired. Higher = more undersampling = harder problem."
    )
    mask_type = st.selectbox("Sampling pattern", ["random", "equispaced"], index=0)

    st.subheader("Algorithm")
    algo_options = [
        "FISTA (compressed sensing)",
        "Learned ISTA (unrolled network)",
        "U-Net (deep learning)",
        "All three (comparison)",
    ]
    if not UNET_AVAILABLE:
        algo_options = [o for o in algo_options if "U-Net" not in o and "All" not in o]
    algo = st.radio("Reconstruction method", algo_options, index=0)

    st.subheader("FISTA settings")
    n_iter = st.slider("Iterations", 20, 120, 60, step=10)
    lam = st.select_slider("Sparsity λ", options=[0.001, 0.002, 0.005, 0.01, 0.02], value=0.005)

    run = st.button("▶  Reconstruct", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "**About:** Built by [Xiaojun Zhang](https://github.com/xjzhang2365) "
        "to demonstrate physics-informed inverse problem solvers. "
        "[GitHub](https://github.com/xjzhang2365/3D-Reconstruction-Low-Dose-Imaging)"
    )

# ── main panel: show phantom / upload preview before running ─────────────────

col_img, col_mask = st.columns(2)

@st.cache_data
def get_phantom():
    return load_phantom(256)

def load_uploaded(file):
    img = Image.open(file).convert("L")
    arr = np.array(img).astype(np.float64)
    arr = resize(arr, (256, 256), anti_aliasing=True)
    return (arr - arr.min()) / (arr.max() - arr.min())

if uploaded:
    ground_truth = load_uploaded(uploaded)
else:
    ground_truth = get_phantom()

with col_img:
    st.subheader("Ground truth")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

_, kunder, mask = simulate_kspace(ground_truth, acceleration, mask_type)

with col_mask:
    st.subheader(f"k-space sampling mask  (R={acceleration})")
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(mask, cmap="gray", aspect="auto")
    ax.set_xlabel("k_x  (frequency encode)")
    ax.set_ylabel("k_y  (phase encode)")
    sampled_pct = mask.mean() * 100
    ax.set_title(f"{sampled_pct:.1f}% of k-space sampled", fontsize=10)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── reconstruction ────────────────────────────────────────────────────────────

if run:
    st.divider()
    st.subheader("Reconstruction results")

    # Zero-filled baseline (always shown)
    zf_raw = zero_filled(kunder)
    zf = (zf_raw - zf_raw.min()) / (zf_raw.max() - zf_raw.min() + 1e-8)
    m_zf = compute_metrics(ground_truth, zf)

    run_fista  = algo in ["FISTA (compressed sensing)", "All three (comparison)"]
    run_lista  = algo in ["Learned ISTA (unrolled network)", "All three (comparison)"]
    run_unet   = algo in ["U-Net (deep learning)", "All three (comparison)"] and UNET_AVAILABLE

    fista_out = lista_out = unet_out = None
    m_f = m_l = m_u = None

    with st.spinner("Running reconstruction..."):
        if run_fista:
            fista_out, t_f = recon_fista(kunder, mask, n_iter=n_iter, lam=lam)
            m_f = compute_metrics(ground_truth, fista_out)
            m_f["time"] = round(t_f, 2)

        if run_lista:
            lista_out, t_l = recon_learned_ista(kunder, mask)
            m_l = compute_metrics(ground_truth, lista_out)
            m_l["time"] = round(t_l, 2)

        if run_unet:
            if "unet_model" not in st.session_state:
                st.session_state["unet_model"] = UNet()
            unet_out, t_u = recon_unet(kunder, mask, st.session_state["unet_model"])
            m_u = compute_metrics(ground_truth, unet_out)
            m_u["time"] = round(t_u, 2)

    # ── image panels ─────────────────────────────────────────────────────────

    def show_panel(ax, img, title, metrics=None):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        if metrics:
            ax.set_xlabel(
                f"SSIM {metrics['SSIM']}   PSNR {metrics['PSNR']} dB"
                + (f"   {metrics.get('time','?')}s" if 'time' in metrics else ""),
                fontsize=9
            )

    def show_error(ax, err, title):
        im = ax.imshow(err, cmap="hot", vmin=0, vmax=0.3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if algo == "All three (comparison)":
        results = [
            (ground_truth, "Ground truth", None),
            (zf,           "Zero-filled",  m_zf),
            (fista_out,    "FISTA",         m_f),
            (lista_out,    "Learned ISTA",  m_l),
            (unet_out,     "U-Net",         m_u),
        ]
        results = [(img, lbl, met) for img, lbl, met in results if img is not None]
        n = len(results)
        fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
        fig.subplots_adjust(hspace=0.35, wspace=0.1)
        for col, (img, lbl, met) in enumerate(results):
            show_panel(axes[0][col], img, lbl, met)
            err = met["error_map"] if met else np.zeros_like(ground_truth)
            show_error(axes[1][col], err, f"|error| {lbl}")

    elif algo == "Both (side by side)":
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.12)
        axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
        show_panel(axes[0][0], ground_truth, "Ground truth")
        show_panel(axes[0][1], zf,          "Zero-filled",  m_zf)
        show_panel(axes[0][2], fista_out,   "FISTA",        m_f)
        show_panel(axes[0][3], lista_out,   "Learned ISTA", m_l)
        show_error(axes[1][0], np.zeros_like(ground_truth), "|error| ground truth")
        show_error(axes[1][1], m_zf["error_map"], "|error| zero-filled")
        show_error(axes[1][2], m_f["error_map"],  "|error| FISTA")
        show_error(axes[1][3], m_l["error_map"],  "|error| Learned ISTA")

    else:
        recon_img = next(x for x in [fista_out, lista_out, unet_out] if x is not None)
        recon_metrics = m_f or m_l or m_u
        algo_label   = (
            "FISTA" if run_fista else
            "Learned ISTA" if run_lista else
            "U-Net"
        )
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.subplots_adjust(hspace=0.35, wspace=0.1)
        show_panel(axes[0][0], ground_truth, "Ground truth")
        show_panel(axes[0][1], zf,           "Zero-filled (baseline)", m_zf)
        show_panel(axes[0][2], recon_img,    algo_label, recon_metrics)
        show_error(axes[1][0], np.zeros_like(ground_truth), "|error| ground truth")
        show_error(axes[1][1], m_zf["error_map"],            "|error| zero-filled")
        show_error(axes[1][2], recon_metrics["error_map"],   f"|error| {algo_label}")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── metrics table ─────────────────────────────────────────────────────────

    st.divider()
    st.subheader("📊 Metrics summary")

    rows = [{"Method": "Zero-filled (baseline)",
             "SSIM": m_zf["SSIM"], "PSNR (dB)": m_zf["PSNR"], "Time (s)": "—"}]
    if m_f:
        rows.append({"Method": "FISTA (compressed sensing)",
                     "SSIM": m_f["SSIM"], "PSNR (dB)": m_f["PSNR"], "Time (s)": m_f["time"]})
    if m_l:
        rows.append({"Method": "Learned ISTA (unrolled network)",
                     "SSIM": m_l["SSIM"], "PSNR (dB)": m_l["PSNR"], "Time (s)": m_l["time"]})
    if m_u:
        rows.append({"Method": "U-Net (deep learning)",
                     "SSIM": m_u["SSIM"], "PSNR (dB)": m_u["PSNR"], "Time (s)": m_u["time"]})

    st.table(rows)

    # ── explainer ─────────────────────────────────────────────────────────────

    with st.expander("📖 What is this demo doing?"):
        st.markdown("""
### The inverse problem

MRI scanners do not directly acquire images. They measure **k-space** — the 2D Fourier
transform of the image. A full scan acquires every k-space line; accelerated MRI skips most
of them to reduce scan time (and patient discomfort).

Recovering a high-quality image from incomplete k-space measurements is an **ill-posed
inverse problem**:

```
y = F_u · x + noise
```

where `y` is the undersampled k-space, `F_u` is the partial Fourier operator, and `x` is
the unknown image we want to recover.

### FISTA (compressed sensing)

Minimises a regularised objective:

```
min_x  0.5 · ‖F_u x − y‖²  +  λ · ‖Wx‖₁
```

The second term enforces **wavelet-domain sparsity** — natural images have compressible
wavelet coefficients. FISTA solves this with an accelerated proximal gradient descent and
a soft-threshold step at each iteration.

### Learned ISTA (unrolled network)

Takes the ISTA iteration and replaces the fixed step size and threshold with **learnable
parameters**, trained end-to-end via 30 steps of Adam on the data-consistency loss. This
is a minimal example of **algorithm unrolling** — a key idea in modern deep MRI
reconstruction (MoDL, E2E-VarNet, etc.).

### Connection to TEM reconstruction

This demo uses the same mathematical skeleton as the author's TEM work:
- Forward model: `F_u` (partial Fourier) ↔ TEM contrast transfer function
- Regulariser: wavelet sparsity ↔ physical plausibility via MD + Tersoff potential
- Optimiser: FISTA / unrolled gradient ↔ Simulated Annealing with MD relaxation
        """)

else:
    st.info("👈 Set parameters in the sidebar and click **Reconstruct** to run.")
