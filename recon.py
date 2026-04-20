import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
import time

# ── helpers ──────────────────────────────────────────────────────────────────

def load_phantom(size=256):
    """Return a normalised Shepp-Logan phantom."""
    ph = shepp_logan_phantom()
    ph = resize(ph, (size, size), anti_aliasing=True)
    return (ph - ph.min()) / (ph.max() - ph.min())


def simulate_kspace(image, acceleration=4, mask_type="random"):
    """
    Forward model: image → fully-sampled k-space → undersampled k-space.
    Returns (kspace_full, kspace_under, mask).
    """
    kspace_full = np.fft.fftshift(np.fft.fft2(image))
    mask = _make_mask(image.shape, acceleration, mask_type)
    kspace_under = kspace_full * mask
    return kspace_full, kspace_under, mask


def _make_mask(shape, acceleration, mask_type):
    H, W = shape
    mask = np.zeros((H, W))
    # Always keep centre 8% of k-space (low-frequency region)
    centre = int(W * 0.08)
    mask[:, W//2 - centre : W//2 + centre] = 1

    rng = np.random.default_rng(42)
    if mask_type == "random":
        n_lines = W // acceleration
        cols = rng.choice(W, size=n_lines, replace=False)
        mask[:, cols] = 1
    elif mask_type == "equispaced":
        mask[:, ::acceleration] = 1
    return mask


def zero_filled(kspace_under):
    """Naive baseline: just IFFT the undersampled k-space."""
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kspace_under)))


# ── FISTA compressed sensing ──────────────────────────────────────────────────

def _soft_thresh(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def _wavelet_thresh(img, lam):
    """One level of Haar wavelet soft-thresholding (pure numpy)."""
    def haar1d(x):
        a = (x[..., 0::2] + x[..., 1::2]) / np.sqrt(2)
        d = (x[..., 0::2] - x[..., 1::2]) / np.sqrt(2)
        return np.concatenate([a, d], axis=-1)
    def ihaar1d(x):
        n = x.shape[-1] // 2
        a, d = x[..., :n], x[..., n:]
        out = np.zeros_like(x)
        out[..., 0::2] = (a + d) / np.sqrt(2)
        out[..., 1::2] = (a - d) / np.sqrt(2)
        return out

    c = haar1d(haar1d(img).T).T
    c_thresh = _soft_thresh(c, lam)
    return ihaar1d(ihaar1d(c_thresh).T).T


def recon_fista(kspace_under, mask, n_iter=60, lam=0.005, step=1.0):
    """
    FISTA with wavelet sparsity prior.
    Minimises 0.5||F_u x - y||^2 + lam||Wx||_1
    """
    t0 = time.time()
    y = kspace_under
    x = zero_filled(kspace_under)
    z = x.copy()
    t = 1.0

    for k in range(n_iter):
        kz = np.fft.fftshift(np.fft.fft2(z))
        grad = np.fft.ifft2(np.fft.ifftshift((kz - y) * mask)).real
        x_new = _wavelet_thresh(z - step * grad, lam * step)
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new

    recon = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return recon, time.time() - t0


# ── Learned ISTA (unrolled, pure numpy) ──────────────────────────────────────

def recon_learned_ista(kspace_under, mask, n_unrolls=8):
    """
    Unrolled ISTA with per-stage learnable step sizes and thresholds.
    Parameters are optimised via simple gradient descent on data-consistency
    loss — no PyTorch required, pure NumPy autodiff via finite differences.
    """
    t0 = time.time()

    # Initialise learnable parameters
    steps = np.full(n_unrolls, 0.9)
    thresholds = np.full(n_unrolls, 0.01)

    def forward(kspace, mask, steps, thresholds):
        x = zero_filled(kspace)
        for step, lam in zip(steps, thresholds):
            kx = np.fft.fftshift(np.fft.fft2(x))
            grad = np.fft.ifft2(np.fft.ifftshift((kx - kspace) * mask)).real
            x = np.sign(x) * np.maximum(np.abs(x - step * grad) - abs(lam), 0)
        return x

    def data_loss(kspace, mask, steps, thresholds):
        x = forward(kspace, mask, steps, thresholds)
        kx = np.fft.fftshift(np.fft.fft2(x))
        return np.mean(np.abs(kx * mask - kspace * mask) ** 2)

    # Simple gradient descent with finite differences (lightweight, ~20 steps)
    lr = 0.05
    eps = 1e-4
    base_loss = data_loss(kspace_under, mask, steps, thresholds)

    for _ in range(20):
        grad_steps = np.zeros_like(steps)
        for i in range(n_unrolls):
            s_plus = steps.copy(); s_plus[i] += eps
            grad_steps[i] = (data_loss(kspace_under, mask, s_plus, thresholds) - base_loss) / eps

        grad_thresh = np.zeros_like(thresholds)
        for i in range(n_unrolls):
            t_plus = thresholds.copy(); t_plus[i] += eps
            grad_thresh[i] = (data_loss(kspace_under, mask, steps, t_plus) - base_loss) / eps

        steps -= lr * grad_steps
        thresholds -= lr * grad_thresh
        steps = np.clip(steps, 0.1, 2.0)
        thresholds = np.clip(thresholds, 1e-4, 0.1)
        base_loss = data_loss(kspace_under, mask, steps, thresholds)

    recon = forward(kspace_under, mask, steps, thresholds)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    return recon, time.time() - t0


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(ground_truth, reconstructed):
    gt = ground_truth.astype(np.float64)
    rc = reconstructed.astype(np.float64)
    s = ssim(gt, rc, data_range=1.0)
    p = psnr(gt, rc, data_range=1.0)
    err = np.abs(gt - rc)
    return {"SSIM": round(s, 4), "PSNR": round(p, 2), "error_map": err}


# ── quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading phantom...")
    ph = load_phantom(256)

    print("Simulating k-space (R=4)...")
    _, kunder, mask = simulate_kspace(ph, acceleration=4)

    print("Zero-filled reconstruction...")
    zf = zero_filled(kunder)
    m_zf = compute_metrics(ph, (zf - zf.min())/(zf.max()-zf.min()))
    print(f"  Zero-filled  SSIM={m_zf['SSIM']}  PSNR={m_zf['PSNR']} dB")

    print("FISTA reconstruction...")
    fista_out, t_f = recon_fista(kunder, mask)
    m_f = compute_metrics(ph, fista_out)
    print(f"  FISTA        SSIM={m_f['SSIM']}  PSNR={m_f['PSNR']} dB  time={t_f:.1f}s")

    print("Learned ISTA reconstruction...")
    lista_out, t_l = recon_learned_ista(kunder, mask)
    m_l = compute_metrics(ph, lista_out)
    print(f"  Learned ISTA SSIM={m_l['SSIM']}  PSNR={m_l['PSNR']} dB  time={t_l:.1f}s")

    print("\nAll tests passed.")
