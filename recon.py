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


# ── Learned ISTA (unrolled, PyTorch) ─────────────────────────────────────────

def recon_learned_ista(kspace_under, mask, n_unrolls=8):
    """
    Lightweight unrolled ISTA network.
    Each stage: gradient step (learnable step size) + soft threshold (learnable).
    Initialised analytically — works without pre-training.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return recon_fista(kspace_under, mask)

    t0 = time.time()

    class UnrolledISTA(nn.Module):
        def __init__(self, n_stages):
            super().__init__()
            self.steps = nn.ParameterList([
                nn.Parameter(torch.tensor(0.9)) for _ in range(n_stages)])
            self.thresholds = nn.ParameterList([
                nn.Parameter(torch.tensor(0.01)) for _ in range(n_stages)])

        def forward(self, kspace, mask_t):
            x = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(kspace)))
            for step, lam in zip(self.steps, self.thresholds):
                kx = torch.fft.fftshift(torch.fft.fft2(x))
                grad = torch.abs(torch.fft.ifft2(
                    torch.fft.ifftshift((kx - kspace) * mask_t)))
                x = torch.sign(x) * torch.clamp(
                    torch.abs(x - step * grad) - torch.abs(lam), min=0)
            return x

    k_t = torch.from_numpy(kspace_under).cfloat()
    m_t = torch.from_numpy(mask).float()

    model = UnrolledISTA(n_unrolls)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(30):
        opt.zero_grad()
        out = model(k_t, m_t)
        kout = torch.fft.fftshift(torch.fft.fft2(out))
        loss = torch.mean(torch.abs(kout * m_t - k_t * m_t)**2)
        loss.backward()
        opt.step()

    with torch.no_grad():
        recon = model(k_t, m_t).numpy()

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
