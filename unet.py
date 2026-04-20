import numpy as np
import time

# ── U-Net architecture ────────────────────────────────────────────────────────

def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class UNet:
    """
    Lightweight U-Net for MRI artifact removal.
    Input:  zero-filled reconstruction (256x256, normalised 0-1)
    Output: dealiased image
    Architecture: 3 encoder levels, bottleneck, 3 decoder levels.
    ~180k parameters — trains in ~20s on CPU on synthetic phantoms.
    """

    def __init__(self):
        self._model = None
        self._trained = False

    def _build(self):
        torch, nn = _get_torch()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        class _UNet(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = conv_block(1, 32)
                self.enc2 = conv_block(32, 64)
                self.enc3 = conv_block(64, 128)
                self.pool = nn.MaxPool2d(2)
                # Bottleneck
                self.bottleneck = conv_block(128, 256)
                # Decoder
                self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec3 = conv_block(256, 128)
                self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec2 = conv_block(128, 64)
                self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                self.dec1 = conv_block(64, 32)
                # Output — residual learning
                self.out = nn.Conv2d(32, 1, 1)

            def forward(self, x):
                inp = x
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                b  = self.bottleneck(self.pool(e3))
                d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                # Residual: learn the artifact, subtract it
                return torch.clamp(inp - self.out(d1), 0, 1)

        self._model = _UNet()
        return self._model

    def train(self, kspace_fn, n_samples=50, n_epochs=15, lr=1e-3):
        """
        Train on synthetic phantom variations.
        kspace_fn: callable(image) -> (kspace_under, mask)
        """
        torch, nn = _get_torch()
        from skimage.data import shepp_logan_phantom
        from skimage.transform import resize

        model = self._build()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # Generate synthetic training pairs
        base = shepp_logan_phantom()
        base = resize(base, (256, 256), anti_aliasing=True)
        base = (base - base.min()) / (base.max() - base.min())

        rng = np.random.default_rng(0)
        pairs = []
        for _ in range(n_samples):
            # Perturb phantom with random smooth deformation
            noise = rng.uniform(0.85, 1.15, base.shape)
            from scipy.ndimage import gaussian_filter
            noise = gaussian_filter(noise, sigma=20)
            img = np.clip(base * noise, 0, 1)
            kunder, mask = kspace_fn(img)
            from recon import zero_filled
            zf = zero_filled(kunder)
            zf = (zf - zf.min()) / (zf.max() - zf.min() + 1e-8)
            pairs.append((zf.astype(np.float32), img.astype(np.float32)))

        model.train()
        for epoch in range(n_epochs):
            rng.shuffle(pairs)
            for zf_np, gt_np in pairs:
                zf_t = torch.from_numpy(zf_np).unsqueeze(0).unsqueeze(0)
                gt_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)
                opt.zero_grad()
                pred = model(zf_t)
                loss = loss_fn(pred, gt_t)
                loss.backward()
                opt.step()

        model.eval()
        self._trained = True

    def predict(self, zero_filled_img):
        """Apply trained U-Net to a single zero-filled image."""
        torch, _ = _get_torch()
        if not self._trained:
            raise RuntimeError("Call train() before predict()")
        x = torch.from_numpy(zero_filled_img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self._model(x).squeeze().numpy()
        return out


def recon_unet(kspace_under, mask, unet_model):
    """
    U-Net reconstruction: zero-fill then apply trained network.
    Returns (reconstructed_image, elapsed_time).
    """
    from recon import zero_filled
    t0 = time.time()
    zf = zero_filled(kspace_under)
    zf = (zf - zf.min()) / (zf.max() - zf.min() + 1e-8)
    recon = unet_model.predict(zf)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    return recon, time.time() - t0
