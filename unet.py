import numpy as np
import os
import time
import onnxruntime as ort

_ONNX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet_mri.onnx")

if not os.path.exists(_ONNX_PATH):
    raise FileNotFoundError(
        f"unet_mri.onnx not found — run  python export_onnx.py  first"
    )


class UNet:
    """
    Lightweight U-Net for MRI artifact removal, loaded from unet_mri.onnx.
    Input:  zero-filled reconstruction (256x256, normalised 0-1)
    Output: dealiased image
    """

    def __init__(self):
        self._session = ort.InferenceSession(_ONNX_PATH)
        self._input_name = self._session.get_inputs()[0].name

    def train(self, **kwargs):
        # Model is pre-exported; training is a no-op at inference time.
        pass

    def predict(self, zero_filled_img):
        """Apply U-Net to a single (256, 256) zero-filled image."""
        x = zero_filled_img.astype(np.float32)[np.newaxis, np.newaxis, ...]
        result = self._session.run(None, {self._input_name: x})
        return result[0].squeeze()


def recon_unet(kspace_under, mask, unet_model):
    """
    U-Net reconstruction: zero-fill then apply network.
    Returns (reconstructed_image, elapsed_time).
    """
    from recon import zero_filled
    t0 = time.time()
    zf = zero_filled(kspace_under)
    zf = (zf - zf.min()) / (zf.max() - zf.min() + 1e-8)
    recon = unet_model.predict(zf)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    return recon, time.time() - t0
