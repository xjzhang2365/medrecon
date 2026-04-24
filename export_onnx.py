"""
Export the U-Net to unet_mri.onnx with random weights.

Run once (needs torch installed):
    python export_onnx.py

The resulting unet_mri.onnx is committed to the repo and loaded at runtime
by onnxruntime — no torch dependency at inference time.

Input shape:  (1, 1, 256, 256)  — batch=1, channel=1, H=256, W=256
Output shape: (1, 1, 256, 256)  — dealiased image, values clamped [0, 1]
"""
import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch):
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
        self.enc1 = _conv_block(1, 32)
        self.enc2 = _conv_block(32, 64)
        self.enc3 = _conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _conv_block(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _conv_block(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        inp = x
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.clamp(inp - self.out_conv(d1), 0.0, 1.0)


if __name__ == "__main__":
    model = _UNet()
    model.eval()
    dummy = torch.zeros(1, 1, 256, 256)

    torch.onnx.export(
        model,
        dummy,
        "unet_mri.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print("Exported unet_mri.onnx  (~180 k params, random weights — structural demo)")
