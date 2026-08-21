"""
AETHER SAR Fusion — FusionUNet Model & Inference Pipeline
Architecture: Same 4-level UNet encoder/decoder structure as the colorizer,
              but with in_channels=4 (SAR + masked/partial optical) and out_channels=2 (ab channels).
Input:  [B, 4, 256, 256]  grayscale SAR image (1ch) concatenated with partial optical image (3ch)
Output: [B, 2, 256, 256]  predicted Lab a, b channels in approx [-1, 1] with Tanh activation
"""

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.color import lab2rgb

# Reuse the existing ConvBlock and despeckling filter from the colorizer module
from model import ConvBlock, apply_despeckling_filter


class FusionUNet(nn.Module):
    """
    Fusion U-Net model for multi-sensor SAR + partial optical image color completion.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 2):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.tanh(self.final(d1))


def fuse_sar_optical(
    sar_array: np.ndarray,
    optical_array: np.ndarray,
    model: nn.Module,
    device: torch.device | str,
    size: int = 256,
    filter_type: str = "enhanced_lee",
) -> tuple[np.ndarray, float]:
    """
    Colorize/reconstruct SAR raster fused with a partial/masked optical reference image.

    Args:
        sar_array: Grayscale numpy array (2D array HxW or 3D single-channel HxWx1)
        optical_array: RGB numpy array (3D array HxWx3 uint8 in range 0..255)
        model: Trained FusionUNet model in evaluation mode
        device: PyTorch device ('cuda', 'cpu', or torch.device)
        size: Target spatial size for model input (default: 256)
        filter_type: Despeckling filter ('enhanced_lee', 'frost', 'deep_despeckle')

    Returns:
        Tuple of (RGB numpy array [size, size, 3] uint8, inference_time_ms float).
    """
    # 0. Ensure 2D SAR array
    if sar_array.ndim == 3 and sar_array.shape[2] == 1:
        sar_array = sar_array.squeeze(2)

    # 1. Despeckle SAR input using existing filter pipeline
    denoised_sar = apply_despeckling_filter(sar_array, filter_type=filter_type)

    # 2. Resize SAR to size x size & normalize to [0, 1]
    sar_resized = cv2.resize(denoised_sar, (size, size), interpolation=cv2.INTER_AREA)
    sar_norm = sar_resized.astype(np.float32) / 255.0  # [size, size]
    sar_tensor = torch.from_numpy(sar_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]

    # 3. Ensure 3D RGB optical array, resize to size x size & normalize to [0, 1]
    if optical_array.ndim == 2:
        optical_array = np.stack([optical_array] * 3, axis=-1)
    elif optical_array.ndim == 3 and optical_array.shape[2] == 4:
        optical_array = optical_array[:, :, :3]  # Drop alpha channel if present

    opt_resized = cv2.resize(optical_array, (size, size), interpolation=cv2.INTER_AREA)
    opt_norm = opt_resized.astype(np.float32) / 255.0  # [size, size, 3]
    opt_norm_chw = np.transpose(opt_norm, (2, 0, 1))  # [3, size, size]
    opt_tensor = torch.from_numpy(opt_norm_chw).unsqueeze(0)  # [1, 3, size, size]

    # 4. Concatenate into 4-channel tensor batch [1, 4, size, size]
    tensor_in = torch.cat([sar_tensor, opt_tensor], dim=1).to(device)

    # 5. Predict Lab a, b channels (measuring forward pass time only)
    model.eval()
    t_start = time.perf_counter()
    with torch.no_grad():
        ab_pred = model(tensor_in)  # Output: [1, 2, size, size] in ~[-1, 1]
    t_end = time.perf_counter()
    inference_time_ms = (t_end - t_start) * 1000.0

    ab_pred = ab_pred.squeeze(0).cpu().numpy()  # [2, size, size]

    # 6. Reconstruct Lab color space (reusing identical colorization logic)
    L = sar_norm * 100.0
    a = ab_pred[0] * 128.0
    b = ab_pred[1] * 128.0

    lab_image = np.stack([L, a, b], axis=-1).astype(np.float64)  # [size, size, 3]

    # 7. Convert Lab back to RGB
    rgb_float = lab2rgb(lab_image)  # [0.0, 1.0] range
    rgb_uint8 = (rgb_float * 255.0).clip(0, 255).astype(np.uint8)

    return rgb_uint8, inference_time_ms
