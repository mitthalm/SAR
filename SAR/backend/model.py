"""
AETHER SAR Colorizer — U-Net Generator Model & Inference Pipeline
Architecture: 4-level encoder/decoder with skip connections + bottleneck (in_channels=1, out_channels=2).
Input:  [B, 1, 256, 256]  grayscale SAR image normalized to [0, 1]
Output: [B, 2, 256, 256]  predicted Lab a, b channels in approx [-1, 1]
"""

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.color import lab2rgb


class ConvBlock(nn.Module):
    """Double convolution block: Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Alias UNetBlock to ConvBlock for compatibility
UNetBlock = ConvBlock


class UNet(nn.Module):
    """
    U-Net Generator for SAR → Lab(a, b) colorization matching the exact checkpoint state_dict.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2):
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


# Alias UNetGenerator to UNet for backwards compatibility
UNetGenerator = UNet


def apply_lee_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Lee adaptive filter for SAR despeckling based on local window mean and variance.
    Calculates adaptive weight W = local_var / (local_var + overall_var) to smooth noise
    while preserving high-contrast edges.
    """
    img_float = img.astype(np.float32)
    mean = cv2.blur(img_float, (window_size, window_size))
    mean_sq = cv2.blur(img_float ** 2, (window_size, window_size))
    var = np.maximum(mean_sq - mean ** 2, 0)
    overall_var = np.var(img_float) + 1e-5
    weight = var / (var + overall_var)
    filtered = mean + weight * (img_float - mean)
    return np.clip(filtered, 0, 255).astype(np.uint8)


def apply_despeckling_filter(img: np.ndarray, filter_type: str = "enhanced_lee") -> np.ndarray:
    """
    Applies the selected despeckling filter to input SAR image.

    NOTE ON ALGORITHM STATUS:
    - 'enhanced_lee' / 'lee': Real adaptive Lee filter algorithm implemented.
    - 'frost': PLACEHOLDER FALLBACK (currently falls back to cv2.medianBlur(img, 5)).
               Full Frost exponential distance-weighted damping filter is planned for future release.
    - 'deep_despeckle': PLACEHOLDER FALLBACK (currently falls back to cv2.medianBlur(img, 5)).
    """
    if filter_type in ("enhanced_lee", "lee"):
        return apply_lee_filter(img, window_size=5)
    elif filter_type == "frost":
        # PLACEHOLDER FALLBACK: Frost algorithm currently falls back to median blur
        return cv2.medianBlur(img, 5)
    else:
        # PLACEHOLDER FALLBACK: deep_despeckle / default currently falls back to median blur
        return cv2.medianBlur(img, 5)



def colorize_sar(
    image_array: np.ndarray,
    model: nn.Module,
    device: torch.device | str,
    size: int = 256,
    filter_type: str = "enhanced_lee",
) -> tuple[np.ndarray, float]:
    """
    Colorize a grayscale SAR image array using a trained UNet model in Lab color space.

    Args:
        image_array: Grayscale numpy array (2D array HxW or 3D single-channel HxWx1)
        model: Trained UNet model in evaluation mode
        device: PyTorch device ('cuda', 'cpu', or torch.device)
        size: Target spatial size for model input (default: 256)
        filter_type: Despeckling filter ('enhanced_lee', 'frost', 'deep_despeckle')

    Returns:
        Tuple of (RGB numpy array [size, size, 3] uint8, inference_time_ms float).
    """
    # 0. Ensure 2D array
    if image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze(2)

    # 1. Despeckling filter
    denoised = apply_despeckling_filter(image_array, filter_type=filter_type)

    # 2. Resize input to size x size
    img_resized = cv2.resize(denoised, (size, size), interpolation=cv2.INTER_AREA)

    # 3. Normalize grayscale to [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0

    # 4. Prepare tensor batch [1, 1, size, size]
    tensor_in = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    # 5. Predict Lab a, b channels (measure forward pass time only)
    model.eval()
    t_start = time.perf_counter()
    with torch.no_grad():
        ab_pred = model(tensor_in)  # Output: [1, 2, size, size] in ~[-1, 1]
    t_end = time.perf_counter()
    inference_time_ms = (t_end - t_start) * 1000.0

    ab_pred = ab_pred.squeeze(0).cpu().numpy()  # [2, size, size]

    # 6. Reconstruct Lab color space
    L = img_norm * 100.0
    a = ab_pred[0] * 128.0
    b = ab_pred[1] * 128.0

    lab_image = np.stack([L, a, b], axis=-1).astype(np.float64)  # [size, size, 3]

    # 7. Convert Lab back to RGB
    rgb_float = lab2rgb(lab_image)  # [0.0, 1.0] range
    rgb_uint8 = (rgb_float * 255.0).clip(0, 255).astype(np.uint8)

    return rgb_uint8, inference_time_ms
