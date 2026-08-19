"""
AETHER SAR Classifier — UNet Land-Cover Classification Model & Inference Pipeline
Architecture: Same 4-level encoder/decoder UNet as the colorizer, but with out_channels=4
             (Water, Vegetation, Urban, Bare Soil) and NO Tanh activation (raw logits).
Input:  [B, 1, 256, 256]  grayscale SAR image normalized to [0, 1]
Output: [B, 4, 256, 256]  raw class logits (use argmax over dim=1 for class map)
"""

import time

import cv2
import numpy as np
import torch
import torch.nn as nn

# Reuse the same ConvBlock and despeckling filter from the colorizer module
from model import ConvBlock, apply_despeckling_filter


# Fixed land-cover class → RGB color mapping
CLASS_COLOR_MAP = {
    0: (0, 0, 255),        # Water → blue
    1: (0, 200, 0),        # Vegetation → green
    2: (150, 150, 150),    # Urban/Built-up → gray
    3: (194, 178, 128),    # Bare Soil/Other → tan
}

CLASS_LABELS = {
    0: "Water",
    1: "Vegetation",
    2: "Urban/Built-up",
    3: "Bare Soil/Other",
}


class UNetClassifier(nn.Module):
    """
    U-Net Classifier for SAR → per-pixel land-cover classification.
    Same encoder-decoder skeleton as the colorizer UNet, but:
      - out_channels=4 (4 land-cover classes)
      - No Tanh activation — outputs raw logits for CrossEntropyLoss / argmax
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 4):
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
        # No activation — raw logits output

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
        return self.final(d1)  # Raw logits [B, 4, H, W]


def classify_sar(
    image_array: np.ndarray,
    model: nn.Module,
    device: torch.device | str,
    size: int = 256,
    filter_type: str = "enhanced_lee",
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Classify a grayscale SAR image into 4 land-cover classes using a trained UNetClassifier.

    Args:
        image_array: Grayscale numpy array (2D array HxW or 3D single-channel HxWx1)
        model: Trained UNetClassifier model in evaluation mode
        device: PyTorch device ('cuda', 'cpu', or torch.device)
        size: Target spatial size for model input (default: 256)
        filter_type: Despeckling filter ('enhanced_lee', 'frost', 'deep_despeckle')

    Returns:
        Tuple of:
          - RGB numpy array [size, size, 3] uint8 with class colors
          - Grayscale confidence heatmap [size, size] uint8 (0=uncertain, 255=confident)
          - inference_time_ms float
          - Per-pixel class map numpy array [size, size] int64 with class indices 0..3
    """
    # 0. Ensure 2D array
    if image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = image_array.squeeze(2)

    # 1. Despeckling filter (same as colorizer — classifier was trained on despeckled input)
    denoised = apply_despeckling_filter(image_array, filter_type=filter_type)

    # 2. Resize input to size x size
    img_resized = cv2.resize(denoised, (size, size), interpolation=cv2.INTER_AREA)

    # 3. Normalize grayscale to [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0

    # 4. Prepare tensor batch [1, 1, size, size]
    tensor_in = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    # 5. Predict class logits (measure forward pass time only)
    model.eval()
    t_start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor_in)  # Output: [1, 4, size, size]
        probs = torch.softmax(logits, dim=1)  # Softmax probabilities [1, 4, size, size]
    t_end = time.perf_counter()
    inference_time_ms = (t_end - t_start) * 1000.0

    # 6. Argmax over class dimension & max probability confidence
    class_map = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [size, size]
    max_probs = probs.max(dim=1).values.squeeze(0).cpu().numpy()  # [size, size] in range [0.25, 1.0]

    # 7. Map each class index to its fixed RGB color
    rgb_output = np.zeros((size, size, 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLOR_MAP.items():
        mask = class_map == class_idx
        rgb_output[mask] = color

    # 8. Normalize confidence to 0-255 uint8 grayscale
    # Theoretical min for 4 classes is 0.25 (total uncertainty), max is 1.0 (certain)
    confidence_norm = np.clip((max_probs - 0.25) / 0.75 * 255.0, 0, 255).astype(np.uint8)

    return rgb_output, confidence_norm, inference_time_ms, class_map
