import base64
import io
import json
import logging
import os
from pathlib import Path
import urllib.request

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import numpy as np
from PIL import Image
from pydantic import BaseModel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

import groq
from groq import Groq

from classifier_model import CLASS_LABELS, UNetClassifier, classify_sar
from database import (
    delete_history_record,
    get_history_record_by_id,
    get_history_records,
    init_db,
    save_history_entry,
)
from fusion_model import FusionUNet, fuse_sar_optical
from model import UNet, colorize_sar

# Load environment variables from .env file if present
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("aether-sar")

# Paths and device setup
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "gan_final_epoch44.pth"
CLASSIFIER_CHECKPOINT_PATH = CHECKPOINT_DIR / "classifier_final_v3.pth"
FUSION_CHECKPOINT_PATH = CHECKPOINT_DIR / "fusion_epoch25.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_URLS = {
    "gan_final_epoch44.pth": "https://huggingface.co/manoj5kumar/aether-sar-checkpoints/resolve/main/gan_final_epoch44.pth",
    "classifier_final_v3.pth": "https://huggingface.co/manoj5kumar/aether-sar-checkpoints/resolve/main/classifier_final_v3.pth",
    "fusion_epoch25.pth": "https://huggingface.co/manoj5kumar/aether-sar-checkpoints/resolve/main/fusion_epoch25.pth",
}


def ensure_checkpoints():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    for filename, url in CHECKPOINT_URLS.items():
        path = CHECKPOINT_DIR / filename
        if not path.exists():
            print(f"Downloading {filename} from Hugging Face...")
            logger.info(f"Downloading {filename} from Hugging Face...")
            try:
                urllib.request.urlretrieve(url, str(path))
                print(f"Downloaded {filename}")
                logger.info(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                logger.error(f"Failed to download {filename}: {e}")


app = FastAPI(title="SAR Colorizer Backend")

# Enable CORS for all origins (without allow_credentials to avoid CORS browser conflicts with wildcard origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Inference-Time-Ms", "X-Image-Size", "X-Model-Checkpoint", "X-Classification-Mode", "X-PSNR", "X-SSIM", "X-History-ID"],
)

# Global model instances
model: UNet | None = None
model_loaded: bool = False
classifier: UNetClassifier | None = None
classifier_loaded: bool = False
fusion_model: FusionUNet | None = None
fusion_loaded: bool = False


@app.on_event("startup")
async def startup_event():
    global model, model_loaded, classifier, classifier_loaded, fusion_model, fusion_loaded

    # Ensure checkpoint files exist locally before loading
    ensure_checkpoints()

    # Initialize SQLite DB and outputs folder
    try:
        init_db()
        logger.info("SQLite database and outputs directory initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)

    # --- Load Colorizer Model ---
    model = UNet(in_channels=1, out_channels=2).to(DEVICE)

    if not CHECKPOINT_PATH.exists():
        logger.warning("Colorizer checkpoint not found at %s", CHECKPOINT_PATH)
        model_loaded = False
    else:
        try:
            state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            model_loaded = True
            logger.info("Colorizer model loaded from %s on %s", CHECKPOINT_PATH, DEVICE)
        except Exception as e:
            logger.error("Failed to load colorizer weights: %s", e)
            model_loaded = False

    # --- Load Classifier Model ---
    classifier = UNetClassifier(in_channels=1, out_channels=4).to(DEVICE)

    if not CLASSIFIER_CHECKPOINT_PATH.exists():
        logger.warning("Classifier checkpoint not found at %s — /classify endpoint will return 503", CLASSIFIER_CHECKPOINT_PATH)
        classifier_loaded = False
    else:
        try:
            cls_state = torch.load(CLASSIFIER_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            classifier.load_state_dict(cls_state)
            classifier.eval()
            classifier_loaded = True
            logger.info("Classifier model loaded from %s on %s", CLASSIFIER_CHECKPOINT_PATH, DEVICE)
        except Exception as e:
            logger.error("Failed to load classifier weights: %s", e)
            classifier_loaded = False

    # --- Load Fusion Model ---
    fusion_model = FusionUNet(in_channels=4, out_channels=2).to(DEVICE)

    if not FUSION_CHECKPOINT_PATH.exists():
        logger.warning("Fusion checkpoint not found at %s", FUSION_CHECKPOINT_PATH)
        fusion_loaded = False
    else:
        try:
            fus_state = torch.load(FUSION_CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
            fusion_model.load_state_dict(fus_state)
            fusion_model.eval()
            fusion_loaded = True
            logger.info("Fusion model loaded from %s on %s", FUSION_CHECKPOINT_PATH, DEVICE)
        except Exception as e:
            logger.error("Failed to load fusion weights: %s", e)
            fusion_loaded = False

    # --- Model Health & Shape Logs ---
    def _shape_summary(m, name):
        params = sum(p.numel() for p in m.parameters())
        return f"{name}: {params:,} parameters"

    logger.info("=== MODEL HEALTH CHECK ===")
    logger.info("  Colorizer (SAR -> Color) : %s | %s", "✓ LOADED" if model_loaded else "✗ MISSING", _shape_summary(model, "UNet(1->2)"))
    logger.info("  Classifier (SAR -> Class): %s | %s", "✓ LOADED" if classifier_loaded else "✗ MISSING", _shape_summary(classifier, "UNetClassifier(1->4)"))
    logger.info("  Fusion (SAR+Opt -> Color): %s | %s", "✓ LOADED" if fusion_loaded else "✗ MISSING", _shape_summary(fusion_model, "FusionUNet(4->2)"))
    logger.info("==========================")

    logger.info("Despeckling Filter Support: Enhanced Lee (Real Adaptive Filter), Frost & Deep Despeckle (Placeholder Fallback to Median Blur).")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "colorizer_loaded": model_loaded,
        "classifier_loaded": classifier_loaded,
        "fusion_loaded": fusion_loaded,
        "checkpoints": {
            "colorizer": CHECKPOINT_PATH.name,
            "classifier": CLASSIFIER_CHECKPOINT_PATH.name,
            "fusion": FUSION_CHECKPOINT_PATH.name
        },
        "despeckling_filters": {
            "enhanced_lee": "Real Adaptive Lee Filter",
            "frost": "Placeholder Fallback (Median Blur k=5)",
            "deep_despeckle": "Placeholder Fallback (Median Blur k=5)"
        }
    }


@app.post("/colorize")
async def colorize(
    file: UploadFile = File(...),
    ground_truth: UploadFile | None = File(None),
    filter_type: str = Form("enhanced_lee"),
):
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded. Please ensure unet_gan_epoch20.pth is present in backend/checkpoints/.",
        )

    try:
        # Read uploaded image bytes
        contents = await file.read()

        # Convert to grayscale numpy array using PIL
        pil_img = Image.open(io.BytesIO(contents)).convert("L")
        img_np = np.array(pil_img)
        original_h, original_w = img_np.shape[:2]

        # Run colorization inference with requested despeckling filter
        rgb_np, inference_time_ms = colorize_sar(img_np, model, DEVICE, size=256, filter_type=filter_type)

        # Optional: Compute PSNR and SSIM if ground truth optical image is uploaded
        psnr_val, ssim_val = None, None
        if ground_truth is not None and ground_truth.filename:
            try:
                gt_contents = await ground_truth.read()
                if gt_contents:
                    gt_pil = Image.open(io.BytesIO(gt_contents)).convert("RGB")
                    # Resize ground truth image to match model output dimensions (256x256)
                    gt_pil_resized = gt_pil.resize((rgb_np.shape[1], rgb_np.shape[0]), Image.Resampling.BILINEAR)
                    gt_np = np.array(gt_pil_resized)

                    psnr_val = float(peak_signal_noise_ratio(gt_np, rgb_np, data_range=255))
                    ssim_val = float(structural_similarity(gt_np, rgb_np, channel_axis=2, data_range=255))
                    logger.info("Computed metrics: PSNR=%.2f dB, SSIM=%.4f", psnr_val, ssim_val)
            except Exception as e:
                logger.warning("Could not calculate ground truth metrics: %s", e)

        # Convert output RGB numpy array to PNG bytes
        out_pil = Image.fromarray(rgb_np)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        buf.seek(0)

        # Save to SQLite history & outputs/ directory as side-effect
        record_id = None
        try:
            input_fn = file.filename or "uploaded_sar.png"
            record_id = save_history_entry(
                mode="colorize",
                filename=input_fn,
                image_bytes=png_bytes,
                inference_time_ms=inference_time_ms,
                psnr=psnr_val,
                ssim=ssim_val,
            )
        except Exception as e:
            logger.warning("Could not save history entry for colorize: %s", e)

        response = StreamingResponse(buf, media_type="image/png")
        response.headers["X-Inference-Time-Ms"] = f"{inference_time_ms:.2f}"
        response.headers["X-Image-Size"] = f"{original_w}x{original_h}"
        response.headers["X-Model-Checkpoint"] = "gan_final_epoch24"

        if record_id:
            response.headers["X-History-ID"] = record_id

        if psnr_val is not None and ssim_val is not None:
            response.headers["X-PSNR"] = f"{psnr_val:.2f}"
            response.headers["X-SSIM"] = f"{ssim_val:.4f}"

        return response

    except Exception as e:
        logger.error("Inference error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Colorization processing failed: {str(e)}",
        )


@app.post("/fuse")
async def fuse(
    sar_file: UploadFile = File(...),
    optical_file: UploadFile = File(...),
    filter_type: str = Form("enhanced_lee"),
):
    """
    Multi-Sensor Fusion endpoint: Accepts SAR image + partial/masked optical reference image,
    concatenates into 4-channel input [SAR, R, G, B], and passes through trained FusionUNet.
    """
    if not fusion_loaded or fusion_model is None:
        raise HTTPException(
            status_code=500,
            detail="Fusion model is not loaded. Please ensure fusion_epoch25.pth is present in backend/checkpoints/.",
        )

    try:
        # Read uploaded SAR raster & convert to grayscale numpy array
        sar_contents = await sar_file.read()
        sar_pil = Image.open(io.BytesIO(sar_contents)).convert("L")
        sar_np = np.array(sar_pil)
        original_h, original_w = sar_np.shape[:2]

        # Read uploaded optical reference raster & convert to RGB numpy array
        opt_contents = await optical_file.read()
        opt_pil = Image.open(io.BytesIO(opt_contents)).convert("RGB")
        opt_np = np.array(opt_pil)

        # Run multi-sensor fusion inference
        rgb_np, inference_time_ms = fuse_sar_optical(
            sar_np,
            opt_np,
            fusion_model,
            DEVICE,
            size=256,
            filter_type=filter_type,
        )

        # Convert output RGB numpy array to PNG bytes
        out_pil = Image.fromarray(rgb_np)
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        buf.seek(0)

        # Save to SQLite history & outputs/ directory as side-effect
        record_id = None
        try:
            input_fn = sar_file.filename or "uploaded_fusion_sar.png"
            record_id = save_history_entry(
                mode="fusion",
                filename=input_fn,
                image_bytes=png_bytes,
                inference_time_ms=inference_time_ms,
                psnr=None,
                ssim=None,
            )
        except Exception as e:
            logger.warning("Could not save history entry for fusion: %s", e)

        response = StreamingResponse(buf, media_type="image/png")
        response.headers["X-Inference-Time-Ms"] = f"{inference_time_ms:.2f}"
        response.headers["X-Image-Size"] = f"{original_w}x{original_h}"
        response.headers["X-Model-Checkpoint"] = "fusion_epoch25"

        if record_id:
            response.headers["X-History-ID"] = record_id

        return response

    except Exception as e:
        logger.error("Fusion inference error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Fusion model error: {str(e)}")


@app.post("/classify")
async def classify(
    file: UploadFile = File(...),
    filter_type: str = Form("enhanced_lee"),
):
    if not classifier_loaded or classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model is not loaded. Please place classifier.pth in backend/checkpoints/.",
        )

    try:
        # Read uploaded image bytes
        contents = await file.read()

        # Convert to grayscale numpy array using PIL
        pil_img = Image.open(io.BytesIO(contents)).convert("L")
        img_np = np.array(pil_img)
        original_h, original_w = img_np.shape[:2]

        # Run classification inference with requested despeckling filter
        rgb_np, confidence_np, inference_time_ms, class_map = classify_sar(
            img_np, classifier, DEVICE, size=256, filter_type=filter_type
        )

        # Compute per-class pixel percentages from class_map
        total_pixels = class_map.size
        class_percentages = {}
        for idx, label in CLASS_LABELS.items():
            count = int(np.sum(class_map == idx))
            class_percentages[label] = round(count / total_pixels * 100, 2)

        # Mean confidence score (normalized from 0-255 heatmap to 0.0-1.0)
        mean_confidence = round(float(np.mean(confidence_np) / 255.0), 4)

        logger.info(
            "Classification breakdown: %s | Mean confidence: %.2f%%",
            ", ".join(f"{k}: {v}%" for k, v in class_percentages.items()),
            mean_confidence * 100,
        )

        # Convert classified output RGB numpy array to PNG bytes & base64
        out_pil = Image.fromarray(rgb_np)
        buf_classified = io.BytesIO()
        out_pil.save(buf_classified, format="PNG")
        png_bytes = buf_classified.getvalue()
        b64_classified = base64.b64encode(png_bytes).decode("utf-8")

        # Convert confidence heatmap grayscale numpy array to PNG base64
        conf_pil = Image.fromarray(confidence_np)
        buf_confidence = io.BytesIO()
        conf_pil.save(buf_confidence, format="PNG")
        b64_confidence = base64.b64encode(buf_confidence.getvalue()).decode("utf-8")

        # Save to SQLite history & outputs/ directory as side-effect
        record_id = None
        try:
            input_fn = file.filename or "uploaded_sar.png"
            record_id = save_history_entry(
                mode="classify",
                filename=input_fn,
                image_bytes=png_bytes,
                inference_time_ms=inference_time_ms,
            )
        except Exception as e:
            logger.warning("Could not save history entry for classify: %s", e)

        headers = {
            "X-Inference-Time-Ms": f"{inference_time_ms:.2f}",
            "X-Image-Size": f"{original_w}x{original_h}",
            "X-Model-Checkpoint": "classifier",
            "X-Classification-Mode": "land_cover",
        }
        if record_id:
            headers["X-History-ID"] = record_id

        return JSONResponse(
            content={
                "id": record_id,
                "classified_image": f"data:image/png;base64,{b64_classified}",
                "confidence_heatmap": f"data:image/png;base64,{b64_confidence}",
                "inference_time_ms": round(inference_time_ms, 2),
                "image_size": f"{original_w}x{original_h}",
                "model_checkpoint": "classifier",
                "classification_mode": "land_cover",
                "class_percentages": class_percentages,
                "mean_confidence": mean_confidence,
            },
            headers=headers,
        )

    except Exception as e:
        logger.error("Classification error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Classification processing failed: {str(e)}",
        )


@app.post("/batch")
async def batch_process(
    files: list[UploadFile] = File(...),
    mode: str = Form("colorize"),
    filter_type: str = Form("enhanced_lee"),
):
    """
    Batch processes multiple SAR images in a single request.
    Calls existing colorize_sar() or classify_sar() per image.
    Saves each result to SQLite history database as a side effect.
    Returns a JSON array of per-file results with isolated error handling.
    """
    if mode == "classify":
        if not classifier_loaded or classifier is None:
            raise HTTPException(
                status_code=503,
                detail="Classifier model is not loaded. Please place classifier.pth in backend/checkpoints/.",
            )
    else:
        if not model_loaded or model is None:
            raise HTTPException(
                status_code=500,
                detail="Colorizer model is not loaded. Please ensure gan_final_epoch24.pth is present in backend/checkpoints/.",
            )

    results = []
    for upload_file in files:
        fn = upload_file.filename or "uploaded_sar.png"
        try:
            contents = await upload_file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("L")
            img_np = np.array(pil_img)
            original_h, original_w = img_np.shape[:2]

            if mode == "classify":
                rgb_np, confidence_np, inference_time_ms, _ = classify_sar(
                    img_np, classifier, DEVICE, size=256, filter_type=filter_type
                )
                out_pil = Image.fromarray(rgb_np)
                buf_c = io.BytesIO()
                out_pil.save(buf_c, format="PNG")
                png_bytes = buf_c.getvalue()
                b64_output = base64.b64encode(png_bytes).decode("utf-8")

                conf_pil = Image.fromarray(confidence_np)
                buf_conf = io.BytesIO()
                conf_pil.save(buf_conf, format="PNG")
                b64_conf = base64.b64encode(buf_conf.getvalue()).decode("utf-8")

                rec_id = None
                try:
                    rec_id = save_history_entry(
                        mode="classify",
                        filename=fn,
                        image_bytes=png_bytes,
                        inference_time_ms=inference_time_ms,
                    )
                except Exception as e:
                    logger.warning("History save failed for batch file %s: %s", fn, e)

                results.append({
                    "filename": fn,
                    "status": "success",
                    "id": rec_id,
                    "mode": "classify",
                    "output_image": f"data:image/png;base64,{b64_output}",
                    "confidence_heatmap": f"data:image/png;base64,{b64_conf}",
                    "inference_time_ms": round(inference_time_ms, 2),
                    "image_size": f"{original_w}x{original_h}",
                })
            else:
                rgb_np, inference_time_ms = colorize_sar(
                    img_np, model, DEVICE, size=256, filter_type=filter_type
                )
                out_pil = Image.fromarray(rgb_np)
                buf_c = io.BytesIO()
                out_pil.save(buf_c, format="PNG")
                png_bytes = buf_c.getvalue()
                b64_output = base64.b64encode(png_bytes).decode("utf-8")

                rec_id = None
                try:
                    rec_id = save_history_entry(
                        mode="colorize",
                        filename=fn,
                        image_bytes=png_bytes,
                        inference_time_ms=inference_time_ms,
                    )
                except Exception as e:
                    logger.warning("History save failed for batch file %s: %s", fn, e)

                results.append({
                    "filename": fn,
                    "status": "success",
                    "id": rec_id,
                    "mode": "colorize",
                    "output_image": f"data:image/png;base64,{b64_output}",
                    "inference_time_ms": round(inference_time_ms, 2),
                    "image_size": f"{original_w}x{original_h}",
                })
        except Exception as e:
            logger.error("Error processing batch item %s: %s", fn, e)
            results.append({
                "filename": fn,
                "status": "error",
                "error": str(e),
            })

    return results


# ==============================================================================
# CHANGE DETECTION ENDPOINT
# ==============================================================================

TRANSITION_COLOR_MAP = {
    # (from_class, to_class): (R, G, B, A)
    (1, 2): (255, 0, 100, 255),    # Vegetation -> Urban (Deforestation / Urbanization) -> Magenta/Red
    (1, 0): (0, 200, 255, 255),    # Vegetation -> Water (Flooding) -> Cyan/Blue
    (1, 3): (255, 140, 0, 255),    # Vegetation -> Bare Soil (Land clearing / drought) -> Orange
    (0, 1): (0, 255, 128, 255),    # Water -> Vegetation (Reclamation / drying) -> Mint Green
    (0, 3): (255, 220, 0, 255),    # Water -> Bare Soil -> Yellow
    (2, 0): (0, 100, 255, 255),    # Urban -> Water -> Deep Blue
    (3, 2): (200, 50, 255, 255),   # Bare Soil -> Urban (New construction) -> Purple
    (3, 1): (50, 205, 50, 255),    # Bare Soil -> Vegetation (Afforestation/Crop growth) -> Lime Green
}
DEFAULT_CHANGE_COLOR = (235, 45, 45, 255)  # High-visibility Red

CLASS_NAMES = {
    0: "Water",
    1: "Vegetation",
    2: "Urban/Built-up",
    3: "Bare Soil/Other",
}


@app.post("/change-detect")
async def change_detect(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    pixel_resolution_m: float = Form(10.0),
    filter_type: str = Form("enhanced_lee"),
):
    """
    Compares two temporal SAR rasters (before vs after), runs both through classify_sar(),
    computes pixel-wise land-cover class transitions, calculates physical area changes (km²),
    and generates a multi-layer change overlay image.
    """
    if not classifier_loaded or classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Classifier model is not loaded. Please place classifier.pth in backend/checkpoints/.",
        )

    try:
        # 1. Read input images
        before_bytes = await before_file.read()
        after_bytes = await after_file.read()

        before_pil = Image.open(io.BytesIO(before_bytes)).convert("L")
        after_pil = Image.open(io.BytesIO(after_bytes)).convert("L")

        before_np = np.array(before_pil)
        after_np = np.array(after_pil)

        # 2. Run classify_sar on both images
        before_rgb, before_conf, t1, before_class = classify_sar(
            before_np, classifier, DEVICE, size=256, filter_type=filter_type
        )
        after_rgb, after_conf, t2, after_class = classify_sar(
            after_np, classifier, DEVICE, size=256, filter_type=filter_type
        )
        total_inference_time_ms = round(t1 + t2, 2)

        # 3. Calculate spatial pixel area & dimensions
        H, W = before_class.shape
        total_pixels = H * W
        pixel_area_m2 = pixel_resolution_m ** 2
        total_area_km2 = (total_pixels * pixel_area_m2) / 1_000_000.0

        # 4. Pixel-wise diff & Change Overlay RGBA
        overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        # Unchanged pixels: semi-transparent gray
        unchanged_mask = before_class == after_class
        overlay_rgba[unchanged_mask] = [128, 128, 128, 40]

        # Changed pixels
        changed_mask = ~unchanged_mask
        changed_pixel_count = int(np.sum(changed_mask))
        percent_changed = (changed_pixel_count / total_pixels) * 100.0

        # Build transitions summary & paint change overlay
        transitions = []
        for from_c in range(4):
            for to_c in range(4):
                if from_c == to_c:
                    continue
                match_mask = (before_class == from_c) & (after_class == to_c)
                count = int(np.sum(match_mask))
                if count > 0:
                    color = TRANSITION_COLOR_MAP.get((from_c, to_c), DEFAULT_CHANGE_COLOR)
                    overlay_rgba[match_mask] = color

                    area_km2 = (count * pixel_area_m2) / 1_000_000.0
                    pct = (count / total_pixels) * 100.0
                    transitions.append({
                        "from_class": CLASS_NAMES[from_c],
                        "to_class": CLASS_NAMES[to_c],
                        "area_km2": round(area_km2, 4),
                        "percent_of_image": round(pct, 2),
                    })

        # Sort transitions by area descending
        transitions.sort(key=lambda x: x["area_km2"], reverse=True)

        # 5. Convert images to base64 PNG data URLs
        def pil_to_b64(pil_image: Image.Image) -> str:
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        b64_before = pil_to_b64(Image.fromarray(before_rgb))
        b64_after = pil_to_b64(Image.fromarray(after_rgb))
        b64_overlay = pil_to_b64(Image.fromarray(overlay_rgba, mode="RGBA"))

        # Save change overlay PNG to SQLite history as side-effect
        record_id = None
        try:
            overlay_pil = Image.fromarray(overlay_rgba, mode="RGBA")
            buf_ov = io.BytesIO()
            overlay_pil.save(buf_ov, format="PNG")
            fn_before = before_file.filename or "before.png"
            fn_after = after_file.filename or "after.png"
            input_fn = f"{fn_before} vs {fn_after}"
            record_id = save_history_entry(
                mode="change_detect",
                filename=input_fn,
                image_bytes=buf_ov.getvalue(),
                inference_time_ms=total_inference_time_ms,
            )
        except Exception as e:
            logger.warning("Could not save history entry for change_detect: %s", e)

        headers = {
            "X-Inference-Time-Ms": f"{total_inference_time_ms:.2f}",
            "X-Model-Checkpoint": "classifier",
            "X-Analysis-Mode": "change_detection",
        }
        if record_id:
            headers["X-History-ID"] = record_id

        return JSONResponse(
            content={
                "id": record_id,
                "before_classified_image": b64_before,
                "after_classified_image": b64_after,
                "change_overlay_image": b64_overlay,
                "transitions": transitions,
                "total_area_km2": round(total_area_km2, 4),
                "percent_changed": round(percent_changed, 2),
                "inference_time_ms": total_inference_time_ms,
            },
            headers=headers,
        )

    except Exception as e:
        logger.error("Change detection error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Change detection processing failed: {str(e)}",
        )


# ==============================================================================
# NARRATION ENDPOINT (Groq LLM Analysis)
# ==============================================================================

class NarrateRequest(BaseModel):
    mode: str  # "classify" or "change-detect"
    stats: dict


@app.post("/narrate")
async def narrate(request: NarrateRequest):
    """
    Generates a 3-5 sentence plain-English geospatial analysis narrative using Groq LLM
    from pre-computed classification or change-detection statistics.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Narration service not configured — set GROQ_API_KEY",
        )

    system_prompt = (
        "You are an expert geospatial remote sensing analyst interpreting Synthetic Aperture Radar (SAR) statistics.\n"
        "Generate a short, clear, plain-English executive analysis paragraph (3-5 sentences).\n"
        "Instructions:\n"
        "1. State the single most important key finding in the very first sentence.\n"
        "2. Reference ONLY numbers and values explicitly present in the input stats. Do NOT invent, hallucinate, or extrapolate numbers.\n"
        "3. If low_confidence_regions or a notably low mean_confidence value is present in the stats, add exactly one sentence explaining where or why model confidence is reduced.\n"
        "4. Keep the tone professional, concise, and focused on environmental or urban insights."
    )

    user_prompt = f"Mode: {request.mode}\nInput Statistics JSON:\n{json.dumps(request.stats, indent=2)}"

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        narrative_text = response.choices[0].message.content.strip()
        # Strip <think>...</think> reasoning blocks from Qwen-style models
        import re
        narrative_text = re.sub(r'<think>.*?</think>', '', narrative_text, flags=re.DOTALL).strip()
        if not narrative_text:
            raise HTTPException(status_code=500, detail="Model returned empty narrative")
        return {"narrative": narrative_text}

    except (groq.APIConnectionError, groq.RateLimitError, groq.APIStatusError) as ge:
        logger.error("Groq API error: %s", ge)
        raise HTTPException(
            status_code=502,
            detail=f"Narration service API error: {str(ge)}",
        )
    except Exception as e:
        logger.error("Narration processing error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Narration generation failed: {str(e)}",
        )


# ==============================================================================
# HISTORY ENDPOINTS (SQLite)
# ==============================================================================


@app.get("/history")
async def get_history():
    """Returns the last 20 history records, most recent first."""
    try:
        records = get_history_records(limit=20)
        # Format response records
        formatted = []
        for r in records:
            formatted.append({
                "id": r["id"],
                "mode": r["mode"],
                "timestamp": r["timestamp"],
                "filename": r["filename"],
                "inference_time_ms": round(r["inference_time_ms"], 2),
                "psnr": r.get("psnr"),
                "ssim": r.get("ssim"),
            })
        return formatted
    except Exception as e:
        logger.error("Error retrieving history: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch history records")


@app.get("/history/{record_id}/image")
async def get_history_image(record_id: str):
    """Returns the stored output image PNG file for the given record ID."""
    record = get_history_record_by_id(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="History record not found")

    img_path = Path(record["image_path"])
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Stored output image file not found on disk")

    return FileResponse(path=img_path, media_type="image/png", filename=f"{record_id}.png")


@app.delete("/history/{record_id}")
async def delete_history(record_id: str):
    """Deletes the history record from SQLite and removes its stored image file."""
    deleted = delete_history_record(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History record not found")

    return {"status": "deleted", "id": record_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
