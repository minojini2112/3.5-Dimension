"""Sketch preprocessing: normalise any drawn/uploaded sketch into a clean, centred line map."""
import cv2
import numpy as np
from PIL import Image


class SketchError(ValueError):
    pass


def to_rgb(img) -> np.ndarray:
    """Any accepted input as uint8 RGB (or a single channel for grayscale images), transparency on white."""
    if isinstance(img, dict):  # gradio ImageEditor/Sketchpad value
        comp = img.get("composite")
        if comp is None:
            layers = img.get("layers") or []
            comp = layers[0] if layers else img.get("background")
        img = comp
    if img is None:
        raise SketchError("No sketch provided. Draw something or upload an image.")
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 4:  # transparent canvas: put strokes on white
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    return img


def _to_gray(img) -> np.ndarray:
    img = to_rgb(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    return img


def preprocess_sketch(img, size: int = 512, margin: float = 0.12):
    """Return (line_map_PIL_RGB, info). Line map = black strokes on white, square, cropped to content.

    Steps: grayscale -> polarity fix (dark-on-light) -> denoise -> adaptive threshold
    -> remove specks -> crop to strokes -> pad to square -> resize -> slight stroke thickening.
    """
    g = _to_gray(img)
    if g.size == 0:
        raise SketchError("Empty image.")
    if int(g.max()) - int(g.min()) < 25:
        raise SketchError("The sketch looks empty. Draw a clearer outline with darker strokes.")
    # Polarity: we want dark strokes on light paper. If most pixels are dark, invert.
    if np.median(g) < 128:
        g = 255 - g
    # Contrast normalise + denoise (handles phone photos of paper sketches)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    h, w = g.shape
    block = max(15, (min(h, w) // 20) | 1)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 10)
    # Also keep globally dark strokes (adaptive threshold hollows thick strokes)
    bw |= (g < 100).astype(np.uint8) * 255
    # Remove small specks (paper texture / jpeg noise)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    min_area = max(8, int(h * w * 2e-5))
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    bw = (keep[lab] * 255).astype(np.uint8)
    ink = int((bw > 0).sum())
    if ink < 30:
        raise SketchError("The sketch looks empty. Draw a clearer outline with darker strokes.")
    ys, xs = np.where(bw > 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = bw[y0:y1, x0:x1]
    side = int(max(crop.shape) * (1 + 2 * margin))
    canvas = np.zeros((side, side), np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    canvas = (canvas > 40).astype(np.uint8) * 255
    # Normalise stroke width so thin pen lines and fat brush lines behave the same
    k = max(1, size // 256)
    canvas = cv2.dilate(canvas, np.ones((k + 1, k + 1), np.uint8))
    lines = 255 - canvas  # black on white
    info = dict(orig_size=(w, h), bbox=(int(x0), int(y0), int(x1), int(y1)), ink_ratio=ink / (h * w))
    return Image.fromarray(lines).convert("RGB"), info


def to_control(line_map: Image.Image) -> Image.Image:
    """ControlNet scribble expects white strokes on black."""
    a = 255 - np.array(line_map.convert("L"))
    return Image.fromarray(a).convert("RGB")


def silhouette_mask(line_map: Image.Image) -> np.ndarray:
    """Filled outer silhouette from a line map (used by the geometric fallback)."""
    a = 255 - np.array(line_map.convert("L"))
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    m = np.zeros_like(a)
    if cnts:
        cv2.drawContours(m, cnts, -1, 255, thickness=cv2.FILLED)
    return m > 0


def is_meaningful_side_sketch(line_map: Image.Image, sil: np.ndarray = None) -> bool:
    """Reject blank Gradio sketchpads and junk fills (they become a solid black square and smash depth).

    An untouched Sketchpad often survives empty-checks, preprocesses into a filled square, then
    `apply_side_sketch` carves the 3D volume into a flat box — which looks like 'coordinates wrong'.
    """
    if sil is None:
        sil = silhouette_mask(line_map)
    fill = float(sil.mean()) if sil is not None and sil.size else 0.0
    if fill < 0.008 or fill > 0.45:
        return False
    ink = float((np.array(line_map.convert("L")) < 128).mean())
    # Real side sketches are stroke-like; a solid black blob is almost all ink.
    if ink > 0.35:
        return False
    # Reject near-square filled blobs (empty-canvas artifact after threshold/crop).
    ys, xs = np.where(sil)
    if len(xs) < 30:
        return False
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    aspect = bw / max(bh, 1)
    bbox_fill = fill * sil.size / max(bw * bh, 1)
    if 0.85 <= aspect <= 1.15 and bbox_fill > 0.85 and fill > 0.2:
        return False
    return True

