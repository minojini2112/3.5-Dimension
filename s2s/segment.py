"""Stage 3: foreground isolation (rembg / U^2-Net) and recentring for the 3D model."""
import numpy as np
from PIL import Image

import os

# isnet-general-use: 0.6 s on CPU and keeps holes (chair legs, handles) open, unlike u2net.
# birefnet-general: best edges, ~17 s on CPU / ~1 s on GPU.
BG_MODEL = os.environ.get("S2S_BG_MODEL", "isnet-general-use")
_SESS = {}


def unload():
    """Drop rembg ONNX sessions (optional; frees RAM between long UI sessions)."""
    _SESS.clear()


def remove_bg(img: Image.Image, model: str = None) -> Image.Image:
    import rembg
    model = model or BG_MODEL
    if model not in _SESS:
        _SESS[model] = rembg.new_session(model)
    out = rembg.remove(img.convert("RGB"), session=_SESS[model])
    a = np.array(out)
    # clean alpha: drop faint halo + keep the largest blob
    import cv2
    alpha = a[..., 3]
    m = (alpha > 127).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        thr = stats[big, cv2.CC_STAT_AREA] * 0.05
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= thr
        alpha = np.where(keep[lab], alpha, 0).astype(np.uint8)
    a[..., 3] = alpha
    return Image.fromarray(a, "RGBA")


def fg_ratio(rgba: Image.Image) -> float:
    return float((np.array(rgba)[..., 3] > 127).mean())


def recentre(rgba: Image.Image, ratio: float = 0.85, size: int = 512) -> Image.Image:
    """Crop to alpha bbox, pad to square so the object fills `ratio` of the frame, composite on grey."""
    a = np.array(rgba)
    ys, xs = np.where(a[..., 3] > 127)
    if len(xs) == 0:
        raise ValueError("Background removal left nothing.")
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    side = int(max(h, w) / ratio)
    canvas = np.zeros((side, side, 4), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    img = Image.fromarray(canvas, "RGBA").resize((size, size), Image.LANCZOS)
    return img


def composite_grey(rgba: Image.Image) -> Image.Image:
    a = np.array(rgba).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + 0.5 * (1 - a[..., 3:4])
    return Image.fromarray((rgb * 255).astype(np.uint8))
