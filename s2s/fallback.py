"""Geometric CPU fallback: inflate the sketch/foreground silhouette into a closed, coloured volume.

Used when no neural backend is available (missing weights, OOM, crash) so the app always returns a
valid 3D asset. Height of each pixel = rounded profile of its distance to the silhouette edge."""
import cv2
import mcubes
import numpy as np
import trimesh
from PIL import Image


def inflate(mask: np.ndarray, color_img: Image.Image = None, res: int = 160, depth: float = 0.35):
    m = cv2.resize(mask.astype(np.uint8) * 255, (res, res), interpolation=cv2.INTER_AREA) > 127
    if m.sum() < 20:
        raise ValueError("Silhouette too small to inflate.")
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    R = d.max()
    h = np.sqrt(np.clip(2 * d * R - d * d, 0, None)) / R  # circular cross-section in [0,1]
    h = cv2.GaussianBlur(h, (5, 5), 0) * depth * (R / res) * 4
    zres = res // 2
    z = np.linspace(-1, 1, zres) * h.max() * 1.1
    vol = (np.abs(z)[None, None, :] < h[:, :, None]).astype(np.float32)
    vol = np.pad(vol, 1)
    vol = mcubes.smooth(vol)
    v, f = mcubes.marching_cubes(vol, 0.0)
    v -= 1
    # (row, col, zi) -> x right, y up, z towards viewer
    x = v[:, 1] / res * 2 - 1
    y = 1 - v[:, 0] / res * 2
    zz = (v[:, 2] / (zres - 1) * 2 - 1) * h.max() * 1.1
    verts = np.stack([x, y, zz], -1)
    mesh = trimesh.Trimesh(verts, f[:, ::-1], process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=10)
    mesh.fix_normals()
    if color_img is not None:
        c = np.array(color_img.convert("RGB").resize((res, res)))
        r = np.clip(((1 - mesh.vertices[:, 1]) / 2 * res).astype(int), 0, res - 1)
        cc = np.clip(((mesh.vertices[:, 0] + 1) / 2 * res).astype(int), 0, res - 1)
        col = c[r, cc]
    else:
        col = np.tile(np.array([[200, 200, 210]], np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=col)
    return mesh
