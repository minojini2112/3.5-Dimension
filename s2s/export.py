"""Stage 5: validation + export (.glb with embedded texture, .obj/.mtl/.png zipped)."""
import json
import os
import zipfile

import numpy as np
import trimesh


def upright(mesh: trimesh.Trimesh, max_deg: float = 30.0, step: float = 2.0) -> trimesh.Trimesh:
    """Undo the camera pitch/roll that single-view reconstruction bakes in (the object 'leans' when the
    source image was shot from above). Picks the X/Z rotation that minimises the axis-aligned box volume."""
    try:
        pts = mesh.convex_hull.vertices
    except Exception:
        pts = mesh.vertices[:: max(1, len(mesh.vertices) // 5000)]
    pts = pts - pts.mean(0)
    angs = np.radians(np.arange(-max_deg, max_deg + 1e-6, step))

    def vol(R):
        q = pts @ R.T
        return np.prod(q.max(0) - q.min(0))

    def rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    base = vol(np.eye(3))
    best = min(((vol(rz(b) @ rx(a)), a, b) for a in angs for b in angs), key=lambda t: t[0])
    if best[0] < 0.97 * base:  # only rotate when it clearly helps
        M = np.eye(4)
        M[:3, :3] = rz(best[2]) @ rx(best[1])
        mesh.apply_transform(M)
    return mesh


def normalise(mesh: trimesh.Trimesh, size: float = 1.0) -> trimesh.Trimesh:
    mesh.apply_translation(-mesh.bounds.mean(0))
    ext = mesh.extents.max()
    if ext > 0:
        mesh.apply_scale(size / ext)
    # sit on the ground plane (y = 0) for nicer viewing
    mesh.apply_translation([0, -mesh.bounds[0][1], 0])
    return mesh


def validate_file(path: str) -> dict:
    """Re-open the exported file from disk and check that it is a usable asset."""
    loaded = trimesh.load(path, force="mesh")
    rep = dict(file=os.path.basename(path), bytes=os.path.getsize(path),
               vertices=int(len(loaded.vertices)), faces=int(len(loaded.faces)))
    rep["non_empty"] = rep["vertices"] > 0 and rep["faces"] > 0
    rep["extents"] = [round(float(e), 4) for e in loaded.extents] if rep["non_empty"] else None
    rep["has_volume"] = bool(rep["non_empty"] and min(loaded.extents) > 0.02 * max(loaded.extents))
    rep["watertight"] = bool(loaded.is_watertight)
    vis = loaded.visual
    rep["texture"] = (vis.kind == "texture" and getattr(vis.material, "baseColorTexture", None) is not None) or \
                     (vis.kind == "texture" and getattr(vis.material, "image", None) is not None)
    rep["vertex_colors"] = vis.kind == "vertex"
    rep["ok"] = bool(rep["non_empty"] and rep["has_volume"])
    return rep


def export_all(mesh: trimesh.Trimesh, out_dir: str, stem: str = "model"):
    os.makedirs(out_dir, exist_ok=True)
    glb = os.path.join(out_dir, f"{stem}.glb")
    mesh.export(glb)
    obj_dir = os.path.join(out_dir, f"{stem}_obj")
    os.makedirs(obj_dir, exist_ok=True)
    obj = os.path.join(obj_dir, f"{stem}.obj")
    mesh.export(obj, include_color=True)  # writes .mtl + texture png next to it when textured
    zpath = os.path.join(out_dir, f"{stem}_obj.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(obj_dir):
            z.write(os.path.join(obj_dir, fn), fn)
    report = {"glb": validate_file(glb), "obj": validate_file(obj)}
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump(report, f, indent=2)
    return glb, zpath, report
