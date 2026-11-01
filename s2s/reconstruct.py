"""Stage 4: single image -> 3D mesh with TripoSR (MIT), plus a pure-CPU UV texture baker."""
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from . import config, memory

if config.TRIPOSR_SRC not in sys.path:
    sys.path.insert(0, config.TRIPOSR_SRC)

_MODEL = None
_DEVICE = None
_COLOR = None


def load(device=None):
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return _MODEL, _DEVICE
    from tsr.system import TSR
    device = device or config.pick_device()
    memory.free_torch()  # SD pipe / prior run may still hold VRAM/RAM
    m = TSR.from_pretrained(config.TRIPOSR, config_name="config.yaml", weight_name="model.ckpt")
    m.renderer.set_chunk_size(8192 if device == "cuda" else 32768)
    m.to(device).eval()
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    _MODEL, _DEVICE = m, device
    return m, device


def unload():
    """Free TripoSR + colour-field tensors after a mesh is exported."""
    global _MODEL, _DEVICE, _COLOR
    m = _MODEL
    _MODEL, _DEVICE, _COLOR = None, None, None
    if m is not None:
        try:
            m.to("cpu")
        except Exception:
            pass
        del m
    memory.free_torch()



def _scene_code(img: Image.Image):
    m, dev = load()
    with torch.no_grad():
        try:
            return m([img], device=dev)
        except torch.cuda.OutOfMemoryError:  # 4 GB cards: retry on CPU
            torch.cuda.empty_cache()
            m.to("cpu")
            globals()["_DEVICE"] = "cpu"
            return m([img], device="cpu")


def _clean(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        biggest = max(p.area for p in parts)
        mesh = trimesh.util.concatenate([p for p in parts if p.area >= 0.03 * biggest])
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > target_faces:
        try:
            import fast_simplification
            v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                                 target_reduction=1 - target_faces / len(mesh.faces))
            mesh = trimesh.Trimesh(v, f, process=True)
        except Exception as e:  # keep full-res mesh if simplifier is unavailable
            print("[reconstruct] simplification skipped:", e)
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=15)
    mesh.fix_normals()
    return mesh


class ColorField:
    """Colour for any surface point, in three layers:
      1. TripoSR's colour field (averaged with the flipped reconstruction when the flip ensemble is on)
      2. points TripoSR never saw (added by a shape prior, or the mirrored back) borrow the colour of the
         visible front at the same screen position
      3. points visible from the input camera take the colour straight from the image (sharp detail),
         by projecting the image onto the front surface."""

    def __init__(self, codes, codes_f, F0v, Fv, rad, z_mirror=None, rgba=None, project=True):
        from scipy.ndimage import distance_transform_edt
        from .priors import R_TV, image_to_grid
        self.codes, self.codes_f, self.rad, self.R = codes, codes_f, rad, R_TV
        self.n = n = F0v.shape[0]
        self.F0v = F0v
        self.z_mirror = z_mirror
        occ0 = F0v > 0
        self.has0 = occ0.any(2)
        self.zf0 = np.where(self.has0, n - 1 - np.argmax(occ0[:, :, ::-1], axis=2), 0)
        _, (self.nx, self.ny) = distance_transform_edt(~self.has0, return_indices=True)
        occ = Fv > 0
        hasf = occ.any(2)
        self.zff = np.where(hasf, n - 1 - np.argmax(occ[:, :, ::-1], axis=2), -10)
        self.proj = None
        if project and rgba is not None:
            a = np.array(rgba).astype(np.float32) / 255.0
            mask = a[..., 3] > 0.5
            chans = [image_to_grid(a[..., c], mask, self.has0) for c in range(4)]
            self.proj = np.stack(chans, -1)                     # (x, y, rgba) in grid space, nan outside

    def _idx(self, v):
        n, rad = self.n, self.rad
        return np.clip(np.round((v + rad) / (2 * rad) * (n - 1)).astype(int), 0, n - 1)

    def _tsr(self, codes, pts):
        return _query_color_raw(codes, pts)

    def __call__(self, pts_tsr: np.ndarray, normals: np.ndarray = None) -> np.ndarray:
        n, rad, R = self.n, self.rad, self.R
        v = pts_tsr @ R.T
        if self.z_mirror is not None:
            zm = self.z_mirror / (n - 1) * 2 * rad - rad
            v = v.copy()
            back = v[:, 2] < zm
            v[back, 2] = 2 * zm - v[back, 2]
        idx = self._idx(v)
        weak = self.F0v[idx[:, 0], idx[:, 1], idx[:, 2]] <= 0
        q = v.copy()
        if weak.any():
            wx, wy = self.nx[idx[weak, 0], idx[weak, 1]], self.ny[idx[weak, 0], idx[weak, 1]]
            q[weak, 0] = wx / (n - 1) * 2 * rad - rad
            q[weak, 1] = wy / (n - 1) * 2 * rad - rad
            q[weak, 2] = (self.zf0[wx, wy] - 1) / (n - 1) * 2 * rad - rad
        col = self._tsr(self.codes, q @ R)
        if self.codes_f is not None:
            qf = q.copy()
            qf[:, 0] = -qf[:, 0]
            col = 0.5 * (col + self._tsr(self.codes_f, qf @ R))
        if self.proj is not None:
            # planar projection of the image along the view axis, weighted by how squarely the surface faces
            # the camera (front) or faces directly away (back, which mirrors the front's colours)
            vi = self._idx(pts_tsr @ R.T)
            pc = self.proj[vi[:, 0], vi[:, 1]]
            ok = np.isfinite(pc[:, 3]) & (pc[:, 3] > 0.5)
            if normals is not None:
                nz = (normals @ R.T)[:, 2]
                nz = nz / (np.linalg.norm(normals, axis=1) + 1e-9)
                front_w = np.clip((nz - 0.15) / 0.45, 0, 1)
                back_w = np.clip((-nz - 0.15) / 0.45, 0, 1) * 0.85
                vis = vi[:, 2] >= self.zff[vi[:, 0], vi[:, 1]] - 3
                w = np.where(vis, front_w, back_w) * ok
            else:
                vis = vi[:, 2] >= self.zff[vi[:, 0], vi[:, 1]] - 2
                w = (vis & ok).astype(np.float32)
            pcc = np.nan_to_num(pc[:, :3])
            col = col * (1 - w[:, None]) + pcc * w[:, None]
        return col


_COLOR = None  # ColorField for the current run


def _query_color_raw(scene_code, pts: np.ndarray, chunk=65536) -> np.ndarray:
    m, dev = _MODEL, _DEVICE
    out = []
    with torch.no_grad():
        for i in range(0, len(pts), chunk):
            p = torch.from_numpy(pts[i:i + chunk].astype(np.float32)).to(dev)
            out.append(m.renderer.query_triplane(m.decoder, p, scene_code)["color"].float().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 3), np.float32)


def _query_color(scene_code, pts: np.ndarray, normals: np.ndarray = None) -> np.ndarray:
    if _COLOR is not None:
        return _COLOR(pts, normals)
    return _query_color_raw(scene_code, pts)


def bake_uv_texture(mesh: trimesh.Trimesh, scene_code, res: int = 1024):
    """xatlas unwrap -> rasterise 3D positions into UV space (CPU) -> query TripoSR colour field."""
    import xatlas
    vmap, idx, uvs = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    verts = mesh.vertices[vmap]
    tri_uv = uvs[idx] * res                          # (F,3,2) in pixels, v up
    tri_px = np.stack([tri_uv[..., 0], res - tri_uv[..., 1]], -1)  # image rows go down
    tri_id = np.full((res, res), -1, np.int32)
    pts_i = np.round(tri_px * 4).astype(np.int32)     # 2 bits sub-pixel precision
    for f in range(len(idx)):
        cv2.fillConvexPoly(tri_id, pts_i[f], int(f), lineType=cv2.LINE_8, shift=2)
    # grow triangle ids into gutters so bilinear lookups at seams stay correct
    pad = max(2, res // 256)
    filled = tri_id >= 0
    for _ in range(pad):
        grown = cv2.dilate((tri_id + 1).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32) - 1
        tri_id = np.where(filled, tri_id, grown)
        filled = tri_id >= 0
    ys, xs = np.nonzero(filled)
    fid = tri_id[ys, xs]
    p = np.stack([xs + 0.5, ys + 0.5], -1)
    a, b, c = tri_px[fid, 0], tri_px[fid, 1], tri_px[fid, 2]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1).sum(-1); d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1); d21 = (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    w = np.clip(np.stack([1 - w1 - w2, w1, w2], -1), 0, 1)
    w /= w.sum(-1, keepdims=True) + 1e-9
    tri_xyz = verts[idx[fid]]                          # (N,3,3)
    xyz = (w[..., None] * tri_xyz).sum(1)
    fn = np.cross(verts[idx[:, 1]] - verts[idx[:, 0]], verts[idx[:, 2]] - verts[idx[:, 0]])
    vn = mesh.vertex_normals[vmap]                     # smooth normals interpolated per texel
    nrm = (w[..., None] * vn[idx[fid]]).sum(1)
    rgb = _query_color(scene_code, xyz, nrm)
    tex = np.zeros((res, res, 3), np.float32)
    tex[ys, xs] = rgb
    tex8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    # fill whatever is still empty so mip-mapping never pulls in black
    hole = (~filled).astype(np.uint8) * 255
    tex8 = cv2.inpaint(tex8, hole, 3, cv2.INPAINT_TELEA) if res <= 1024 else tex8
    tm = trimesh.Trimesh(vertices=verts, faces=idx, process=False)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex8),
                                                   metallicFactor=0.0, roughnessFactor=0.8)
    tm.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    return tm


def to_cpu():
    global _DEVICE
    if _MODEL is not None:
        _MODEL.to("cpu")
        _DEVICE = "cpu"
        memory.free_torch()



def image_to_mesh(rgba: Image.Image, mc_resolution: int = 256, target_faces: int = 60000,
                  texture: bool = True, texture_res: int = 1024, prior: str = "auto", prompt: str = "",
                  side_mask=None, ensemble: bool = True):
    """rgba: recentred foreground. Returns (trimesh, timings). Retries on CPU if the GPU runs out of memory."""
    try:
        return _image_to_mesh(rgba, mc_resolution, target_faces, texture, texture_res, prior, prompt, side_mask, ensemble)
    except torch.cuda.OutOfMemoryError:
        print("[reconstruct] CUDA OOM, retrying TripoSR on CPU")
        to_cpu()
        return _image_to_mesh(rgba, mc_resolution, target_faces, texture, texture_res, prior, prompt, side_mask, ensemble)


def _density_field(scene_code, res: int, threshold: float = 25.0) -> np.ndarray:
    """TripoSR density on a res^3 grid over [-radius, radius]^3, as (density - threshold), clipped."""
    m, dev = _MODEL, _DEVICE
    rad = m.renderer.cfg.radius
    lin = torch.linspace(-rad, rad, res)
    g = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    out = []
    with torch.no_grad():
        for i in range(0, len(g), 2 ** 20):
            d = m.renderer.query_triplane(m.decoder, g[i:i + 2 ** 20].to(dev), scene_code)["density_act"]
            out.append(d.float().cpu().numpy().reshape(-1))
    F = np.concatenate(out).reshape(res, res, res) - threshold
    return np.clip(F, -threshold, 4 * threshold).astype(np.float32), rad


def _field_to_mesh(F_view: np.ndarray, rad: float) -> trimesh.Trimesh:
    import mcubes
    v, f = mcubes.marching_cubes(np.pad(F_view, 1, constant_values=-1e3), 0.0)
    v = (v - 1) / (F_view.shape[0] - 1) * 2 * rad - rad          # view-frame coordinates
    from .priors import view_points_to_tsr
    return trimesh.Trimesh(view_points_to_tsr(v), f, process=True)


def _cues(rgba: Image.Image, img_grey: Image.Image, P0: np.ndarray) -> dict:
    """CLIP symmetry belief + Depth Anything depth resampled into the volume grid (None if weights missing)."""
    from . import understand
    from .priors import image_to_grid
    if not understand.available():
        return None
    a = np.array(rgba)
    white = Image.fromarray((a[..., :3] * (a[..., 3:4] / 255.0) + 255 * (1 - a[..., 3:4] / 255.0)).astype(np.uint8))
    sym = understand.symmetry_probs(white)
    depth = understand.relative_depth(img_grey)
    return dict(probs=sym["probs"], label=sym["label"], label_p=sym["label_p"],
                depth_grid=image_to_grid(depth, a[..., 3] > 127, P0))


def _image_to_mesh(rgba, mc_resolution, target_faces, texture, texture_res, prior="auto", prompt="",
                   side_mask=None, ensemble=True, project_texture=True):
    t = {}
    load()
    a = np.array(rgba).astype(np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + 0.5 * (1 - a[..., 3:4])  # TripoSR expects grey background
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    t0 = time.time()
    codes = _scene_code(img)
    t["triplane"] = time.time() - t0
    t0 = time.time()
    from . import priors
    F, rad = _density_field(codes[0], mc_resolution)
    t["density_grid"] = time.time() - t0
    t1 = time.time()
    global _COLOR
    F0v = priors.tsr_to_view(F)
    codes_f = None
    if ensemble:
        # test-time flip augmentation: reconstruct the mirrored image, mirror the result back, average.
        # Cancels much of TripoSR's one-sided bias (e.g. tops bending backwards).
        t1 = time.time()
        codes_f = _scene_code(img.transpose(Image.FLIP_LEFT_RIGHT))
        Ff, _ = _density_field(codes_f[0], mc_resolution)
        F0v = 0.5 * (F0v + priors.tsr_to_view(Ff)[::-1])
        t["flip_ensemble"] = time.time() - t1
    cues = None
    if prior == "auto":
        t1 = time.time()
        try:
            cues = _cues(rgba, img, (F0v > 0).any(2))
        except Exception as e:
            print("[reconstruct] cue models unavailable, using keyword rules:", e)
        t["understand"] = time.time() - t1
    Fv, used, stats = priors.apply(F0v, prior if prior else "none", prompt, side_mask, cues)
    if cues:
        stats["clip_label"] = f"{cues['label']} ({cues['label_p']:.2f})"
    t["shape_prior"] = time.time() - t1
    _COLOR = ColorField(codes[0], codes_f[0] if codes_f is not None else None, F0v, Fv, rad,
                        stats.get("z_mirror"), rgba, project=project_texture)
    t["prior_mode"] = used
    print(f"[reconstruct] shape prior: {used}  stats={ {k: v for k, v in stats.items() if k != 'y_range'} }")
    t["prior_report"] = dict(mode=used, clip=stats.get("clip_label"), candidates=stats.get("candidates"))
    t2 = time.time()
    mesh = _field_to_mesh(Fv, rad)
    if len(mesh.faces) == 0:
        raise RuntimeError("TripoSR produced an empty surface.")
    mesh = _clean(mesh, target_faces)
    t["marching_cubes+cleanup"] = time.time() - t2
    t0 = time.time()
    if texture:
        try:
            mesh = bake_uv_texture(mesh, codes[0], texture_res)
        except Exception as e:  # never fail the run because of texturing
            print("[reconstruct] UV bake failed, using vertex colours:", e)
            texture = False
    if not texture:
        col = _query_color(codes[0], mesh.vertices, mesh.vertex_normals)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=(np.clip(col, 0, 1) * 255).astype(np.uint8))
    t["texture"] = time.time() - t0
    # TripoSR frame -> glTF Y-up, facing +Z (same as the official demo)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    # face +Z (glTF "front"), so viewers open on the side that matches the sketch
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    return mesh, t
