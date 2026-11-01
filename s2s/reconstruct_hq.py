"""HQ Stage 4: single image -> 3D mesh with Hunyuan3D-2mini (shape only) + front-projected UV texture.

TripoSR stays as the fast path. This module is heavier (~0.6B DiT, ~6 GB class VRAM); on 4 GB cards we
enable accelerate CPU offload and a milder octree so generation still fits.
"""
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image

from . import config, memory

if config.HUNYUAN3D_SRC not in sys.path:
    sys.path.insert(0, config.HUNYUAN3D_SRC)

_PIPE = None
_DEVICE = None
_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"  # distilled; fits 4 GB with CPU offload
TARGET_FACES = 120000


def _rgba_to_hunyuan_image(rgba: Image.Image) -> Image.Image:
    """Hunyuan expects RGBA with transparent background (or RGB). Keep alpha from rembg."""
    if rgba.mode != "RGBA":
        rgba = rgba.convert("RGBA")
    return rgba


def _load_without_double_allocation(pipeline_cls, model_path, sub, use_st):
    """from_pretrained builds every model with random weights, then reads a 3.8 GB checkpoint on top: about 16 GB
    of commit, more than a 16 GB laptop has spare. Build them empty and adopt the memory-mapped tensors instead."""
    from accelerate import init_empty_weights
    from hy3dgen.shapegen import pipelines as upstream

    real_load, real_instantiate, real_state = torch.load, upstream.instantiate_from_config, torch.nn.Module.load_state_dict

    def empty_instance(cfg, **kwargs):
        with init_empty_weights():
            return real_instantiate(cfg, **kwargs)

    torch.load = lambda f, **kw: real_load(f, mmap=True, **kw)
    upstream.instantiate_from_config = empty_instance

    def adopt(self, sd, strict=True, **kw):
        result = real_state(self, sd, strict=strict, assign=True)
        for mod in self.modules():  # upstream loads the VAE with strict=False: fill what the file does not have
            for name, param in list(mod._parameters.items()):
                if param is not None and param.is_meta:
                    mod._parameters[name] = torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype),
                                                               requires_grad=False)
        return result

    torch.nn.Module.load_state_dict = adopt
    try:
        pipe = pipeline_cls.from_pretrained(model_path, subfolder=sub, use_safetensors=use_st, device="cpu")
    finally:
        torch.load, upstream.instantiate_from_config, torch.nn.Module.load_state_dict = \
            real_load, real_instantiate, real_state
    return pipe


def load(device=None, turbo: bool = None):
    global _PIPE, _DEVICE
    if _PIPE is not None:
        return _PIPE, _DEVICE
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    device = device or config.pick_device()
    vram = config.vram_gb()
    sub = _SUBFOLDER
    local_ckpt = os.path.join(config.HUNYUAN3D_MINI, sub, "model.fp16.ckpt")
    model_path = config.HUNYUAN3D_MINI if os.path.exists(local_ckpt) else "tencent/Hunyuan3D-2mini"
    # Prefer .ckpt (already downloaded); safetensors twin is optional and large.
    use_st = os.path.exists(os.path.join(config.HUNYUAN3D_MINI, sub, "model.fp16.safetensors"))

    memory.free_torch()
    print(f"[reconstruct_hq] loading Hunyuan3D-2mini ({sub}, safetensors={use_st}) from {model_path} ...", flush=True)
    # Always construct on CPU first, then optionally offload-hook to CUDA (4 GB friendly).
    pipe = _load_without_double_allocation(Hunyuan3DDiTFlowMatchingPipeline, model_path, sub, use_st)
    if device == "cuda":
        # Upstream offload expects a diffusers-style `.components` dict (not always present).
        if not hasattr(pipe, "components"):
            pipe.components = {
                "conditioner": pipe.conditioner,
                "model": pipe.model,
                "vae": pipe.vae,
            }
        try:
            if vram < 6.5:
                pipe.enable_model_cpu_offload(device="cuda")
                pipe.device = torch.device("cuda:0")  # __call__ reads self.device, which offload leaves on cpu
                print("[reconstruct_hq] enable_model_cpu_offload (low VRAM)", flush=True)
            else:
                pipe.to("cuda")
        except Exception as e:
            print(f"[reconstruct_hq] CUDA path failed ({e}); staying on CPU", flush=True)
            try:
                pipe.to("cpu")
            except Exception:
                pass
            device = "cpu"
    else:
        pipe.to("cpu")
    _PIPE, _DEVICE = pipe, device
    return pipe, device


def unload():
    global _PIPE, _DEVICE
    pipe = _PIPE
    _PIPE, _DEVICE = None, None
    if pipe is not None:
        try:
            pipe.to("cpu")
        except Exception:
            pass
        del pipe
    memory.free_torch()


def _drop_floaters(mesh: trimesh.Trimesh, keep_ratio: float = 0.03) -> trimesh.Trimesh:
    """Remove disconnected scraps smaller than keep_ratio of the largest part (same rule as TripoSR's _clean).
    Works on face labels: mesh.split() runs out of memory on Hunyuan's raw meshes."""
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    areas = np.bincount(labels, weights=mesh.area_faces)
    keep = areas >= keep_ratio * areas.max()
    if keep.all():
        return mesh
    print(f"[reconstruct_hq] dropped {int((~keep).sum())} of {len(keep)} disconnected parts", flush=True)
    mesh.update_faces(keep[labels])
    mesh.remove_unreferenced_vertices()
    return mesh


def _tripo_color_field(mesh: trimesh.Trimesh, rgba: Image.Image):
    """Colour for surfaces the input image cannot see: TripoSR's colour field for the same image, aligned to
    the mesh by bounding box (centre + per-axis scale). Returns (points -> rgb or None, cleanup)."""
    from . import priors, reconstruct
    try:
        reconstruct.load()
        a = np.array(rgba.convert("RGBA")).astype(np.float32) / 255.0
        rgb = a[..., :3] * a[..., 3:4] + 0.5 * (1 - a[..., 3:4])  # TripoSR expects grey background
        code = reconstruct._scene_code(Image.fromarray((rgb * 255).astype(np.uint8)))[0]
        res = 64
        field, rad = reconstruct._density_field(code, res)
        idx = np.argwhere(priors.tsr_to_view(field) > 0)
        if len(idx) == 0:
            raise RuntimeError("TripoSR field is empty")
        tmin, tmax = (idx.min(0) / (res - 1) * 2 - 1) * rad, (idx.max(0) / (res - 1) * 2 - 1) * rad
        mmin, mmax = mesh.bounds

        def query(pts):
            view = tmin + (pts - mmin) / np.maximum(mmax - mmin, 1e-9) * (tmax - tmin)
            return np.clip(reconstruct._query_color_raw(code, priors.view_points_to_tsr(view)), 0, 1)

        return query, reconstruct.unload
    except Exception as e:
        print(f"[reconstruct_hq] hidden-side colour unavailable ({e}); using front projection only", flush=True)
        return None, reconstruct.unload


def _bake_image_texture(mesh: trimesh.Trimesh, rgba: Image.Image, res: int = 1024, hidden_rgb=None) -> trimesh.Trimesh:
    """xatlas unwrap + project the input photo onto +Z facing surfaces (same convention as TripoSR export).
    hidden_rgb(points) colours whatever faces away from the camera."""
    import xatlas
    img = np.array(rgba.convert("RGBA")).astype(np.float32) / 255.0
    solid = img[..., 3] > 0.5
    ys0, xs0 = np.nonzero(solid)
    if len(ys0):  # the mesh spans the object, not the frame: crop the margins or they sample as background
        img = img[ys0.min():ys0.max() + 1, xs0.min():xs0.max() + 1]
        solid = solid[ys0.min():ys0.max() + 1, xs0.min():xs0.max() + 1]
    # Points that land just outside the object borrow the nearest object colour instead of a flat grey.
    from scipy.ndimage import distance_transform_edt
    near = distance_transform_edt(~solid, return_distances=False, return_indices=True)
    img = img[near[0], near[1]]
    h, w = img.shape[:2]
    vmap, idx, uvs = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    verts = mesh.vertices[vmap]
    # mesh should already be Y-up facing +Z; project XY into image with Z as depth cue
    vmin, vmax = verts.min(0), verts.max(0)
    center = 0.5 * (vmin + vmax)
    extent = np.maximum(vmax - vmin, 1e-6)
    tri_uv = uvs[idx] * res
    tri_px = np.stack([tri_uv[..., 0], res - tri_uv[..., 1]], -1)
    tri_id = np.full((res, res), -1, np.int32)
    pts_i = np.round(tri_px * 4).astype(np.int32)
    for f in range(len(idx)):
        cv2.fillConvexPoly(tri_id, pts_i[f], int(f), lineType=cv2.LINE_8, shift=2)
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
    wt = np.clip(np.stack([1 - w1 - w2, w1, w2], -1), 0, 1)
    wt /= wt.sum(-1, keepdims=True) + 1e-9
    tri_xyz = verts[idx[fid]]
    xyz = (wt[..., None] * tri_xyz).sum(1)
    vn = mesh.vertex_normals[vmap]
    nrm = (wt[..., None] * vn[idx[fid]]).sum(1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
    # Front-facing: sample image; backsides get a darkened copy of the same UV sample
    u = (xyz[:, 0] - center[0]) / extent[0] + 0.5
    v = 1.0 - ((xyz[:, 1] - center[1]) / extent[1] + 0.5)
    ui = np.clip((u * (w - 1)).astype(np.int32), 0, w - 1)
    vi = np.clip((v * (h - 1)).astype(np.int32), 0, h - 1)
    samp = img[vi, ui]
    rgb = samp[..., :3]
    if hidden_rgb is None:
        rgb = np.where((nrm[:, 2] > 0.05)[:, None], rgb, rgb * 0.65)
    else:
        w = np.clip(nrm[:, 2], 0, 1)[:, None] ** 2
        rgb = w * rgb + (1 - w) * hidden_rgb(xyz)
    tex = np.zeros((res, res, 3), np.float32)
    tex[ys, xs] = rgb
    tex8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    hole = (~filled).astype(np.uint8) * 255
    if res <= 1024:
        tex8 = cv2.inpaint(tex8, hole, 3, cv2.INPAINT_TELEA)
    tm = trimesh.Trimesh(vertices=verts, faces=idx, process=False)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex8),
                                                   metallicFactor=0.0, roughnessFactor=0.8)
    tm.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    return tm


def image_to_mesh(rgba: Image.Image, seed: int = 42, steps: int = None, texture_res: int = 1024,
                  octree_resolution: int = None):
    """Returns (trimesh, timings)."""
    t = {}
    pipe, device = load()
    image = _rgba_to_hunyuan_image(rgba)
    vram = config.vram_gb()
    if steps is None:
        steps = 12 if (vram and vram < 6.5) else 30
    if octree_resolution is None:
        octree_resolution = 384
    num_chunks = 8000 if (vram and vram < 6.5) else 20000

    t0 = time.time()
    memory.free_torch()
    with torch.no_grad():
        try:
            out = pipe(
                image=image,
                num_inference_steps=int(steps),
                octree_resolution=int(octree_resolution),
                num_chunks=int(num_chunks),
                generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                output_type="trimesh",
            )
        except torch.cuda.OutOfMemoryError:
            steps, octree_resolution = 5, 256
            print(f"[reconstruct_hq] CUDA OOM - retrying with {octree_resolution} / {steps} steps", flush=True)
            memory.free_torch()
            out = pipe(
                image=image,
                num_inference_steps=int(steps),
                octree_resolution=int(octree_resolution),
                num_chunks=int(num_chunks),
                generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                output_type="trimesh",
            )
    mesh = out[0] if isinstance(out, (list, tuple)) else out
    if isinstance(mesh, list):
        mesh = mesh[0]
    t["hunyuan_shape"] = time.time() - t0

    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=True)
        else:
            raise RuntimeError(f"Hunyuan3D returned unsupported mesh type: {type(mesh)}")
    if len(mesh.faces) == 0:
        raise RuntimeError("Hunyuan3D-2mini produced an empty surface.")

    # Hunyuan already outputs the glTF convention (Y-up, facing +Z): no rotation needed.
    t1 = time.time()
    mesh = _drop_floaters(mesh)
    if len(mesh.faces) > TARGET_FACES:  # the pure-Python texture bake scales with face count
        import fast_simplification
        v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
                                            target_count=TARGET_FACES)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    unload()  # free the card for TripoSR's colour field
    hidden, close_hidden = _tripo_color_field(mesh, rgba)
    try:
        mesh = _bake_image_texture(mesh, rgba, res=int(texture_res), hidden_rgb=hidden)
    except Exception as e:
        print("[reconstruct_hq] UV bake failed, using grey vertices:", e, flush=True)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.full((len(mesh.vertices), 4), 200, np.uint8))
    close_hidden()
    t["texture"] = time.time() - t1
    t["prior_mode"] = "hunyuan3d-2mini"
    t["prior_report"] = dict(mode="hunyuan3d-2mini", device=device, steps=steps,
                             octree_resolution=octree_resolution)
    return mesh, t


def image_to_mesh_isolated(rgba: Image.Image, out_dir: str, seed: int = 42, texture_res: int = 1024):
    """Run HQ in a child process: loading peaks near the commit limit of a 16 GB machine, and a native
    crash there would otherwise take the whole app down. Everything it used is released on exit."""
    png, glb, info = (os.path.join(out_dir, n) for n in ("hq_input.png", "hq_mesh.glb", "hq_timings.json"))
    rgba.save(png)
    p = subprocess.run([sys.executable, "-m", "s2s.reconstruct_hq", png, glb, info, str(seed), str(texture_res)],
                       cwd=config.ROOT, timeout=1500)
    if p.returncode != 0 or not os.path.exists(glb):
        raise RuntimeError(f"Hunyuan3D process exited with code {p.returncode}")
    with open(info, encoding="utf-8") as f:
        t = json.load(f)
    mesh = trimesh.load(glb, force="mesh")
    os.remove(glb)
    return mesh, t


if __name__ == "__main__":
    png, glb, info, seed, res = sys.argv[1:6]
    m, timings = image_to_mesh(Image.open(png), seed=int(seed), texture_res=int(res))
    m.export(glb)
    with open(info, "w", encoding="utf-8") as f:
        json.dump(timings, f)
