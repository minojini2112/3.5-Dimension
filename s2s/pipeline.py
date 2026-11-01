"""End-to-end orchestration with per-stage timing, candidate selection and graceful fallbacks."""
import json
import os
import platform
import time
import traceback
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image

from . import config, export, fallback, preprocess, segment


@dataclass
class Result:
    prompt_used: str = ""
    glb: str = None
    obj_zip: str = None
    line_map: Image.Image = None
    candidates: list = field(default_factory=list)
    chosen: Image.Image = None
    foreground: Image.Image = None
    backend: str = ""
    timings: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    fidelity_iou: float = None
    prior_mode: str = "none"
    prior_report: dict = None
    warnings: list = field(default_factory=list)
    out_dir: str = None

    def summary(self) -> dict:
        return dict(prompt_used=self.prompt_used, backend=self.backend, shape_prior=self.prior_mode, shape_prior_report=self.prior_report, timings_s={k: round(v, 2) for k, v in self.timings.items()},
                    fidelity_iou=None if self.fidelity_iou is None else round(self.fidelity_iou, 3),
                    validation=self.validation, warnings=self.warnings, glb=self.glb, obj_zip=self.obj_zip,
                    hardware=hardware())


def hardware() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.get_device_name(0)} ({config.vram_gb():.1f} GB) + {os.cpu_count()} CPU threads"
    return f"CPU only: {platform.processor() or platform.machine()} x{os.cpu_count()} threads"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni else 0.0


def _mask_of(rgba: Image.Image, size=256) -> np.ndarray:
    a = np.array(rgba.resize((size, size)))[..., 3] > 127
    return a


def run(sketch, prompt: str = "", seed: int = 42, steps: int = 6, control_scale: float = 0.9,
        guidance: float = 1.5, n_candidates: int = 1, mc_resolution: int = 256, texture_res: int = 1024,
        backend: str = "auto", upright: bool = True, shape_prior: str = "auto", side_sketch=None,
        auto_retry: int = 1, retry_below: float = 0.55, ensemble: bool = True, out_root: str = None,
        quality: str = "hq", progress=None) -> Result:
    """backend: 'auto' (neural, fall back to geometric), 'neural', 'geometric'.
    quality: 'hq' = Hunyuan3D-2mini shape, 'fast' = TripoSR (+ shape priors)."""
    def step(frac, msg):
        print(f"[s2s] {msg}")
        if progress is not None:
            try:
                progress(frac, desc=msg)
            except Exception:
                pass

    r = Result()
    T = r.timings
    out_root = out_root or config.OUTPUTS
    r.out_dir = os.path.join(out_root, time.strftime("%Y%m%d-%H%M%S") + f"-s{seed}")
    os.makedirs(r.out_dir, exist_ok=True)
    t_all = time.time()

    # 1. preprocess
    step(0.02, "Preprocessing sketch")
    t0 = time.time()
    r.line_map, info = preprocess.preprocess_sketch(sketch)
    r.line_map.save(os.path.join(r.out_dir, "1_linemap.png"))
    sil = preprocess.silhouette_mask(r.line_map)
    T["preprocess"] = time.time() - t0

    mesh = None
    try:
        if backend in ("auto", "neural"):
            try:
                # 2. sketch-guided generation
                if not (prompt or "").strip():
                    step(0.05, "Working out what the drawing shows")
                    from . import understand
                    try:
                        prompt = understand.name_isolated(r.line_map, preprocess.to_rgb(sketch), r.out_dir)
                    except Exception as e:
                        r.warnings.append(f"Could not name the drawing ({e}); generating without a prompt.")
                r.prompt_used = prompt or ""
                step(0.08, "Loading image generator" if True else "")
                t0 = time.time()
                from . import generate
                T["load_generator"] = time.time() - t0
                step(0.15, f"Generating {n_candidates} image(s) from sketch")
                t0 = time.time()
                sil_small = cv2.resize(sil.astype(np.uint8), (256, 256)) > 0
                best, best_iou, t_seg = None, -1, 0.0
                cur_seed, cur_scale = seed, control_scale
                for attempt in range(1 + max(0, int(auto_retry))):
                    cands = generate.generate_isolated(preprocess.to_control(r.line_map), prompt, r.out_dir,
                                                       seed=cur_seed, steps=steps,
                                                       guidance=guidance, control_scale=cur_scale, n=n_candidates)
                    # 3. background removal + keep the candidate that best matches the sketch silhouette
                    step(0.45, "Removing background / checking sketch fidelity")
                    t1 = time.time()
                    for c in cands:
                        i = len(r.candidates)
                        r.candidates.append(c)
                        c.save(os.path.join(r.out_dir, f"2_candidate{i}.png"))
                        fg = segment.remove_bg(c)
                        iou = _iou(_mask_of(fg), sil_small)
                        if iou > best_iou and segment.fg_ratio(fg) > 0.01:
                            best, best_iou, r.chosen = fg, iou, c
                    t_seg += time.time() - t1
                    if best_iou >= retry_below:
                        break
                    # self-check failed: the image drifted from the sketch -> new seed, stronger sketch control
                    r.warnings.append(f"Attempt {attempt + 1}: outline match {best_iou:.2f} < {retry_below}; "
                                      "retrying with a new seed and stronger sketch adherence.")
                    cur_seed, cur_scale = cur_seed + 1000, min(1.5, cur_scale + 0.2)
                    step(0.3, "Retrying generation (outline mismatch)")
                T["generate"] = time.time() - t0 - t_seg
                T["segment"] = t_seg
                if best is None:
                    raise RuntimeError("Background removal found no object in the generated image(s).")
                r.fidelity_iou = best_iou
                if best_iou < 0.5:
                    r.warnings.append(f"Generated shape only loosely matches the sketch outline (IoU {best_iou:.2f}); "
                                      "try another seed or raise sketch adherence.")
                r.foreground = segment.recentre(best)
                r.foreground.save(os.path.join(r.out_dir, "3_foreground.png"))

                # Free SD/ControlNet before 3D â€” both do not fit together on ~4 GB cards.
                step(0.52, "Freeing image generator memory")
                generate.unload()

                # 4. image -> 3D
                quality = (quality or "hq").lower()
                if quality == "hq":
                    step(0.55, "Reconstructing 3D (Hunyuan3D-2mini HQ)")
                    from . import reconstruct_hq
                    try:
                        mesh, t3 = reconstruct_hq.image_to_mesh_isolated(r.foreground, r.out_dir, seed=seed,
                                                                         texture_res=texture_res)
                        r.prior_mode = t3.pop("prior_mode", "hunyuan3d-2mini")
                        r.prior_report = t3.pop("prior_report", None)
                        T.update({f"3d_{k}": v for k, v in t3.items()})
                        r.backend = "neural: DreamShaper8+ControlNet-scribble+LCM -> rembg -> Hunyuan3D-2mini"
                    except Exception as e:
                        r.warnings.append(f"HQ 3D failed ({e}); used the fast TripoSR path instead.")
                        quality = "fast"
                if quality != "hq":
                    step(0.55, "Reconstructing 3D (TripoSR fast)")
                    t0 = time.time()
                    from . import reconstruct
                    reconstruct.load()
                    T["load_3d_model"] = time.time() - t0
                    side_mask = None
                    if side_sketch is not None:
                        try:
                            side_lines, _ = preprocess.preprocess_sketch(side_sketch)
                            cand = preprocess.silhouette_mask(side_lines)
                            # Blank Gradio pads preprocess into a solid square and wreck depth/"coordinates".
                            if preprocess.is_meaningful_side_sketch(side_lines, cand):
                                side_mask = cand
                                side_lines.save(os.path.join(r.out_dir, "1b_side_linemap.png"))
                            else:
                                r.warnings.append(
                                    "Ignored side-view input (looked empty or like a solid fill, not a sketch).")
                        except preprocess.SketchError:
                            side_mask = None  # empty side canvas: ignore
                    mesh, t3 = reconstruct.image_to_mesh(r.foreground, mc_resolution=mc_resolution,
                                                         texture_res=texture_res, prior=shape_prior, prompt=prompt,
                                                         side_mask=side_mask, ensemble=ensemble)
                    r.prior_mode = t3.pop("prior_mode", "none")
                    r.prior_report = t3.pop("prior_report", None)
                    T.update({f"3d_{k}": v for k, v in t3.items()})
                    r.backend = "neural: DreamShaper8+ControlNet-scribble+LCM -> rembg -> TripoSR"
                    reconstruct.unload()
            except Exception as e:
                if backend == "neural":
                    raise
                traceback.print_exc()
                r.warnings.append(f"Neural pipeline failed ({type(e).__name__}: {e}); used geometric fallback.")
                mesh = None

        if mesh is None:
            step(0.6, "Geometric fallback: inflating silhouette")
            t0 = time.time()
            mask, col = sil, None
            if r.foreground is not None:
                mask = np.array(r.foreground)[..., 3] > 127
                col = segment.composite_grey(r.foreground)
            mesh = fallback.inflate(mask, col)
            T["fallback_inflate"] = time.time() - t0
            r.backend = "geometric: silhouette inflation (CPU)"

        # 5. export + validate
        step(0.92, "Exporting and validating")
        t0 = time.time()
        if r.backend.startswith("neural") and upright:
            mesh = export.upright(mesh)
        mesh = export.normalise(mesh)
        r.glb, r.obj_zip, r.validation = export.export_all(mesh, r.out_dir)
        T["export+validate"] = time.time() - t0
        T["total"] = time.time() - t_all
        if not r.validation["glb"]["ok"]:
            r.warnings.append("Exported mesh failed validation; see validation.json.")
        with open(os.path.join(r.out_dir, "run.json"), "w") as f:
            json.dump(dict(prompt=prompt, seed=seed, shape_prior_requested=shape_prior, steps=steps, control_scale=control_scale, guidance=guidance,
                           n_candidates=n_candidates, mc_resolution=mc_resolution, texture_res=texture_res,
                           **r.summary()), f, indent=2, default=str)
        step(1.0, "Done")
        return r
    finally:
        # Always leave RAM/VRAM clear so the next UI/CLI run can load models again.
        try:
            from . import generate, reconstruct, reconstruct_hq, understand, memory as mem
            generate.unload()
            reconstruct.unload()
            reconstruct_hq.unload()
            understand.unload()
            mem.free_torch()
        except Exception:
            pass

