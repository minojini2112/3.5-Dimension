"""Stage 2: sketch-guided image synthesis (DreamShaper-8 SD1.5 + ControlNet scribble + LCM-LoRA)."""
import json
import os
import subprocess
import sys

import torch
from PIL import Image

from . import config, memory

_PIPE = None


def load(device=None):
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, LCMScheduler

    device = device or config.pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    cn = ControlNetModel.from_pretrained(config.CONTROLNET, variant="fp16", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        config.SD_MODEL, controlnet=cn, variant="fp16", torch_dtype=dtype,
        safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(config.LCM_LORA, weight_name="pytorch_lora_weights.safetensors")
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        if config.vram_gb() < 7:  # e.g. RTX 2050 4 GB: stream modules to GPU on demand
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.enable_vae_slicing()
    else:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        pipe.to("cpu")
    pipe.enable_attention_slicing()
    _PIPE = pipe
    return pipe


def unload():
    """Drop the SD/ControlNet pipeline so TripoSR (or a later run) can fit in RAM/VRAM."""
    global _PIPE
    pipe = _PIPE
    _PIPE = None
    if pipe is None:
        return
    try:
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
    except Exception:
        pass
    try:
        pipe.to("cpu")
    except Exception:
        pass
    del pipe
    memory.free_torch()



def build_prompt(user_prompt: str) -> str:
    subject = (user_prompt or "").strip() or "object"
    return config.DEFAULT_STYLE.format(subject=subject)


def generate(control: Image.Image, prompt: str, seed: int = 42, steps: int = 6,
             guidance: float = 1.5, control_scale: float = 0.9, n: int = 1, size: int = 512):
    pipe = load()
    gen = torch.Generator("cpu").manual_seed(int(seed))
    out = pipe(prompt=build_prompt(prompt), negative_prompt=config.NEGATIVE,
               image=control.resize((size, size)), num_inference_steps=int(steps),
               guidance_scale=float(guidance), controlnet_conditioning_scale=float(control_scale),
               num_images_per_prompt=int(n), generator=gen, height=size, width=size)
    return out.images


def generate_isolated(control: Image.Image, prompt: str, out_dir: str, seed: int = 42, steps: int = 6,
                      guidance: float = 1.5, control_scale: float = 0.9, n: int = 1):
    """Run generate() in a child process. unload() does not return this memory to Windows, and the 3D stage
    that follows needs nearly all of the machine's commit limit, so it has to exit instead."""
    ctl, req = os.path.join(out_dir, "gen_control.png"), os.path.join(out_dir, "gen_request.json")
    control.save(ctl)
    with open(req, "w", encoding="utf-8") as f:
        json.dump(dict(prompt=prompt, seed=seed, steps=steps, guidance=guidance, control_scale=control_scale, n=n), f)
    prefix = os.path.join(out_dir, "gen_output")
    p = subprocess.run([sys.executable, "-m", "s2s.generate", ctl, req, prefix], cwd=config.ROOT, timeout=900)
    paths = [f"{prefix}{i}.png" for i in range(int(n))]
    if p.returncode != 0 or not all(os.path.exists(q) for q in paths):
        raise RuntimeError(f"Image generation process exited with code {p.returncode}")
    images = []
    for q in paths:
        images.append(Image.open(q).convert("RGB"))
        os.remove(q)
    return images


if __name__ == "__main__":
    ctl_path, req_path, out_prefix = sys.argv[1:4]
    with open(req_path, encoding="utf-8") as fh:
        kw = json.load(fh)
    for idx, im in enumerate(generate(Image.open(ctl_path).convert("RGB"), kw.pop("prompt"), **kw)):
        im.save(f"{out_prefix}{idx}.png")
