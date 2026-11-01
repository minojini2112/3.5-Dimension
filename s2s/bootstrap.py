"""First-run setup: make sure the TripoSR source and every model weight exist; download whatever is missing.

Called automatically by app.py and run.py, so a fresh checkout only needs `pip install -r requirements.txt`
and then `python app.py`. Already-present files are detected offline (no network needed after the first run)."""
import io
import os
import shutil
import sys
import urllib.request
import zipfile

from . import config

# name -> (repo_id, allow_patterns, key file that proves the download finished)
MODELS = {
    "dreamshaper-8": ("Lykon/dreamshaper-8",
                      ["*.json", "*.txt", "tokenizer/*", "*fp16.safetensors"],
                      "unet/diffusion_pytorch_model.fp16.safetensors"),
    "controlnet-scribble": ("lllyasviel/control_v11p_sd15_scribble",
                            ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
                            "diffusion_pytorch_model.fp16.safetensors"),
    "lcm-lora-sdv1-5": ("latent-consistency/lcm-lora-sdv1-5", ["*.safetensors"], "pytorch_lora_weights.safetensors"),
    "TripoSR": ("stabilityai/TripoSR", ["config.yaml", "model.ckpt"], "model.ckpt"),
    "clip-vit-b-32": ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                      ["config.json", "model.safetensors", "tokenizer*", "vocab.json", "merges.txt",
                       "special_tokens_map.json", "preprocessor_config.json"], "model.safetensors"),
    "depth-anything-v2-small": ("depth-anything/Depth-Anything-V2-Small-hf",
                                ["config.json", "model.safetensors", "preprocessor_config.json"], "model.safetensors"),
    # HQ image->3D (Hunyuan3D-2mini turbo — ckpt is enough; safetensors twin optional).
    "Hunyuan3D-2mini": ("tencent/Hunyuan3D-2mini",
                        ["hunyuan3d-dit-v2-mini-turbo/config.yaml", "hunyuan3d-dit-v2-mini-turbo/model.fp16.ckpt"],
                        "hunyuan3d-dit-v2-mini-turbo/model.fp16.ckpt"),
}
REMBG_MODELS = ["isnet-general-use"]
TRIPOSR_GIT = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_ZIP = "https://github.com/VAST-AI-Research/TripoSR/archive/refs/heads/main.zip"
HUNYUAN_GIT = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git"
HUNYUAN_ZIP = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/refs/heads/main.zip"


def _log(msg):
    print(f"[setup] {msg}", flush=True)


def missing_models():
    return [n for n, (_, _, key) in MODELS.items() if not os.path.exists(os.path.join(config.WEIGHTS, n, key))]


def ensure_models(names=None):
    from huggingface_hub import snapshot_download
    todo = names or missing_models()
    for n in todo:
        repo, patterns, key = MODELS[n]
        _log(f"downloading {n} from huggingface.co/{repo} (one-time) ...")
        snapshot_download(repo_id=repo, local_dir=os.path.join(config.WEIGHTS, n), allow_patterns=patterns,
                          ignore_patterns=["safety_checker/*"])  # never loaded (safety_checker=None); saves ~600 MB
        if not os.path.exists(os.path.join(config.WEIGHTS, n, key)):
            raise RuntimeError(f"Download of {n} finished but {key} is missing; check your connection and retry.")
        _log(f"{n} ready")


def ensure_triposr_source():
    """Fetch TripoSR's code (MIT) as a zip if it isn't there (no git needed), then apply the PyMCubes patch."""
    dst = config.TRIPOSR_SRC
    if not os.path.exists(os.path.join(dst, "tsr", "system.py")):
        _log("downloading TripoSR source (one-time) ...")
        if shutil.which("git"):
            import subprocess
            r = subprocess.run(["git", "clone", "--depth", "1", TRIPOSR_GIT, dst], capture_output=True, text=True)
            if r.returncode == 0:
                _log("TripoSR source ready")
                _patch()
                return
            _log(f"git clone failed ({r.stderr.strip()[:200]}), trying the zip archive")
            shutil.rmtree(dst, ignore_errors=True)
        data = urllib.request.urlopen(TRIPOSR_ZIP, timeout=120).read()
        tmp = dst + "_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(tmp)
        inner = os.path.join(tmp, os.listdir(tmp)[0])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(inner, dst)
        shutil.rmtree(tmp, ignore_errors=True)
        _log("TripoSR source ready")
    _patch()


def _patch():
    sys.path.insert(0, os.path.join(config.ROOT, "tools"))
    import patch_triposr
    patch_triposr.patch()


def ensure_rembg():
    from rembg import new_session
    home = os.path.expanduser(os.environ.get("U2NET_HOME", "~/.u2net"))
    for m in REMBG_MODELS:
        if not os.path.exists(os.path.join(home, f"{m}.onnx")):
            _log(f"downloading background-removal model {m} (one-time) ...")
        new_session(m)


def ensure_hunyuan_source():
    """Fetch Hunyuan3D-2 Python package (hy3dgen) if missing."""
    dst = config.HUNYUAN3D_SRC
    if os.path.exists(os.path.join(dst, "hy3dgen", "shapegen", "pipelines.py")):
        return
    _log("downloading Hunyuan3D-2 source (one-time) ...")
    if shutil.which("git"):
        import subprocess
        r = subprocess.run(["git", "clone", "--depth", "1", HUNYUAN_GIT, dst], capture_output=True, text=True)
        if r.returncode == 0:
            _log("Hunyuan3D-2 source ready")
            return
        _log(f"git clone failed ({r.stderr.strip()[:200]}), trying the zip archive")
        shutil.rmtree(dst, ignore_errors=True)
    data = urllib.request.urlopen(HUNYUAN_ZIP, timeout=180).read()
    tmp = dst + "_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(tmp)
    inner = os.path.join(tmp, os.listdir(tmp)[0])
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(inner, dst)
    shutil.rmtree(tmp, ignore_errors=True)
    _log("Hunyuan3D-2 source ready")


def ensure_all(need_hq: bool = True):
    os.makedirs(config.WEIGHTS, exist_ok=True)
    ensure_triposr_source()
    if need_hq:
        ensure_hunyuan_source()
    miss = missing_models()
    if not need_hq:
        miss = [n for n in miss if n != "Hunyuan3D-2mini"]
    if miss:
        _log(f"missing models: {', '.join(miss)}; about {_size(miss):.1f} GB to download")
        ensure_models(miss)
    ensure_rembg()
    _log("all models present")


def _size(names):
    gb = {"dreamshaper-8": 2.6, "controlnet-scribble": 0.7, "lcm-lora-sdv1-5": 0.13, "TripoSR": 1.7,
          "clip-vit-b-32": 0.6, "depth-anything-v2-small": 0.1, "Hunyuan3D-2mini": 2.5}
    return sum(gb.get(n, 0.5) for n in names)


if __name__ == "__main__":
    ensure_all()
