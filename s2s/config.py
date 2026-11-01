import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.environ.get("S2S_WEIGHTS", os.path.join(ROOT, "weights"))
OUTPUTS = os.environ.get("S2S_OUTPUTS", os.path.join(ROOT, "outputs"))
TRIPOSR_SRC = os.path.join(ROOT, "third_party", "TripoSR")
HUNYUAN3D_SRC = os.path.join(ROOT, "third_party", "Hunyuan3D-2")

SD_MODEL = os.path.join(WEIGHTS, "dreamshaper-8")
CONTROLNET = os.path.join(WEIGHTS, "controlnet-scribble")
LCM_LORA = os.path.join(WEIGHTS, "lcm-lora-sdv1-5")
TRIPOSR = os.path.join(WEIGHTS, "TripoSR")
HUNYUAN3D_MINI = os.path.join(WEIGHTS, "Hunyuan3D-2mini")



def pick_device() -> str:
    forced = os.environ.get("S2S_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


DEFAULT_STYLE = ("3d render of a single {subject}, product photography, eye-level straight-on view, centered, full object visible, "
                 "isolated on a plain white background, soft studio lighting, clean materials, high detail, 8k")
NEGATIVE = ("multiple objects, cropped, cut off, text, watermark, logo, frame, border, busy background, "
            "scenery, ground shadow, people, hands, lowres, blurry, noisy, deformed, sketch, line drawing, "
            "monochrome")
