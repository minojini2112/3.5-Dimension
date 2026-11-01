"""Make TripoSR use PyMCubes when torchmcubes (needs a CUDA/C++ build) is not installed. Idempotent."""
import os

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "third_party", "TripoSR", "tsr", "models", "isosurface.py")
NEW = '''try:
    from torchmcubes import marching_cubes as _tmc
    _BACKEND = "torchmcubes"
except ImportError:  # portable fallback (no CUDA build needed): PyMCubes
    import mcubes
    _tmc = None
    _BACKEND = "pymcubes"


def marching_cubes(level, thresh):
    if _tmc is not None:
        return _tmc(level, thresh)
    vol = level.detach().cpu().numpy().astype(np.float32)
    v, f = mcubes.marching_cubes(vol, thresh)
    # torchmcubes returns (x=k, y=j, z=i); mimic it so the caller's swap stays valid
    v = v[:, [2, 1, 0]]
    return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(f.astype(np.int64))
'''


def patch(verbose=False):
    s = open(P, encoding="utf-8").read()
    if "_BACKEND" in s:
        if verbose:
            print("already patched")
        return
    s = s.replace("from torchmcubes import marching_cubes\n", NEW)
    open(P, "w", encoding="utf-8").write(s)
    print("patched", P)


if __name__ == "__main__":
    patch(verbose=True)
