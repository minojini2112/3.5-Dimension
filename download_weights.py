"""Download everything Sketch2Shape needs (TripoSR source + ~6 GB of weights) into ./third_party and ./weights.
Optional: app.py and run.py do this automatically on first start."""
from s2s.bootstrap import ensure_all

if __name__ == "__main__":
    ensure_all()
