# Download one model, retrying (with resume) after connection drops until its key file exists.
m="$1"
cd "$(dirname "$0")"
for i in $(seq 1 200); do
  .venv/Scripts/python.exe -u -c "import sys;from s2s.bootstrap import ensure_models,missing_models
if '$m' in missing_models(): ensure_models(['$m'])" >> "dl_$m.log" 2>&1 && { echo "[retry] $m done" >> "dl_$m.log"; exit 0; }
  echo "[retry] $m attempt $i failed, retrying in 10s" >> "dl_$m.log"; sleep 10
done
exit 1
