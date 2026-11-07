python -m pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
python -m pip install build
python -m pip check
python - <<'PY'
import sys
import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
assert torch.version.cuda is None, "The CPU Torch build is required"
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}; CUDA build: {torch.version.cuda}")
PY
python -m pip freeze --all > "$RUN_ROOT/python-packages.txt"
