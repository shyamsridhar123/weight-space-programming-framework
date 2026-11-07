import sys
import torch

assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__.split("+", 1)[0] == "2.8.0", torch.__version__
assert torch.version.cuda is None, "The CPU Torch build is required"
print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}; CUDA build: {torch.version.cuda}")
