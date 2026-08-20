"""Quick check that CUDA and transformers are available before a long run."""

import torch
import transformers

print("== Environment check ==")
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers : {transformers.__version__}")
