"""Psychopomp: A torch.compile backend for the luminal runtime."""

# Import the native Rust module
from psychopomp._psychopomp_rs import process_onnx, OnnxGraphResult

# Import and register the torch.compile backends (registration happens on import)
from psychopomp.backend import psychopomp, psychopomp_cuda

# Import the compiled model wrapper
from psychopomp.compiled_model import CompiledModel

__all__ = [
    "process_onnx",
    "OnnxGraphResult",
    "psychopomp",
    "psychopomp_cuda",
    "CompiledModel",
]
