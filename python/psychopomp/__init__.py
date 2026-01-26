"""Psychopomp: A torch.compile backend for the luminal runtime."""

# Import the native Rust module
from psychopomp._psychopomp_rs import process_onnx, OnnxGraphResult

# Import and register the torch.compile backend (registration happens on import)
from psychopomp.backend import psychopomp

# Import the compiled model wrapper
from psychopomp.compiled_model import CompiledModel

__all__ = [
    "process_onnx",
    "OnnxGraphResult",
    "psychopomp",
    "CompiledModel",
]
