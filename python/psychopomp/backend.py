"""torch.compile backend for psychopomp."""

import os
import tempfile
from typing import Callable, List

import torch
import torch._dynamo

from psychopomp._psychopomp_rs import process_onnx
from psychopomp.compiled_model import CompiledModel


def _create_backend(backend: str = "native") -> Callable:
    """Create a torch.compile backend function for the specified runtime backend.

    Args:
        backend: Backend to use for execution ("native" or "cuda").

    Returns:
        A backend function suitable for torch.compile.
    """

    def backend_fn(
        gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
    ) -> Callable:
        """torch.compile backend that compiles to luminal via ONNX.

        Args:
            gm: The traced GraphModule from TorchDynamo
            example_inputs: Example input tensors used for tracing

        Returns:
            A callable that executes through the luminal runtime
        """
        # Export the GraphModule to ONNX
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            # Export to ONNX
            torch.onnx.export(
                gm,
                tuple(example_inputs),
                tmp_path,
                input_names=[f"input_{i}" for i in range(len(example_inputs))],
                dynamic_axes=None,
                opset_version=17,
                # dynamo=True, // For now dyanmo is off so that the older exported is used
            )

            # Process through Rust with the specified backend
            graph_result = process_onnx(tmp_path, backend)

        finally:
            os.unlink(tmp_path)

        # Create the compiled model wrapper
        compiled = CompiledModel(graph_result)

        return compiled

    return backend_fn


# Register the native backend (default)
@torch._dynamo.register_backend
def psychopomp(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable:
    """torch.compile backend using the native (CPU) runtime."""
    return _create_backend("native")(gm, example_inputs)


# Register the CUDA backend
@torch._dynamo.register_backend
def psychopomp_cuda(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
) -> Callable:
    """torch.compile backend using the CUDA (GPU) runtime."""
    return _create_backend("cuda")(gm, example_inputs)
