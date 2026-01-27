//! Runtime abstraction module for backend selection.
//!
//! This module provides a unified interface for different luminal runtimes
//! (Native, CUDA) using an enum wrapper that allows runtime selection.

use luminal::prelude::*;
use luminal_cuda::cudarc::driver::CudaContext;
use luminal_cuda::runtime::CudaRuntime;
use rustc_hash::FxHashMap;

/// Enum wrapper for runtime backends allowing runtime selection.
pub enum RuntimeBackend {
    Native(NativeRuntime),
    Cuda(CudaRuntime),
}

impl RuntimeBackend {
    /// Set input data for a tensor node.
    pub fn set_data(&mut self, node: NodeIndex, data: Vec<f32>) {
        match self {
            RuntimeBackend::Native(rt) => rt.set_data(node, data),
            RuntimeBackend::Cuda(rt) => rt.set_data(node, data),
        }
    }

    /// Execute the compiled graph.
    pub fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        match self {
            RuntimeBackend::Native(rt) => rt.execute(dyn_map),
            RuntimeBackend::Cuda(rt) => rt.execute(dyn_map),
        }
    }

    /// Get output data from a tensor node.
    pub fn get_f32(&self, node: NodeIndex) -> Vec<f32> {
        match self {
            RuntimeBackend::Native(rt) => rt.get_f32(node).to_vec(),
            RuntimeBackend::Cuda(rt) => rt.get_f32(node),
        }
    }

    /// Get the name of the active backend.
    pub fn name(&self) -> &'static str {
        match self {
            RuntimeBackend::Native(_) => "native",
            RuntimeBackend::Cuda(_) => "cuda",
        }
    }
}

/// Initialize a native (CPU) runtime for the given graph context.
pub fn initialize_native(context: &mut Graph) -> Result<RuntimeBackend, String> {
    context.build_search_space::<NativeRuntime>();
    let rt = context.search(NativeRuntime::default(), 1);
    Ok(RuntimeBackend::Native(rt))
}

/// Initialize a CUDA (GPU) runtime for the given graph context.
pub fn initialize_cuda(context: &mut Graph) -> Result<RuntimeBackend, String> {
    let cuda_ctx = CudaContext::new(0)
        .map_err(|e| format!("Failed to init CUDA context: {}", e))?;
    let stream = cuda_ctx.default_stream();
    context.build_search_space::<CudaRuntime>();
    let rt = context.search(CudaRuntime::initialize(stream), 1);
    Ok(RuntimeBackend::Cuda(rt))
}

/// Initialize a runtime based on the backend name.
pub fn initialize_runtime(context: &mut Graph, backend: &str) -> Result<RuntimeBackend, String> {
    match backend {
        "cuda" => initialize_cuda(context),
        "native" | _ => initialize_native(context),
    }
}
