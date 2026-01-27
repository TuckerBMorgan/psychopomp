mod common_types;
mod onnx_parse;
mod ops_parse;
mod runtime;
mod utils;

use common_types::OnnxGraphResult;
use onnx_parse::build_onnx_graph;
use pyo3::prelude::*;

/// Process an ONNX file and return a compiled graph result.
///
/// Args:
///     path: Path to the ONNX model file.
///     backend: Backend to use for execution ("native" or "cuda"). Defaults to "native".
#[pyfunction]
#[pyo3(signature = (path, backend="native"))]
fn process_onnx(path: &str, backend: &str) -> PyResult<OnnxGraphResult> {
    build_onnx_graph(path, backend).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
}

/// Python module for the psychopomp backend.
#[pymodule]
fn _psychopomp_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_onnx, m)?)?;
    m.add_class::<OnnxGraphResult>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use luminal::prelude::*;

    fn build_linear() {
        let mut cx = Graph::new();
        let _ = cx.named_tensor("input".to_string(), vec![10, 10]);
    }

    #[test]
    fn test_basic() {
        build_linear();
    }
}
