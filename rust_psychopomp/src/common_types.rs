use std::collections::HashMap;

use luminal::prelude::*;
use pyo3::prelude::*;

use crate::runtime::RuntimeBackend;

#[pyclass(unsendable)]
pub struct OnnxGraphResult {
    pub context: Graph,
    pub runtime: RuntimeBackend,
    pub tensor_ids: HashMap<String, NodeIndex>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_shapes: Vec<Vec<usize>>,
}

#[pymethods]
impl OnnxGraphResult {
    /// Get the list of input tensor names.
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.input_names.clone()
    }

    /// Get the list of output tensor names.
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.output_names.clone()
    }

    /// Get the output shapes.
    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.output_shapes.clone()
    }

    /// Get all tensor names in the graph.
    #[getter]
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_ids.keys().cloned().collect()
    }

    /// Get the name of the active backend (native or cuda).
    #[getter]
    fn backend(&self) -> &'static str {
        self.runtime.name()
    }

    /// Set input tensor data by name.
    fn set_input(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let node_id = self
            .tensor_ids
            .get(name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown input tensor: {}", name)))?;
        self.runtime.set_data(*node_id, data);
        Ok(())
    }

    /// Execute the graph.
    fn run(&mut self) {
        self.runtime.execute(&self.context.dyn_map);
    }

    /// Get output tensor data by name.
    fn get_output(&self, name: &str) -> PyResult<Vec<f32>> {
        let node_id = self
            .tensor_ids
            .get(name)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Unknown output tensor: {}", name)))?;
        Ok(self.runtime.get_f32(*node_id))
    }
}
