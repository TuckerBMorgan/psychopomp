use crate::ops_parse::*;
use crate::utils::*;
use luminal::prelude::GraphTensor;
use luminal::prelude::*;
use onnx_protobuf::ModelProto;
use onnx_protobuf::NodeProto;
use protobuf::Message;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::Hash;
use std::path::Path;

use crate::common_types::OnnxGraphResult;

pub fn build_onnx_graph(path: &str) -> Result<OnnxGraphResult, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
    let model_directory = Path::new(path).parent().unwrap_or(Path::new("."));
    // TODO: this loads the entire model into memory, need to look into having this mmapeds
    let model = ModelProto::parse_from_bytes(&data)
        .map_err(|e| format!("Failed to parse Onnx Model: {}", e))?;

    return OnnxGraphResult::parse_model(model, model_directory);
}

/// Process all nodes in the ONNX graph, dispatching each to its parse function.
///
/// Iterates over nodes in topological order (as provided by the ONNX model) and
/// translates each operator into luminal graph operations. Returns an error if
/// an unsupported op type is encountered.
fn process_onnx_nodes(
    nodes: &[NodeProto],
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    for node in nodes {
        match node.op_type.as_str() {
            "Gemm" => parse_gemm_node(node, tensors, cx)?,
            "Add" => parse_add_node(node, tensors, cx, weight_data, known_values)?,
            "Mul" => parse_mul_node(node, tensors, cx, weight_data, known_values)?,
            "Div" => parse_div_node(node, tensors, cx, weight_data, known_values)?,
            "Sqrt" => parse_sqrt_node(node, tensors, known_values)?,
            "Softmax" => parse_softmax_node(node, tensors, cx, weight_data)?,
            "Erf" => parse_erf_node(node, tensors, cx, weight_data)?,
            "Gelu" => parse_gelu_node(node, tensors, cx, weight_data)?,
            "LayerNormalization" => parse_layer_normalization_node(node, tensors, cx, weight_data)?,
            "Constant" => parse_constant(node, tensors, cx, weight_data, known_values)?,
            "ConstantOfShape" => {
                parse_constant_of_shape(node, tensors, cx, weight_data, known_values)?
            }

            "Identity" => parse_identity(node, tensors, known_values)?,
            "Dropout" => parse_dropout_node(node, tensors, known_values)?,
            "Equal" => parse_equal_node(node, tensors, known_values)?,
            "Where" => parse_where_node(node, tensors, cx, weight_data, known_values)?,
            "Expand" => parse_expand_node(node, tensors, known_values)?,
            "Gather" => parse_gather_node(node, tensors, cx, weight_data, known_values)?,
            "Trilu" => parse_trilu_node(node, tensors, cx, weight_data, known_values)?,
            "Cast" => parse_cast_node(node, tensors, known_values)?,

            "Transpose" => parse_transpose_node(node, tensors)?,
            "MatMul" => parse_matmul_node(node, tensors)?,
            "Mod" => parse_mod_node(node, tensors, known_values)?,
            "Shape" => parse_shape_node(node, tensors, cx, weight_data, known_values)?,
            "Reshape" => parse_reshape_node(node, tensors, known_values)?,
            "Slice" => parse_slice_node(node, tensors, cx, weight_data, known_values)?,
            "Concat" => parse_concat_node(node, tensors, cx, weight_data, known_values)?,
            "Unsqueeze" => parse_unsqueeze_node(node, tensors, known_values)?,
            "Squeeze" => parse_squeeze_node(node, tensors, known_values)?,

            other => return Err(format!("Unsupported ONNX op type: {:?}", other)),
        }
    }
    Ok(())
}

impl OnnxGraphResult {
    pub fn parse_model(
        model: ModelProto,
        model_directory: &Path,
    ) -> Result<OnnxGraphResult, String> {
        let onnx_graph = &model.graph;

        // Setup our Luminal Graph
        let mut context = Graph::new();
        // We will need to track the tensors we allocate so we can match up inputs and outputs in the graph
        let mut tensors: HashMap<String, GraphTensor> = HashMap::new();

        // This is the name of all of the tensors we will need to fill in parameters for
        let initializer_names: HashSet<&str> = onnx_graph
            .initializer
            .iter()
            .map(|t| t.name.as_str())
            .collect();

        // Input is an overloaded term in Onnx, it both means the inputs into the model, like the next token
        // and the parameters of the layers, for this we don't want any of the parameters
        let input_names: Vec<String> = onnx_graph
            .input
            .iter()
            .filter(|inp| !initializer_names.contains(inp.name.as_str()))
            .map(|inp| inp.name.clone())
            .collect();

        // Create "holding" tensors for the input
        // this way they can be considered in the graph computation, and later as we do mutiple runs we can target them and swap out the values
        // in them and not need to recompile the network
        for input in &onnx_graph.input {
            let shape = get_shape_for_onnx_value(input);
            if shape.is_empty() {
                /*
                log_syntax!(
                    "{} missing a shape, will not create tensor for it in graph",
                    input.name
                );
                */
                continue;
            }
            let tensor = context.named_tensor(input.name.clone(), shape);
            tensors.insert(input.name.clone(), tensor);
        }

        // Create the tensors for all of the paramaters in the model, this DOES NOT fill in the values, just creates placeholders
        for init in &onnx_graph.initializer {
            if !tensors.contains_key(&init.name) {
                let shape: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                let tensor = context.named_tensor(init.name.clone(), shape);
                tensors.insert(init.name.clone(), tensor);
            }
        }

        let mut weight_data = Vec::new();

        let mut known_values: HashMap<String, Vec<f32>> = HashMap::new();

        for init in &onnx_graph.initializer {
            let n_elements: usize = init
                .dims
                .iter()
                .map(|&d| d as usize)
                .product::<usize>()
                .max(1);
            // MAGIC_NUMBER:
            if n_elements <= 32 {
                if let Some(floats) = load_initializer_as_f32(init) {
                    known_values.insert(init.name.clone(), floats);
                } else {
                    // Questions
                    // Should this be fatal
                    // Should this be a print or a log
                    println!("Unable to initializer values for {:?}", init.name);
                }
            }
        }

        // Process computation nodes (Constant nodes add to weight_data)
        process_onnx_nodes(
            &onnx_graph.node,
            &mut tensors,
            &mut context,
            &mut weight_data,
            &mut known_values,
        )
        .unwrap();

        // Mark graph outputs (must happen before build_search_space)
        let mut output_names = Vec::new();
        let mut output_shapes = Vec::new();
        for output_vi in &onnx_graph.output {
            if let Some(gt) = tensors.get(&output_vi.name) {
                gt.output();
                let shape = get_shape_for_onnx_value(output_vi);
                if shape.is_empty() {
                    return Err(format!(
                        "Output tensor '{}' has no shape information in the ONNX model",
                        output_vi.name
                    ));
                }
                output_names.push(output_vi.name.clone());
                output_shapes.push(shape);
            }
        }

        // Extract weight data from initializers (handles inline + external storage)
        for init in &onnx_graph.initializer {
            let floats = match load_tensor_floats(init, model_directory) {
                Some(f) => f,
                None => {
                    continue;
                }
            };
            weight_data.push((init.name.clone(), floats.clone()));
            // Also store transposed data for _kn tensors
            let kn_name = format!("{}_kn", &init.name);
            if tensors.contains_key(&kn_name) {
                let dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                if dims.len() == 2 {
                    let transposed = transpose_weight_data(&floats, dims[0], dims[1]);
                    weight_data.push((kn_name, transposed));
                }
            }
        }

        // Handle _kn tensors from Identity-aliased weights (e.g., layer 1 sharing layer 0 weights).
        // For each _kn tensor without weight_data, find data via the known_values propagation.
        let weight_data_names: HashSet<String> =
            weight_data.iter().map(|(n, _)| n.clone()).collect();
        let kn_tensors: Vec<String> = tensors
            .keys()
            .filter(|name| name.ends_with("_kn") && !weight_data_names.contains(*name))
            .cloned()
            .collect();
        for kn_name in kn_tensors {
            // Strip "_kn" to get the base weight name
            let base_name = &kn_name[..kn_name.len() - 3];
            // The base tensor's GraphTensor may point to an initializer (via Identity alias).
            // Look for the data by checking which initializer this tensor's node ID matches.
            if let Some(base_gt) = tensors.get(base_name) {
                // Find which weight_data entry has this same node ID (the aliased initializer)
                for (wd_name, wd_data) in &weight_data.clone() {
                    if let Some(wd_gt) = tensors.get(wd_name) {
                        if wd_gt.id == base_gt.id {
                            // Found the source data, generate transposed version
                            if let Some(kn_gt) = tensors.get(&kn_name) {
                                let kn_dims = kn_gt.dims();
                                if kn_dims.len() == 2 {
                                    let k = kn_dims[0].to_usize().unwrap();
                                    let n = kn_dims[1].to_usize().unwrap();
                                    // The _kn shape is [K, N] where original is [N, K]
                                    let transposed = transpose_weight_data(wd_data, n, k);
                                    weight_data.push((kn_name.clone(), transposed));
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Collect tensor name -> NodeIndex mapping
        let tensor_ids: HashMap<String, NodeIndex> = tensors
            .iter()
            .map(|(name, gt)| (name.clone(), gt.id))
            .collect();

        // Track which tensor names are Input nodes (includes those created during process_onnx_nodes)
        let input_tensor_names: HashSet<String> = tensors.keys().cloned().collect();

        // Build and optimize with CudaRuntime.
        // CudaRuntime's profile() actually executes the graph during search,
        // so ALL Input nodes must have buffers before cx.search().
        //        let ctx = CudaContext::new(0).map_err(|e| format!("Failed to init CUDA context: {}", e))?;
        //  let stream = ctx.default_stream();
        context.build_search_space::<NativeRuntime>();
        let mut rt = context.search(NativeRuntime::default(), 1);

        // Step 1: Set dummy zero data for ALL input tensors
        for (name, gt) in &tensors {
            if !input_tensor_names.contains(name) {
                continue;
            }
            let n_elements = if let Some(vi) = onnx_graph.input.iter().find(|i| &i.name == name) {
                let shape = get_shape_for_onnx_value(vi);
                shape.iter().product::<usize>()
            } else if let Some(init) = onnx_graph.initializer.iter().find(|i| &i.name == name) {
                init.dims.iter().map(|&d| d as usize).product::<usize>()
            } else if let Some((_, data)) = weight_data.iter().find(|(n, _)| n == name) {
                data.len()
            } else {
                continue;
            };
            if n_elements > 0 {
                rt.set_data(gt.id, vec![0.0f32; n_elements]);
            }
        }

        // Step 2: Overwrite with real initializer data (for accurate profiling)
        for init in &onnx_graph.initializer {
            let floats = match load_tensor_floats(init, model_directory) {
                Some(f) => f,
                None => continue,
            };
            if let Some(gt) = tensors.get(&init.name) {
                rt.set_data(gt.id, floats.clone());
            }
            let kn_name = format!("{}_kn", &init.name);
            if let Some(gt_kn) = tensors.get(&kn_name) {
                let dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
                if dims.len() == 2 {
                    let transposed = transpose_weight_data(&floats, dims[0], dims[1]);
                    rt.set_data(gt_kn.id, transposed);
                }
            }
        }

        // Step 3: Load constant node data (from Constant, ConstantOfShape, Where masks, etc.)
        for (name, floats) in &weight_data {
            if let Some(gt) = tensors.get(name) {
                rt.set_data(gt.id, floats.clone());
            }
        }

        context.build_search_space::<NativeRuntime>();

        Ok(OnnxGraphResult {
            context,
            runtime: rt,
            tensor_ids,
            input_names,
            output_names,
            output_shapes,
        })
    }
}
