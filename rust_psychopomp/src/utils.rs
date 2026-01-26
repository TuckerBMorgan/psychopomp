use std::{fs, path::Path};

use luminal::{prelude::GraphTensor, shape::Expression};
use onnx_protobuf::*;

// Given a Value from the Onnx proto return its tensor Shape, if it exists
// Note: some times pytorch will create tensors with a 0 shape
// we might want to handle, 0 shape and No shape as seperate ideas
pub fn get_shape_for_onnx_value(value: &onnx_protobuf::ValueInfoProto) -> Vec<usize> {
    if let Some(type_proto) = value.type_.as_ref() {
        if let Some(onnx_protobuf::type_proto::Value::TensorType(tensor)) = &type_proto.value {
            if let Some(shape) = tensor.shape.as_ref() {
                return shape
                    .dim
                    .iter()
                    .map(|dimension| {
                        if let Some(
                            onnx_protobuf::tensor_shape_proto::dimension::Value::DimValue(v),
                        ) = &dimension.value
                        {
                            *v as usize
                        } else {
                            1
                        }
                    })
                    .collect();
            }
        }
    }

    vec![]
}

/// Load initializer data as f32 values, handling multiple ONNX data types.
/// Used to seed known_values with small constant initializers for constant folding.
pub fn load_initializer_as_f32(init: &onnx_protobuf::TensorProto) -> Option<Vec<f32>> {
    match init.data_type {
        1 => {
            // FLOAT
            if !init.float_data.is_empty() {
                Some(init.float_data.clone())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            } else {
                None
            }
        }
        7 => {
            // INT64
            if !init.int64_data.is_empty() {
                Some(init.int64_data.iter().map(|&v| v as f32).collect())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        6 => {
            // INT32
            if !init.int32_data.is_empty() {
                Some(init.int32_data.iter().map(|&v| v as f32).collect())
            } else if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                        .collect(),
                )
            } else {
                None
            }
        }
        11 => {
            // FLOAT64
            if !init.raw_data.is_empty() {
                Some(
                    init.raw_data
                        .chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                                as f32
                        })
                        .collect(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Get an integer attribute from a node, with a default value
pub fn get_int_attr(node: &NodeProto, name: &str, default: i64) -> i64 {
    for attr in &node.attribute {
        if attr.name == name {
            return attr.i;
        }
    }
    default
}

/// Get a float attribute from a node, with a default value
pub fn get_f32_attr(node: &NodeProto, name: &str, default: f32) -> f32 {
    for attr in &node.attribute {
        if attr.name == name {
            return attr.f;
        }
    }

    return default;
}

/// Element-wise binary operation with scalar broadcasting support
pub fn broadcast_binop(a: &[f32], b: &[f32], op: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    match (a.len(), b.len()) {
        (1, _) => b.iter().map(|bv| op(a[0], *bv)).collect(),
        (_, 1) => a.iter().map(|av| op(*av, b[0])).collect(),
        _ => a
            .iter()
            .zip(b.iter())
            .map(|(av, bv)| op(*av, *bv))
            .collect(),
    }
}

/// Compute the broadcast output shape for two tensors (numpy rules: element-wise max).
pub fn compute_broadcast_shape(a: &[Expression], b: &[Expression]) -> Vec<usize> {
    let max_rank = a.len().max(b.len());
    let mut result = vec![1usize; max_rank];

    for i in 0..max_rank {
        let a_dim = if i < max_rank - a.len() {
            1
        } else {
            a[i - (max_rank - a.len())]
                .to_usize()
                .expect("broadcast: dim must be concrete")
        };
        let b_dim = if i < max_rank - b.len() {
            1
        } else {
            b[i - (max_rank - b.len())]
                .to_usize()
                .expect("broadcast: dim must be concrete")
        };
        result[i] = a_dim.max(b_dim);
    }
    result
}

/// Broadcast a tensor's shape to match a target shape (numpy-style broadcasting).
/// Left-pads with size-1 dims, then expands dims that are 1 to match target.
pub fn broadcast_to(mut tensor: GraphTensor, target_shape: &[usize]) -> GraphTensor {
    let src_dims = tensor.dims();
    let src_len = src_dims.len();
    let tgt_len = target_shape.len();

    if src_len == tgt_len {
        // Same rank: just expand dims that are 1
        tensor.shape.expand(target_shape.to_vec());
        return tensor;
    }

    // Left-pad with size-1 dims using expand_dim (adds dim with stride 0)
    for _ in 0..(tgt_len - src_len) {
        tensor = tensor.expand_dim(0, 1);
    }

    // Now expand each dim that is 1 to match target
    tensor.shape.expand(target_shape.to_vec());
    tensor
}

/// Load float data from a TensorProto, handling inline (float_data/raw_data) and external storage.
pub fn load_tensor_floats(init: &onnx_protobuf::TensorProto, model_dir: &Path) -> Option<Vec<f32>> {
    // Try inline float_data first
    if !init.float_data.is_empty() {
        return Some(init.float_data.clone());
    }
    // Try inline raw_data
    if !init.raw_data.is_empty() {
        let floats: Vec<f32> = init
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        return Some(floats);
    }
    // Try external data (data_location == EXTERNAL = 1)
    if !init.external_data.is_empty() {
        let mut location: Option<&str> = None;
        let mut offset: u64 = 0;
        let mut length: Option<u64> = None;
        for entry in &init.external_data {
            match entry.key.as_str() {
                "location" => location = Some(&entry.value),
                "offset" => offset = entry.value.parse().unwrap_or(0),
                "length" => length = entry.value.parse().ok(),
                _ => {}
            }
        }
        if let Some(loc) = location {
            let ext_path = model_dir.join(loc);
            match fs::read(&ext_path) {
                Ok(file_data) => {
                    let start = offset as usize;
                    let end = match length {
                        Some(len) => start + len as usize,
                        None => file_data.len(),
                    };
                    if end > file_data.len() {
                        return None;
                    }
                    let slice = &file_data[start..end];
                    let floats: Vec<f32> = slice
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    return Some(floats);
                }
                Err(_) => {
                    return None;
                }
            }
        }
    }
    None
}

/// Transpose weight data from [rows, cols] to [cols, rows] row-major layout
pub fn transpose_weight_data(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}
