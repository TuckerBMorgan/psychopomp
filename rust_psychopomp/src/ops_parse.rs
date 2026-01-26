use std::collections::HashMap;

use luminal::prelude::*;
use onnx_protobuf::NodeProto;
use std::ops::Add;
use std::ops::Mul;

use crate::utils::*;

/// Handle Gemm node: Y = alpha * A' * B' + beta * Cx
/// Default: transA=0, transB=0, alpha=1.0, beta=1.0
pub fn parse_gemm_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
) -> Result<(), String> {
    let trans_b = get_int_attr(node, "transB", 0);
    assert!(node.input.len() >= 2, "Gemm should have at least 2 inputs");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Gemm: missing input tensor '{}'", node.input[0]))?;
    let weight = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Gemm: missing weight tensor '{}'", node.input[1]))?;

    let matmul_result = if trans_b != 0 {
        // Weight is [N, K]. Create a new tensor with shape [K, N] to avoid the
        // double-permute issue. Store weight as [K, N] directly.
        let (n, k) = weight.dims2();
        let weight_kn_name = format!("{}_kn", &node.input[1]);
        let weight_kn = cx.named_tensor(weight_kn_name.clone(), vec![k, n]);
        tensors.insert(weight_kn_name, weight_kn);
        input.matmul(weight_kn)
    } else {
        input.matmul(weight)
    };

    let output = if node.input.len() > 2 && !node.input[2].is_empty() {
        let bias = *tensors
            .get(&node.input[2])
            .ok_or_else(|| format!("Gemm: missing bias tensor '{}'", node.input[2]))?;
        // Bias is [out_features], matmul_result is [batch, out_features].
        // Expand bias to [batch, out_features] by adding batch dim with stride 0.
        let (batch_dim, _) = matmul_result.dims2();
        let bias_expanded = bias.expand_dim(0, batch_dim);
        matmul_result.add(bias_expanded)
    } else {
        matmul_result
    };

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), output);
    Ok(())
}

/// Handle Add node: output = input[0] + input[1]
///
/// Supports numpy-style broadcasting and constant folding when both inputs
/// have known values at graph-build time.
pub fn parse_add_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Add should only have two inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Add: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Add: missing input tensor '{}'", node.input[1]))?;

    let output_name = &node.output[0];

    // Constant-fold when both inputs are known
    if let (Some(va), Some(vb)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let folded = broadcast_binop(&va, &vb, |a, b| a + b);
        let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
        let tensor = cx.named_tensor(output_name.clone(), broadcast_shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        return Ok(());
    }

    // Dynamic path: broadcast both operands to the output shape (numpy rules).
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.add(b_bc);
    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Mul node: output = input[0] * input[1]
///
/// Supports numpy-style broadcasting and constant folding. NaN values produced
/// by 0 * inf (common in attention masks) are replaced with 0.0.
pub fn parse_mul_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Mul should only have two inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Mul: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Mul: missing input tensor '{}'", node.input[1]))?;

    let output_name = &node.output[0];

    // Constant-fold when both inputs are known
    if let (Some(va), Some(vb)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let mut folded = broadcast_binop(&va, &vb, |a, b| a * b);
        // Replace NaN with 0.0 (handles 0 * -inf = NaN in attention masks)
        // TODO: Discuss if this is the best solution
        for v in &mut folded {
            if v.is_nan() {
                *v = 0.0;
            }
        }
        let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
        let tensor = cx.named_tensor(output_name.clone(), broadcast_shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        return Ok(());
    }

    // Dynamic path: broadcast both operands to the output shape (numpy rules).
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc.mul(b_bc);
    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Div node: element-wise division with numpy-style broadcasting.
///
/// When the divisor is a known constant, we pre-compute the reciprocal on the CPU
/// and use Mul with the reciprocal values (avoiding a runtime Recip kernel launch).
/// Otherwise, falls back to luminal's `/` operator (Recip + Mul decomposition).
pub fn parse_div_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Div should have exactly 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Div: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Div: missing input tensor '{}'", node.input[1]))?;

    let output_name = &node.output[0];
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());

    // Constant-fold when both inputs are known
    if let (Some(va), Some(vb)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let folded = broadcast_binop(&va, &vb, |a, b| a / b);
        let tensor = cx.named_tensor(output_name.clone(), broadcast_shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        return Ok(());
    }

    // When divisor is a known constant, pre-compute reciprocal and use Mul
    if let Some(vb) = known_values.get(&node.input[1]).cloned() {
        let recip_values: Vec<f32> = vb.iter().map(|v| 1.0 / v).collect();
        let recip_name = format!("{}_recip", node.input[1]);
        let b_shape: Vec<usize> = b.dims().iter().map(|e| e.to_usize().unwrap()).collect();
        let recip_tensor = cx.named_tensor(recip_name.clone(), b_shape);
        tensors.insert(recip_name.clone(), recip_tensor);
        weight_data.push((recip_name, recip_values));

        let a_bc = broadcast_to(a, &broadcast_shape);
        let b_recip_bc = broadcast_to(recip_tensor, &broadcast_shape);
        let result = a_bc.mul(b_recip_bc);
        tensors.insert(output_name.clone(), result);
        return Ok(());
    }

    // Dynamic fallback: both inputs unknown at graph-build time.
    // Uses Recip + Mul decomposition (luminal's `/` operator).
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc / b_bc;
    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Sqrt node: element-wise square root.
///
/// Supports constant folding when the input has known values.
pub fn parse_sqrt_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Sqrt should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sqrt: missing input tensor '{}'", node.input[0]))?;

    let result = input.sqrt();
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded: Vec<f32> = vals.iter().map(|&v| v.sqrt()).collect();
        known_values.insert(output_name.clone(), folded);
    }
    Ok(())
}
