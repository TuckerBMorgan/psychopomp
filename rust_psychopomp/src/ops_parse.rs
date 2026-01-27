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

/// Handle Pow node: element-wise power (base^exponent).
///
/// Supports broadcasting and constant folding.
pub fn parse_pow_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Pow should have exactly 2 inputs");
    let base = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Pow: missing base tensor '{}'", node.input[0]))?;
    let exp = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Pow: missing exponent tensor '{}'", node.input[1]))?;

    // Broadcast both operands to the same shape
    let broadcast_shape = compute_broadcast_shape(&base.dims(), &exp.dims());
    let base_bc = broadcast_to(base, &broadcast_shape);
    let exp_bc = broadcast_to(exp, &broadcast_shape);

    let result = base_bc.pow(exp_bc);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Constant folding: if both inputs have known values, compute the result
    if let (Some(vb), Some(ve)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let folded = broadcast_binop(&vb, &ve, |b, e| b.powf(e));
        known_values.insert(output_name.clone(), folded);
    }

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

/// Handle Cos node: element-wise cosine.
///
/// Supports constant folding when the input has known values.
pub fn parse_cos_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Cos should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Cos: missing input tensor '{}'", node.input[0]))?;

    let result = input.cos();
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded: Vec<f32> = vals.iter().map(|&v| v.cos()).collect();
        known_values.insert(output_name.clone(), folded);
    }
    Ok(())
}

/// Handle Sin node: element-wise sine.
///
/// Supports constant folding when the input has known values.
pub fn parse_sin_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Sin should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sin: missing input tensor '{}'", node.input[0]))?;

    let result = input.sin();
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded: Vec<f32> = vals.iter().map(|&v| v.sin()).collect();
        known_values.insert(output_name.clone(), folded);
    }
    Ok(())
}

/// Handle Neg node: element-wise negation.
///
/// Supports constant folding when the input has known values.
pub fn parse_neg_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Neg should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Neg: missing input tensor '{}'", node.input[0]))?;

    let result = -input;
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded: Vec<f32> = vals.iter().map(|&v| -v).collect();
        known_values.insert(output_name.clone(), folded);
    }
    Ok(())
}

/// Handle Tanh node: compute hyperbolic tangent.
/// tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
pub fn parse_tanh_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Tanh should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Tanh: missing input tensor '{}'", node.input[0]))?;

    let output_name = &node.output[0];

    // Constant fold if input is known
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded: Vec<f32> = vals.iter().map(|&v| v.tanh()).collect();
        known_values.insert(output_name.clone(), folded.clone());

        let input_shape: Vec<usize> = input.dims().iter().map(|e| e.to_usize().unwrap()).collect();
        let tensor = cx.named_tensor(output_name.clone(), input_shape);
        tensors.insert(output_name.clone(), tensor);
        weight_data.push((output_name.clone(), folded));
        return Ok(());
    }

    let input_shape: Vec<usize> = input.dims().iter().map(|e| e.to_usize().unwrap()).collect();

    // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    // Using exp2: exp(z) = exp2(z * log2(e))

    // -2.0 constant
    let neg2_name = format!("{}_tanh_neg2", output_name);
    let neg2 = cx.named_tensor(neg2_name.clone(), vec![1usize]);
    tensors.insert(neg2_name.clone(), neg2);
    weight_data.push((neg2_name, vec![-2.0f32]));
    let neg2_bc = broadcast_to(neg2, &input_shape);

    // log2(e) for exp2 conversion
    let log2e_name = format!("{}_tanh_log2e", output_name);
    let log2e = cx.named_tensor(log2e_name.clone(), vec![1usize]);
    tensors.insert(log2e_name.clone(), log2e);
    weight_data.push((log2e_name, vec![1.0f32 / 2.0f32.ln()]));
    let log2e_bc = broadcast_to(log2e, &input_shape);

    // 1.0 constant
    let one_name = format!("{}_tanh_one", output_name);
    let one = cx.named_tensor(one_name.clone(), vec![1usize]);
    tensors.insert(one_name.clone(), one);
    weight_data.push((one_name, vec![1.0f32]));
    let one_bc = broadcast_to(one, &input_shape);

    // -1.0 constant
    let neg_one_name = format!("{}_tanh_neg_one", output_name);
    let neg_one = cx.named_tensor(neg_one_name.clone(), vec![1usize]);
    tensors.insert(neg_one_name.clone(), neg_one);
    weight_data.push((neg_one_name, vec![-1.0f32]));
    let neg_one_bc = broadcast_to(neg_one, &input_shape);

    // Step 1: -2x
    let neg2x = neg2_bc.mul(input);

    // Step 2: -2x * log2(e) for exp2 conversion
    let exp_arg = neg2x.mul(log2e_bc);

    // Step 3: exp(-2x) = exp2(-2x * log2(e))
    let exp_neg2x = exp_arg.exp2();

    // Step 4: 1 + exp(-2x)
    let one_plus_exp = one_bc.add(exp_neg2x);

    // Step 5: 1 - exp(-2x) = 1 + (-1) * exp(-2x)
    let neg_exp = neg_one_bc.mul(exp_neg2x);
    let one_minus_exp = one_bc.add(neg_exp);

    // Step 6: (1 - exp(-2x)) / (1 + exp(-2x)) = (1 - exp(-2x)) * recip(1 + exp(-2x))
    let recip_denom = one_plus_exp.reciprocal();
    let result = one_minus_exp.mul(recip_denom);

    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Softmax node: compute softmax along the specified axis (default: last).
///
/// Softmax(x)_i = exp(x_i) / sum(exp(x_j)) for all j along the axis.
pub fn parse_softmax_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Softmax should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Softmax: missing input tensor '{}'", node.input[0]))?;

    let axis = get_int_attr(node, "axis", -1);
    let ndim = input.dims().len();
    let resolved_axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    let output_name = &node.output[0];
    let input_shape: Vec<usize> = input.dims().iter().map(|e| e.to_usize().unwrap()).collect();

    // 1. MaxReduce along axis (numerical stability)
    let max_val = input.max(resolved_axis);
    let expanded_max = max_val.expand_to_shape_on_axes(input_shape.clone(), &[resolved_axis]);

    // 2. Subtraction: input - max
    //    Use named_tensor for -1.0 to avoid Constant node (Constant has cleanup:true
    //    with no kernel counterpart, causing e-graph extraction failure).
    let neg_name = format!("{}_neg_one", output_name);
    let neg_tensor = cx.named_tensor(neg_name.clone(), vec![1usize]);
    tensors.insert(neg_name.clone(), neg_tensor);
    weight_data.push((neg_name, vec![-1.0f32]));
    let neg_bc = broadcast_to(neg_tensor, &input_shape);
    let neg_max = expanded_max.mul(neg_bc);
    let diff = input.add(neg_max);

    // 3. Scale by log2(e) for exp2 conversion: exp(x) = exp2(x * log2(e))
    let log2e_name = format!("{}_log2e", output_name);
    let log2e_tensor = cx.named_tensor(log2e_name.clone(), vec![1usize]);
    tensors.insert(log2e_name.clone(), log2e_tensor);
    weight_data.push((log2e_name, vec![1.0f32 / 2.0f32.ln()]));
    let log2e_bc = broadcast_to(log2e_tensor, &input_shape);
    let scaled = diff.mul(log2e_bc);

    // 4. exp2
    let exp_val = scaled.exp2();

    // 5. SumReduce along axis
    let sum_val = exp_val.sum(resolved_axis);
    let expanded_sum = sum_val.expand_to_shape_on_axes(input_shape, &[resolved_axis]);

    // 6. Division: exp / sum = exp * recip(sum)
    let recip_sum = expanded_sum.reciprocal();
    let result = exp_val.mul(recip_sum);

    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle ReduceMean node: compute mean along specified axes.
///
/// Supports both opset 18+ (axes as input) and older (axes as attribute).
/// keepdims attribute controls whether reduced dimensions are kept as size 1.
pub fn parse_reduce_mean_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("ReduceMean: missing input tensor '{}'", node.input[0]))?;

    let ndim = input.dims().len();
    let keepdims = get_int_attr(node, "keepdims", 1) != 0;

    // Get axes: from second input (opset 18+) or attribute (older)
    let axes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_data = known_values
            .get(&node.input[1])
            .ok_or_else(|| format!("ReduceMean: axes '{}' must be known", node.input[1]))?;
        axes_data
            .iter()
            .map(|&v| {
                let a = v as i64;
                if a < 0 {
                    (ndim as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes") {
        attr.ints
            .iter()
            .map(|&v| {
                if v < 0 {
                    (ndim as i64 + v) as usize
                } else {
                    v as usize
                }
            })
            .collect()
    } else {
        // No axes: reduce all dimensions
        (0..ndim).collect()
    };

    let result = input.mean(axes.clone());

    // Handle keepdims by expanding back to original shape
    let output = if keepdims {
        let input_shape: Vec<usize> = input
            .dims()
            .iter()
            .map(|e| e.to_usize().unwrap())
            .collect();
        result.expand_to_shape_on_axes(input_shape, axes)
    } else {
        result
    };

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), output);
    Ok(())
}

/// Handle Erf node: approximate the error function using a sigmoid-based formula.
///
/// erf(x) ≈ 2*sigmoid(2.2567583342 * x * (1 + 0.0885*x²)) - 1
/// Uses sigmoid since CudaRuntime provides sigmoid kernels but not erf directly.
/// All constants are created as named tensors to avoid Constant nodes (which have
/// cleanup:true but no kernel counterpart, causing e-graph extraction failure).
pub fn parse_erf_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Erf should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Erf: missing input tensor '{}'", node.input[0]))?;

    let output_name = &node.output[0];
    let input_shape: Vec<usize> = input.dims().iter().map(|e| e.to_usize().unwrap()).collect();

    // Create named tensors for all constants to avoid Constant nodes
    // erf(x) ≈ 2*sigmoid(2.2567583342 * x * (1 + 0.0885*x²)) - 1

    // Coefficient a = 2.2567583342
    let coeff_a_name = format!("{}_erf_coeff_a", output_name);
    let coeff_a = cx.named_tensor(coeff_a_name.clone(), vec![1usize]);
    tensors.insert(coeff_a_name.clone(), coeff_a);
    weight_data.push((coeff_a_name, vec![2.2567583342f32]));
    let coeff_a_bc = broadcast_to(coeff_a, &input_shape);

    // Coefficient b = 0.0885
    let coeff_b_name = format!("{}_erf_coeff_b", output_name);
    let coeff_b = cx.named_tensor(coeff_b_name.clone(), vec![1usize]);
    tensors.insert(coeff_b_name.clone(), coeff_b);
    weight_data.push((coeff_b_name, vec![0.0885f32]));
    let coeff_b_bc = broadcast_to(coeff_b, &input_shape);

    // One = 1.0
    let one_name = format!("{}_erf_one", output_name);
    let one = cx.named_tensor(one_name.clone(), vec![1usize]);
    tensors.insert(one_name.clone(), one);
    weight_data.push((one_name, vec![1.0f32]));
    let one_bc = broadcast_to(one, &input_shape);

    // Negative one = -1.0
    let neg_one_name = format!("{}_erf_neg_one", output_name);
    let neg_one = cx.named_tensor(neg_one_name.clone(), vec![1usize]);
    tensors.insert(neg_one_name.clone(), neg_one);
    weight_data.push((neg_one_name, vec![-1.0f32]));
    let neg_one_bc = broadcast_to(neg_one, &input_shape);

    // Two = 2.0
    let two_name = format!("{}_erf_two", output_name);
    let two = cx.named_tensor(two_name.clone(), vec![1usize]);
    tensors.insert(two_name.clone(), two);
    weight_data.push((two_name, vec![2.0f32]));
    let two_bc = broadcast_to(two, &input_shape);

    // log2(e) for exp2 conversion: exp(z) = exp2(z * log2(e))
    let log2e_name = format!("{}_erf_log2e", output_name);
    let log2e = cx.named_tensor(log2e_name.clone(), vec![1usize]);
    tensors.insert(log2e_name.clone(), log2e);
    weight_data.push((log2e_name, vec![1.0f32 / 2.0f32.ln()]));
    let log2e_bc = broadcast_to(log2e, &input_shape);

    // Compute: erf(x) ≈ 2*sigmoid(2.2567583342 * x * (1 + 0.0885*x²)) - 1
    // where sigmoid(z) = 1/(1 + exp(-z))

    // Step 1: x² = input * input
    let x_sq = input.mul(input);

    // Step 2: 0.0885 * x²
    let term = coeff_b_bc.mul(x_sq);

    // Step 3: 1 + 0.0885*x²
    let inner_add = one_bc.add(term);

    // Step 4: x * (1 + 0.0885*x²)
    let mul1 = input.mul(inner_add);

    // Step 5: 2.2567583342 * x * (1 + 0.0885*x²)
    let scaled = coeff_a_bc.mul(mul1);

    // Step 6: -1 * scaled (for sigmoid: 1/(1+exp(-z)))
    let neg_scaled = neg_one_bc.mul(scaled);

    // Step 7: Convert to base 2: exp(-z) = exp2(-z * log2(e))
    let exp_input = neg_scaled.mul(log2e_bc);

    // Step 8: exp2(...)
    let exp_val = exp_input.exp2();

    // Step 9: 1 + exp(...)
    let one2_name = format!("{}_erf_one2", output_name);
    let one2 = cx.named_tensor(one2_name.clone(), vec![1usize]);
    tensors.insert(one2_name.clone(), one2);
    weight_data.push((one2_name, vec![1.0f32]));
    let one2_bc = broadcast_to(one2, &input_shape);
    let one_plus_exp = one2_bc.add(exp_val);

    // Step 10: sigmoid = 1 / (1 + exp(-z))
    let sigmoid = one_plus_exp.reciprocal();

    // Step 11: 2 * sigmoid
    let two_sigmoid = two_bc.mul(sigmoid);

    // Step 12: 2*sigmoid - 1 (using neg_one for subtraction via addition)
    let neg_one2_name = format!("{}_erf_neg_one2", output_name);
    let neg_one2 = cx.named_tensor(neg_one2_name.clone(), vec![1usize]);
    tensors.insert(neg_one2_name.clone(), neg_one2);
    weight_data.push((neg_one2_name, vec![-1.0f32]));
    let neg_one2_bc = broadcast_to(neg_one2, &input_shape);
    let result = two_sigmoid.add(neg_one2_bc);

    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Sigmoid node: logistic sigmoid activation.
///
/// sigmoid(x) = 1 / (1 + exp(-x))
/// Uses exp2 with log2(e) scaling: exp(z) = exp2(z * log2(e))
pub fn parse_sigmoid_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Sigmoid should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Sigmoid: missing input tensor '{}'", node.input[0]))?;

    let output_name = &node.output[0];
    let input_shape: Vec<usize> = input.dims().iter().map(|e| e.to_usize().unwrap()).collect();

    // Create named tensors for constants
    // One = 1.0
    let one_name = format!("{}_sigmoid_one", output_name);
    let one = cx.named_tensor(one_name.clone(), vec![1usize]);
    tensors.insert(one_name.clone(), one);
    weight_data.push((one_name, vec![1.0f32]));
    let one_bc = broadcast_to(one, &input_shape);

    // Negative one = -1.0 (for computing -x)
    let neg_one_name = format!("{}_sigmoid_neg_one", output_name);
    let neg_one = cx.named_tensor(neg_one_name.clone(), vec![1usize]);
    tensors.insert(neg_one_name.clone(), neg_one);
    weight_data.push((neg_one_name, vec![-1.0f32]));
    let neg_one_bc = broadcast_to(neg_one, &input_shape);

    // log2(e) for exp2 conversion: exp(z) = exp2(z * log2(e))
    let log2e_name = format!("{}_sigmoid_log2e", output_name);
    let log2e = cx.named_tensor(log2e_name.clone(), vec![1usize]);
    tensors.insert(log2e_name.clone(), log2e);
    weight_data.push((log2e_name, vec![1.0f32 / 2.0f32.ln()]));
    let log2e_bc = broadcast_to(log2e, &input_shape);

    // sigmoid(x) = 1 / (1 + exp(-x))

    // Step 1: -x = input * (-1)
    let neg_x = input.mul(neg_one_bc);

    // Step 2: -x * log2(e) (for exp2)
    let scaled = neg_x.mul(log2e_bc);

    // Step 3: exp(-x) = exp2(scaled)
    let exp_neg_x = scaled.exp2();

    // Step 4: 1 + exp(-x)
    let one_plus_exp = one_bc.add(exp_neg_x);

    // Step 5: sigmoid = 1 / (1 + exp(-x))
    let result = one_plus_exp.reciprocal();

    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Gelu node: Gaussian Error Linear Unit activation.
///
/// Uses the tanh approximation: GELU(x) = x * sigmoid(1.5957691216 * x * (1 + 0.044715*x²))
/// Implemented via sigmoid since CudaRuntime provides sigmoid kernels but not tanh.
/// All constants are created as named tensors to avoid Constant nodes (which have
/// cleanup:true but no kernel counterpart, causing e-graph extraction failure).
pub fn parse_gelu_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Gelu should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Gelu: missing input tensor '{}'", node.input[0]))?;

    let output_name = &node.output[0];
    let input_shape: Vec<usize> = input.dims().iter().map(|e| e.to_usize().unwrap()).collect();

    // GELU(x) = x * sigmoid(1.5957691216 * x * (1 + 0.044715*x²))
    // where sigmoid(z) = 1/(1+exp(-z))
    // Create named tensors for all constants to avoid Constant nodes

    // Coefficient a = 1.5957691216
    let coeff_a_name = format!("{}_gelu_coeff_a", output_name);
    let coeff_a = cx.named_tensor(coeff_a_name.clone(), vec![1usize]);
    tensors.insert(coeff_a_name.clone(), coeff_a);
    weight_data.push((coeff_a_name, vec![1.5957691216f32]));
    let coeff_a_bc = broadcast_to(coeff_a, &input_shape);

    // Coefficient b = 0.044715
    let coeff_b_name = format!("{}_gelu_coeff_b", output_name);
    let coeff_b = cx.named_tensor(coeff_b_name.clone(), vec![1usize]);
    tensors.insert(coeff_b_name.clone(), coeff_b);
    weight_data.push((coeff_b_name, vec![0.044715f32]));
    let coeff_b_bc = broadcast_to(coeff_b, &input_shape);

    // One = 1.0
    let one_name = format!("{}_gelu_one", output_name);
    let one = cx.named_tensor(one_name.clone(), vec![1usize]);
    tensors.insert(one_name.clone(), one);
    weight_data.push((one_name, vec![1.0f32]));
    let one_bc = broadcast_to(one, &input_shape);

    // Negative one = -1.0
    let neg_one_name = format!("{}_gelu_neg_one", output_name);
    let neg_one = cx.named_tensor(neg_one_name.clone(), vec![1usize]);
    tensors.insert(neg_one_name.clone(), neg_one);
    weight_data.push((neg_one_name, vec![-1.0f32]));
    let neg_one_bc = broadcast_to(neg_one, &input_shape);

    // log2(e) for exp2 conversion: exp(z) = exp2(z * log2(e))
    let log2e_name = format!("{}_gelu_log2e", output_name);
    let log2e = cx.named_tensor(log2e_name.clone(), vec![1usize]);
    tensors.insert(log2e_name.clone(), log2e);
    weight_data.push((log2e_name, vec![1.0f32 / 2.0f32.ln()]));
    let log2e_bc = broadcast_to(log2e, &input_shape);

    // Step 1: x² = input * input
    let x_sq = input.mul(input);

    // Step 2: coeff_b * x²
    let term = coeff_b_bc.mul(x_sq);

    // Step 3: 1 + coeff_b*x²
    let inner_add = one_bc.add(term);

    // Step 4: x * (1 + coeff_b*x²)
    let mul1 = input.mul(inner_add);

    // Step 5: coeff_a * x * (1 + coeff_b*x²)
    let scaled = coeff_a_bc.mul(mul1);

    // Step 6: -1 * scaled (for sigmoid: 1/(1+exp(-z)))
    let neg_scaled = neg_one_bc.mul(scaled);

    // Step 7: Convert to base 2: exp(-z) = exp2(-z * log2(e))
    let exp_input = neg_scaled.mul(log2e_bc);

    // Step 8: exp2(...)
    let exp_val = exp_input.exp2();

    // Step 9: 1 + exp(...)
    let one2_name = format!("{}_gelu_one2", output_name);
    let one2 = cx.named_tensor(one2_name.clone(), vec![1usize]);
    tensors.insert(one2_name.clone(), one2);
    weight_data.push((one2_name, vec![1.0f32]));
    let one2_bc = broadcast_to(one2, &input_shape);
    let one_plus_exp = one2_bc.add(exp_val);

    // Step 10: sigmoid = 1 / (1 + exp(-z))
    let sigmoid = one_plus_exp.reciprocal();

    // Step 11: x * sigmoid = final GELU result
    let result = input.mul(sigmoid);

    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle LayerNormalization node: normalize along the last N axes, then scale and shift.
///
/// Normalizes axes [axis..ndim] with the given epsilon. Applies element-wise
/// scale and optional bias after normalization.
pub fn parse_layer_normalization_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    assert!(
        node.input.len() >= 2,
        "LayerNormalization should have at least 2 inputs"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("LayerNorm: missing input tensor '{}'", node.input[0]))?;
    let scale = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("LayerNorm: missing scale tensor '{}'", node.input[1]))?;

    let epsilon = get_f32_attr(node, "epsilon", 1e-5);
    let axis = get_int_attr(node, "axis", -1);

    let ndim = input.dims().len();
    let resolved_axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    let output_name = &node.output[0];
    let input_shape: Vec<usize> = input.dims().iter().map(|e| e.to_usize().unwrap()).collect();
    let norm_axes: Vec<usize> = (resolved_axis..ndim).collect();

    // 1. Compute mean along norm_axes
    let mean_val = input.mean(norm_axes.clone());
    let expanded_mean = mean_val.expand_to_shape_on_axes(input_shape.clone(), norm_axes.clone());

    // 2. Subtract mean: centered = input + mean * (-1)
    //    Use named_tensor for -1.0 to avoid Constant node (Constant has cleanup:true
    //    with no kernel counterpart, causing e-graph extraction failure).
    let neg_name = format!("{}_neg_one", output_name);
    let neg_tensor = cx.named_tensor(neg_name.clone(), vec![1usize]);
    tensors.insert(neg_name.clone(), neg_tensor);
    weight_data.push((neg_name, vec![-1.0f32]));
    let neg_bc = broadcast_to(neg_tensor, &input_shape);
    let neg_mean = expanded_mean.mul(neg_bc);
    let centered = input.add(neg_mean);

    // 3. Compute variance: mean(centered^2)
    let squared = centered.mul(centered);
    let variance = squared.mean(norm_axes.clone());

    // 4. Add epsilon (use named_tensor to avoid Constant cleanup issue)
    let eps_name = format!("{}_epsilon", output_name);
    let eps_tensor = cx.named_tensor(eps_name.clone(), vec![1usize]);
    tensors.insert(eps_name.clone(), eps_tensor);
    weight_data.push((eps_name, vec![epsilon]));
    let var_shape: Vec<usize> = variance
        .dims()
        .iter()
        .map(|e| e.to_usize().unwrap())
        .collect();
    let eps_bc = broadcast_to(eps_tensor, &var_shape);
    let var_eps = variance.add(eps_bc);

    // 5. inv_std = 1/sqrt(var + eps)
    let sqrt_val = var_eps.sqrt();
    let inv_std = sqrt_val.reciprocal();
    let expanded_inv_std = inv_std.expand_to_shape_on_axes(input_shape.clone(), norm_axes.clone());

    // 6. Normalize: centered * inv_std
    let normalized = centered.mul(expanded_inv_std);

    // 7. Scale by weight (scale is already an Input node from ONNX)
    let input_dims = input.dims();
    let mut scale_expanded = scale;
    for i in (0..resolved_axis).rev() {
        scale_expanded = scale_expanded.expand_dim(0, input_dims[i].clone());
    }
    let mut result = normalized.mul(scale_expanded);

    // 8. Optional bias (bias is already an Input node from ONNX)
    if node.input.len() > 2 && !node.input[2].is_empty() {
        let bias = *tensors
            .get(&node.input[2])
            .ok_or_else(|| format!("LayerNorm: missing bias tensor '{}'", node.input[2]))?;
        let mut bias_expanded = bias;
        for i in (0..resolved_axis).rev() {
            bias_expanded = bias_expanded.expand_dim(0, input_dims[i].clone());
        }
        result = result.add(bias_expanded);
    }

    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Constant node: creates a tensor from embedded data in the node attributes.
///
/// Supports FLOAT, INT64, INT32, and FLOAT64 data types (all converted to f32).
/// The resulting tensor is registered as a known constant for downstream folding.
pub fn parse_constant(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(
        node.output.len() == 1,
        "Constant should have exactly one output"
    );

    // Find the "value" attribute (type TENSOR)
    let value_attr = node
        .attribute
        .iter()
        .find(|a| a.name == "value")
        .ok_or_else(|| "Constant node missing 'value' attribute".to_string())?;

    let tensor_proto = value_attr
        .t
        .as_ref()
        .ok_or_else(|| "Constant 'value' attribute has no TensorProto".to_string())?;

    // Determine shape: empty dims = scalar = [1] for luminal
    let shape: Vec<usize> = if tensor_proto.dims.is_empty() {
        vec![1]
    } else {
        tensor_proto.dims.iter().map(|&d| d as usize).collect()
    };

    // Extract float data based on data_type
    let floats: Vec<f32> = match tensor_proto.data_type {
        1 => {
            // FLOAT (f32)
            if !tensor_proto.float_data.is_empty() {
                tensor_proto.float_data.clone()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
        }
        7 => {
            // INT64
            // There is a cast from Int64 -> f32 here because Luminal does not support f32
            if !tensor_proto.int64_data.is_empty() {
                tensor_proto.int64_data.iter().map(|&v| v as f32).collect()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect()
            }
        }
        6 => {
            // INT32
            // There is a cast from Int32 -> f32 here because Luminal does not support f32
            if !tensor_proto.int32_data.is_empty() {
                tensor_proto.int32_data.iter().map(|&v| v as f32).collect()
            } else {
                tensor_proto
                    .raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect()
            }
        }
        9 => {
            // Bool
            // Bools are stored as bytes in raw_data or as int32 in int32_data
            if !tensor_proto.int32_data.is_empty() {
                tensor_proto
                    .int32_data
                    .iter()
                    .map(|&v| if v != 0 { 1.0 } else { 0.0 })
                    .collect()
            } else {
                tensor_proto
                    .raw_data
                    .iter()
                    .map(|&b| if b != 0 { 1.0 } else { 0.0 })
                    .collect()
            }
        }
        11 => {
            // FLOAT64 (f64)
            // There is a cast from f64 -> f32 here because Luminal does not support f32
            // TODO: add f64 as this will loss information, this is a bad approach
            tensor_proto
                .raw_data
                .chunks_exact(8)
                .map(|c| {
                    f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                })
                .collect()
        }
        dt => return Err(format!("Constant node: unsupported data_type {}", dt)),
    };

    let output_name = &node.output[0];
    let tensor = cx.named_tensor(output_name.clone(), shape);
    tensors.insert(output_name.clone(), tensor);
    known_values.insert(output_name.clone(), floats.clone());
    weight_data.push((output_name.clone(), floats));

    Ok(())
}

/// Handle ConstantOfShape node: creates a tensor of a given shape filled with a constant value.
///
/// The shape is read from the first input (must be a known constant, typically from a
/// Constant node). The fill value comes from the "value" attribute (defaults to 0.0).
pub fn parse_constant_of_shape(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(
        node.input.len() == 1,
        "ConstantOfShape should have exactly one input"
    );
    assert!(
        node.output.len() == 1,
        "ConstantOfShape should have exactly one output"
    );

    // Resolve shape statically from weight_data (input comes from a Constant node)
    let shape_name = &node.input[0];
    let shape_data = weight_data.iter()
        .find(|(n, _)| n == shape_name)
        .ok_or_else(|| format!("ConstantOfShape: shape input '{}' not found in weight_data (must come from a Constant node)", shape_name))?;
    let shape: Vec<usize> = shape_data.1.iter().map(|&v| v as usize).collect();
    let n_elements: usize = shape.iter().product();

    // Parse the fill value from the "value" attribute (default: 0.0f32)
    // TODO: this, along with similar code above in Constant parse, might be a good
    // idea to turn into a single function for getting data out of an attribute
    let fill_value: f32 = if let Some(attr) = node.attribute.iter().find(|a| a.name == "value") {
        if let Some(tp) = attr.t.as_ref() {
            match tp.data_type {
                1 => {
                    if !tp.float_data.is_empty() {
                        tp.float_data[0]
                    } else if !tp.raw_data.is_empty() {
                        f32::from_le_bytes([
                            tp.raw_data[0],
                            tp.raw_data[1],
                            tp.raw_data[2],
                            tp.raw_data[3],
                        ])
                    } else {
                        0.0
                    }
                }
                7 => {
                    if !tp.int64_data.is_empty() {
                        tp.int64_data[0] as f32
                    } else if tp.raw_data.len() >= 8 {
                        i64::from_le_bytes([
                            tp.raw_data[0],
                            tp.raw_data[1],
                            tp.raw_data[2],
                            tp.raw_data[3],
                            tp.raw_data[4],
                            tp.raw_data[5],
                            tp.raw_data[6],
                            tp.raw_data[7],
                        ]) as f32
                    } else {
                        0.0
                    }
                }
                6 => {
                    if !tp.int32_data.is_empty() {
                        tp.int32_data[0] as f32
                    } else if tp.raw_data.len() >= 4 {
                        i32::from_le_bytes([
                            tp.raw_data[0],
                            tp.raw_data[1],
                            tp.raw_data[2],
                            tp.raw_data[3],
                        ]) as f32
                    } else {
                        0.0
                    }
                }
                11 => {
                    if tp.raw_data.len() >= 8 {
                        f64::from_le_bytes([
                            tp.raw_data[0],
                            tp.raw_data[1],
                            tp.raw_data[2],
                            tp.raw_data[3],
                            tp.raw_data[4],
                            tp.raw_data[5],
                            tp.raw_data[6],
                            tp.raw_data[7],
                        ]) as f32
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            }
        } else {
            0.0
        }
    } else {
        0.0
    };

    let output_name = &node.output[0];
    let tensor = cx.named_tensor(output_name.clone(), shape);
    tensors.insert(output_name.clone(), tensor);
    let values = vec![fill_value; n_elements];
    known_values.insert(output_name.clone(), values.clone());
    weight_data.push((output_name.clone(), values));

    Ok(())
}

// ============================================================================
// Comparison / conditional ops (Equal, Where, Trilu)
// ============================================================================

/// Handle Equal node: element-wise equality comparison.
///
/// Outputs 1.0 where inputs are equal, 0.0 otherwise. Supports broadcasting
/// and constant folding.
pub fn parse_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Equal should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Equal: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Equal: missing input tensor '{}'", node.input[1]))?;

    // Broadcast both operands to the same shape
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);

    let result = a_bc.eq(b_bc);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Constant folding: if both inputs have known values, compute the result
    if let (Some(va), Some(vb)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let folded = broadcast_binop(&va, &vb, |a, b| if a == b { 1.0 } else { 0.0 });
        known_values.insert(output_name.clone(), folded);
    }

    Ok(())
}

/// Parse LessOrEqual node (ONNX element-wise less-than-or-equal comparison).
///
/// Outputs 1.0 where a <= b, 0.0 otherwise. Supports broadcasting
/// and constant folding.
pub fn parse_less_or_equal_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "LessOrEqual should have 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("LessOrEqual: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("LessOrEqual: missing input tensor '{}'", node.input[1]))?;

    // Broadcast both operands to the same shape
    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);

    // LessOrEqual: a <= b is equivalent to NOT(b < a), i.e., 1.0 - b.lt(a)
    let one = a_bc.graph().constant_float(1.0).expand_rhs(a_bc.shape);
    let result = one - b_bc.lt(a_bc);

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Constant folding: if both inputs have known values, compute the result
    if let (Some(va), Some(vb)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let folded = broadcast_binop(&va, &vb, |a, b| if a <= b { 1.0 } else { 0.0 });
        known_values.insert(output_name.clone(), folded);
    }

    Ok(())
}

/// Compute 3-way broadcast shape for Where(condition, x, y).
fn compute_where_broadcast_shape(
    condition: &GraphTensor,
    x: &GraphTensor,
    y: &GraphTensor,
) -> Vec<usize> {
    let shape_cx = compute_broadcast_shape(&condition.dims(), &x.dims());
    compute_broadcast_shape(
        &shape_cx
            .iter()
            .map(|&d| Expression::from(d as i32))
            .collect::<Vec<_>>(),
        &y.dims(),
    )
}

/// If tensor has known inf values, create a clamped copy (±1e9); then broadcast to target shape.
/// This avoids 0 * inf = NaN in luminal's cond decomposition (cond * x + (1-cond) * y).
fn clamp_and_broadcast(
    tensor: GraphTensor,
    input_name: &str,
    broadcast_shape: &[usize],
    output_name: &str,
    label: &str,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &HashMap<String, Vec<f32>>,
) -> GraphTensor {
    if let Some(vals) = known_values.get(input_name) {
        if vals.iter().any(|v| v.is_infinite()) {
            let clamped: Vec<f32> = vals.iter().map(|&v| v.clamp(-1e9, 1e9)).collect();
            let clamped_name = format!("{}_{}_clamped", output_name, label);
            let shape: Vec<usize> = tensor
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("dim must be concrete"))
                .collect();
            let clamped_tensor = cx.named_tensor(clamped_name.clone(), shape);
            weight_data.push((clamped_name.clone(), clamped));
            return broadcast_to(clamped_tensor, broadcast_shape);
        }
    }
    broadcast_to(tensor, broadcast_shape)
}

/// Handle Where node: conditional select — output[i] = condition[i] ? x[i] : y[i]
///
/// Supports full constant folding when all three inputs are known. For the dynamic
/// path, infinite values in x or y are clamped to ±1e9 to avoid NaN from luminal's
/// cond decomposition (cond * x + (1-cond) * y where 0 * inf = NaN).
pub fn parse_where_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 3, "Where should have 3 inputs");
    let condition = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Where: missing condition tensor '{}'", node.input[0]))?;
    let x = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Where: missing X tensor '{}'", node.input[1]))?;
    let y = *tensors
        .get(&node.input[2])
        .ok_or_else(|| format!("Where: missing Y tensor '{}'", node.input[2]))?;

    let output_name = &node.output[0];

    // Full constant folding: if all inputs have known values, compute the result directly
    if let (Some(vc), Some(vx), Some(vy)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
        known_values.get(&node.input[2]).cloned(),
    ) {
        let out_len = vc.len().max(vx.len()).max(vy.len());
        let folded: Vec<f32> = (0..out_len)
            .map(|i| {
                let c = vc[if vc.len() == 1 { 0 } else { i }];
                let xv = vx[if vx.len() == 1 { 0 } else { i }];
                let yv = vy[if vy.len() == 1 { 0 } else { i }];
                if c != 0.0 { xv } else { yv }
            })
            .collect();
        let shape_cx = compute_broadcast_shape(&condition.dims(), &x.dims());
        let broadcast_shape = compute_broadcast_shape(
            &shape_cx
                .iter()
                .map(|&d| Expression::from(d as i32))
                .collect::<Vec<_>>(),
            &y.dims(),
        );
        let tensor = cx.named_tensor(output_name.clone(), broadcast_shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        return Ok(());
    }

    // Dynamic path: broadcast and use luminal's cond decomposition.
    // If x or y is a known constant with inf values, clamp to ±1e9
    // to avoid 0 * inf = NaN in cond's algebraic expansion (cond * x + (1-cond) * y).
    // For attention masking, -1e9 is equivalent to -inf after softmax.
    let broadcast_shape = compute_where_broadcast_shape(&condition, &x, &y);

    let x_final = clamp_and_broadcast(
        x,
        &node.input[1],
        &broadcast_shape,
        output_name,
        "x",
        cx,
        weight_data,
        known_values,
    );
    let y_final = clamp_and_broadcast(
        y,
        &node.input[2],
        &broadcast_shape,
        output_name,
        "y",
        cx,
        weight_data,
        known_values,
    );

    let cond_bc = broadcast_to(condition, &broadcast_shape);
    let result = x_final.cond(cond_bc, y_final);
    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Expand node: broadcast a tensor to a larger shape.
///
/// The target shape is read from the second input (must be a known constant).
/// Only dimensions of size 1 in the input are expanded to match the target.
/// Propagates known_values by broadcasting the input values to the output shape.
pub fn parse_expand_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Expand should have 2 inputs");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Expand: missing input tensor '{}'", node.input[0]))?;

    let shape_name = &node.input[1];
    let shape_data = known_values
        .get(shape_name)
        .ok_or_else(|| format!("Expand: missing shape data for '{}'", shape_name))?;
    let target_shape: Vec<usize> = shape_data.iter().map(|&v| v as usize).collect();

    // Compute broadcast output shape: max(input_dim, target_dim) per dimension.
    // Per ONNX spec, if input has fewer dims than target, it's right-aligned
    // (leading dims are treated as 1).
    let input_dims = input.dims();
    let input_rank = input_dims.len();
    let target_rank = target_shape.len();

    let mut result = input;

    // If input has fewer dimensions, prepend dimensions of size 1
    if input_rank < target_rank {
        let dims_to_add = target_rank - input_rank;
        for _ in 0..dims_to_add {
            result = result.expand_dim(0, 1);
        }
    }

    // Now compute broadcast shape with matching ranks
    let padded_input_dims = result.dims();
    let broadcast_shape: Vec<usize> = padded_input_dims
        .iter()
        .zip(target_shape.iter())
        .map(|(id, &td)| {
            let id_val = id.to_usize().expect("Expand: input dim must be concrete");
            id_val.max(td)
        })
        .collect();

    result.shape.expand(broadcast_shape.clone());

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Propagate known values by broadcasting to the output shape
    if let Some(input_vals) = known_values.get(&node.input[0]).cloned() {
        let output_size: usize = broadcast_shape.iter().product();
        let expanded = if input_vals.len() == 1 {
            // Scalar broadcast: replicate the single value
            vec![input_vals[0]; output_size]
        } else {
            // General broadcast: tile the input values
            // For simplicity, only handle the common case where input is broadcastable
            input_vals
                .iter()
                .cycle()
                .take(output_size)
                .cloned()
                .collect()
        };
        known_values.insert(output_name.clone(), expanded);
    }

    Ok(())
}

/// Handle Gather node: index into a tensor along a specified axis.
///
/// For 1D data, performs simple element-wise gathering. For ND data, gathers
/// slices along the specified axis. Supports arbitrary axis values.
/// Supports constant folding when both data and indices are known.
pub fn parse_gather_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Gather should have 2 inputs");

    let data = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Gather: missing data tensor '{}'", node.input[0]))?;

    let data_dims = data.dims();
    let data_ndim = data_dims.len();

    // Handle axis (support negative indexing)
    let axis_raw = get_int_attr(node, "axis", 0);
    let axis = if axis_raw < 0 {
        (data_ndim as i64 + axis_raw) as usize
    } else {
        axis_raw as usize
    };

    // If both inputs are known, fully constant-fold
    if let (Some(vdata), Some(vidx)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let output_name = &node.output[0];

        // Get concrete data dimensions
        let data_shape: Vec<usize> = data_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: data dims must be concrete"))
            .collect();

        // Get indices tensor shape
        let indices_tensor = tensors.get(&node.input[1]);
        let indices_shape: Vec<usize> = if let Some(idx_tensor) = indices_tensor {
            idx_tensor
                .dims()
                .iter()
                .map(|d| d.to_usize().expect("Gather: indices dims must be concrete"))
                .collect()
        } else {
            vec![vidx.len()]
        };

        // Compute output shape: data_shape[0..axis] + indices_shape + data_shape[axis+1..]
        let mut output_shape: Vec<usize> = data_shape[..axis].to_vec();
        output_shape.extend(&indices_shape);
        output_shape.extend(&data_shape[axis + 1..]);

        // Gather with arbitrary axis
        let output_size: usize = output_shape.iter().product();
        let mut folded = vec![0.0f32; output_size.max(1)];

        if output_size > 0 {
            let pre_size: usize = data_shape[..axis].iter().product::<usize>().max(1);
            let post_size: usize = data_shape[axis + 1..].iter().product::<usize>().max(1);
            let indices_size: usize = vidx.len();
            let axis_dim = data_shape[axis] as i64;

            for pre in 0..pre_size {
                for (idx_pos, &idx_val) in vidx.iter().enumerate() {
                    // Normalize negative indices: ONNX allows -1 for last element, etc.
                    let idx_raw = idx_val as i64;
                    let idx = (((idx_raw % axis_dim) + axis_dim) % axis_dim) as usize;
                    for post in 0..post_size {
                        let data_flat =
                            pre * (data_shape[axis] * post_size) + idx * post_size + post;
                        let out_flat = pre * (indices_size * post_size) + idx_pos * post_size + post;
                        folded[out_flat] = vdata[data_flat];
                    }
                }
            }
        }

        let shape = if output_shape.is_empty() {
            vec![1]
        } else {
            output_shape
        };
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), folded.clone());
        weight_data.push((output_name.clone(), folded));
        return Ok(());
    }

    let indices_raw = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Gather: missing indices tensor '{}'", node.input[1]))?;

    // Normalize negative indices: ONNX allows -1 for last element, -2 for second-to-last, etc.
    // Use conditional normalization instead of modulo to avoid floating-point precision loss:
    // if index < 0 then index + axis_dim else index
    let axis_dim = data_dims[axis]
        .to_usize()
        .ok_or_else(|| "Gather: axis dimension must be concrete for index normalization".to_string())?;
    let axis_dim_f32 = axis_dim as f32;
    let zero = indices_raw.graph().constant_float(0.0).expand_rhs(indices_raw.shape);
    let adjustment = indices_raw.graph().constant_float(axis_dim_f32).expand_rhs(indices_raw.shape);
    let is_negative = indices_raw.lt(zero); // Returns 1.0 for negative, 0.0 for non-negative
    let indices = indices_raw + (is_negative * adjustment);

    let result = if data_ndim == 1 {
        // 1D data: simple flat element-wise gather
        data.gather(indices.cast(DType::Int))
    } else if axis == 0 {
        // ND data with axis=0: gather entire slices along first dimension
        gather_axis0(data, indices, &data_dims)?
    } else {
        // General case: axis != 0
        // Strategy: permute to bring axis to front, gather, permute back

        // Build permutation to move axis to position 0
        let mut perm: Vec<usize> = (0..data_ndim).collect();
        perm.remove(axis);
        perm.insert(0, axis);

        // Permute data
        let permuted_data = data.permute(perm);
        let permuted_dims = permuted_data.dims();

        // Gather on axis 0 of permuted data
        let gathered = gather_axis0(permuted_data, indices, &permuted_dims)?;

        // Compute gathered shape
        let idx_dims = indices.dims();
        let mut gathered_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &permuted_dims[1..] {
            gathered_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(gathered_shape.clone());

        // Build inverse permutation to restore original axis order
        let indices_rank = idx_dims.len();
        let mut inv_perm: Vec<usize> = Vec::with_capacity(reshaped.dims().len());

        // Dimensions before original axis (they're after indices in gathered result)
        for i in 0..axis {
            inv_perm.push(indices_rank + i);
        }
        // The indices dimensions
        for i in 0..indices_rank {
            inv_perm.push(i);
        }
        // Dimensions after original axis
        for i in axis..(data_ndim - 1) {
            inv_perm.push(indices_rank + i);
        }

        reshaped.permute(inv_perm)
    };

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    Ok(())
}

/// Helper function for gathering along axis 0
fn gather_axis0(
    data: GraphTensor,
    indices: GraphTensor,
    data_dims: &[Expression],
) -> Result<GraphTensor, String> {
    let inner_dim: usize = data_dims[1..]
        .iter()
        .map(|d| {
            d.to_usize()
                .ok_or_else(|| "Gather: inner dimensions must be concrete".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .product();

    // Compute flat_indices = indices * inner_dim + offsets
    // IMPORTANT: Use Int arithmetic to avoid f32 precision loss for large indices.
    // f32 only has 24-bit mantissa, so indices > 16.7M lose precision.
    // For vocab_size=50304, embed_dim=768: max flat index = 38.6M
    let indices_int = indices.cast(DType::Int);
    let inner_dim_tensor = indices.graph().constant(inner_dim as i32).expand_rhs(indices_int.shape);
    let scaled = indices_int * inner_dim_tensor;
    let idx_ndim = indices.dims().len();
    let scaled_expanded = scaled.expand_dim(idx_ndim, inner_dim);

    // Create column offsets [0, 1, ..., inner_dim-1] and broadcast to indices shape
    // Keep as Int for precise arithmetic
    let offsets = data.graph().arange(inner_dim); // arange returns Int
    let idx_dims = indices.dims();
    let mut offsets_expanded = offsets;
    for i in (0..idx_dims.len()).rev() {
        offsets_expanded = offsets_expanded.expand_dim(0, idx_dims[i].clone());
    }

    let flat_indices = scaled_expanded + offsets_expanded; // Both Int, result is Int

    // Gather using flat indices into contiguous data buffer
    let gathered = data.gather(flat_indices);

    // Reshape from [..., inner_dim] to [..., D1, D2, ..., Dk]
    if data_dims.len() > 2 {
        let mut output_shape: Vec<usize> = idx_dims
            .iter()
            .map(|d| d.to_usize().expect("Gather: index dims must be concrete"))
            .collect();
        for d in &data_dims[1..] {
            output_shape.push(d.to_usize().expect("Gather: data dims must be concrete"));
        }
        let mut reshaped = gathered;
        reshaped.shape = ShapeTracker::new(output_shape);
        Ok(reshaped)
    } else {
        Ok(gathered)
    }
}

/// Handle Cast node: convert a tensor's data type.
///
/// Maps ONNX data type codes to luminal DType values. Since the internal
/// representation is f32, most casts are logical (bool→0.0/1.0, int→truncate).
/// Propagates known values with appropriate type conversion.
pub fn parse_cast_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Cast should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Cast: missing input tensor '{}'", node.input[0]))?;

    // ONNX data type enum → luminal DType
    let to = get_int_attr(node, "to", 1);
    let dtype = match to {
        1 => DType::F32,     // FLOAT
        10 => DType::F16,    // FLOAT16
        16 => DType::Bf16,   // BFLOAT16
        6 | 7 => DType::Int, // INT32, INT64
        9 => DType::F32,     // BOOL → treat as F32 (0.0/1.0)
        11 => DType::F32,    // DOUBLE → F32 (downcast)
        _ => DType::F32,     // fallback
    };

    let result = input.cast(dtype);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Propagate known values (cast is a no-op for our f32 storage)
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        let folded = if to == 9 {
            // Bool cast: non-zero → 1.0, zero → 0.0
            vals.iter()
                .map(|&v| if v != 0.0 { 1.0 } else { 0.0 })
                .collect()
        } else if to == 6 || to == 7 {
            // Int cast: truncate
            vals.iter().map(|&v| (v as i64) as f32).collect()
        } else {
            vals
        };
        known_values.insert(output_name.clone(), folded);
    }

    Ok(())
}

/// Handle Unsqueeze node: insert size-1 dimensions at the specified axes.
///
/// Axes are read from the second input (must be a known constant).
/// Negative axis values are resolved relative to the output rank.
pub fn parse_unsqueeze_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(
        node.input.len() == 2,
        "Unsqueeze should have exactly 2 inputs"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Unsqueeze: missing input tensor '{}'", node.input[0]))?;

    let axes_data = known_values
        .get(&node.input[1])
        .ok_or_else(|| format!("Unsqueeze: axes '{}' must be known", node.input[1]))?;

    let ndim = input.dims().len();
    let out_ndim = ndim + axes_data.len();
    let mut axes: Vec<usize> = axes_data
        .iter()
        .map(|&v| {
            let a = v as i64;
            if a < 0 {
                (out_ndim as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    axes.sort();

    let mut result = input;
    for &axis in &axes {
        result = result.unsqueeze(axis);
    }

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    Ok(())
}

/// Handle Concat node: concatenate multiple tensors along a given axis.
///
/// When all inputs have known values (common for shape manipulation), the
/// concatenation is constant-folded. Otherwise, uses luminal's `concat_along`.
pub fn parse_concat_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(
        !node.input.is_empty(),
        "Concat should have at least 1 input"
    );
    let output_name = &node.output[0];

    // If all inputs have known values, fully constant-fold (common for shape manipulation)
    let all_known = node
        .input
        .iter()
        .all(|name| known_values.contains_key(name));
    if all_known {
        let mut result: Vec<f32> = Vec::new();
        for name in &node.input {
            result.extend(known_values.get(name).unwrap());
        }
        let shape = vec![result.len().max(1)];
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), result.clone());
        weight_data.push((output_name.clone(), result));
        return Ok(());
    }

    let axis = get_int_attr(node, "axis", 0);

    // Build graph tensor via chained concat_along
    let first = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Concat: missing input tensor '{}'", node.input[0]))?;

    let ndim = first.dims().len();
    let resolved_axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    let mut result = first;
    for name in &node.input[1..] {
        let t = *tensors
            .get(name)
            .ok_or_else(|| format!("Concat: missing input tensor '{}'", name))?;
        result = result.concat_along(t, resolved_axis);
    }
    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Slice node: extract a sub-tensor using start/end indices along specified axes.
///
/// Inputs: data, starts, ends, [axes], [steps]. The starts, ends, axes, and steps
/// must be known constants. Only step=1 is currently supported. Constant input
/// tensors are folded at build time.
pub fn parse_slice_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() >= 3, "Slice should have at least 3 inputs");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Slice: missing input tensor '{}'", node.input[0]))?;

    let starts = known_values
        .get(&node.input[1])
        .ok_or_else(|| format!("Slice: starts '{}' must be known", node.input[1]))?
        .iter()
        .map(|&v| v as i64)
        .collect::<Vec<_>>();
    let ends = known_values
        .get(&node.input[2])
        .ok_or_else(|| format!("Slice: ends '{}' must be known", node.input[2]))?
        .iter()
        .map(|&v| v as i64)
        .collect::<Vec<_>>();

    let input_dims = input.dims();
    let ndim = input_dims.len();

    // Optional axes (default: 0, 1, 2, ...)
    let axes: Vec<usize> = if node.input.len() > 3 && !node.input[3].is_empty() {
        known_values
            .get(&node.input[3])
            .ok_or_else(|| format!("Slice: axes '{}' must be known", node.input[3]))?
            .iter()
            .map(|&v| {
                let a = v as i64;
                if a < 0 {
                    (ndim as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else {
        (0..starts.len()).collect()
    };

    // Optional steps (default: 1)
    let steps: Vec<i64> = if node.input.len() > 4 && !node.input[4].is_empty() {
        known_values
            .get(&node.input[4])
            .ok_or_else(|| format!("Slice: steps '{}' must be known", node.input[4]))?
            .iter()
            .map(|&v| v as i64)
            .collect()
    } else {
        vec![1i64; starts.len()]
    };

    // Build slice ranges for each axis
    let mut slice_ranges: Vec<(Expression, Expression)> = input_dims
        .iter()
        .map(|d| (Expression::from(0), d.clone()))
        .collect();

    for i in 0..axes.len() {
        let axis = axes[i];
        let dim_size = input_dims[axis]
            .to_usize()
            .ok_or_else(|| "Slice: dimension must be concrete".to_string())?
            as i64;

        let start = if starts[i] < 0 {
            (dim_size + starts[i]).max(0)
        } else {
            starts[i].min(dim_size)
        };
        let end = if ends[i] < 0 {
            (dim_size + ends[i]).max(0)
        } else if ends[i] > dim_size {
            dim_size
        } else {
            ends[i]
        };

        assert_eq!(steps[i], 1, "Slice: only step=1 is currently supported");
        slice_ranges[axis] = (Expression::from(start as i32), Expression::from(end as i32));
    }

    let output_name = &node.output[0];

    // If input is in known_values (constant), fully constant-fold to avoid graph operations
    if let Some(input_vals) = known_values.get(&node.input[0]).cloned() {
        if ndim == 1 && axes.len() == 1 {
            let start = slice_ranges[0].0.to_usize().unwrap_or(0);
            let end = slice_ranges[0].1.to_usize().unwrap_or(input_vals.len());
            let folded: Vec<f32> = input_vals[start..end].to_vec();
            // Use max(1) for shape to avoid 0-dim tensors in luminal
            let shape = vec![folded.len().max(1)];
            // Ensure weight_data matches shape (pad with 0 if empty)
            let wd = if folded.is_empty() {
                vec![0.0]
            } else {
                folded.clone()
            };
            let tensor = cx.named_tensor(output_name.clone(), shape);
            tensors.insert(output_name.clone(), tensor);
            known_values.insert(output_name.clone(), folded);
            weight_data.push((output_name.clone(), wd));
            return Ok(());
        }
    }

    let result = input.slice(slice_ranges);
    tensors.insert(output_name.clone(), result);

    Ok(())
}

/// Handle Reshape node: change the tensor's shape without modifying data.
///
/// The target shape is read from the second input (must be a known constant).
/// Supports -1 (infer from total elements) and 0 (copy from input dimension).
/// Non-contiguous tensors (e.g., from Expand) are materialized before reshaping.
pub fn parse_reshape_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(
        node.input.len() == 2,
        "Reshape should have exactly 2 inputs"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Reshape: missing input tensor '{}'", node.input[0]))?;

    let shape_data = known_values.get(&node.input[1]).ok_or_else(|| {
        format!(
            "Reshape: shape input '{}' must be a known constant",
            node.input[1]
        )
    })?;

    // Compute total elements for resolving -1
    let input_dims = input.dims();
    let total_elements: usize = input_dims
        .iter()
        .map(|d| d.to_usize().expect("Reshape: input dims must be concrete"))
        .product();

    // Resolve target shape, handling -1 (infer) and 0 (copy from input)
    let mut target_shape: Vec<i64> = shape_data.iter().map(|&v| v as i64).collect();
    // First pass: resolve 0 (copy from input at same position)
    for i in 0..target_shape.len() {
        if target_shape[i] == 0 {
            target_shape[i] = input_dims[i].to_usize().unwrap_or(1) as i64;
        }
    }
    // Second pass: resolve -1 (infer from total elements)
    let known_product: i64 = target_shape.iter().filter(|&&d| d > 0).product();
    for d in target_shape.iter_mut() {
        if *d == -1 {
            *d = total_elements as i64 / known_product;
        }
    }
    let final_shape: Vec<usize> = target_shape.iter().map(|&d| d as usize).collect();

    let mut result = input;
    // If tensor is not contiguous (e.g., has broadcast strides from Expand),
    // materialize it before reshaping by multiplying by 1.0.
    // This forces a contiguous copy through the binary op mechanism.
    if !result.shape.is_contiguous() {
        let one = result.graph().constant_float(1.0);
        let src_dims = result.dims();
        let broadcast_shape: Vec<usize> = src_dims
            .iter()
            .map(|d| d.to_usize().expect("dim must be concrete"))
            .collect();
        let one_expanded = broadcast_to(one, &broadcast_shape);
        result = result * one_expanded;
    }
    result.shape = ShapeTracker::new(final_shape);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    // Propagate known values (reshape doesn't change data, just layout)
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    Ok(())
}

/// Handle Shape node: extract the shape of the input tensor as a 1D constant.
///
/// All dimensions must be statically known. The shape values are stored as
/// known constants for downstream operations (Reshape, Expand, etc.).
pub fn parse_shape_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Shape should have exactly 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Shape: missing input tensor '{}'", node.input[0]))?;

    let dims = input.dims();
    let shape_values: Vec<f32> = dims
        .iter()
        .map(|d| {
            d.to_usize()
                .expect("Shape: all dimensions must be concrete") as f32
        })
        .collect();

    let output_name = &node.output[0];
    let tensor = cx.named_tensor(output_name.clone(), vec![shape_values.len()]);
    tensors.insert(output_name.clone(), tensor);
    known_values.insert(output_name.clone(), shape_values.clone());
    weight_data.push((output_name.clone(), shape_values));
    Ok(())
}

/// Handle Mod node: element-wise modulo with numpy-style broadcasting.
///
/// Supports constant folding when both inputs have known values.
pub fn parse_mod_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "Mod should have exactly 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Mod: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("Mod: missing input tensor '{}'", node.input[1]))?;

    let broadcast_shape = compute_broadcast_shape(&a.dims(), &b.dims());
    let a_bc = broadcast_to(a, &broadcast_shape);
    let b_bc = broadcast_to(b, &broadcast_shape);
    let result = a_bc % b_bc;
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let (Some(va), Some(vb)) = (
        known_values.get(&node.input[0]).cloned(),
        known_values.get(&node.input[1]).cloned(),
    ) {
        let folded = broadcast_binop(&va, &vb, |a, b| a % b);
        known_values.insert(output_name.clone(), folded);
    }
    Ok(())
}

/// Handle MatMul node: standard matrix multiplication of two tensors.
///
/// Unlike Gemm, MatMul does not support transposing inputs or adding bias.
pub fn parse_matmul_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(node.input.len() == 2, "MatMul should have exactly 2 inputs");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("MatMul: missing input tensor '{}'", node.input[0]))?;
    let b = *tensors
        .get(&node.input[1])
        .ok_or_else(|| format!("MatMul: missing input tensor '{}'", node.input[1]))?;

    let result = a.matmul(b);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Transpose node: permute the dimensions of a tensor.
///
/// If no `perm` attribute is specified, reverses all dimensions (default ONNX behavior).
pub fn parse_transpose_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
) -> Result<(), String> {
    assert!(
        node.input.len() == 1,
        "Transpose should have exactly 1 input"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Transpose: missing input tensor '{}'", node.input[0]))?;

    // Get perm attribute; if not specified, reverse all dimensions
    let perm: Vec<usize> = node
        .attribute
        .iter()
        .find(|a| a.name == "perm")
        .map(|a| a.ints.iter().map(|&v| v as usize).collect())
        .unwrap_or_else(|| {
            let ndim = input.dims().len();
            (0..ndim).rev().collect()
        });

    let result = input.permute(perm);
    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);
    Ok(())
}

/// Handle Squeeze node: remove size-1 dimensions from a tensor.
///
/// Opset 13+: axes come from the second input. Opset 11: from the "axes" attribute.
/// If no axes are specified, all dimensions of size 1 are removed.
/// Axes are processed in reverse order to avoid index shifting.
pub fn parse_squeeze_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(
        !node.input.is_empty(),
        "Squeeze should have at least 1 input"
    );
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Squeeze: missing input tensor '{}'", node.input[0]))?;

    // Opset 13+: axes come from second input; opset 11: from attribute
    let axes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
        let axes_data = known_values
            .get(&node.input[1])
            .ok_or_else(|| format!("Squeeze: axes '{}' must be known", node.input[1]))?;
        let ndim = input.dims().len();
        axes_data
            .iter()
            .map(|&v| {
                let a = v as i64;
                if a < 0 {
                    (ndim as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect()
    } else if let Some(attr) = node.attribute.iter().find(|a| a.name == "axes") {
        let ndim = input.dims().len();
        attr.ints
            .iter()
            .map(|&v| {
                if v < 0 {
                    (ndim as i64 + v) as usize
                } else {
                    v as usize
                }
            })
            .collect()
    } else {
        // No axes specified: squeeze all dims of size 1
        input
            .dims()
            .iter()
            .enumerate()
            .filter(|(_, d)| d.to_usize() == Some(1))
            .map(|(i, _)| i)
            .collect()
    };

    // Sort in reverse order so removing earlier axes doesn't shift later ones
    let mut sorted_axes = axes.clone();
    sorted_axes.sort();
    sorted_axes.reverse();

    let mut result = input;
    for &axis in &sorted_axes {
        result = result.squeeze(axis);
    }

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), result);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }
    Ok(())
}

/// Handle Identity node: output is a direct alias of the input tensor.
///
/// Propagates known constant values for downstream constant folding.
pub fn parse_identity(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(node.input.len() == 1, "Identity should only have one input");
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Identity: missing input tensor '{}'", node.input[0]))?;

    assert!(
        node.output.len() == 1,
        "Identity should only have a single output"
    );

    let output_name = &node.output[0];
    // Direct alias: the output tensor IS the input tensor
    tensors.insert(output_name.clone(), a);

    // Propagate known values
    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }

    Ok(())
}

/// Handle Dropout node: a no-op during inference (output = input).
///
/// Dropout is only active during training; at inference time it passes
/// the input through unchanged.
pub fn parse_dropout_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    // Dropout is a no-op during inference: output = input
    let a = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Dropout: missing input tensor '{}'", node.input[0]))?;

    let output_name = &node.output[0];
    tensors.insert(output_name.clone(), a);

    if let Some(vals) = known_values.get(&node.input[0]).cloned() {
        known_values.insert(output_name.clone(), vals);
    }

    Ok(())
}

/// Handle Trilu node: extract the upper or lower triangular part of a matrix.
///
/// `upper=1` (default) keeps the upper triangle; `upper=0` keeps the lower triangle.
/// The diagonal offset `k` shifts which diagonal is included (0 = main diagonal).
/// Constant inputs are folded at build time; dynamic inputs require square matrices.
pub fn parse_trilu_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    cx: &mut Graph,
    weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(!node.input.is_empty(), "Trilu should have at least 1 input");
    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Trilu: missing input tensor '{}'", node.input[0]))?;

    let upper = get_int_attr(node, "upper", 1);

    // Get diagonal offset k (default 0)
    let k: i32 = if node.input.len() > 1 && !node.input[1].is_empty() {
        let k_values = known_values.get(&node.input[1]).ok_or_else(|| {
            format!(
                "Trilu: k input '{}' must be a known constant",
                node.input[1]
            )
        })?;
        k_values[0] as i32
    } else {
        0
    };

    let input_dims = input.dims();
    assert!(input_dims.len() >= 2, "Trilu: input must be at least 2D");
    let rows = input_dims[input_dims.len() - 2]
        .to_usize()
        .ok_or_else(|| "Trilu: second-to-last dimension must be concrete".to_string())?;
    let cols = input_dims[input_dims.len() - 1]
        .to_usize()
        .ok_or_else(|| "Trilu: last dimension must be concrete".to_string())?;

    let output_name = &node.output[0];

    // If input has known values, constant-fold the entire operation
    if let Some(input_vals) = known_values.get(&node.input[0]).cloned() {
        let mut result = vec![0.0f32; input_vals.len()];
        let matrix_size = rows * cols;
        let num_matrices = input_vals.len() / matrix_size;
        for m in 0..num_matrices {
            for i in 0..rows {
                for j in 0..cols {
                    let idx = m * matrix_size + i * cols + j;
                    let keep = if upper == 1 {
                        (j as i32 - i as i32) >= k
                    } else {
                        (j as i32 - i as i32) <= k
                    };
                    result[idx] = if keep { input_vals[idx] } else { 0.0 };
                }
            }
        }
        let shape: Vec<usize> = input_dims
            .iter()
            .map(|d| {
                d.to_usize()
                    .expect("Trilu: all dims must be concrete for constant folding")
            })
            .collect();
        let tensor = cx.named_tensor(output_name.clone(), shape);
        tensors.insert(output_name.clone(), tensor);
        known_values.insert(output_name.clone(), result.clone());
        weight_data.push((output_name.clone(), result));
    } else {
        // Dynamic case: generate mask and multiply
        assert_eq!(
            rows, cols,
            "Trilu: luminal's triu/tril requires square matrices"
        );
        let mask = if upper == 1 {
            cx.triu(rows, k)
        } else {
            cx.tril(rows, k)
        };
        // Broadcast mask to match input dimensions (handles batched inputs)
        let broadcast_shape = compute_broadcast_shape(&input.dims(), &mask.dims());
        let mask_bc = broadcast_to(mask, &broadcast_shape);
        let result = input.mul(mask_bc);
        tensors.insert(output_name.clone(), result);
    }

    Ok(())
}

/// Handle Split node: split a tensor into multiple output tensors along an axis.
/// Inputs: input, [split sizes]
/// Attributes: axis (default 0), num_outputs (for equal division)
/// Outputs: Multiple tensors, one for each split chunk
pub fn parse_split_node(
    node: &NodeProto,
    tensors: &mut HashMap<String, GraphTensor>,
    _cx: &mut Graph,
    _weight_data: &mut Vec<(String, Vec<f32>)>,
    known_values: &mut HashMap<String, Vec<f32>>,
) -> Result<(), String> {
    assert!(
        !node.input.is_empty(),
        "Split should have at least 1 input"
    );

    let input = *tensors
        .get(&node.input[0])
        .ok_or_else(|| format!("Split: missing input tensor '{}'", node.input[0]))?;

    // Get axis attribute (default 0, handle negative indices)
    let axis = get_int_attr(node, "axis", 0);
    let input_dims = input.dims();
    let ndim = input_dims.len();
    let resolved_axis = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };

    // Get dimension size at split axis
    let dim_size = input_dims[resolved_axis]
        .to_usize()
        .ok_or_else(|| "Split: dimension must be concrete".to_string())?;

    // Determine split sizes
    let split_sizes: Vec<usize> = if node.input.len() > 1 && !node.input[1].is_empty() {
        // Split sizes from input tensor (must be known constant)
        known_values
            .get(&node.input[1])
            .ok_or_else(|| format!("Split: split sizes '{}' must be known", node.input[1]))?
            .iter()
            .map(|&v| v as usize)
            .collect()
    } else {
        // Use num_outputs attribute for equal division
        let num_outputs = get_int_attr(node, "num_outputs", node.output.len() as i64) as usize;
        let base_size = dim_size / num_outputs;
        let remainder = dim_size % num_outputs;
        // Last chunk may be smaller if not evenly divisible
        (0..num_outputs)
            .map(|i| {
                if i == num_outputs - 1 {
                    base_size + remainder
                } else {
                    base_size
                }
            })
            .collect()
    };

    // Validate split sizes sum to dimension size
    let total: usize = split_sizes.iter().sum();
    if total != dim_size {
        return Err(format!(
            "Split: sum of split sizes ({}) != dimension size ({})",
            total, dim_size
        ));
    }

    // Create output tensors using slice operations
    let mut offset = 0usize;
    for (i, &size) in split_sizes.iter().enumerate() {
        let output_name = &node.output[i];

        // Build slice ranges (full range for all dims except split axis)
        let slice_ranges: Vec<(Expression, Expression)> = input_dims
            .iter()
            .enumerate()
            .map(|(dim_idx, d)| {
                if dim_idx == resolved_axis {
                    (
                        Expression::from(offset as i32),
                        Expression::from((offset + size) as i32),
                    )
                } else {
                    (Expression::from(0), d.clone())
                }
            })
            .collect();

        let result = input.slice(slice_ranges);
        tensors.insert(output_name.clone(), result);

        // Handle constant folding for known inputs (1D case)
        if let Some(input_vals) = known_values.get(&node.input[0]).cloned() {
            if ndim == 1 {
                let folded: Vec<f32> = input_vals[offset..offset + size].to_vec();
                known_values.insert(output_name.clone(), folded);
            }
        }

        offset += size;
    }

    Ok(())
}
