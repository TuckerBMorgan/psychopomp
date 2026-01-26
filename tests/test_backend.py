"""Tests for the psychopomp torch.compile backend."""

import psychopomp
import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.linear_2 = nn.Linear(10, 10)

    def forward(self, x):
        first_output = self.linear(x)
        second_output = x + first_output
        return self.linear_2(second_output)


class MulDivModel(nn.Module):
    """Element-wise Mul and Div with broadcasting."""

    def __init__(self, features):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(features))
        self.divisor = nn.Parameter(torch.rand(1) + 0.5)

    def forward(self, x):
        scaled = x * self.scale
        return scaled / self.divisor


class SqrtDivModel(nn.Module):
    """Sqrt and Div in a chain (ensures positive input to sqrt)."""

    def __init__(self, features):
        super().__init__()
        self.offset = nn.Parameter(torch.rand(features) + 0.1)

    def forward(self, x):
        x_pos = x * x + self.offset
        root = torch.sqrt(x_pos)
        return x / root


def check(name, result, expected, tol):
    """Compare result vs expected and assert within tolerance."""
    diff = (result - expected).abs().max().item()
    mean_diff = (result - expected).abs().mean().item()
    print(f"  Output shape: {result.shape} (expected {expected.shape})")
    print(f"  Max absolute difference: {diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    assert result.shape == expected.shape, (
        f"Shape mismatch: {result.shape} vs {expected.shape}"
    )
    assert diff < tol, f"{name}: difference {diff:.2e} exceeds tolerance {tol}"
    print("  PASSED")


def test_simple_linear():
    """Test Mul and Div with broadcasting."""
    print("=== Test: Elementwise Mul/Div ===")
    model = SimpleModel()
    model.eval()
    x = torch.randn(2, 10)
    compiled = torch.compile(model, backend="psychopomp")
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_simple_linear", result, expected, tol=1e-5)


def test_elementwise_mul_div():
    """Test Mul and Div with broadcasting."""
    print("=== Test: Elementwise Mul/Div ===")
    model = MulDivModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend="psychopomp")
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_elementwise_mul_div", result, expected, tol=1e-5)


def test_sqrt_div_model():
    """Test Sqrt and Div in a chain."""
    print("=== Test: Sqrt/Div ===")
    model = SqrtDivModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend="psychopomp")
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_sqrt_div_model", result, expected, tol=1e-5)


def main():
    tests = [
        test_simple_linear,
        test_elementwise_mul_div,
        test_sqrt_div_model,
    ]
    """
    test_softmax_standalone,
    test_layer_norm_standalone,
    test_erf_standalone,
    test_gelu_activation,
    test_concat_model,
    test_slice_model,
    test_concat_slice_roundtrip,
    test_residual_block,
    test_matmul_transpose_attention,
    test_multi_layer_mlp,
    """

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn, str(e)))
            print(f"  FAILED: {e}")
        print()

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
