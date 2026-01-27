"""Tests for the psychopomp torch.compile backend."""

import os

import psychopomp
import torch
from nanogpt import *
from torch import nn

# Backend selection via environment variable
# Default: "psychopomp" (native), can be set to "psychopomp_cuda" for CUDA testing
BACKEND = os.environ.get("PSYCHOPOMP_BACKEND", "psychopomp")


class BasicTransformerLM(nn.Module):
    """
    Minimal encoder-style Transformer for language-model-ish next-token logits.
    Input:  (B, T) token ids
    Output: (B, T, vocab_size) logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, T, C)
            norm_first=True,  # pre-norm = a bit more stable
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        x: LongTensor (B, T)
        """
        B, T = x.shape
        device = x.device

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)  # (B, T, C)
        return h
        # Padding mask: True where tokens should be ignored
        key_padding_mask = x == self.pad_id  # (B, T)

        # Causal mask so each position can't see the future (LM-style)
        # True/inf above diagonal => disallow attention
        causal_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

        h = self.encoder(
            h,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        logits = self.lm_head(h)  # (B, T, vocab)
        return logits


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


class SoftmaxModel(nn.Module):
    """Softmax on the last dimension."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


class LayerNormModel(nn.Module):
    """LayerNorm applied to linear output."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        return self.norm(self.linear(x))


class ErfModel(nn.Module):
    """torch.erf applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.erf(self.linear(x))


class GeluModel(nn.Module):
    """GELU activation (decomposes to Erf ops in ONNX)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.linear(x))


class ConcatModel(nn.Module):
    """Two parallel branches concatenated along feature dim."""

    def __init__(self, in_features, hidden):
        super().__init__()
        self.branch_a = nn.Linear(in_features, hidden)
        self.branch_b = nn.Linear(in_features, hidden)

    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        return torch.cat([a, b], dim=-1)


class SliceModel(nn.Module):
    """Slices first half of features, then applies linear."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.half = in_features // 2
        self.linear = nn.Linear(self.half, out_features)

    def forward(self, x):
        sliced = x[:, : self.half]
        return self.linear(sliced)


class ConcatSliceModel(nn.Module):
    """Concat two branches then slice back to original hidden size."""

    def __init__(self, features, hidden):
        super().__init__()
        self.hidden = hidden
        self.proj_a = nn.Linear(features, hidden)
        self.proj_b = nn.Linear(features, hidden)
        self.final_proj = nn.Linear(hidden, features)

    def forward(self, x):
        a = self.proj_a(x)
        b = self.proj_b(x)
        combined = torch.cat([a, b], dim=-1)
        sliced = combined[:, : self.hidden]
        return self.final_proj(sliced)


class ResidualBlock(nn.Module):
    """Residual Add + LayerNorm pattern."""

    def __init__(self, features):
        super().__init__()
        self.linear1 = nn.Linear(features, features)
        self.linear2 = nn.Linear(features, features)
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = x + residual
        x = self.norm(x)
        return self.linear2(x)


class SimpleAttention(nn.Module):
    """Single-head attention: Q@K^T/sqrt(d), softmax, @V."""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model**0.5

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        k_t = k.transpose(-2, -1)
        scores = torch.matmul(q, k_t) / self.scale
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class MultiLayerMLP(nn.Module):
    """3-layer MLP with GELU activations."""

    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_features)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


# --- New ONNX Op Test Models ---


class TanhModel(nn.Module):
    """Tanh activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class CosModel(nn.Module):
    """Cos activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.cos(self.linear(x))


class SinModel(nn.Module):
    """Sin activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sin(self.linear(x))


class PowModel(nn.Module):
    """Pow operation: base raised to exponent power."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # Use abs to avoid NaN from negative bases with fractional exponents
        base = torch.abs(self.linear(x)) + 0.1
        return torch.pow(base, 2.0)


class ReduceMeanModel(nn.Module):
    """ReduceMean along last dimension."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return torch.mean(x, dim=-1, keepdim=True)


class NegModel(nn.Module):
    """Negation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.neg(self.linear(x))


class SigmoidModel(nn.Module):
    """Sigmoid activation applied to linear output."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class SplitEqualModel(nn.Module):
    """Split tensor into equal parts along feature dimension."""

    def __init__(self, in_features, num_splits):
        super().__init__()
        self.num_splits = num_splits
        self.linear = nn.Linear(in_features, in_features)

    def forward(self, x):
        x = self.linear(x)
        splits = torch.split(x, x.shape[-1] // self.num_splits, dim=-1)
        return splits[0] + splits[1]


class SplitUnequalModel(nn.Module):
    """Split tensor into unequal parts."""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)

    def forward(self, x):
        x = self.linear(x)
        # Split into [4, 8, 4] for 16 features
        splits = torch.split(x, [4, 8, 4], dim=-1)
        return splits[1]  # Return middle chunk


class GatherEmbeddingModel(nn.Module):
    """Gather elements using indices (embedding lookup pattern)."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, indices):
        return self.embedding(indices)


class ExpandBroadcastModel(nn.Module):
    """Expand tensor via broadcasting addition."""

    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1, features))

    def forward(self, x):
        # x: (batch, seq, features), bias: (1, features)
        # This triggers Expand to broadcast bias
        return x + self.bias


class ReshapeViewModel(nn.Module):
    """Reshape tensor to different dimensions via view."""

    def __init__(self, in_features, hidden):
        super().__init__()
        self.hidden = hidden
        self.linear = nn.Linear(in_features, hidden * 4)

    def forward(self, x):
        batch = x.shape[0]
        x = self.linear(x)
        return x.view(batch, 4, self.hidden)


class ReshapeFlattenModel(nn.Module):
    """Flatten tensor dimensions using view."""

    def __init__(self, seq_len, features):
        super().__init__()
        self.seq_len = seq_len
        self.features = features
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        # x: (batch, seq, features) -> (batch, seq * features)
        batch = x.shape[0]
        x = self.linear(x)
        return x.view(batch, self.seq_len * self.features)


class UnsqueezeModel(nn.Module):
    """Add dimension via unsqueeze."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear(x)
        # (batch, features) -> (batch, 1, features)
        return x.unsqueeze(1)


class SqueezeModel(nn.Module):
    """Remove dimension via squeeze."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear(x)
        # (batch, 1, features) -> (batch, features)
        return x.squeeze(1)


class EqualModel(nn.Module):
    """Equality comparison."""

    def __init__(self):
        super().__init__()
        self.register_buffer("target", torch.tensor([0.0]))

    def forward(self, x):
        return (x == self.target).float()


class ModModel(nn.Module):
    """Element-wise modulo operation."""

    def __init__(self, divisor):
        super().__init__()
        self.register_buffer("divisor", torch.tensor([float(divisor)]))

    def forward(self, x):
        return torch.fmod(x, self.divisor)


class CastModel(nn.Module):
    """Type casting operations (float -> int -> float)."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, x):
        x = self.linear(x)
        x_int = x.int()
        return x_int.float()


# --- Trilu (Triangular) Models ---


class TriluUpperModel(nn.Module):
    """Extract upper triangular part of matrix."""

    def __init__(self, size):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        x = self.linear(x)
        # Reshape to square matrix for triu
        batch = x.shape[0]
        x = x.view(batch, int(self.size**0.5), int(self.size**0.5))
        return torch.triu(x)


class TriluLowerModel(nn.Module):
    """Extract lower triangular part of matrix."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return torch.tril(x)


class TriluDiagonalOffsetModel(nn.Module):
    """Trilu with diagonal offset k."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # k=1 means one diagonal above main
        return torch.triu(x, diagonal=1)


# --- Where (Conditional) Models ---


class WhereEqualModel(nn.Module):
    """Where with Equal condition (selecting based on zero elements)."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.register_buffer("zero", torch.tensor([0.0]))

    def forward(self, x):
        # Condition: where x equals 0, use replacement value
        condition = x == self.zero
        replacement = torch.ones_like(x) * -1.0
        return torch.where(condition, replacement, x)


class WhereBroadcastModel(nn.Module):
    """Where with broadcasting boolean mask."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        # Create a mask that broadcasts
        self.register_buffer("mask", torch.tensor([True, False] * (features // 2)))
        # Alternative value (not using Neg since it's unsupported)
        self.register_buffer("alt_value", torch.zeros(features))

    def forward(self, x):
        # mask: (features,) broadcasts to (batch, features)
        return torch.where(self.mask, x, self.alt_value)


class WhereTwoTensorModel(nn.Module):
    """Where selecting between two tensors based on Equal condition."""

    def __init__(self, features):
        super().__init__()
        self.linear_a = nn.Linear(features, features)
        self.linear_b = nn.Linear(features, features)
        self.register_buffer("zero", torch.tensor([0.0]))

    def forward(self, x):
        a = self.linear_a(x)
        b = self.linear_b(x)
        # Condition based on Equal
        condition = x == self.zero
        return torch.where(condition, a, b)


# --- ConstantOfShape Models ---


class ConstantOfShapeZerosModel(nn.Module):
    """Creates zeros tensor with computed shape."""

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        batch = x.shape[0]
        # Create zeros and add to input (tests ConstantOfShape with value=0)
        zeros = torch.zeros(batch, self.features, device=x.device, dtype=x.dtype)
        return x + zeros


class ConstantOfShapeOnesModel(nn.Module):
    """Creates ones tensor and multiplies."""

    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        batch = x.shape[0]
        # Create ones and multiply (tests ConstantOfShape with value=1)
        ones = torch.ones(batch, self.features, device=x.device, dtype=x.dtype)
        return x * ones


class ConstantOfShapeFullModel(nn.Module):
    """Creates tensor filled with arbitrary constant."""

    def __init__(self, features, fill_value):
        super().__init__()
        self.features = features
        self.fill_value = fill_value

    def forward(self, x):
        batch = x.shape[0]
        # Create filled tensor (tests ConstantOfShape with custom value)
        filled = torch.full(
            (batch, self.features), self.fill_value, device=x.device, dtype=x.dtype
        )
        return x + filled


# --- Shape-based Model ---


class ShapeBasedReshapeModel(nn.Module):
    """Model that uses tensor shape for reshaping."""

    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features * 2)

    def forward(self, x):
        # Uses shape internally: reshape to (batch, 2, features)
        batch_size = x.shape[0]
        x = self.linear(x)
        return x.view(batch_size, 2, -1)


# --- Gather Edge Case Models ---


class GatherLargeVocabModel(nn.Module):
    """Gather with large vocabulary (tests index precision)."""

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, indices):
        return self.embedding(indices)


# --- Expand Edge Case Models ---


class ExpandMultiDimModel(nn.Module):
    """Expand tensor across multiple dimensions."""

    def __init__(self, features):
        super().__init__()
        # Scalar-like parameter that expands to full tensor
        self.scale = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # scale: (1,) expands to x's shape (batch, seq, features)
        return x * self.scale


class ExpandPrependDimsModel(nn.Module):
    """Expand by prepending dimensions."""

    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(features))

    def forward(self, x):
        # bias: (features,) needs batch and seq dims prepended
        # x: (batch, seq, features)
        return x + self.bias


# --- Split Edge Case Models ---


class SplitBatchDimModel(nn.Module):
    """Split along axis 0 (batch dimension)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split batch in half
        splits = torch.split(x, x.shape[0] // 2, dim=0)
        return splits[0] + splits[1]


class SplitMultipleChunksModel(nn.Module):
    """Split into 3 chunks and concatenate first two."""

    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
        # in_features should be divisible by 3

    def forward(self, x):
        x = self.linear(x)
        # Split into 3 equal chunks
        chunk_size = x.shape[-1] // 3
        chunks = torch.split(x, chunk_size, dim=-1)
        # Concatenate first two chunks
        return torch.cat([chunks[0], chunks[1]], dim=-1)


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
    compiled = torch.compile(model, backend=BACKEND)
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
    compiled = torch.compile(model, backend=BACKEND)
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
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_sqrt_div_model", result, expected, tol=1e-5)


def test_softmax_standalone():
    """Test Softmax applied to linear output."""
    print("=== Test: Softmax ===")
    model = SoftmaxModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_softmax_standalone", result, expected, tol=1e-5)


def test_layer_norm_standalone():
    """Test LayerNorm in isolation."""
    print("=== Test: LayerNorm ===")
    model = LayerNormModel(features=32)
    model.eval()
    x = torch.randn(4, 32)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_layer_norm_standalone", result, expected, tol=1e-4)


def test_erf_standalone():
    """Test torch.erf (approximated in Rust)."""
    print("=== Test: Erf ===")
    model = ErfModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_erf_standalone", result, expected, tol=5e-3)


def test_gelu_activation():
    """Test GELU activation (Erf-based decomposition in ONNX)."""
    print("=== Test: GELU ===")
    model = GeluModel(in_features=16, out_features=32)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_gelu_activation", result, expected, tol=5e-3)


def test_concat_model():
    """Test Concat along feature dimension."""
    print("=== Test: Concat ===")
    model = ConcatModel(in_features=16, hidden=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_concat_model", result, expected, tol=1e-5)


def test_slice_model():
    """Test Slice on feature dimension."""
    print("=== Test: Slice ===")
    model = SliceModel(in_features=16, out_features=6)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_slice_model", result, expected, tol=1e-5)


def test_concat_slice_roundtrip():
    """Test Concat followed by Slice."""
    print("=== Test: Concat+Slice ===")
    model = ConcatSliceModel(features=16, hidden=12)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_concat_slice_roundtrip", result, expected, tol=1e-5)


def test_residual_block():
    """Test residual Add + LayerNorm pattern."""
    print("=== Test: Residual Block ===")
    model = ResidualBlock(features=32)
    model.eval()
    x = torch.randn(4, 32)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_residual_block", result, expected, tol=1e-4)


def test_matmul_transpose_attention():
    """Test MatMul, Transpose, Div, Softmax via single-head attention."""
    print("=== Test: Attention (MatMul+Transpose+Softmax) ===")
    model = SimpleAttention(d_model=32)
    model.eval()
    x = torch.randn(2, 8, 32)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_matmul_transpose_attention", result, expected, tol=1e-4)


def test_multi_layer_mlp():
    """Test 3-layer MLP with GELU activations."""
    print("=== Test: Multi-layer MLP ===")
    model = MultiLayerMLP(in_features=16, hidden=32, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_multi_layer_mlp", result, expected, tol=5e-3)


def test_simple_transfomer():
    """Test Simple transformer."""
    print("=== Test: Simple Transformer ===")
    vocab_size = 10_000
    model = BasicTransformerLM(
        vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, max_len=256
    )
    model.eval()
    x = torch.randint(0, vocab_size, (2, 32))  # (batch=2, seq=32)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_simple_transformer", result, expected, tol=1e-5)


# --- New ONNX Op Tests ---


def test_tanh_standalone():
    """Test torch.tanh activation."""
    print("=== Test: Tanh ===")
    model = TanhModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_tanh_standalone", result, expected, tol=1e-4)


def test_cos_standalone():
    """Test torch.cos function."""
    print("=== Test: Cos ===")
    model = CosModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_cos_standalone", result, expected, tol=1e-4)


def test_sin_standalone():
    """Test torch.sin function."""
    print("=== Test: Sin ===")
    model = SinModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_sin_standalone", result, expected, tol=1e-4)


def test_pow_standalone():
    """Test torch.pow function."""
    print("=== Test: Pow ===")
    model = PowModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_pow_standalone", result, expected, tol=1e-4)


def test_reduce_mean_standalone():
    """Test torch.mean reduction."""
    print("=== Test: ReduceMean ===")
    model = ReduceMeanModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_reduce_mean_standalone", result, expected, tol=1e-4)


def test_neg_standalone():
    """Test torch.neg function."""
    print("=== Test: Neg ===")
    model = NegModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_neg_standalone", result, expected, tol=1e-5)


def test_sigmoid_standalone():
    """Test torch.sigmoid function."""
    print("=== Test: Sigmoid ===")
    model = SigmoidModel(in_features=16, out_features=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_sigmoid_standalone", result, expected, tol=1e-4)


def test_split_equal():
    """Test Split with equal division."""
    print("=== Test: Split (Equal) ===")
    model = SplitEqualModel(in_features=16, num_splits=4)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_split_equal", result, expected, tol=1e-5)


def test_split_unequal():
    """Test Split with unequal split sizes."""
    print("=== Test: Split (Unequal) ===")
    model = SplitUnequalModel(in_features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_split_unequal", result, expected, tol=1e-5)


def test_gather_embedding():
    """Test Gather via embedding lookup (axis=0)."""
    print("=== Test: Gather (Embedding) ===")
    model = GatherEmbeddingModel(num_embeddings=100, embedding_dim=32)
    model.eval()
    indices = torch.randint(0, 100, (4, 8))
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(indices)
    with torch.no_grad():
        expected = model(indices)
    check("test_gather_embedding", result, expected, tol=1e-5)


def test_expand_broadcast():
    """Test Expand via broadcasting."""
    print("=== Test: Expand (Broadcast) ===")
    model = ExpandBroadcastModel(features=32)
    model.eval()
    x = torch.randn(4, 8, 32)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_expand_broadcast", result, expected, tol=1e-5)


def test_reshape_view():
    """Test Reshape via view operation."""
    print("=== Test: Reshape (View) ===")
    model = ReshapeViewModel(in_features=16, hidden=8)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_reshape_view", result, expected, tol=1e-5)


def test_reshape_flatten():
    """Test Reshape via view to flatten."""
    print("=== Test: Reshape (Flatten) ===")
    model = ReshapeFlattenModel(seq_len=8, features=16)
    model.eval()
    x = torch.randn(4, 8, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_reshape_flatten", result, expected, tol=1e-5)


def test_unsqueeze():
    """Test Unsqueeze dimension insertion."""
    print("=== Test: Unsqueeze ===")
    model = UnsqueezeModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_unsqueeze", result, expected, tol=1e-5)


def test_squeeze():
    """Test Squeeze dimension removal."""
    print("=== Test: Squeeze ===")
    model = SqueezeModel(features=16)
    model.eval()
    x = torch.randn(4, 1, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_squeeze", result, expected, tol=1e-5)


def test_equal():
    """Test Equal comparison."""
    print("=== Test: Equal ===")
    model = EqualModel()
    model.eval()
    x = torch.tensor([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0]])
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_equal", result, expected, tol=1e-5)


def test_mod():
    """Test Mod element-wise modulo."""
    print("=== Test: Mod ===")
    model = ModModel(divisor=3.0)
    model.eval()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_mod", result, expected, tol=1e-5)


def test_cast():
    """Test Cast type conversion."""
    print("=== Test: Cast ===")
    model = CastModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_cast", result, expected, tol=1e-5)


def test_llama():
    """Test Llama."""
    print("=== Test: LLama ===")
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128256,  # Llama 3 tokenizer vocab
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA (grouped-query attention)
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
    )

    model = LlamaForCausalLM(config)
    model.eval()
    batch_size = 2
    seq_len = 128

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_llama", result, expected, tol=1e-5)


# --- Trilu Tests ---


def test_trilu_upper():
    """Test Trilu upper triangular extraction."""
    print("=== Test: Trilu (Upper) ===")
    model = TriluUpperModel(size=16)
    model.eval()
    x = torch.randn(2, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_trilu_upper", result, expected, tol=1e-5)


def test_trilu_lower():
    """Test Trilu lower triangular extraction."""
    print("=== Test: Trilu (Lower) ===")
    model = TriluLowerModel(size=8)
    model.eval()
    x = torch.randn(4, 8, 8)  # 3D with square last two dims
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_trilu_lower", result, expected, tol=1e-5)


def test_trilu_diagonal_offset():
    """Test Trilu with diagonal offset."""
    print("=== Test: Trilu (Diagonal Offset) ===")
    model = TriluDiagonalOffsetModel()
    model.eval()
    x = torch.randn(2, 6, 6)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_trilu_diagonal_offset", result, expected, tol=1e-5)


# --- Where Tests ---


def test_where_equal():
    """Test Where with Equal condition."""
    print("=== Test: Where (Equal) ===")
    model = WhereEqualModel(features=16)
    model.eval()
    # Use tensor with some zeros to trigger the condition
    x = torch.tensor([[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0,
                       0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0]])
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_where_equal", result, expected, tol=1e-5)


def test_where_broadcast():
    """Test Where with broadcasting condition."""
    print("=== Test: Where (Broadcast) ===")
    model = WhereBroadcastModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_where_broadcast", result, expected, tol=1e-5)


def test_where_two_tensors():
    """Test Where selecting between two computed tensors."""
    print("=== Test: Where (Two Tensors) ===")
    model = WhereTwoTensorModel(features=16)
    model.eval()
    # Use tensor with some zeros to trigger the condition
    x = torch.tensor([[0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0,
                       0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0],
                      [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0,
                       5.0, 0.0, 6.0, 0.0, 7.0, 0.0, 8.0, 0.0]])
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_where_two_tensors", result, expected, tol=1e-5)


# --- ConstantOfShape Tests ---


def test_constant_of_shape_zeros():
    """Test ConstantOfShape with zeros."""
    print("=== Test: ConstantOfShape (Zeros) ===")
    model = ConstantOfShapeZerosModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_constant_of_shape_zeros", result, expected, tol=1e-5)


def test_constant_of_shape_ones():
    """Test ConstantOfShape with ones."""
    print("=== Test: ConstantOfShape (Ones) ===")
    model = ConstantOfShapeOnesModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_constant_of_shape_ones", result, expected, tol=1e-5)


def test_constant_of_shape_full():
    """Test ConstantOfShape with custom fill value."""
    print("=== Test: ConstantOfShape (Full) ===")
    model = ConstantOfShapeFullModel(features=16, fill_value=0.5)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_constant_of_shape_full", result, expected, tol=1e-5)


# --- Shape Test ---


def test_shape_based_reshape():
    """Test Shape operation via reshape with dynamic batch."""
    print("=== Test: Shape (via Reshape) ===")
    model = ShapeBasedReshapeModel(features=16)
    model.eval()
    x = torch.randn(4, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_shape_based_reshape", result, expected, tol=1e-5)


# --- Gather Edge Case Tests ---


def test_gather_large_vocab():
    """Test Gather with large vocabulary (tests index precision)."""
    print("=== Test: Gather (Large Vocab) ===")
    vocab_size = 50304  # GPT-2 vocab size
    model = GatherLargeVocabModel(vocab_size=vocab_size, embedding_dim=32)
    model.eval()
    # Include indices near the end of vocab to stress precision
    indices = torch.randint(0, vocab_size, (2, 16))
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(indices)
    with torch.no_grad():
        expected = model(indices)
    check("test_gather_large_vocab", result, expected, tol=1e-5)


# --- Expand Edge Case Tests ---


def test_expand_multi_dim():
    """Test Expand across multiple dimensions."""
    print("=== Test: Expand (Multi-dim) ===")
    model = ExpandMultiDimModel(features=16)
    model.eval()
    x = torch.randn(2, 8, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_expand_multi_dim", result, expected, tol=1e-5)


def test_expand_prepend_dims():
    """Test Expand with prepended dimensions."""
    print("=== Test: Expand (Prepend Dims) ===")
    model = ExpandPrependDimsModel(features=16)
    model.eval()
    x = torch.randn(2, 8, 16)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_expand_prepend_dims", result, expected, tol=1e-5)


# --- Split Edge Case Tests ---


def test_split_batch_dim():
    """Test Split along batch dimension."""
    print("=== Test: Split (Batch Dim) ===")
    model = SplitBatchDimModel()
    model.eval()
    x = torch.randn(4, 16)  # 4 batches, split into 2+2
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_split_batch_dim", result, expected, tol=1e-5)


def test_split_multiple_chunks():
    """Test Split into multiple chunks."""
    print("=== Test: Split (Multiple Chunks) ===")
    model = SplitMultipleChunksModel(in_features=24)
    model.eval()
    x = torch.randn(4, 24)
    compiled = torch.compile(model, backend=BACKEND)
    result = compiled(x)
    with torch.no_grad():
        expected = model(x)
    check("test_split_multiple_chunks", result, expected, tol=1e-5)


def test_nanogpt():
    """Test Nanogpt."""
    print("=== Test: NanoGpt ===")
    config = GPTConfig()
    model = GPT(config)
    model.eval()  # Put in eval mode

    batch_size = 1
    seq_length = 128  # Start with smaller sequence for export

    x = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_length), dtype=torch.long
    )
    print(f"  Input shape: {x.shape}, dtype: {x.dtype}")

    print("  Compiling model...")
    compiled = torch.compile(model, backend=BACKEND)

    print("  Running compiled model...")
    result = compiled(x)

    print("  Running PyTorch model...")
    with torch.no_grad():
        expected = model(x)
    print(f"  PyTorch result type: {type(expected)}")

    # Handle tuple outputs (nanogpt returns (logits, loss))
    if isinstance(result, tuple) and isinstance(expected, tuple):
        # Compare only the logits (first element), loss is None during inference
        result = result[0]
        expected = expected[0]
        print(f"  Comparing first element of tuples...")

    check("test_nanogpt", result, expected, tol=1e-5)


def main():
    print(f"Running tests with backend: {BACKEND}")
    print("=" * 50)
    print()

    tests = [
        test_llama,
        test_nanogpt,
        test_simple_linear,
        test_elementwise_mul_div,
        test_sqrt_div_model,
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
        test_simple_transfomer,
        # New ONNX op tests
        test_tanh_standalone,
        test_cos_standalone,
        test_sin_standalone,
        test_pow_standalone,
        test_reduce_mean_standalone,
        test_neg_standalone,
        test_sigmoid_standalone,
        test_split_equal,
        test_split_unequal,
        test_gather_embedding,
        test_expand_broadcast,
        test_reshape_view,
        test_reshape_flatten,
        test_unsqueeze,
        test_squeeze,
        test_equal,
        test_mod,
        test_cast,
        # Trilu tests
        test_trilu_upper,
        test_trilu_lower,
        test_trilu_diagonal_offset,
        # Where tests
        test_where_equal,
        test_where_broadcast,
        test_where_two_tensors,
        # ConstantOfShape tests
        test_constant_of_shape_zeros,
        test_constant_of_shape_ones,
        test_constant_of_shape_full,
        # Shape test
        test_shape_based_reshape,
        # Gather edge cases
        test_gather_large_vocab,
        # Expand edge cases
        test_expand_multi_dim,
        test_expand_prepend_dims,
        # Split edge cases
        test_split_batch_dim,
        test_split_multiple_chunks,
    ]

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
