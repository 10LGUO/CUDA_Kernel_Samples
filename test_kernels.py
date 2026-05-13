"""
Unit tests for every CUDA kernel in this repository, verified against PyTorch.

What this does
--------------
Each kernel (and each of its optimized versions) is JIT-compiled into a small
PyTorch C++/CUDA extension using ``torch.utils.cpp_extension.load``. The binding
``#include``s the *real* kernel source files from this repo (no copy/paste), wraps
each ``__global__`` with a thin launcher that reproduces the exact grid/block
configuration used by the repo, and exposes it as a Python-callable function that
takes and returns ``torch.Tensor``. Every test then compares the kernel output to
a trusted PyTorch reference (``a + b``, ``A @ x``, ``softmax``, ``x.T``, ...).

Coverage
--------
  elementwise      : add (float4)
  gemv             : sgemv_k32
  reduce/sum       : v0, v1, v2, v3, v4, v5
  reduce/max       : max_kernel (warp-shuffle + atomicMax)
  reduce/softmax   : max -> sum -> softmax pipeline (numerically stable)
  reduce/softmax_matrix : row / row2 / col / col2
  transpose        : v0, v1, v2, v3, v4, v5
  sgemm            : v1, v2, v3, v4, v5, v6, v7

Requirements
------------
  * An NVIDIA GPU and a CUDA-enabled PyTorch build (``torch.cuda.is_available()``).
  * The CUDA toolkit (``nvcc``) on PATH, plus a host C++ compiler.
  * ``pytest``.
If CUDA is unavailable every test is skipped (rather than failed), so the file is
safe to collect on any machine.

Usage
-----
    pip install torch pytest
    pytest test_kernels.py -v

The first run compiles the extensions (cached under ``.cuda_test_build/``), so it
is slow; subsequent runs reuse the cached ``.so`` files.
"""

import os

import pytest

torch = pytest.importorskip("torch")

REPO = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(REPO, ".cuda_test_build")

# The sgemm/matmul kernels do plain fp32 FMAs; make the PyTorch reference do the
# same (disable TF32) so the comparison is apples-to-apples.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-capable GPU"
)


# --------------------------------------------------------------------------- #
# Extension bindings.
#
# For each "group" we build one extension. The CUDA source below is assembled
# from a common preamble + a block that #includes the repo's kernel file(s) +
# thin torch wrappers + the pybind module. `@@REPO@@` is replaced with the repo
# path at build time (we use str.replace, never str.format, because the C++ is
# full of braces).
# --------------------------------------------------------------------------- #

_PREAMBLE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cfloat>

// Launch everything on the tensor's current stream so ordering vs. torch is safe.
#define CKS_STREAM (at::cuda::getCurrentCUDAStream().stream())

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

static torch::Tensor as_cuda_f32(torch::Tensor t) {
    TORCH_CHECK(t.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, "input must be float32");
    return t.contiguous();
}
"""


def _include(rel_path, disable_main):
    """A block that pulls a repo .cu into this TU, neutralizing its main()."""
    inc = '#include "@@REPO@@/%s"' % rel_path
    if disable_main:
        tag = "_disabled_main_" + os.path.basename(rel_path).replace(".", "_")
        return "#define main %s\n%s\n#undef main\n" % (tag, inc)
    return inc + "\n"


# ---- elementwise: add.cu -------------------------------------------------- #
_ELEMENTWISE_BODY = r"""
// elementwise_add_float4 is defined by the #included add.cu above.

torch::Tensor add_float4(torch::Tensor a, torch::Tensor b) {
    a = as_cuda_f32(a); b = as_cuda_f32(b);
    TORCH_CHECK(a.numel() == b.numel(), "size mismatch");
    TORCH_CHECK(a.numel() % 4 == 0, "float4 kernel needs a multiple of 4 elements");
    int N = a.numel();
    auto c = torch::empty_like(a);
    int block = 1024;
    int grid = cdiv(cdiv(N, 4), block);
    elementwise_add_float4<<<grid, block, 0, CKS_STREAM>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_float4", &add_float4);
}
"""

# ---- gemv: sgemv_k32.cu --------------------------------------------------- #
_GEMV_BODY = r"""
// sgemv_k32 is defined by the #included sgemv_k32.cu above.

torch::Tensor sgemv(torch::Tensor A, torch::Tensor x) {
    A = as_cuda_f32(A); x = as_cuda_f32(x);
    TORCH_CHECK(A.dim() == 2 && x.dim() == 1 && A.size(1) == x.size(0), "shape mismatch");
    int M = A.size(0), K = A.size(1);
    auto y = torch::empty({M}, A.options());
    // one block per row, one warp (32 threads) per block.
    sgemv_k32<<<M, 32, 0, CKS_STREAM>>>(
        A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), M, K);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemv", &sgemv);
}
"""

# ---- reduce/sum: sum.cu (v0..v5) ------------------------------------------ #
# v0/v1/v2 emit one partial sum per block (reduced on the python side);
# v3/v4/v5 atomicAdd into a single slot. Either way the wrapper returns a
# tensor whose .sum() is the total, so the test is uniform.
_REDUCE_SUM_BODY = r"""
// device_reduce_v0..v5 are defined by the #included sum.cu above.

static const int BS = 128;

torch::Tensor sum_v0(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    TORCH_CHECK(N % BS == 0, "v0 requires N to be a multiple of BLOCK_SIZE(128)");
    int grid = cdiv(N, BS);
    auto xin = x.clone();                        // v0 reduces in place, so copy
    auto y = torch::zeros({grid}, x.options());
    device_reduce_v0<<<grid, BS, 0, CKS_STREAM>>>(xin.data_ptr<float>(), y.data_ptr<float>());
    return y;
}
torch::Tensor sum_v1(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    int grid = cdiv(N, BS);
    auto y = torch::zeros({grid}, x.options());
    device_reduce_v1<BS><<<grid, BS, 0, CKS_STREAM>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}
torch::Tensor sum_v2(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    int grid = cdiv(N, BS);
    auto y = torch::zeros({grid}, x.options());
    device_reduce_v2<<<grid, BS, BS * sizeof(float), CKS_STREAM>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}
torch::Tensor sum_v3(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    int grid = cdiv(N, BS);
    auto y = torch::zeros({1}, x.options());     // atomicAdd into a single slot
    device_reduce_v3<<<grid, BS, BS * sizeof(float), CKS_STREAM>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}
torch::Tensor sum_v4(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    int grid = cdiv(N, BS);
    auto y = torch::zeros({1}, x.options());
    device_reduce_v4<<<grid, BS, 0, CKS_STREAM>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}
torch::Tensor sum_v5(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    TORCH_CHECK(N % 4 == 0, "v5 (float4) requires N to be a multiple of 4");
    int grid = cdiv(cdiv(N, BS), 4);             // each thread handles 4 elements
    auto y = torch::zeros({1}, x.options());
    device_reduce_v5<<<grid, BS, 0, CKS_STREAM>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_v0", &sum_v0);
    m.def("sum_v1", &sum_v1);
    m.def("sum_v2", &sum_v2);
    m.def("sum_v3", &sum_v3);
    m.def("sum_v4", &sum_v4);
    m.def("sum_v5", &sum_v5);
}
"""

# ---- reduce/max: max.cu --------------------------------------------------- #
_REDUCE_MAX_BODY = r"""
// max_kernel is defined by the #included max.cu above.

torch::Tensor max_reduce(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    int grid = cdiv(N, 128);
    // atomicMax refines *output, so it must start at the identity (-FLT_MAX).
    auto out = torch::full({1}, -FLT_MAX, x.options());
    max_kernel<<<grid, 128, 0, CKS_STREAM>>>(x.data_ptr<float>(), out.data_ptr<float>(), N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_reduce", &max_reduce);
}
"""

# ---- reduce/softmax: softmax.cu (stable 3-kernel pipeline) ---------------- #
_REDUCE_SOFTMAX_BODY = r"""
// setToNegativeMax, max_kernel, sum_kernel, softmax_kernel are defined by the
// #included softmax.cu above.

torch::Tensor softmax(torch::Tensor x) {
    x = as_cuda_f32(x);
    int N = x.numel();
    const int BS = 256;
    int grid = cdiv(N, BS);
    auto mx  = torch::empty({1}, x.options());
    auto sm  = torch::zeros({1}, x.options());
    auto out = torch::empty({N}, x.options());
    setToNegativeMax<<<1, 1, 0, CKS_STREAM>>>(mx.data_ptr<float>());
    max_kernel<<<grid, BS, 0, CKS_STREAM>>>(x.data_ptr<float>(), mx.data_ptr<float>(), N);
    sum_kernel<<<grid, BS, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), sm.data_ptr<float>(), mx.data_ptr<float>(), N);
    softmax_kernel<<<grid, BS, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), sm.data_ptr<float>(), mx.data_ptr<float>(), N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax);
}
"""

# ---- reduce/softmax_matrix: softmax_matrix.cu ----------------------------- #
_REDUCE_SOFTMAX_MATRIX_BODY = r"""
// softmax_row_kernel[2] / softmax_col_kernel[2] are defined by the #included
// softmax_matrix.cu above.

// one warp per row -> softmax along dim=1
torch::Tensor softmax_row(torch::Tensor x) {
    x = as_cuda_f32(x);
    TORCH_CHECK(x.dim() == 2, "expected a 2D matrix");
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    softmax_row_kernel<<<M, 32, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
torch::Tensor softmax_row2(torch::Tensor x) {
    x = as_cuda_f32(x);
    TORCH_CHECK(x.dim() == 2, "expected a 2D matrix");
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    softmax_row_kernel2<<<M, 32, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
// one warp per column -> softmax along dim=0
torch::Tensor softmax_col(torch::Tensor x) {
    x = as_cuda_f32(x);
    TORCH_CHECK(x.dim() == 2, "expected a 2D matrix");
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    softmax_col_kernel<<<N, 32, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
torch::Tensor softmax_col2(torch::Tensor x) {
    x = as_cuda_f32(x);
    TORCH_CHECK(x.dim() == 2, "expected a 2D matrix");
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    softmax_col_kernel2<<<N, 32, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_row",  &softmax_row);
    m.def("softmax_row2", &softmax_row2);
    m.def("softmax_col",  &softmax_col);
    m.def("softmax_col2", &softmax_col2);
}
"""

# ---- transpose: transpose.cu (v0..v5) ------------------------------------- #
_TRANSPOSE_BODY = r"""
// device_transpose_v0..v5 are defined by the #included transpose.cu above
// (v3/v4/v5 are templated on TILE_DIM).

static torch::Tensor make_out(torch::Tensor x, int M, int N) {
    return torch::empty({N, M}, x.options());
}

torch::Tensor t_v0(torch::Tensor x) {
    x = as_cuda_f32(x); int M = x.size(0), N = x.size(1);
    auto out = make_out(x, M, N);
    dim3 block(32, 32), grid(cdiv(N, 32), cdiv(M, 32));  // tile by input shape
    device_transpose_v0<<<grid, block, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
torch::Tensor t_v1(torch::Tensor x) {
    x = as_cuda_f32(x); int M = x.size(0), N = x.size(1);
    auto out = make_out(x, M, N);
    dim3 block(32, 32), grid(cdiv(M, 32), cdiv(N, 32));  // tile by output shape
    device_transpose_v1<<<grid, block, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
torch::Tensor t_v2(torch::Tensor x) {
    x = as_cuda_f32(x); int M = x.size(0), N = x.size(1);
    auto out = make_out(x, M, N);
    dim3 block(32, 32), grid(cdiv(M, 32), cdiv(N, 32));
    device_transpose_v2<<<grid, block, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
torch::Tensor t_v3(torch::Tensor x) {
    x = as_cuda_f32(x); int M = x.size(0), N = x.size(1);
    auto out = make_out(x, M, N);
    dim3 block(32, 32), grid(cdiv(N, 32), cdiv(M, 32));
    device_transpose_v3<32><<<grid, block, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
torch::Tensor t_v4(torch::Tensor x) {
    x = as_cuda_f32(x); int M = x.size(0), N = x.size(1);
    auto out = make_out(x, M, N);
    dim3 block(32, 32), grid(cdiv(N, 32), cdiv(M, 32));
    device_transpose_v4<32><<<grid, block, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}
torch::Tensor t_v5(torch::Tensor x) {
    x = as_cuda_f32(x); int M = x.size(0), N = x.size(1);
    auto out = make_out(x, M, N);
    dim3 block(32, 32), grid(cdiv(N, 32), cdiv(M, 32));
    device_transpose_v5<32><<<grid, block, 0, CKS_STREAM>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("t_v0", &t_v0);
    m.def("t_v1", &t_v1);
    m.def("t_v2", &t_v2);
    m.def("t_v3", &t_v3);
    m.def("t_v4", &t_v4);
    m.def("t_v5", &t_v5);
}
"""

# ---- sgemm: kernel1..7 (C = A @ B, alpha=1, beta=0) ----------------------- #
_SGEMM_BODY = r"""
// sgemm_v1..v7 are declared/defined by the #included kernel1.cu..kernel7.cu
// above (each also carries the explicit template instantiation used below).

// Returns C = A @ B. C starts at zero so the kernels' beta*C term is a clean 0.
static void check_mm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && A.size(1) == B.size(0), "shape mismatch");
}

torch::Tensor mm_v1(torch::Tensor A, torch::Tensor B) {
    A = as_cuda_f32(A); B = as_cuda_f32(B); check_mm(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(32, 32), grid(cdiv(N, 32), cdiv(M, 32));
    sgemm_v1<<<grid, block, 0, CKS_STREAM>>>(
        M, N, K, 1.0f, A.data_ptr<float>(), B.data_ptr<float>(), 0.0f, C.data_ptr<float>());
    return C;
}
torch::Tensor mm_v2(torch::Tensor A, torch::Tensor B) {
    A = as_cuda_f32(A); B = as_cuda_f32(B); check_mm(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(1024), grid(cdiv(N, 32), cdiv(M, 32));
    sgemm_v2<32><<<grid, block, 0, CKS_STREAM>>>(
        M, N, K, 1.0f, A.data_ptr<float>(), B.data_ptr<float>(), 0.0f, C.data_ptr<float>());
    return C;
}
torch::Tensor mm_v3(torch::Tensor A, torch::Tensor B) {
    A = as_cuda_f32(A); B = as_cuda_f32(B); check_mm(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(512), grid(cdiv(N, 64), cdiv(M, 64));
    sgemm_v3<64, 64, 8, 8><<<grid, block, 0, CKS_STREAM>>>(
        M, N, K, 1.0f, A.data_ptr<float>(), B.data_ptr<float>(), 0.0f, C.data_ptr<float>());
    return C;
}
torch::Tensor mm_v4(torch::Tensor A, torch::Tensor B) {
    A = as_cuda_f32(A); B = as_cuda_f32(B); check_mm(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(256), grid(cdiv(N, 128), cdiv(M, 128));
    sgemm_v4<128, 128, 8, 8, 8><<<grid, block, 0, CKS_STREAM>>>(
        M, N, K, 1.0f, A.data_ptr<float>(), B.data_ptr<float>(), 0.0f, C.data_ptr<float>());
    return C;
}
torch::Tensor mm_v5(torch::Tensor A, torch::Tensor B) {
    A = as_cuda_f32(A); B = as_cuda_f32(B); check_mm(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(256), grid(cdiv(N, 128), cdiv(M, 128));
    sgemm_v5<128, 128, 8, 8, 8><<<grid, block, 0, CKS_STREAM>>>(
        M, N, K, 1.0f, A.data_ptr<float>(), B.data_ptr<float>(), 0.0f, C.data_ptr<float>());
    return C;
}
torch::Tensor mm_v6(torch::Tensor A, torch::Tensor B) {
    A = as_cuda_f32(A); B = as_cuda_f32(B); check_mm(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(256), grid(cdiv(N, 128), cdiv(M, 128));
    sgemm_v6<128, 128, 8, 8, 8><<<grid, block, 0, CKS_STREAM>>>(
        M, N, K, 1.0f, A.data_ptr<float>(), B.data_ptr<float>(), 0.0f, C.data_ptr<float>());
    return C;
}
torch::Tensor mm_v7(torch::Tensor A, torch::Tensor B) {
    A = as_cuda_f32(A); B = as_cuda_f32(B); check_mm(A, B);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(256), grid(cdiv(N, 128), cdiv(M, 128));
    sgemm_v7<128, 128, 8, 8, 8><<<grid, block, 0, CKS_STREAM>>>(
        M, N, K, 1.0f, A.data_ptr<float>(), B.data_ptr<float>(), 0.0f, C.data_ptr<float>());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm_v1", &mm_v1);
    m.def("mm_v2", &mm_v2);
    m.def("mm_v3", &mm_v3);
    m.def("mm_v4", &mm_v4);
    m.def("mm_v5", &mm_v5);
    m.def("mm_v6", &mm_v6);
    m.def("mm_v7", &mm_v7);
}
"""


# name -> spec. `includes` = kernel files pulled via #include (with main disabled
# where the file has one); `sources` = *additional* real .cu files that must be
# compiled/linked (e.g. the repo's utils.cu that define helpers the file's main
# references); `inc_dirs` = include paths; `ldflags` = extra linker flags.
_GROUPS = {
    "elementwise": dict(
        includes=[("elementwise/add.cu", True)],
        body=_ELEMENTWISE_BODY, sources=[], inc_dirs=[], ldflags=[],
    ),
    "gemv": dict(
        includes=[("gemv/sgemv_k32.cu", True)],
        body=_GEMV_BODY, sources=[], inc_dirs=[], ldflags=["-lcublas"],
    ),
    "reduce_sum": dict(
        includes=[("reduce/sum/sum.cu", True)],
        body=_REDUCE_SUM_BODY, sources=["reduce/sum/src/utils.cu"],
        inc_dirs=["reduce/sum/include"], ldflags=[],
    ),
    "reduce_max": dict(
        includes=[("reduce/max/max.cu", True)],
        body=_REDUCE_MAX_BODY, sources=["reduce/max/src/utils.cu"],
        inc_dirs=["reduce/max/include"], ldflags=[],
    ),
    "reduce_softmax": dict(
        includes=[("reduce/softmax/softmax.cu", True)],
        body=_REDUCE_SOFTMAX_BODY, sources=["reduce/softmax/src/utils.cu"],
        inc_dirs=["reduce/softmax/include"], ldflags=[],
    ),
    "reduce_softmax_matrix": dict(
        includes=[("reduce/softmax_matrix/softmax_matrix.cu", True)],
        body=_REDUCE_SOFTMAX_MATRIX_BODY, sources=["reduce/softmax_matrix/src/utils.cu"],
        inc_dirs=["reduce/softmax_matrix/include"], ldflags=[],
    ),
    "transpose": dict(
        includes=[("transpose/transpose.cu", True)],
        body=_TRANSPOSE_BODY, sources=["transpose/src/utils.cu"],
        inc_dirs=["transpose/include"], ldflags=[],
    ),
    "sgemm": dict(
        includes=[("sgemm/src/kernel%d.cu" % i, False) for i in range(1, 8)],
        body=_SGEMM_BODY, sources=[], inc_dirs=["sgemm/include"], ldflags=[],
    ),
}

_MODULE_CACHE = {}


def _module(name):
    """Build (once) and return the compiled extension for a kernel group."""
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    from torch.utils.cpp_extension import load

    spec = _GROUPS[name]
    parts = [_PREAMBLE]
    for rel, disable_main in spec["includes"]:
        parts.append(_include(rel, disable_main))
    parts.append(spec["body"])
    source = "\n".join(parts).replace("@@REPO@@", REPO)

    os.makedirs(BUILD_DIR, exist_ok=True)
    binding_path = os.path.join(BUILD_DIR, "cks_%s_binding.cu" % name)
    with open(binding_path, "w") as fh:
        fh.write(source)

    sources = [binding_path] + [os.path.join(REPO, s) for s in spec["sources"]]
    module = load(
        name="cks_%s" % name,
        sources=sources,
        extra_include_paths=[os.path.join(REPO, d) for d in spec["inc_dirs"]],
        extra_cuda_cflags=["-O2"],
        extra_ldflags=spec["ldflags"],
        build_directory=BUILD_DIR,
        verbose=False,
    )
    _MODULE_CACHE[name] = module
    return module


@pytest.fixture(scope="session")
def cuda_dev():
    torch.manual_seed(0)
    return torch.device("cuda")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

@requires_cuda
def test_elementwise_add(cuda_dev):
    m = _module("elementwise")
    a = torch.randn(4096, device=cuda_dev)
    b = torch.randn(4096, device=cuda_dev)
    torch.testing.assert_close(m.add_float4(a, b), a + b, rtol=0, atol=0)


@requires_cuda
def test_gemv(cuda_dev):
    m = _module("gemv")
    M, K = 256, 128
    A = torch.randn(M, K, device=cuda_dev)
    x = torch.randn(K, device=cuda_dev)
    torch.testing.assert_close(m.sgemv(A, x), A @ x, rtol=1e-3, atol=1e-3)


@requires_cuda
@pytest.mark.parametrize("ver", ["v0", "v1", "v2", "v3", "v4", "v5"])
def test_reduce_sum(cuda_dev, ver):
    m = _module("reduce_sum")
    N = 128 * 100  # multiple of BLOCK_SIZE(128) and of 4 (for the float4 v5)
    # Non-negative inputs: the sum is ~N/2, large enough that relative tolerance
    # is meaningful and float32 accumulation order doesn't cause cancellation.
    x = torch.rand(N, device=cuda_dev)
    got = getattr(m, "sum_" + ver)(x).sum().item()
    ref = x.double().sum().item()
    assert got == pytest.approx(ref, rel=1e-4, abs=1e-1)


@requires_cuda
def test_reduce_max(cuda_dev):
    m = _module("reduce_max")
    x = torch.randn(128 * 100, device=cuda_dev)
    got = m.max_reduce(x).item()
    assert got == pytest.approx(x.max().item(), rel=0, abs=0)


@requires_cuda
def test_reduce_softmax(cuda_dev):
    m = _module("reduce_softmax")
    x = torch.randn(2048, device=cuda_dev) * 5  # large spread stresses the max-subtraction
    torch.testing.assert_close(m.softmax(x), torch.softmax(x, dim=0), rtol=1e-4, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("fn,dim", [("softmax_row", 1), ("softmax_row2", 1),
                                    ("softmax_col", 0), ("softmax_col2", 0)])
def test_softmax_matrix(cuda_dev, fn, dim):
    m = _module("reduce_softmax_matrix")
    x = torch.randn(2048, 64, device=cuda_dev) * 3
    torch.testing.assert_close(getattr(m, fn)(x), torch.softmax(x, dim=dim),
                               rtol=1e-4, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("ver", ["v0", "v1", "v2", "v3", "v4", "v5"])
def test_transpose(cuda_dev, ver):
    m = _module("transpose")
    x = torch.randn(128, 64, device=cuda_dev)  # non-square, both dims multiples of 32
    got = getattr(m, "t_" + ver)(x)
    torch.testing.assert_close(got, x.t().contiguous(), rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("ver", ["v1", "v2", "v3", "v4", "v5", "v6", "v7"])
def test_sgemm(cuda_dev, ver):
    m = _module("sgemm")
    M = N = K = 256  # multiple of every tile size used (128/64/32, K multiple of 8)
    A = torch.randn(M, K, device=cuda_dev)
    B = torch.randn(K, N, device=cuda_dev)
    got = getattr(m, "mm_" + ver)(A, B)
    torch.testing.assert_close(got, A @ B, rtol=1e-2, atol=1e-2)
