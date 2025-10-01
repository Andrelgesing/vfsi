import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import numba
import numpy as np
from numba import cuda

@cuda.jit
def fused_scale_spmv_numba(weight, indptr, indices, data, p_in, p_out):
    row, k = cuda.grid(2)
    n_rows = indptr.size - 1
    n_rhs = p_in.shape[1]

    if row >= n_rows or k >= n_rhs:
        return

    acc_r = 0.0
    acc_i = 0.0

    start = indptr[row]
    end = indptr[row + 1]

    for idx in range(start, end):
        col = indices[idx]
        w = data[idx]
        pval = p_in[col, k]
        scaled_pval = weight[col] * pval
        acc_r += w * scaled_pval.real
        acc_i += w * scaled_pval.imag

    p_out[row, k] = complex(acc_r, acc_i)


def wp_product_numba(weight, w_csr, p):
    """
    Applies fused scaling and sparse-matrix vector multiplication on GPU using Numba.

    Computes: p_out = w @ (diag(weight) * p)

    Parameters
    ----------
    weight : cp.ndarray (float32), shape (n_cols,)
        Integration weights on GPU.
    w_csr : cupyx.scipy.sparse.csr_matrix
        Sparse matrix (GPU) in CSR format.
    p : cp.ndarray (complex64), shape (n_cols, n_rhs)
        Input dense matrix to be scaled and multiplied.

    Returns
    -------
    p_out : cp.ndarray (complex64), shape (n_rows, n_rhs)
        Output after fused scaling and multiplication.
    """
    # Ensure input is complex64 Fortran-order
    p_in = cp.asfortranarray(p.astype(cp.complex64))

    n_rows = w_csr.shape[0]
    n_rhs = p.shape[1]

    # Allocate output
    p_out = cp.zeros((n_rows, n_rhs), dtype=cp.complex64)

    # Kernel launch parameters
    threadsperblock = (16, 16)
    blockspergrid_x = (n_rows + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (n_rhs + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    fused_scale_spmv_numba[blockspergrid, threadsperblock](
        weight.astype(cp.float32),
        w_csr.indptr,
        w_csr.indices,
        w_csr.data,
        p_in,
        p_out
    )

    # Return as C-contiguous array for downstream compatibility
    return p_out.copy(order='C')



cuda_kernel = r'''
#include <cuComplex.h>

extern "C" __global__
void scale_and_spmv(
    const float* __restrict__ weight,
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const cuComplex* __restrict__ p_in,
    cuComplex* __restrict__ p_out,
    int n_rhs,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int row_start = indptr[row];
    int row_end = indptr[row + 1];

    for (int k = 0; k < n_rhs; ++k) {
        cuComplex acc = make_cuComplex(0.0f, 0.0f);
        for (int jj = row_start; jj < row_end; ++jj) {
            int col = indices[jj];
            float w = weight[col];
            float val = data[jj];

            cuComplex pval = p_in[col * n_rhs + k];
            pval.x *= w * val;
            pval.y *= w * val;
            acc.x += pval.x;
            acc.y += pval.y;
        }
        p_out[row * n_rhs + k] = acc;
    }
}

'''

scale_spmv_kernel = cp.RawKernel(cuda_kernel, 'scale_and_spmv')

def wp_product_cuda(weight, w_csr, p_in):
    assert isinstance(w_csr, csr_matrix)
    assert isinstance(p_in, cp.ndarray)
    assert weight.shape[0] == w_csr.shape[1]

    n_rows, n_rhs = w_csr.shape[0], p_in.shape[1]
    p_out = cp.zeros((n_rows, n_rhs), dtype=cp.complex64)
    p_in = cp.asfortranarray(p_in.astype(cp.complex64))
    threads_per_block = 128
    blocks = (n_rows + threads_per_block - 1) // threads_per_block

    scale_spmv_kernel(
        (blocks,), (threads_per_block,),
        (
            weight.astype(cp.float32),
            w_csr.indptr,
            w_csr.indices,
            w_csr.data,
            p_in.astype(cp.complex64),
            p_out,
            n_rhs, 
            n_rows
        )
    )

    return p_out.copy(order='C')


cuda_kernel = r'''
#include <cuComplex.h>

extern "C" __global__
void row_scale(
    cuComplex* __restrict__ p,
    const float* __restrict__ weights,
    int n_rows,
    int n_cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows && col < n_cols) {
        int idx = row * n_cols + col;
        float w = weights[row];
        p[idx].x *= w;
        p[idx].y *= w;
    }
}
'''

row_scale_kernel = cp.RawKernel(cuda_kernel, 'row_scale')

def wp_product_v1(p, weights):
    p = p.astype(cp.complex64, copy=False)
    weights = weights.astype(cp.float32, copy=False)
    assert p.shape[0] == weights.shape[0]

    n_rows, n_cols = p.shape
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (n_cols + threads_per_block[0] - 1) // threads_per_block[0],
        (n_rows + threads_per_block[1] - 1) // threads_per_block[1]
    )

    row_scale_kernel(
        blocks_per_grid, threads_per_block,
        (
            p,
            weights,
            n_rows,
            n_cols
        )
    )
    return p


cuda_kernel_low = r'''
#include <cuComplex.h>

extern "C" __global__
void csr_spmm(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const float* __restrict__ data,
    const cuComplex* __restrict__ p_in,
    cuComplex* __restrict__ p_out,
    int n_rhs,
    int n_cols,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int row_start = indptr[row];
    int row_end = indptr[row + 1];

    for (int k = 0; k < n_rhs; ++k) {
        cuComplex acc = make_cuComplex(0.0f, 0.0f);
        for (int jj = row_start; jj < row_end; ++jj) {
            int col = indices[jj];
            float val = data[jj];

            // Column-major layout: k * n_cols + col
            cuComplex p_val = p_in[k * n_cols + col];

            // Multiply real * complex
            cuComplex scaled = make_cuComplex(val * p_val.x, val * p_val.y);
            acc = cuCaddf(acc, scaled);
        }
        // Column-major output
        p_out[k * n_rows + row] = acc;
    }
}
'''
csr_spmm_low = cp.RawKernel(cuda_kernel_low, 'csr_spmm')

def wp_product_v4(w_csr, p_in):
    assert isinstance(w_csr, csr_matrix)
    assert isinstance(p_in, cp.ndarray)

    n_rows, n_cols = w_csr.shape
    n_rhs = p_in.shape[1]

    # Ensure correct types and column-major layout
    p_in = cp.asfortranarray(p_in.astype(cp.complex64))  # shape (n_cols, n_rhs)
    p_out = cp.zeros((n_rows, n_rhs), dtype=cp.complex64, order='F')

    # CSR components
    indptr = w_csr.indptr
    indices = w_csr.indices
    data = w_csr.data.astype(cp.float32)

    threads_per_block = 128
    blocks = (n_rows + threads_per_block - 1) // threads_per_block

    csr_spmm_low(
        (blocks,), (threads_per_block,),
        (
            indptr,
            indices,
            data,
            p_in,
            p_out,
            n_rhs,
            n_cols,
            n_rows
        )
    )

    return p_out.copy(order='C')  # Return in C-order for downstream compatibility


cuda_kernel = r'''
#include <cuComplex.h>

extern "C" __global__
void csr_spmm(
    const int* __restrict__ indptr,
    const int* __restrict__ indices,
    const double* __restrict__ data,
    const cuDoubleComplex* __restrict__ p_in,
    cuDoubleComplex* __restrict__ p_out,
    int n_rhs,
    int n_cols,
    int n_rows
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    int row_start = indptr[row];
    int row_end = indptr[row + 1];

    for (int k = 0; k < n_rhs; ++k) {
        cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
        for (int jj = row_start; jj < row_end; ++jj) {
            int col = indices[jj];
            double val = data[jj];

            // Corrected indexing: column-major access
            cuDoubleComplex p_val = p_in[k * n_cols + col];

            cuDoubleComplex scaled = make_cuDoubleComplex(val * p_val.x, val * p_val.y);
            acc = cuCadd(acc, scaled);
        }
        p_out[k * n_rows + row] = acc;  // Also column-major layout for output
    }
}
'''

csr_spmm_kernel = cp.RawKernel(cuda_kernel, 'csr_spmm')

def wp_product_v2(w_csr, p_in):
    assert isinstance(w_csr, csr_matrix)
    assert isinstance(p_in, cp.ndarray)
    #assert p_in.dtype == cp.complex128

    n_rows, n_cols = w_csr.shape
    n_rhs = p_in.shape[1]

    # Ensure p_in is column-major: shape (n_cols, n_rhs)
    p_in = cp.asfortranarray(p_in, dtype=cp.complex128,)

    # Output array: shape (n_rows, n_rhs), also Fortran order
    p_out = cp.zeros((n_rows, n_rhs), dtype=cp.complex128, order='F')

    # Extract CSR components (GPU arrays, all float64)
    indptr = w_csr.indptr
    indices = w_csr.indices
    data = w_csr.data

    threads_per_block = 128
    blocks = (n_rows + threads_per_block - 1) // threads_per_block

    csr_spmm_kernel(
        (blocks,), (threads_per_block,),
        (
            indptr,
            indices,
            data,
            p_in,
            p_out,
            n_rhs,
            n_cols,  # required for correct indexing in kernel
            n_rows
        )
    )

    return p_out.copy(order='C')  # still in Fortran-order, matching layout

cuda_kernel_dense_mm = r'''
#include <cuComplex.h>

extern "C" __global__
void dense_matmul(
    const double* __restrict__ W,
    const cuComplex* __restrict__ p_in,
    cuComplex* __restrict__ p_out,
    int n_rows,
    int n_cols,
    int n_rhs
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows && col < n_rhs) {
        cuComplex acc = make_cuComplex(0.0f, 0.0f);
        for (int k = 0; k < n_cols; ++k) {
            double w_val = W[row * n_cols + k];
            cuComplex p_val = p_in[k * n_rhs + col];

            // Multiply real w_val with complex p_val
            cuComplex w_times_p;
            w_times_p.x = w_val * p_val.x;
            w_times_p.y = w_val * p_val.y;
            acc = cuCaddf(acc, w_times_p);
        }
        p_out[row * n_rhs + col] = acc;
    }
}
'''

dense_matmul_kernel = cp.RawKernel(cuda_kernel_dense_mm, 'dense_matmul')

def wp_product_v2_dense(W, p_in):
    assert W.dtype == cp.float64
    assert p_in.dtype == cp.complex64
    assert W.shape[1] == p_in.shape[0]

    n_rows, n_cols = W.shape
    n_rhs = p_in.shape[1]
    p_out = cp.zeros((n_rows, n_rhs), dtype=cp.complex64)

    W = cp.ascontiguousarray(W)
    p_in = cp.asfortranarray(p_in)  # (cols, rhs) is more efficient in col-major
    p_out = cp.ascontiguousarray(p_out)

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (n_rhs + threads_per_block[0] - 1) // threads_per_block[0],
        (n_rows + threads_per_block[1] - 1) // threads_per_block[1]
    )

    dense_matmul_kernel(
        blocks_per_grid, threads_per_block,
        (
            W,
            p_in,
            p_out,
            n_rows,
            n_cols,
            n_rhs
        )
    )
    return p_out.copy(order='C')
