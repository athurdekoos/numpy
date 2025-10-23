// /*
//  * This file provides optimized sum of product implementations used internally
//  * by einsum.
//  *
//  * Copyright (c) 2025 by Amelia Thurdekoos(athurdek@gmail.com)
//  * This templated file is based on previous work 
//  * by Mark Wiebe (mwwiebe@gmail.com)
//  * 
//  * See LICENSE.txt for the license.
//  */


// //WIP Notes
// //This file provides optimized sum of product implementations used internally by einsum
// //NumPy Einstein Summation is a cool word
// //np.einsum("ij,jk->ik", A, B)
// //C[i, k] = Σ_j A[i, j] * B[j, k]
// // ┌──────────────────────────────────────────────────────────────────────────┐
// // │                        NumPy Einstein Summation                          │
// // │                              np.einsum()                                 │
// // └──────────────────────────────────────────────────────────────────────────┘
// //                                       │
// //                                       ▼
// // ┌──────────────────────────────────────────────────────────────────────────┐
// // │                                einsum.c                                  │
// // │  High-level logic:                                                       │
// // │  ─────────────────                                                       │
// // │  • Parses subscript strings like "ij,jk->ik"                             │
// // │  • Determines how to broadcast operands                                  │
// // │  • Allocates output array                                                │
// // │  • Sets up iteration over "outer" indices                                │
// // │  • For each inner block, calls numeric kernel →                          │
// // │        npy_einsum_sum_of_products_<dtype>()                              │
// // └──────────────────────────────────────────────────────────────────────────┘
// //                                       │
// //                                       ▼
// // ┌──────────────────────────────────────────────────────────────────────────┐
// // │                        einsum_sumprod.h (Header)                         │
// // │  Interface / declarations:                                               │
// // │  ─────────────────────────                                               │
// // │  • Declares:                                                             │
// // │       npy_einsum_sum_of_arr_<dtype>()                                    │
// // │       npy_einsum_sum_of_products_<dtype>()                               │
// // │  • Included by einsum.c                                                  │
// // │  • Exposes C linkage for each dtype                                      │
// // │  • Each function computes sum or sum-of-products                         │
// // └──────────────────────────────────────────────────────────────────────────┘
// //                                       │
// //                                       ▼
// // ┌──────────────────────────────────────────────────────────────────────────┐
// // │                   einsum_sumprod.c (Implementation)                      │
// // │  Numeric inner-loop kernel:                                              │
// // │  ──────────────────────────                                              │
// // │                                                                          │
// // │  • Implements the "sum of products" core math:                           │
// // │       result = Σ (A[i] * B[i] * C[i] …)                                  │
// // │                                                                          │
// // │  • Optimized per dtype using templating macros                           │
// // │       byte, short, int, float, double, complex                           │
// // │                                                                          │
// // │  • SIMD vectorized for float32/float64 via npyv_* intrinsics             │
// // │       - Uses npyv_load / npyv_loada for aligned or unaligned reads       │
// // │       - Adds vector lanes (4x or 8x at once)                             │
// // │       - Accumulates partial sums                                         │
// // │                                                                          │
// // │  • Scalar fallback path                                                  │
// // │       - Triggered if SIMD unavailable or disabled                        │
// // │       - Uses manual loop unrolling for partial speedup                   │
// // │                                                                          │
// // │  • Alignment logic:                                                      │
// // │       #ifdef NPY_HAVE_NEON                                               │
// // │           EINSUM_IS_ALIGNED(x) = 0  (NEON doesn’t require it)            │
// // │       #else                                                              │
// // │           EINSUM_IS_ALIGNED(x) = npy_is_aligned(x, NPY_SIMD_WIDTH)       │
// // │       #endif                                                             │
// // │                                                                          │
// // │  • Output: returns scalar accumulation to einsum.c                       │
// // └──────────────────────────────────────────────────────────────────────────┘
// //                                       │
// //                                       ▼
// // ┌──────────────────────────────────────────────────────────────────────────┐
// // │                              SIMD Backend                                │
// // │     simd/simd.h + npyv_* API                                             │
// // │     ──────────────────────                                               │
// // │     • Provides platform abstraction (SSE, AVX, NEON, etc.)               │
// // │     • Defines vector types (npyv_f32, npyv_f64, etc.)                    │
// // │     • Handles differences in alignment and instruction width             │
// // └──────────────────────────────────────────────────────────────────────────┘
// //                                       │
// //                                       ▼
// // ┌──────────────────────────────────────────────────────────────────────────┐
// // │                         Output Accumulation                              │
// // │  einsum.c receives scalar results from einsum_sumprod.c and writes them  │
// // │  into the correct output array position for each combination of indices. │
// // └──────────────────────────────────────────────────────────────────────────┘


// #pragma once
// #ifndef NUMPY_CORE_SRC_MULTIARRAY_EINSUM_SUMPROD_H_
// #define NUMPY_CORE_SRC_MULTIARRAY_EINSUM_SUMPROD_H_

// //athurdekoos debug utility
// #define athurdek_DEBUG 1

// #include <numpy/npy_common.h>

// #ifdef __cplusplus

// //input from einsum
// //sop = npy_einsum_simprod_experimental(nop,
//                         // NpyIter_GetDescrArray(iter)[0]->type_num,
//                         // NpyIter_GetDescrArray(iter)[0]->elsize,
//                         // stride);

// using sum_of_products_fn = void (*)(int nop, char** data, 
//     const npy_intp* strides,npy_intp count);

// class SumOfProducts
// {
// private:
//     // nop tells the kernel how many input and output pointers it should expect
//     int _numOfOperands;
//     // Array of raw memory pointers
//     char** _data;
//     std::size_t _dataSize;
//     // Pointer to an array of strides (in bytes) for each operand
//     const npy_intp* _strides;
//     std::size_t _strideCount;
//     // Number of elements to process (loop length)
//     npy_intp _count;

// public:
//     SumOfProducts()
//       : _numOfOperands(0),
//           _data(nullptr),
//           _dataSize(0),
//           _strides(nullptr),
//           _strideCount(0),
//           _count(0){};

// };


// //TODO athurdekoos remove 'extern "C" when einsum.c is refactored to einsum.cpp
// //namespace np::multiarray::einsum {
// extern "C" {
// //athurdekoos sanity check
// #ifdef athurdek_DEBUG 1

//     //internal nested experimental functions

// sum_of_products_fn npy_einsum_simprod_experimental(int nop, int type_num, 
//     npy_intp itemsize, npy_intp const *fixed_strides);

// SumOfProducts internal_simprod_experimental(int nop, int type_num, 
//     npy_intp itemsize, npy_intp const *fixed_strides);


// #endif



// //TODO athurdekoos remove 'extern "C" when einsum.c is refactored to einsum.cpp
// //change name to get_sum_of_products_function also
// NPY_VISIBILITY_HIDDEN
// sum_of_products_fn get_sum_of_products_function(int nop, int type_num, 
//     npy_intp itemsize, npy_intp const *fixed_strides);
// }; // extern "C"
// //} // namespace np::multiarray::einsum 
// #endif //__cplusplus
// #endif //NUMPY_CORE_SRC_MULTIARRAY_EINSUM_SUMPROD_H_