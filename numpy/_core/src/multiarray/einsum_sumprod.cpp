/*
 * This file provides optimized sum of product implementations used internally
 * by einsum.
 *
 * Copyright (c) 2025 by Amelia Thurdekoos(athurdek@gmail.com)
 * This templated file is based on previous work 
 * by Mark Wiebe (mwwiebe@gmail.com)
 * 
 * See LICENSE.txt for the license.
 */


//WIP Notes
//This file provides optimized sum of product implementations used internally by einsum
//NumPy Einstein Summation is a cool word
//np.einsum("ij,jk->ik", A, B)
//C[i, k] = Σ_j A[i, j] * B[j, k]
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                        NumPy Einstein Summation                          │
// │                              np.einsum()                                 │
// └──────────────────────────────────────────────────────────────────────────┘
//                                       │
//                                       ▼
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                                einsum.c                                  │
// │  High-level logic:                                                       │
// │  ─────────────────                                                       │
// │  • Parses subscript strings like "ij,jk->ik"                             │
// │  • Determines how to broadcast operands                                  │
// │  • Allocates output array                                                │
// │  • Sets up iteration over "outer" indices                                │
// │  • For each inner block, calls numeric kernel →                          │
// │        npy_einsum_sum_of_products_<dtype>()                              │
// └──────────────────────────────────────────────────────────────────────────┘
//                                       │
//                                       ▼
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                        einsum_sumprod.h (Header)                         │
// │  Interface / declarations:                                               │
// │  ─────────────────────────                                               │
// │  • Declares:                                                             │
// │       npy_einsum_sum_of_arr_<dtype>()                                    │
// │       npy_einsum_sum_of_products_<dtype>()                               │
// │  • Included by einsum.c                                                  │
// │  • Exposes C linkage for each dtype                                      │
// │  • Each function computes sum or sum-of-products                         │
// └──────────────────────────────────────────────────────────────────────────┘
//                                       │
//                                       ▼
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                   einsum_sumprod.c (Implementation)                      │
// │  Numeric inner-loop kernel:                                              │
// │  ──────────────────────────                                              │
// │                                                                          │
// │  • Implements the "sum of products" core math:                           │
// │       result = Σ (A[i] * B[i] * C[i] …)                                  │
// │                                                                          │
// │  • Optimized per dtype using templating macros                           │
// │       byte, short, int, float, double, complex                           │
// │                                                                          │
// │  • SIMD vectorized for float32/float64 via npyv_* intrinsics             │
// │       - Uses npyv_load / npyv_loada for aligned or unaligned reads       │
// │       - Adds vector lanes (4x or 8x at once)                             │
// │       - Accumulates partial sums                                         │
// │                                                                          │
// │  • Scalar fallback path                                                  │
// │       - Triggered if SIMD unavailable or disabled                        │
// │       - Uses manual loop unrolling for partial speedup                   │
// │                                                                          │
// │  • Alignment logic:                                                      │
// │       #ifdef NPY_HAVE_NEON                                               │
// │           EINSUM_IS_ALIGNED(x) = 0  (NEON doesn’t require it)            │
// │       #else                                                              │
// │           EINSUM_IS_ALIGNED(x) = npy_is_aligned(x, NPY_SIMD_WIDTH)       │
// │       #endif                                                             │
// │                                                                          │
// │  • Output: returns scalar accumulation to einsum.c                       │
// └──────────────────────────────────────────────────────────────────────────┘
//                                       │
//                                       ▼
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                              SIMD Backend                                │
// │     simd/simd.h + npyv_* API                                             │
// │     ──────────────────────                                               │
// │     • Provides platform abstraction (SSE, AVX, NEON, etc.)               │
// │     • Defines vector types (npyv_f32, npyv_f64, etc.)                    │
// │     • Handles differences in alignment and instruction width             │
// └──────────────────────────────────────────────────────────────────────────┘
//                                       │
//                                       ▼
// ┌──────────────────────────────────────────────────────────────────────────┐
// │                         Output Accumulation                              │
// │  einsum.c receives scalar results from einsum_sumprod.c and writes them  │
// │  into the correct output array position for each combination of indices. │
// └──────────────────────────────────────────────────────────────────────────┘

//athurdekoos why is this here?
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define _MULTIARRAYMODULE

#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>  /* for NPY_NTYPES_LEGACY */
#include <numpy/halffloat.h>



#include "simd/simd.h"
#include "common.h"


#include "einsum_sumprod.hpp"

#include <iostream>
#include <optional>

//TODO: Debug
//#include "einsum_debug.h"

// ARM/Neon don't have instructions for aligned memory access
// Athurdekoos notes: Need to verify what this does, why, and if it's still needed

#ifdef NPY_HAVE_NEON
    #define EINSUM_IS_ALIGNED(x) 0
#else
    #define EINSUM_IS_ALIGNED(x) npy_is_aligned(x, NPY_SIMD_WIDTH)
#endif



//todo need to verify this is correct and pobably change name. Also add private
namespace npy_trait{
    struct BaseTraits{
        static constexpr bool is_complex = false;
        static constexpr bool is_float32 = false;
        static constexpr bool is_float64 = false;
        static constexpr const char* to_func = "";
        static constexpr const char* from_func = "";
        static constexpr const char* sfx = "";
    };
    
    template <typename T>
    struct TypeTraits : BaseTraits {
        using type = T;
        using temp_type = T;
        static constexpr const char* sfx = [] {
            if constexpr (std::is_same_v<T, npy_byte>) return "s8";
            else if constexpr (std::is_same_v<T, npy_short>) return "s16";
            else if constexpr (std::is_same_v<T, npy_int>) return "s32";
            else if constexpr (std::is_same_v<T, npy_long>) return "long";
            else if constexpr (std::is_same_v<T, npy_longlong>) return "s64";
            else if constexpr (std::is_same_v<T, npy_ubyte>) return "u8";
            else if constexpr (std::is_same_v<T, npy_ushort>) return "u16";
            else if constexpr (std::is_same_v<T, npy_uint>) return "u32";
            else if constexpr (std::is_same_v<T, npy_ulong>) return "ulong";
            else if constexpr (std::is_same_v<T, npy_ulonglong>) return "u64";
            else if constexpr (std::is_same_v<T, npy_longdouble>) return "longdouble";
            else return "";
    }();
    };

    template <> struct TypeTraits<npy_half> : BaseTraits{
        using type = npy_half;
        using temp_type = npy_half;
        static constexpr bool is_float32 = true;
        static constexpr const char* sfx = "half";
        static constexpr const char* to_func = "npy_float_to_half";
        static constexpr const char* from_func = "npy_half_to_float";
    };

    template <> struct TypeTraits<npy_float> : BaseTraits{
        using type = npy_float;
        using temp_type = npy_float;
        static constexpr bool is_float32 = true;
        static constexpr const char* sfx = "f32";
    };

    template <> struct TypeTraits<npy_double> : BaseTraits{
        using type = npy_double;
        using temp_type = npy_double;
        static constexpr bool is_float64 = true;
        static constexpr const char* sfx = "f64";
    };

    template <> struct TypeTraits<npy_cfloat> : BaseTraits{
        using type = npy_cfloat;
        using temp_type = npy_float;
        static constexpr bool is_complex = true;
        static constexpr bool is_float32 = true;
        static constexpr const char* sfx = "f32";
    };

    template <> struct TypeTraits<npy_cdouble> : BaseTraits{
        using type = npy_cdouble;
        using temp_type = npy_double;
        static constexpr bool is_complex = true;
        static constexpr bool is_float64 = true;
        static constexpr const char* sfx = "f64";
    };

    template <> struct TypeTraits<npy_clongdouble> : BaseTraits{
        using type = npy_clongdouble;
        using temp_type = npy_longdouble;
        static constexpr bool is_complex = true;
        static constexpr const char* sfx = "clongdouble";
    };
};//namespace npy_trais


#ifdef athurdek_DEBUG 1

sum_of_products_fn npy_einsum_simprod_experimental(int nop, int type_num, 
npy_intp itemsize, npy_intp const *fixed_strides){
    
    auto testtype = (NPY_TYPES)type_num;
    
    auto test = npy_trait::TypeTraits<npy_byte>::is_complex;

    std::cout << test;

    sum_of_products_fn ret;
    return ret;
}

#endif

template <typename T, typename AccT = T>
void sum_of_products_contig_outstride0_one(
    int nop, char **dataptr, npy_intp const *strides, npy_intp count)
    {
        NPY_EINSUM_DBG_PRINT1(T + "_sum_of_products_contig_outstride0_one (%d)\n", (int)count);

    }



//athurdekoos update extern 'c' after refactor and namespace
NPY_VISIBILITY_HIDDEN
sum_of_products_fn get_sum_of_products_function(int nop, int type_num, 
    npy_intp itemsize, npy_intp const *fixed_strides)
    {
        int iop;
        
    
        //athurdekoos TODO: need to verify if the nullptr is okay
        if (type_num >= NPY_NTYPES_LEGACY) {
            return nullptr;
        }

        //contiguous reduction
        if (nop == 1 && fixed_strides[0] == itemsize && fixed_strides[1] == 0){
            sum_of_products_fn ret = sum_of_products_contig_outstride0_one(sum_of_products_fn ret = sum_of_products_contig_outstride0_one());
            if(ret) return ret;
        }

    
    }

