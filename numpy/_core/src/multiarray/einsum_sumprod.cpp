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

#ifdef athurdek_DEBUG 1
void npy_einsum_simprod_experimental() {
        std::cout << "einsum_simprod.hpp successfully linked!\n";
    }
#endif

namespace npy_trais{
    template <typename T>
    struct TypeTraits {
        using type = T;
        using temp_type = T;
        static constexpr bool is_complex = false;
        static constexpr bool is_float32 = false;
        static constexpr bool is_float64 = false;
        static constexpr const char* sfx = "";
        static constexpr const char* to_func = "";
        static constexpr const char* from_func = "";
    };

    template <> struct TypeTraits<npy_byte> {
        using type = npy_byte;
        using temp_type = npy_byte;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "s8";
        static constexpr const char* to_func = "";
        static constexpr const char* from_func = "";
    };

    template <> struct TypeTraits<npy_short> {
        using type = npy_short;
        using temp_type = npy_short;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "s16";
        static constexpr const char* to_func = "";
        static constexpr const char* from_func = "";
    };

    template <> struct TypeTraits<npy_int> {
        using type = npy_int;
        using temp_type = npy_int;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "s32";
    };

    template <> struct TypeTraits<npy_long> {
        using type = npy_long;
        using temp_type = npy_long;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "long";
    };

    template <> struct TypeTraits<npy_longlong> {
        using type = npy_longlong;
        using temp_type = npy_longlong;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "s64";
    };

    template <> struct TypeTraits<npy_ubyte> {
        using type = npy_ubyte;
        using temp_type = npy_ubyte;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "u8";
    };

    template <> struct TypeTraits<npy_ushort> {
        using type = npy_ushort;
        using temp_type = npy_ushort;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "u16";
    };

    template <> struct TypeTraits<npy_uint> {
        using type = npy_uint;
        using temp_type = npy_uint;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "u32";
    };

    template <> struct TypeTraits<npy_ulong> {
        using type = npy_ulong;
        using temp_type = npy_ulong;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "ulong";
    };

    template <> struct TypeTraits<npy_ulonglong> {
        using type = npy_ulonglong;
        using temp_type = npy_ulonglong;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "u64";
    };

    template <> struct TypeTraits<npy_half> {
        using type = npy_half;
        using temp_type = npy_float;
        static constexpr bool is_complex = false;
        static constexpr bool is_float32 = true;
        static constexpr const char* sfx = "half";
        static constexpr const char* to_func = "npy_float_to_half";
        static constexpr const char* from_func = "npy_half_to_float";
    };

    template <> struct TypeTraits<npy_float> {
        using type = npy_float;
        using temp_type = npy_float;
        static constexpr bool is_complex = false;
        static constexpr bool is_float32 = true;
        static constexpr const char* sfx = "f32";
    };

    template <> struct TypeTraits<npy_double> {
        using type = npy_double;
        using temp_type = npy_double;
        static constexpr bool is_complex = false;
        static constexpr bool is_float64 = true;
        static constexpr const char* sfx = "f64";
    };

    template <> struct TypeTraits<npy_longdouble> {
        using type = npy_longdouble;
        using temp_type = npy_longdouble;
        static constexpr bool is_complex = false;
        static constexpr const char* sfx = "longdouble";
    };

    template <> struct TypeTraits<npy_cfloat> {
        using type = npy_cfloat;
        using temp_type = npy_float;
        static constexpr bool is_complex = true;
        static constexpr bool is_float32 = true;
        static constexpr const char* sfx = "f32";
    };

    template <> struct TypeTraits<npy_cdouble> {
        using type = npy_cdouble;
        using temp_type = npy_double;
        static constexpr bool is_complex = true;
        static constexpr bool is_float64 = true;
        static constexpr const char* sfx = "f64";
    };

    template <> struct TypeTraits<npy_clongdouble> {
        using type = npy_clongdouble;
        using temp_type = npy_longdouble;
        static constexpr bool is_complex = true;
        static constexpr const char* sfx = "clongdouble";
    };
}//namespace npy_trais


template <typename T, typename AccT = T>
void sum_of_products_contig_outstride0_one(
    int nop, char **dataptr, npy_intp const *strides, npy_intp count)
    {
        NPY_EINSUM_DBG_PRINT1(T + "_sum_of_products_contig_outstride0_one (%d)\n", (int)count);

    }



//athurdekoos update extern 'c' after refactor and namespace
NPY_VISIBILITY_HIDDEN
extern "C" sum_of_products_fn get_sum_of_products_function(int nop, int type_num, 
    npy_intp itemsize, npy_intp const *fixed_strides)
    {
        int iop;
        

        

        //athurdekoos TODO: need to verify if the nullptr is okay
        if (type_num >= NPY_NTYPES_LEGACY) {
            return nullptr;
        }

        //contiguous reduction
        if (nop == 1 && fixed_strides[0] == itemsize && fixed_strides[1] == 0){
            if(sum_of_products_fn ret = sum_of_products_contig_outstride0_one(); ret) return ret;
        }

        
        if (nop == 1 && fixed_strides[0] == itemsize && fixed_strides[1] == 0){
            std::optional<sum_of_products_fn>




            
            _contig_outstride0_unary_specialization_table[type_num];

        }
    
    }

