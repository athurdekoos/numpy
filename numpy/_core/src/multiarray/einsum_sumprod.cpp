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

// //athurdekoos why is this here?
// #define NPY_NO_DEPRECATED_API NPY_API_VERSION

// #define _MULTIARRAYMODULE

// #include <numpy/npy_common.h>
// #include <numpy/ndarraytypes.h>  /* for NPY_NTYPES_LEGACY */
// #include <numpy/halffloat.h>



// #include "simd/simd.h"
// #include "common.h"


// #include "einsum_sumprod.hpp"

// #include <iostream>
// #include <optional>
// #include <unordered_set>

// //TODO: Debug
// //#include "einsum_debug.h"

// // ARM/Neon don't have instructions for aligned memory access
// // Athurdekoos notes: Need to verify what this does, why, and if it's still needed

// #ifdef NPY_HAVE_NEON
//     #define EINSUM_IS_ALIGNED(x) 0
// #else
//     #define EINSUM_IS_ALIGNED(x) npy_is_aligned(x, NPY_SIMD_WIDTH)
// #endif


// // namespace einsum_traits{

// //     //Base class of traits.
// //     template <typename T>
// //     struct BaseTraits{
// //     protected:
// //         using type = T;
// //         bool is_complex = false;
// //         bool is_float32 = false;
// //         bool is_float64 = false;
// //         std::string to_func = "";
// //         std::string from_func = "";
// //         std::string sfx = "";
// //     };
// //     //Generic template of Basetraits
// //     template <typename T>
// //     struct Traits : BaseTraits<T> {};

// //     template <> struct Traits<npy_byte> : BaseTraits<npy_byte> {
// //         Traits() { this->sfx = "s8"; }
// //     };

// //     template <> struct Traits<npy_short> : BaseTraits<npy_short> {
// //         Traits() { this->sfx = "s16"; }
// //     };

// //     template <> struct Traits<npy_int> : BaseTraits<npy_int> {
// //         Traits() { this->sfx = "s32"; }
// //     };

// //     template <> struct Traits<npy_long> : BaseTraits<npy_long> {
// //         Traits() { this->sfx = "long"; }
// //     };

// //     template <> struct Traits<npy_longlong> : BaseTraits<npy_longlong> {
// //         Traits() { this->sfx = "s64"; }
// //     };

// //     template <> struct Traits<npy_ubyte> : BaseTraits<npy_ubyte> {
// //         Traits() { this->sfx = "u8"; }
// //     };

// //     template <> struct Traits<npy_ushort> : BaseTraits<npy_ushort> {
// //         Traits() { this->sfx = "u16"; }
// //     };

// //     template <> struct Traits<npy_uint> : BaseTraits<npy_uint> {
// //         Traits() { this->sfx = "u32"; }
// //     };

// //     template <> struct Traits<npy_ulong> : BaseTraits<npy_ulong> {
// //         Traits() { this->sfx = "ulong"; }
// //     };

// //     template <> struct Traits<npy_ulonglong> : BaseTraits<npy_ulonglong> {
// //         Traits() { this->sfx = "u64"; }
// //     };

// //     template <> struct Traits<npy_half> : BaseTraits<npy_half> {
// //         Traits() {
// //             this->is_float32 = true;
// //             this->sfx = "half";
// //             this->to_func = "npy_float_to_half";
// //             this->from_func = "npy_half_to_float";
// //         }
// //     };

// //     template <> struct Traits<npy_float> : BaseTraits<npy_float> {
// //         Traits() {
// //             this->is_float32 = true;
// //             this->sfx = "f32";
// //         }
// //     };

// //     template <> struct Traits<npy_double> : BaseTraits<npy_double> {
// //         Traits() {
// //             this->is_float64 = true;
// //             this->sfx = "f64";
// //         }
// //     };

// //     template <> struct Traits<npy_longdouble> : BaseTraits<npy_longdouble> {
// //         Traits() { this->sfx = "longdouble"; }
// //     };

// //     template <> struct Traits<npy_cfloat> : BaseTraits<npy_cfloat> {
// //     Traits() {
// //         this->is_complex = true;
// //         this->is_float32 = true;
// //         this->sfx = "f32";
// //     }
// //     };

// //     template <> struct Traits<npy_cdouble> : BaseTraits<npy_cdouble> {
// //         Traits() {
// //             this->is_complex = true;
// //             this->is_float64 = true;
// //             this->sfx = "f64";
// //         }
// //     };

// //     template <> struct Traits<npy_clongdouble> : BaseTraits<npy_clongdouble> {
// //         Traits() {
// //             this->is_complex = true;
// //             this->sfx = "clongdouble";
// //         }
// //     };

// // };//namespace einsum_traits


// #ifdef athurdek_DEBUG 1

// sum_of_products_fn npy_einsum_simprod_experimental(int nop, int type_num, 
// npy_intp itemsize, npy_intp const *fixed_strides){
    
//     SumOfProducts dummySumOfProductsRet = internal_simprod_experimental(nop,type_num,itemsize,fixed_strides);
    
// };

// SumOfProducts internal_simprod_experimental(int nop, int type_num, 
// npy_intp itemsize, npy_intp const *fixed_strides){
    



//     //einsum_traits::Traits<npy_cfloat> test;

//     // TypeTraits<npy_cfloat> cfloat_test;
//     // cfloat_test.is_complex = true;
//     // cfloat_test.is_float32 = true;
//     // cfloat_test.sfx = "f32";


//     // auto testtype = (NPY_TYPES)type_num;
    
//     // auto test = npy_trait::TypeTraits<npy_byte>::is_complex;

//     // std::cout << test;

//     SumOfProducts ret;
//     return ret;
// }

// #endif
//     // NPY_BOOL        = 0,
//     // NPY_BYTE        = 1,
//     // NPY_UBYTE       = 2,
//     // NPY_SHORT       = 3,
//     // NPY_USHORT      = 4,
//     // NPY_INT         = 5,
//     // NPY_UINT        = 6,
//     // NPY_LONG        = 7,
//     // NPY_ULONG       = 8,
//     // NPY_LONGLONG    = 9,
//     // NPY_ULONGLONG   = 10,
//     // NPY_FLOAT       = 11,
//     // NPY_DOUBLE      = 12,
//     // NPY_LONGDOUBLE  = 13,
//     // NPY_CFLOAT      = 14,
//     // NPY_CDOUBLE     = 15,
//     // NPY_CLONGDOUBLE = 16,
//     // NPY_OBJECT      = 17,
//     // NPY_STRING      = 18,
//     // NPY_UNICODE     = 19,
//     // NPY_VOID        = 20,
//     // NPY_DATETIME    = 21,
//     // NPY_TIMEDELTA   = 22,
//     // NPY_HALF        = 23,
//     // NPY_NTYPES      = 24,
//     // NPY_NOTYPE      = 25,
//     // NPY_CHAR        = 26


// // template <typename T, typename AccT = T>
// // void sum_of_products_contig_outstride0_one(int nop, char **dataptr, npy_intp const *strides, npy_intp count)
// //     {

// //     }



// // //athurdekoos update extern 'c' after refactor and namespace
// // NPY_VISIBILITY_HIDDEN
// // sum_of_products_fn get_sum_of_products_function(int nop, int type_num, 
// //     npy_intp itemsize, npy_intp const *fixed_strides)
// // {
// //     int iop;

// //     //Casting type_num to a type. 
// //     //TODO athrudekoos remove after einsum is updated
// //     NPY_TYPES curr_type = static_cast<NPY_TYPES>(type_num);

// //     //athurdekoos TODO: need to verify if the nullptr is okay
// //     if (type_num >= NPY_NTYPES_LEGACY) {
// //         return nullptr;
// //     }

// //     //contiguous reduction
// //     if (nop == 1 && fixed_strides[0] == itemsize && fixed_strides[1] == 0){
// //         std::unordered_set<NPY_TYPES> nonallowed_types = {NPY_BOOL, NPY_OBJECT, NPY_STRING, NPY_UNICODE, NPY_VOID, NPY_DATETIME, NPY_TIMEDELTA};
// //         if(nonallowed_types.find(curr_type) == nonallowed_types.end()){
// //             sum_of_products_fn ret = &sum_of_products_contig_outstride0_one<curr_type>;
// //         }
        
// //     }

    
// // }

