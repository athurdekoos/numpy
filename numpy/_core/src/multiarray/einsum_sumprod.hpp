#pragma once
#include <numpy/ndarraytypes.h> 

//In work
//
//TODO 
//Do I need to update?
//
// ARM/Neon don't have instructions for aligned memory access
#ifdef NPY_HAVE_NEON
    #define EINSUM_IS_ALIGNED(x) 0
#else
    #define EINSUM_IS_ALIGNED(x) npy_is_aligned(x, NPY_SIMD_WIDTH)
#endif


namespace np{

//@type@ = the element type = T, @temptype@ = the accumulator = Acct.
template <typename T, typename AccT = T>
inline AccT sum_of_arr(T* data, npy_intp count)

AccT accum = AccT{};

//simd.h always creates NPY_SIMD, float and double expand out 
#if NPY_SIMD 

//next todo  
//    const int is_aligned = EINSUM_IS_ALIGNED(data);
//    const int vstep = npyv_nlanes_@sfx@;
//    npyv_@sfx@ v_accum = npyv_zero_@sfx@();
ei
#else
#ifndef NPY_DISABLE_OPTIMIZATION
#endif // !NPY_DISABLE_OPTIMIZATION
#endif // NPY_SIMD check

}
