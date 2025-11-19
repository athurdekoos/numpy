

#include <iostream>
#include "numpy/ndarraytypes.h"
#include "static_nhalffloat.h"

/*NUMPY_API
 * Test of static_nhalffloat.cpp
 */
NPY_NO_EXPORT void
NpyHalffloat_test()
{
    std::cout << "I am static_nhalffloat.cpp" << std::endl;
}

//make sure numpy itself is comming the c api 
//add c++ tests 
//new meson project can use the new c++ project correctly 
//also have 
//can't have people use import array
//generate depricate warnings
//add macro to generate warning - potentially - look back to numpy 2
