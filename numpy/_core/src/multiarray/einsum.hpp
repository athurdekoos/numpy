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

#pragma once
#define PY_SSIZE_T_CLEAN

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include <Python.h>
#include <structmember.h>

#define _MULTIARRAYMODULE
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

#include <array_assign.h>   //PyArray_AssignRawScalar

#include <ctype.h>

#include "convert.h"
#include "common.h"
#include "ctors.h"

//athurdekoos debug utility
#define athurdek_DEBUG 1

#ifdef __cplusplus
#include <iostream>


extern "C" {
#endif
PyObject *PyArray_EinsteinSum_Experimental(
    char *subscripts, int nop, PyArrayObject **op_in, PyArray_Descr *dtype,
    NPY_ORDER order, NPY_CASTING casting, PyObject *out);
#ifdef __cplusplus
}

class SumOfProducts
{
private:
    // nop tells the kernel how many input and output pointers it should expect
    int _numOfOperands;
    // Array of raw memory pointers (not owned)
    char** _data;
    std::size_t _dataSize;
    // Pointer to an array of strides (in bytes) for each operand
    const npy_intp* _strides;
    std::size_t _strideCount;
    // Number of elements to process (loop length)
    npy_intp _count;

public:
    //default constructor
    SumOfProducts()
      : _numOfOperands(0), _data(nullptr), _dataSize(0),
        _strides(nullptr), _strideCount(0), _count(0){};

    //Paramterized Constructor 
    SumOfProducts(int numOfOperands, char** data, std::size_t dataSize,
                const npy_intp* strides, std::size_t strideCount, npy_intp count)
        : _numOfOperands(numOfOperands), _data(data), _dataSize(dataSize), 
        _strides(strides), _strideCount(strideCount), _count(count) {}       

    //Getters
    int numOfOperands() const noexcept { return _numOfOperands; }
    char** data() const noexcept { return _data; }
    std::size_t dataSize() const noexcept { return _dataSize; }
    const npy_intp* strides() const noexcept { return _strides; }
    std::size_t strideCount() const noexcept { return _strideCount; }
    npy_intp count() const noexcept { return _count; }

    //Setters
    void setNumOfOperands(int num) noexcept { _numOfOperands = num; }
    void setData(char** data, std::size_t size) noexcept {
        _data = data;
        _dataSize = size;
    }
    void setStrides(const npy_intp* strides, std::size_t count) noexcept {
        _strides = strides;
        _strideCount = count;
    }
    void setCount(npy_intp count) noexcept { _count = count; }

    //Debug Utility
    void debugPrint(){
        std::cout << "SumOfProducts {\n"
                  << "  _numOfOperands: " << _numOfOperands << "\n"
                  << "  _data: " << static_cast<const void*>(_data)
                  << " (size=" << _dataSize << ")\n"
                  << "  _strides: " << static_cast<const void*>(_strides)
                  << " (count=" << _strideCount << ")\n"
                  << "  _count: " << _count << "\n"
                  << "}" << std::endl;
    };
};

static int parse_operand_subscripts(char *subscripts, int length,
                         int ndim, int iop, char *op_labels,
                         char *label_counts, int *min_label, int *max_label);
#endif
