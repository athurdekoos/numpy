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

#include "einsum.hpp"
#include <iostream>

#ifdef athurdek_DEBUG 1
#define PyArray_RUNTIME_VERSION NPY_2_0_API_VERSION
#endif

/*
 * Parses the subscripts for one operand into an output of 'ndim'
 * labels. The resulting 'op_labels' array will have:
 *  - the ASCII code of the label for the first occurrence of a label;
 *  - the (negative) offset to the first occurrence of the label for
 *    repeated labels;
 *  - zero for broadcast dimensions, if subscripts has an ellipsis.
 * For example:
 *  - subscripts="abbcbc",  ndim=6 -> op_labels=[97, 98, -1, 99, -3, -2]
 *  - subscripts="ab...bc", ndim=6 -> op_labels=[97, 98, 0, 0, -3, 99]
 */

static int parse_operand_subscripts(char *subscripts, int length,
                         int ndim, int iop, char *op_labels,
                         char *label_counts, int *min_label, int *max_label)
{
    int idim = 0;
    int ellipsis = -1;

    /* Process all labels for this operand */
    for (int i = 0; i < length; ++i) {
        int label = subscripts[i];

        /* A proper label for an axis. */
        if (label > 0 && isalpha(label)) {
            /* Check we don't exceed the operator dimensions. */
            if (idim >= ndim) {
                PyErr_Format(PyExc_ValueError,
                             "einstein sum subscripts string contains "
                             "too many subscripts for operand %d", iop);
                return -1;
            }

            op_labels[idim++] = label;
            if (label < *min_label) {
                *min_label = label;
            }
            if (label > *max_label) {
                *max_label = label;
            }
            label_counts[label]++;
        }
        /* The beginning of the ellipsis. */
        else if (label == '.') {
            /* Check it's a proper ellipsis. */
            if (ellipsis != -1 || i + 2 >= length
                    || subscripts[++i] != '.' || subscripts[++i] != '.') {
                PyErr_Format(PyExc_ValueError,
                             "einstein sum subscripts string contains a "
                             "'.' that is not part of an ellipsis ('...') "
                             "in operand %d", iop);
                return -1;
            }

            ellipsis = idim;
        }
        else if (label != ' ') {
            PyErr_Format(PyExc_ValueError,
                         "invalid subscript '%c' in einstein sum "
                         "subscripts string, subscripts must "
                         "be letters", (char)label);
            return -1;
        }
    }

    /* No ellipsis found, labels must match dimensions exactly. */
    if (ellipsis == -1) {
        if (idim != ndim) {
            PyErr_Format(PyExc_ValueError,
                         "operand has more dimensions than subscripts "
                         "given in einstein sum, but no '...' ellipsis "
                         "provided to broadcast the extra dimensions.");
            return -1;
        }
    }
    /* Ellipsis found, may have to add broadcast dimensions. */
    else if (idim < ndim) {
        /* Move labels after ellipsis to the end. */
        for (int i = 0; i < idim - ellipsis; ++i) {
            op_labels[ndim - i - 1] = op_labels[idim - i - 1];
        }
        /* Set all broadcast dimensions to zero. */
        for (int i = 0; i < ndim - idim; ++i) {
            op_labels[ellipsis + i] = 0;
        }
    }

    /*
     * Find any labels duplicated for this operand, and turn them
     * into negative offsets to the axis to merge with.
     *
     * In C, the char type may be signed or unsigned, but with
     * twos complement arithmetic the char is ok either way here, and
     * later where it matters the char is cast to a signed char.
     */
    for (idim = 0; idim < ndim - 1; ++idim) {
        int label = (signed char)op_labels[idim];
        /* If it is a proper label, find any duplicates of it. */
        if (label > 0) {
            /* Search for the next matching label. */
            char *next = static_cast<char*>(std::memchr(op_labels + idim + 1, label, ndim - idim - 1));

            while (next != NULL) {
                /* The offset from next to op_labels[idim] (negative). */
                *next = (char)((op_labels + idim) - next);
                /* Search for the next matching label. */
                next = static_cast<char*>(std::memchr(next + 1, label, op_labels + ndim - 1 - next));
            }
        }
    }

    return 0;
}

/*NUMPY_API
 * This function provides summation of array elements according to
 * the Einstein summation convention.  For example:
 *  - trace(a)        -> einsum("ii", a)
 *  - transpose(a)    -> einsum("ji", a)
 *  - multiply(a,b)   -> einsum(",", a, b)
 *  - inner(a,b)      -> einsum("i,i", a, b)
 *  - outer(a,b)      -> einsum("i,j", a, b)
 *  - matvec(a,b)     -> einsum("ij,j", a, b)
 *  - matmat(a,b)     -> einsum("ij,jk", a, b)
 *
 * subscripts: The string of subscripts for einstein summation.
 * nop:        The number of operands
 * op_in:      The array of operands
 * dtype:      Either NULL, or the data type to force the calculation as.
 * order:      The order for the calculation/the output axes.
 * casting:    What kind of casts should be permitted.
 * out:        Either NULL, or an array into which the output should be placed.
 *
 * By default, the labels get placed in alphabetical order
 * at the end of the output. So, if c = einsum("i,j", a, b)
 * then c[i,j] == a[i]*b[j], but if c = einsum("j,i", a, b)
 * then c[i,j] = a[j]*b[i].
 *
 * Alternatively, you can control the output order or prevent
 * an axis from being summed/force an axis to be summed by providing
 * indices for the output. This allows us to turn 'trace' into
 * 'diag', for example.
 *  - diag(a)         -> einsum("ii->i", a)
 *  - sum(a, axis=0)  -> einsum("i...->", a)
 *
 * Subscripts at the beginning and end may be specified by
 * putting an ellipsis "..." in the middle.  For example,
 * the function einsum("i...i", a) takes the diagonal of
 * the first and last dimensions of the operand, and
 * einsum("ij...,jk...->ik...") takes the matrix product using
 * the first two indices of each operand instead of the last two.
 *
 * When there is only one operand, no axes being summed, and
 * no output parameter, this function returns a view
 * into the operand instead of making a copy.
 */
extern "C" PyObject *PyArray_EinsteinSum_Experimental(
    char *subscripts, int nop, PyArrayObject **op_in, PyArray_Descr *dtype,
    NPY_ORDER order, NPY_CASTING casting, PyObject *out){
        int label;
        int min_label = 127;
        int max_label = 0;
        char label_counts[128];
        char op_labels[NPY_MAXARGS][NPY_MAXDIMS];
        char output_labels[NPY_MAXDIMS]; 
        char* iter_labels;
        int idim;
        int ndim_output;
        int ndim_broadcast;
        int ndim_iter;

        PyArrayObject *op[NPY_MAXARGS];
        PyArrayObject *ret = NULL;
        PyArray_Descr *op_dtypes_array[NPY_MAXARGS];
        PyArray_Descr **op_dtypes;


        int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
        int *op_axes[NPY_MAXARGS];
        npy_uint32 iter_flags;
        npy_uint32 op_flags[NPY_MAXARGS];

        NpyIter *iter = NULL;
        npy_intp *stride;

        //Replaces sum_of_products_fn because I hated it. 
        //TODO need to change codebase 
        SumOfProducts sop;

        // nop+1 (+1 is for the output) must fit in NPY_MAXARGS 
        if (nop >= NPY_MAXARGS) {
            PyErr_SetString(PyExc_ValueError, "too many operands provided to einstein sum function");
            return NULL;
        }
        else if (nop < 1) { 
            PyErr_SetString(PyExc_ValueError, "not enough operands provided to einstein sum function");
            return NULL;
        }

        // Parse the subscripts string into label_counts and op_labels
        //TODO: Optimize this
        //label_counts is the global histogram of label usage.
        // Count of how often each label character appears globally
        //op_labels is the per-operand label table.
        // Labels per operand, mapping each array dimension to a subscript character
        //NEED TO UNDERSTAND THE ALGO ON PAPER FIRST, JESUS
        memset(label_counts, 0, sizeof(label_counts));
        for(int iop = 0; iop < nop; ++iop){
            int length = (int)strcspn(subscripts, ",-");
        
            if (iop == nop-1 && subscripts[length] == ',') {
                PyErr_SetString(PyExc_ValueError,
                            "more operands provided to einstein sum function "
                            "than specified in the subscripts string");
                return NULL;
            }
            else if(iop < nop-1 && subscripts[length] != ',') {
                PyErr_SetString(PyExc_ValueError,
                            "fewer operands provided to einstein sum function "
                            "than specified in the subscripts string");
                return NULL;
            }
        
            if (parse_operand_subscripts(subscripts, length,
                        PyArray_NDIM(op_in[iop]),iop, op_labels[iop], 
                        label_counts, &min_label, &max_label) < 0) {
                return NULL;
            }
            
            /* Move subscripts to the start of the labels for the next op */
            subscripts += length;
            if (iop < nop-1) {
                subscripts++;
            }
        
        }
    }

