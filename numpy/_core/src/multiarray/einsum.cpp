/*
 * This file provides optimized sum of product implementations used internally
 * by einsum.
 *
 * Copyright (c) 2025 by Amelia Thurdekoos(athurdek@gmail.com)
 * This file is based on previous work 
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

static int parse_operand_subscripts(
    //char *subscripts = a text string that describes how the arrays’ axes are labeled and combined (for example "ij,jk->ik").
    char *subscripts, 
    //Find how many characters appear in subscripts before we hit either a comma (,) or a dash (-).
    //Then store that number in length.
    //subscripts is the string of subscripts for einstein summation.
    //
    //ij,jk->ik
    //"ij" and "jk" are individual subscripts
    //input1_labels,input2_labels->output_labels
    //operand1     ,operand2     ->result
    int length,
    //a counter or index representing the current dimension being processed.
    int ndim, 
    //current position we're at
    int iop, 
    //op_labels is a 2D array that stores the labels 
    //op_labels[iop][NPY_MAXDIMS]
    //(like 'i', 'j', 'k', etc.) used by each 
    //operand (input array) — one row per operand, 
    //one column per dimension (axis).
    //iop = poisiton we're at
    //NPY_MAXDIMS = the maximum number of dimensions a NumPy array can have.
    char *op_labels,
    //label_counts is an array that keeps track of
    // how many times each label (like ‘i’, ‘j’, ‘k’, etc.)
    // appears across all operands in the einsum expression.
    char *label_counts, 
    //Keep track of the smallest label character (alphabetically / ASCII-wise) that we’ve seen so far while parsing the subscripts
    int *min_label, 
    //Keep track of the largest label character (alphabetically / ASCII-wise) that we’ve seen so far while parsing the subscripts
    int *max_label)
{
    // idim is a counter that tracks which dimension (axis) of the current operand is being processed
    int idim = 0;

    // ellipsis marks the position of "..." in the subscripts string; -1 means no ellipsis has been found yet
    int ellipsis = -1;

    /* Process all labels for this operand */
    //Go through every character (label) in the subscripts 
    //string that belongs to this particular operand,
    //and record what each one means
    for (int i = 0; i < length; ++i) {

        //label = subscripts[i]
        //subscript = string that belongs to this particular operand,
        //so label is the string at i position 
        int label = subscripts[i];

        /* A proper label for an axis. */
        //make sure label is not null, and 
        //isalpha(checks if the character stored in label is an alphabet letter (A–Z or a–z))
        //Will continue to hit this so long as the elipse is not present
        if (label > 0 && isalpha(label)) {
            /* Check we don't exceed the operator dimensions. */
            if (idim >= ndim) {
                PyErr_Format(PyExc_ValueError,
                             "einstein sum subscripts string contains "
                             "too many subscripts for operand %d", iop);
                return -1;
            }

            //op_labels at position idim = label, 
            //Label is equal to subscripts[i],
            //then increase idim by 1
            op_labels[idim++] = label;

            //Keep track of the smallest label 
            //character (alphabetically / ASCII-wise) 
            //that we’ve seen so far while parsing the subscripts
            if (label < *min_label) {
                *min_label = label;
            }

            //If label is larger than we have already seen, update counter 
            if (label > *max_label) {
                *max_label = label;
            }
            //increase the counter at position label 
            label_counts[label]++;
        }

        /* The beginning of the ellipsis. */
        //The first check was for if it's greater than 0 an is alpha
        //this check assumes that label exists and then check for a dot
        else if (label == '.') {
            /* Check it's a proper ellipsis. */
            //If a elispe has already been found;
            //If no elipse has been found, and current poition + 2 is shorter than the length
            // so the end is to close to form a ellipse
            //Move to the next char, if it is not a '.'
            //Move to the next char, if it is not a '.'
            //CLEAVER^^^^^
            if (ellipsis != -1 || i + 2 >= length
                    || subscripts[++i] != '.' || subscripts[++i] != '.') {
                PyErr_Format(PyExc_ValueError,
                             "einstein sum subscripts string contains a "
                             "'.' that is not part of an ellipsis ('...') "
                             "in operand %d", iop);
                return -1;
            }
            //Elippse found! 
            //Set position of elipse to idim
            ellipsis = idim;
        }
        //checks if the label is not empty, if it's not empty that it means that is nonAlpha, Nonzero char
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
    // the "..." was detected in the subscripts string,
    // and the code might need to insert extra dimensions in the array shapes
    // so they can align correctly for broadcasting (to match sizes across operands).
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
    //char *subscripts = a text string that describes how the arrays’ axes are labeled and combined (for example "ij,jk->ik").
    char *subscripts, 
    //int nop = the number of input arrays (operands) being used.
    int nop, 
    //PyArrayObject **op_in = a list of pointers to the input NumPy arrays.
    PyArrayObject **op_in, 
    //PyArray_Descr *dtype = the data type to use for the calculation (for example float64); can be NULL to use the inputs’ types.
    PyArray_Descr *dtype,
    //NPY_ORDER order = tells whether the output array should be stored in C order (row-major) or Fortran order (column-major).
    NPY_ORDER order,
    //NPY_CASTING casting = defines what kind of data conversions (casts) are allowed between input and output types.
    NPY_CASTING casting, 
    //PyObject *out = an optional existing array where the result will be written; if NULL, a new output array is created.
    PyObject *out){
        
        //label is an integer variable that temporarily 
        //stores the ASCII code of each subscript 
        //character (like 'i', 'j', 'k', etc.) as 
        //the subscripts string is being parsed.
        int label;

        //	min_label starts at 127, which is higher than any normal ASCII letter.
        int min_label = 127;

        //max_label = 0 is lower than any normal ASCII letter.
        int max_label = 0;

        //label_counts is an array that keeps track of 
        //how many times each label (like ‘i’, ‘j’, ‘k’, etc.)
        // appears across all operands in the einsum expression.
        char label_counts[128];

        //op_labels is a 2D array that stores the labels 
        //op_labels[NPY_MAXARGS][NPY_MAXDIMS]
        //(like 'i', 'j', 'k', etc.) used by each 
        //operand (input array) — one row per operand, 
        //one column per dimension (axis).
        //NPY_MAXARGS = the maximum number of input arrays you can pass into this operation.
        //NPY_MAXDIMS = the maximum number of dimensions a NumPy array can have.
        char op_labels[NPY_MAXARGS][NPY_MAXDIMS];

        //output_labels is an array that stores the 
        //labels (like 'i', 'j', 'k', etc.) used for 
        //each dimension of the output array.
        //Each element corresponds to one output axis, 
        //and its value is the ASCII code of that label.
        //NPY_MAXDIMS = the maximum number of dimensions a NumPy array can have.
        char output_labels[NPY_MAXDIMS]; 

        //iter_label = pointer to the list of labels used by the 
        //iterator (the combined or broadcasted set of all 
        //labels across operands).
        char* iter_labels;

        //a counter or index representing the current dimension being processed.
        int idim;

        //the number of dimensions (axes) in the output array.
        int ndim_output;

        //the number of dimensions created or extended by broadcasting to match array shapes.
        int ndim_broadcast;

        //the total number of dimensions that the iterator will loop over when performing the operation.
        int ndim_iter;

        //PyArrayObject is the C struct that represents a NumPy array in C code.
        //See bookmark for infor on PyArryObject
        //NPY_MAXARG = the maximum number of input arrays you can pass into this operation.
        //op = pointer of the list of all NumPy array objects involved in the operation.
        PyArrayObject *op[NPY_MAXARGS];

        //ret = return object?
        PyArrayObject *ret = NULL;

        //NPY_MAXARG = the maximum number of input arrays you can pass into this operation.
        //op_dtypes[i] tells you what kind of data (float, int, complex, etc.) 
        //the i-th array (op[i]) contains or should be converted to
        PyArray_Descr *op_dtypes_array[NPY_MAXARGS];

        //a pointer to an array of pointers, where each pointer 
        //refers to the data type (dtype) description 
        //of one operand (input or output array).
        PyArray_Descr **op_dtypes;

        // a table that stores how each operand’s axes map to the iterator’s axes
        int op_axes_arrays[NPY_MAXARGS][NPY_MAXDIMS];
        
        // pointers that each point to one row of op_axes_arrays, one per operand
        int *op_axes[NPY_MAXARGS];

        //pointer that each point to one row of op_axes_arrays, one per operand
        npy_uint32 iter_flags;

        // flags for each operand describing how it is used in the iterator (read-only, writeable, or read-write)
        //NPY_MAXARG = the maximum number of input arrays you can pass into this operation.
        npy_uint32 op_flags[NPY_MAXARGS];

        // iter = apointer to a NumPy iterator object used to loop over array elements efficiently
        NpyIter *iter = NULL;

        //stride = apointer to the number of bytes to move in memory to go from one element to the next along an axis
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
        //
        //Fill the entire label_counts array with zeros.
        memset(label_counts, 0, sizeof(label_counts));

        for(int iop = 0; iop < nop; ++iop){
            //Find how many characters appear in subscripts before we hit either a comma (,) or a dash (-).
            //Then store that number in length.
            //subscripts is the string of subscripts for einstein summation.
            //
            //ij,jk->ik
            //"ij" and "jk" are individual subscripts
            //input1_labels,input2_labels->output_labels
            //operand1     ,operand2     ->result
            //             ^             ^ 
            int length = (int)strcspn(subscripts, ",-");
            
            //If this is the last operand, but there’s still a comma in the subscripts string something’s wrong.
            if (iop == nop-1 && subscripts[length] == ',') {
                PyErr_SetString(PyExc_ValueError,
                            "more operands provided to einstein sum function "
                            "than specified in the subscripts string");
                return NULL;
            }

            //If we’re not yet on the last operand, but there’s no comma separating this one from the next that’s an error.
            else if(iop < nop-1 && subscripts[length] != ',') {
                PyErr_SetString(PyExc_ValueError,
                            "fewer operands provided to einstein sum function "
                            "than specified in the subscripts string");
                return NULL;
            }
            //if parse_operand_subscripts returns less than zero return null
            //parse_operand_subscripts
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

