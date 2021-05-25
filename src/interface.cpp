#include "FLOPART.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include <numpy/arrayobject.h>


static PyObject * FLOPARTInterface(PyObject *self, PyObject *args)
{
     PyArrayObject *inputData, *weight_vec, *input_label_start, *input_label_end, *input_label_changes;

    int lenData, lenLabels;
    double penalty;

    if(!PyArg_ParseTuple(args, "O!O!iO!O!O!id",
                         &PyArray_Type, &inputData,
                         &PyArray_Type, &weight_vec,
                         &lenData,
                         &PyArray_Type, &input_label_start,
                         &PyArray_Type, &input_label_end,
                         &PyArray_Type, &input_label_changes,
                         &lenLabels,
                         &penalty))
    {
        PyErr_SetString(PyExc_TypeError, "Parse Tuple Error");
        return NULL;
    }
    if(PyArray_TYPE(inputData) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "Input Data must be numpy.ndarray type int");
        return NULL;
    }
    if(PyArray_TYPE(weight_vec) != NPY_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "weight vec must be numpy.ndarray type double");
        return NULL;
    }
    if(PyArray_TYPE(input_label_start) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "Input Label Start Data must be numpy.ndarray type int");
        return NULL;
    }
    if(PyArray_TYPE(input_label_end) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "Input Label End Data must be numpy.ndarray type int");
        return NULL;
    }
    if(PyArray_TYPE(input_label_changes) != NPY_INT)
    {
        PyErr_SetString(PyExc_TypeError, "Input Label changes Data must be numpy.ndarray type int");
        return NULL;
    }

    // Input Data Formatting
    int *inputDataA = (int*)PyArray_DATA(inputData);
    double *weight_vecA = (double*)PyArray_DATA(weight_vec);
    int *input_label_startA = (int*)PyArray_DATA(input_label_start);
    int *input_label_endA = (int*)PyArray_DATA(input_label_end);
    int *input_label_changesA = (int*)PyArray_DATA(input_label_changes);

    // Output Data Formatting
    npy_intp col_dim = PyArray_DIM(inputData, 0);
    npy_intp doubleCol_dim = col_dim * 2;

    PyArrayObject *out_end, *out_cost, *out_mean, *out_intervals_mat, *out_state_vec;

    out_end = (PyArrayObject*)PyArray_ZEROS(1, &col_dim, NPY_INT, 0);
    int *out_endA = (int*)PyArray_DATA(out_end);

    // 2x Data count
    out_cost = (PyArrayObject*)PyArray_ZEROS(1, &doubleCol_dim, NPY_DOUBLE, 0);
    double *out_costA = (double*)PyArray_DATA(out_cost);

    out_mean = (PyArrayObject*)PyArray_ZEROS(1, &col_dim, NPY_DOUBLE, 0);
    double *out_meanA = (double*)PyArray_DATA(out_mean);

    // 2x Data count
    out_intervals_mat = (PyArrayObject*)PyArray_ZEROS(1, &doubleCol_dim, NPY_INT, 0);
    int *out_intervals_matA = (int*)PyArray_DATA(out_intervals_mat);

    out_state_vec = (PyArrayObject*)PyArray_ZEROS(1, &col_dim, NPY_INT, 0);
    int *out_state_vecA = (int*)PyArray_DATA(out_state_vec);

    int status = FLOPART(inputDataA,
                         weight_vecA,
                         lenData,
                         penalty,
                         input_label_changesA,
                         input_label_startA,
                         input_label_endA,
                         lenLabels,
                         out_costA,
                         out_endA,
                         out_meanA,
                         out_intervals_matA,
                         out_state_vecA);

    if(status == ERROR_MIN_MAX_SAME)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Min and max same in data");

        return NULL;
    }
    else if(status == ERROR_UNRECOGNIZED_LABEL_TYPE)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Unrecognized Label Type");

        return NULL;
    }
    else if(status == ERROR_LABEL_START_SHOULD_BE_GREATER_THAN_PREVIOUS_LABEL_END)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Label Start should be greater than previous label end");

        return NULL;
    }
    else if(status == ERROR_LABEL_END_MUST_BE_AT_LEAST_LABEL_START)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Label end must be at least label start");

        return NULL;
    }
    else if(status == ERROR_LABEL_END_MUST_BE_LESS_THAN_DATA_SIZE)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Label End must be less than data size");

        return NULL;
    }
    else if(status == ERROR_LABEL_END_MUST_BE_LESS_THAN_DATA_SIZE)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Label start must be at least 0");

        return NULL;
    }
    else if(status != 0){
        PyErr_SetString(PyExc_ValueError,
                        "Error");

        return NULL;
    }

    PyArrayObject *seg_mean_vec, *seg_start_vec, *seg_end_vec, *seg_state_vec;

    //convert to segments data table.
    npy_intp seg_count=1;
    while(seg_count < lenData && 0 <= out_state_vecA[seg_count-1]){
    seg_count++;
    }

    seg_mean_vec = (PyArrayObject*)PyArray_ZEROS(1, &seg_count, NPY_INT, 0);
    int *seg_mean_vecA = (int*)PyArray_DATA(seg_mean_vec);

    seg_start_vec = (PyArrayObject*)PyArray_ZEROS(1, &seg_count, NPY_INT, 0);
    int *seg_start_vecA = (int*)PyArray_DATA(seg_start_vec);

    seg_end_vec = (PyArrayObject*)PyArray_ZEROS(1, &seg_count, NPY_INT, 0);
    int *seg_end_vecA = (int*)PyArray_DATA(seg_end_vec);

    seg_state_vec = (PyArrayObject*)PyArray_ZEROS(1, &seg_count, NPY_INT, 0);
    int *seg_state_vecA = (int*)PyArray_DATA(seg_state_vec);

    for(int seg_i=0; seg_i < seg_count; seg_i++){
    int mean_index = seg_count-1-seg_i;
    seg_mean_vecA[seg_i] = out_meanA[mean_index];
    seg_state_vecA[seg_i] = out_state_vecA[mean_index];
    if(mean_index==0){
      seg_end_vecA[seg_i] = lenData;
    }else{
      seg_end_vecA[seg_i] = out_endA[mean_index-1]+1;
    }
    if(seg_i==0){
      seg_start_vecA[seg_i] = 1;
    }else{
      seg_start_vecA[seg_i] = out_endA[mean_index]+2;
    }
    }

    PyObject * output = Py_BuildValue("{s:O,s:O,s:O,s:O,s:O,s:O}",
                                      "cost_mat", out_cost,
                                      "intervals_mat", out_intervals_mat,
                                      "segment_starts", seg_start_vec,
                                      "segment_ends", seg_end_vec,
                                      "segment_means", seg_mean_vec,
                                      "segment_states", seg_state_vec);

    return output;
}

static PyMethodDef Methods[] = {
        {"interface", FLOPARTInterface, METH_VARARGS,
                        "the interface for FLOPART"},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduleDef =
        {
        PyModuleDef_HEAD_INIT,
        "FLOPARTInterface",
        "A Python extension for FLOPART",
        -1,
        Methods
        };


PyMODINIT_FUNC
PyInit_FLOPARTInterface(void)
{
    import_array();
    PyObject *m;
    m = PyModule_Create(&moduleDef);
    if (m == NULL)
        return NULL;
    if (PyErr_Occurred()) return NULL;
    return m;
}
