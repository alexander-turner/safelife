#define NO_IMPORT_ARRAY

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "random.h"

#define RAND_BUFFER_SIZE 10000

PyObject *rand_buffer_obj = NULL;
double *rand_buffer = NULL;
int buffer_pos = RAND_BUFFER_SIZE;

static void reset_buffer(void) {
    // Note that we're not doing error checking here.
    // We're just going to assume that numpy is installed and accessible.
    Py_XDECREF(rand_buffer_obj);
    PyObject *numpy = PyImport_ImportModule("numpy.random");
    PyObject *np_random = PyObject_GetAttrString(numpy, "random");
    rand_buffer_obj = PyObject_CallFunction(np_random, "i", RAND_BUFFER_SIZE);
    rand_buffer = (double *)PyArray_DATA((PyArrayObject *)rand_buffer_obj);
    buffer_pos = 0;

    Py_DECREF(numpy);
    Py_DECREF(np_random);
}

void random_seed(uint64_t seed) {
    PyObject *numpy = PyImport_ImportModule("numpy.random");
    PyObject *np_random = PyObject_GetAttrString(numpy, "seed");
    PyObject *rval = PyObject_CallFunction(np_random, "K", seed);
    Py_DECREF(numpy);
    Py_DECREF(np_random);
    Py_DECREF(rval);

    reset_buffer();
}

double random_float(void) {
    if (buffer_pos >= RAND_BUFFER_SIZE) {
        reset_buffer();
    }
    return rand_buffer[buffer_pos++];
}

int32_t random_int(int32_t high) {

    return random_float() * high;
}
