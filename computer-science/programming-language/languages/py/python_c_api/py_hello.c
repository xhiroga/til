// #include <Python.h>
#include "Python.h"

// C function 'hello world'
static PyObject* c_helloworld(PyObject* self, PyObject* args)
{
    printf("Hello World\n");
    return Py_None;
}

// myModule definition(python's name)
static PyMethodDef myMethods[] = {
    { "helloworld", c_helloworld, METH_NOARGS, "Prints Hello World" },
    { NULL }
};

// myModule definition struct
static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "myModule",
    "Python3 C API Module(Sample 1)",
    -1,
    myMethods
};

// Initializes myModule
PyMODINIT_FUNC PyInit_myModule(void)
{
    return PyModule_Create(&myModule);
}
