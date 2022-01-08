from distutils.core import setup, Extension

setup(name = 'myModule', version = '1.0.0',  \
   ext_modules = [Extension('myModule', ['py_hello.c'])])
