from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

os.environ['CFLAGS'] = '-O3 -ffast-math -std=c99 -march=native'

setup(name='candid',
      version='0.3.1',
      py_modules=['candid'],
      author='Antoine Merand',
      author_email='antoine.merand@gmail.com',
      url='https://github.com/amerand/CANDID',
      ext_modules = cythonize('cyvis.pyx',
                              include_path=[numpy.get_include()]),
      include_dirs=[numpy.get_include()],
      )
