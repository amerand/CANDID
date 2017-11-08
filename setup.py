from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='candid',
      py_modules=['candid'],
      author='Antoine Merand',
      author_email='antoine.merand@gmail.com',
      url='https://github.com/amerand/CANDID',
      ext_modules = cythonize('cvis.pyx',
                              include_path=[numpy.get_include()],
                              libraries=["m"],
                              extra_compile_args = ["-ffast-math"]),
      include_dirs=[numpy.get_include()],
      )
