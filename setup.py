#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils import build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension
import numpy as np


ext_modules = [Extension("malis_large_volumes.malis_cython",
               ["malis_large_volumes/malis_cython.pyx"],
               language='c++',
               extra_compile_args=["-std=c++11"],
               extra_link_args=["-std=c++14"],
               include_dirs=[np.get_include()])]

setup(name="malis_large_volumes",
      version="0.0.1",
      cmdclass = {'build_ext': build_ext},
      ext_modules=ext_modules,
      packages=find_packages()
)
