from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("malis_cython",
                               ["malis_cython.pyx"],
                               language='c++',
                               extra_compile_args=["-std=c++11"],
                               extra_link_args=["-std=c++11"],
                               include_dirs=[np.get_include()])]
)
