from Cython.Distutils import build_ext
import setuptools
import numpy as np

ext_modules = [setuptools.extension.Extension("malis_large_volumes.malis_cython",
               ["malis_large_volumes/malis_cython.pyx"],
               language='c++',
               extra_compile_args=["-std=c++14"],
               extra_link_args=["-std=c++14"],
               include_dirs=[np.get_include()])]

setuptools.setup(name="malis_large_volumes",
                 version="0.0.1",
                 cmdclass = {'build_ext': build_ext},
                 ext_modules=ext_modules,
                 packages=["malis_large_volumes"]
)
