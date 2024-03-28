'''
COMPILE: python build_cython.py build_ext --inplace
'''

from setuptools import setup
from Cython.Build import cythonize
import numpy
from setuptools.command.build_ext import build_ext as _build_ext
import shutil
import os

class build_ext(_build_ext):
    def run(self):
        _build_ext.run(self)

        # Move .pyd file to lib/
        for ext in self.extensions:
            ext_path = self.get_ext_fullpath(ext.name)
            shutil.move(ext_path, os.path.join('lib', os.path.basename(ext_path)))

        # Delete lib/process_row.c
        if os.path.exists('lib/process_row.c'):
            os.remove('lib/process_row.c')

        # Delete build directory
        if os.path.exists('build'):
            shutil.rmtree('build')

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules = cythonize("lib/process_row.pyx"),
    include_dirs=[numpy.get_include()]
)
