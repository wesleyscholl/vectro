from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Cython extension for high-performance quantization
extensions = [
    Extension(
        "vectro.quantizer_cython",
        ["src/quantizer_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
)
