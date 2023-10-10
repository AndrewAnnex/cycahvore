import numpy
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize


extensions = [
    Extension(
        name="cycahvore.cahv",
        sources=[
            './src/VICAR/vos/p2/sub/cahvor/cmod_cahv.c',
            './src/VICAR/vos/p2/sub/cahvor/cmod_error_unique.c',
            './src/VICAR/vos/p2/sub/mat3/mat3.c',
            './src/cycahvore/cahv.pyx',

        ],
        include_dirs=[
            './src/VICAR/vos/p2/inc/',
            './src/VICAR/vos/p2/sub/mat3/',
            './src/VICAR/vos/p2/sub/cahvor/',
             numpy.get_include(),
        ],
        language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=[],
    ),
    Extension(
        name="cycahvore.cahvor",
        sources=[
            './src/VICAR/vos/p2/sub/cahvor/cmod_cahvor.c',
            './src/VICAR/vos/p2/sub/cahvor/cmod_cahv.c',
            './src/VICAR/vos/p2/sub/cahvor/cmod_error_unique.c',
            './src/VICAR/vos/p2/sub/mat3/mat3.c',
            './src/cycahvore/cahvor.pyx',
        ],
        include_dirs=[
            './src/VICAR/vos/p2/inc/',
            './src/VICAR/vos/p2/sub/mat3/',
            './src/VICAR/vos/p2/sub/cahvor/',
            numpy.get_include(),
        ],
        language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=[],
    ),
    Extension(
        name="cycahvore.cahvore",
        sources=[
            './src/VICAR/vos/p2/sub/cahvor/cmod_cahvore.c',
            './src/VICAR/vos/p2/sub/cahvor/cmod_cahvor.c',
            './src/VICAR/vos/p2/sub/cahvor/cmod_cahv.c',
            './src/VICAR/vos/p2/sub/cahvor/cmod_psph.c',
            './src/VICAR/vos/p2/sub/cahvor/cmod_error_unique.c',
            './src/VICAR/vos/p2/sub/mat3/mat3.c',
            './src/cycahvore/cahvore.pyx',
        ],
        include_dirs=[
            './src/VICAR/vos/p2/inc/',
            './src/VICAR/vos/p2/sub/mat3/',
            './src/VICAR/vos/p2/sub/cahvor/',
            numpy.get_include(),
        ],
        language="c",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=[],
    )
]

setup(
    name="cycahvore",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=cythonize(extensions, language_level=3, annotate=True, nthreads=1),
)
