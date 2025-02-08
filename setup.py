from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='CPUOptimizer',  # Changed to match your package name
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            'CPUOptimizer.bindings',
            ['CPUOptimizer/bindings.cpp'],
            extra_compile_args=['-lm', '-O3', '-march=native', '-fno-math-errno'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    setup_requires=['torch'],
    install_requires=['torch']
)
