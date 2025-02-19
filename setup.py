from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

DEBUG = False

setup(
    name='CPUOptimizer',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            'CPUOptimizer.bindings',
            ['CPUOptimizer/bindings.cpp'],
            extra_compile_args=['-O3', '-march=native', '-fno-math-errno', '-Wno-aggressive-loop-optimizations'] +
                              (['-g', '-fno-omit-frame-pointer',] if DEBUG else []),
            extra_link_args=['-lm', '-lquadmath'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    setup_requires=['torch'],
    install_requires=['torch']
)
