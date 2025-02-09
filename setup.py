from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='CPUOptimizer',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            'CPUOptimizer.bindings',
            ['CPUOptimizer/bindings.cpp'],
            extra_compile_args=['-lm', '-O3', '-march=native', '-fno-math-errno'], # '-fsanitize=address', '-g', '-fno-omit-frame-pointer', '-fsanitize=undefined'
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    setup_requires=['torch'],
    install_requires=['torch']
)
