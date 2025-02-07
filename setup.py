from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='offload_adam',
    ext_modules=[
        CppExtension(
            'offload_adam',
            ['offload_adam_bindings.cpp'],
            extra_compile_args=['-lm', '-O3', '-march=native', '-fno-math-errno'], #  '-fsanitize=address', '-g', '-fno-omit-frame-pointer', '-fsanitize=undefined'
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
