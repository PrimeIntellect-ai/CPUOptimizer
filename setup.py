from setuptools import setup
from setuptools.command.build_ext import build_ext
import sys

class BuildExt(build_ext):
    def build_extensions(self):
        try:
            # Import PyTorch build tools only when building extensions
            from torch.utils.cpp_extension import BuildExtension, CppExtension
            
            # Create the PyTorch extension
            extension = CppExtension(
                'cpu_optimizer_bindings',
                ['bindings.cpp'],
                extra_compile_args=['-lm', '-O3', '-march=native', '-fno-math-errno']
            )
            
            # Use PyTorch's BuildExtension
            torch_build_ext = BuildExtension()
            torch_build_ext.extensions = [extension]
            torch_build_ext.build_extensions()
            
        except ImportError:
            sys.exit(
                "Error: PyTorch is required to build this extension.\n"
                "Please install PyTorch first: pip install torch"
            )

setup(
    name='cpu_optimizer_bindings',
    version='0.1.0',
    install_requires=['torch'],
    cmdclass={'build_ext': BuildExt}
)
