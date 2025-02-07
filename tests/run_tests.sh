#!/bin/sh

echo "\033[35mTraining test model, and comparing against PyTorch's Adam implementation.\033[0m"
python test_cpu_adam.py
echo

echo "\033[35mBenchmarking vectorized implementations vs naive C implementation.\033[0m"
gcc benchmark_vs_naive.c -lm -O3 -march=native -fno-math-errno -o benchmark_vs_naive && ./benchmark_vs_naive && rm benchmark_vs_naive
echo
