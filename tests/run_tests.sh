#!/bin/sh

python test_cpu_adam.py

c++ benchmark_vs_naive.cpp -lm -O3 -march=native -fno-math-errno -o benchmark_vs_naive && ./benchmark_vs_naive && rm benchmark_vs_naive
