#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <stdint.h>
#include <immintrin.h>

// Size of arrays (100 GB)
#define ARRAY_SIZE (100ULL * 1024 * 1024 * 1024)
#define ELEMENT_SIZE sizeof(float)
#define NUM_ELEMENTS (ARRAY_SIZE / ELEMENT_SIZE)
#define AVX512_ALIGN 64  // AVX-512 requires 64-byte alignment
#define VECTOR_SIZE 16   // Number of floats in AVX-512 vector

// Function to get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Function to run bandwidth test with given alignment
void run_bandwidth_test(const char* test_name, int offset) {
    printf("\nRunning %s...\n", test_name);
    
    // Allocate memory based on test type
    float* a = (float*)aligned_alloc(AVX512_ALIGN, ARRAY_SIZE + AVX512_ALIGN);
    float* b = (float*)aligned_alloc(AVX512_ALIGN, ARRAY_SIZE + AVX512_ALIGN);
    
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }

    // Apply offset if needed
    if (offset > 0) {
        a += offset;
        b += offset;
    }
    
    // Initialize arrays
    printf("Initializing arrays...\n");
    for (uint64_t i = 0; i < NUM_ELEMENTS; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Perform bandwidth test
    printf("Starting bandwidth test...\n");
    double start_time = get_time();

    // Main computation loop using AVX-512
    if (offset == 0) {
        // Aligned access
        for (uint64_t i = 0; i < NUM_ELEMENTS; i += VECTOR_SIZE) {
            __m512 va = _mm512_load_ps(&a[i]);
            __m512 vb = _mm512_load_ps(&b[i]);
            __m512 vc = _mm512_add_ps(va, vb);
            _mm512_store_ps(&a[i], vc);
        }
    } else {
        // Unaligned access
        for (uint64_t i = 0; i < NUM_ELEMENTS; i += VECTOR_SIZE) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vc = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(&a[i], vc);
        }
    }

    double end_time = get_time();
    double elapsed_time = end_time - start_time;

    // Calculate bandwidth
    double bytes_processed = 3.0 * ARRAY_SIZE; // Two reads and one write
    double bandwidth = bytes_processed / (elapsed_time * 1024*1024*1024); // GB/s

    printf("Test completed in %.2f seconds\n", elapsed_time);
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);

    // Verify results (sample check)
    printf("Verifying results...\n");
    int errors = 0;
    for (uint64_t i = 0; i < NUM_ELEMENTS; i += 1000000) {
        if (a[i] != 3.0f) {
            errors++;
            if (errors < 10) {
                printf("Value at index %lu: %.2f\n", i, a[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("Verification passed\n");
    } else {
        printf("Verification failed: %d errors found\n", errors);
    }

    // Clean up (adjust pointer back before freeing if offset was used)
    if (offset > 0) {
        a -= offset;
        b -= offset;
    }
    free(a);
    free(b);
}

int main() {
    printf("Running memory bandwidth tests on %.2f GB arrays\n", (double)ARRAY_SIZE / (1024*1024*1024));
    
    // Run aligned test
    run_bandwidth_test("Aligned AVX-512 Test", 0);

    // Run unaligned test (offset by one float)
    run_bandwidth_test("Unaligned AVX-512 Test", 1);

    return 0;
}
