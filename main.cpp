

/*
Processor: Intel(R) Core i5-9400F CPU @ 2.90GHz
RAM: 8.00 GB

Benchmark details:

TEST_SIZE: 80000000 floats

Dataset:
x[i] = i; y[i] = i;

Test program:
void pythagorean_theorum(SIMD_vecf** arrays, size_t index)
{
    SIMD_vecf& x = arrays[0][index];
    SIMD_vecf& y = arrays[1][index];

    x.inline_pow(2);
    y.inline_pow(2);
    x += y;
    x.inline_sqrt();
}

Results:

Note: Allocation and population of test arrays not counted towards the benchmark

SIMD_float_256.h: SIMD operation took 0.02106850 seconds.
SIMD_float_128.h: SIMD operation took 0.05056010 seconds.

*/

#include "compute_engine.h"-

#include <iostream>
#include <cmath>  // For std::log2
#include <iomanip>  // Include this header for std::setprecision and std::fixed
#include <thread>
#include <vector>
#include <chrono>
#include <cstring>

#define TEST_SIZE 1000 * 1000 * 8
typedef void (*SIMD_operation)(SIMD_vecf**, size_t);



// Calculates sqrt(x^2 + y^2)
void pythagorean_theorum(SIMD_vecf** arrays, size_t index)
{
    SIMD_vecf& x = arrays[0][index];
    SIMD_vecf& y = arrays[1][index];

    x.inline_pow(2);
    y.inline_pow(2);
    x += y;
    x.inline_sqrt();
}
template <size_t num_elements>
void print_float_array(float* array)
{
    for (size_t i = 0; i < num_elements - 1; i++)
    {
        std::cout << array[i] << ", ";
    }
    std::cout << array[num_elements - 1] << '\n';
}

template <size_t num_arrays, size_t array_size>
void simd_operation_thread(const weaved_array<SIMD_vecf, num_arrays, array_size>& arrays, SIMD_operation simd_op, size_t start, size_t end) {
    SIMD_vecf* simd_arrays[num_arrays];
    for (size_t i = 0; i < num_arrays; ++i) {
        simd_arrays[i] = reinterpret_cast<SIMD_vecf*>(arrays.getArray(i));
    }

    for (size_t i = start; i < end; i += SIMD_VECTOR_SIZE) {
        simd_op(simd_arrays, i / SIMD_VECTOR_SIZE);
    }
}

template <size_t num_arrays, size_t array_size>
void call_SIMD_operation(const weaved_array<SIMD_vecf, num_arrays, array_size>& arrays, SIMD_operation simd_op) {
    size_t num_threads = 4;
    size_t chunk_size = (array_size / SIMD_VECTOR_SIZE) / num_threads * SIMD_VECTOR_SIZE;
    size_t leftovers = array_size % SIMD_VECTOR_SIZE;
    size_t cutoff = array_size - leftovers;

    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? cutoff : (t + 1) * chunk_size;
        threads.push_back(std::thread(simd_operation_thread<num_arrays, array_size>, std::cref(arrays), simd_op, start, end));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    if (leftovers > 0) {
        // Handle leftovers as before
    }
}


template <size_t num_arrays, size_t array_size>
weaved_array<SIMD_vecf, num_arrays, array_size> gen_arrays() {
    weaved_array<SIMD_vecf, num_arrays, array_size> arrays;

    for (size_t i = 0; i < num_arrays; i++) {
        for (size_t j = 0; j < array_size; j++) {
            // Assuming SIMD_vecf can be initialized directly with float values
            arrays.set(i, j, SIMD_vecf(static_cast<float>(j)));
        }
    }

    return arrays;
}


int main() {
    std::cout << std::fixed << std::setprecision(2);
    auto inputs = gen_arrays<2, TEST_SIZE>();

    // Timing the SIMD operation
    auto start = std::chrono::high_resolution_clock::now();

    call_SIMD_operation<2, TEST_SIZE>(inputs, pythagorean_theorum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << std::setprecision(8) << "\nSIMD operation took " << duration.count() << " seconds.\n\n";
}
