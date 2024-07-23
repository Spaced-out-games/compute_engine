

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

#include "SIMD_float.h"

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
template <size_t num_arrays, size_t array_size>
void simd_operation_thread(float** arrays, SIMD_operation simd_op, size_t start, size_t end) {
    SIMD_vecf* simd_arrays[num_arrays];
    for (size_t i = 0; i < num_arrays; ++i) {
        simd_arrays[i] = reinterpret_cast<SIMD_vecf*>(arrays[i]);
    }

    for (size_t i = start; i < end; i += SIMD_VECTOR_SIZE) {
        simd_op(simd_arrays, i / SIMD_VECTOR_SIZE);
    }
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
void call_SIMD_operation(float** arrays, SIMD_operation simd_op) {
    size_t num_threads = 4;
    size_t chunk_size = (array_size / SIMD_VECTOR_SIZE) / num_threads * SIMD_VECTOR_SIZE;
    size_t leftovers = array_size % SIMD_VECTOR_SIZE;
    size_t cutoff = array_size - leftovers;

    std::vector<std::thread> threads;
     for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? cutoff : (t + 1) * chunk_size;
        threads.push_back(std::thread(simd_operation_thread<num_arrays, array_size>, arrays, simd_op, start, end));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // If there is leftover data to work through, copy the leftover arguments into an array of arguments, create pointers to each element, call the function using the argument array, then copy back to the main array
    if (leftovers > 0) {
        // Array to store leftover elements
        SIMD_vecf leftover_arguments[num_arrays];
        SIMD_vecf* arg_pointers[num_arrays];
        for (size_t i = 0; i < num_arrays; i++)
        {
            memcpy(&(leftover_arguments[i].data), &arrays[i][cutoff], sizeof(float) * leftovers);
            arg_pointers[i] = &leftover_arguments[i];
        }
        
        simd_op(arg_pointers, 0);
  
        for (size_t i = 0; i < num_arrays; i++)
        {
            memcpy(&arrays[i][cutoff], &(leftover_arguments[i].data), sizeof(float) * leftovers);
        }


    }
}

template <size_t num_arrays>
float** gen_arrays() {
    float* xy = new float[TEST_SIZE * num_arrays];
    float** inputs = new float* [num_arrays];

    for (size_t i = 0; i < num_arrays; i++) {
        inputs[i] = xy + TEST_SIZE * i;
        for (size_t j = 0; j < TEST_SIZE; j++) {
            inputs[i][j] = static_cast<float>(j); // Ensuring type consistency
        }
    }
    return inputs;
}

int main()
{
    std::cout << std::fixed << std::setprecision(2);
    auto inputs = gen_arrays<2>();
    // Timing the SIMD operation
    auto start = std::chrono::high_resolution_clock::now();

    call_SIMD_operation<2, TEST_SIZE>(inputs, pythagorean_theorum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << std::setprecision(8) << "\nSIMD operation took " << duration.count() << " seconds.\n\n";

    
}
