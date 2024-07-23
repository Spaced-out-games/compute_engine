#include "SIMD_float.h"


template <typename T, size_t num_arrays, size_t array_size>
class weaved_array {
public:
    weaved_array();
    ~weaved_array();

    T* getArray(size_t index) const;
    T get(size_t array_index, size_t element_index) const;
    void set(size_t array_index, size_t element_index, const T& value);

private:
    T* data;
    T** arrays;
};

template <typename T, size_t num_arrays, size_t array_size>
weaved_array<T, num_arrays, array_size>::weaved_array() {
    // Allocate a contiguous block of memory for all arrays interleaved
    data = new T[num_arrays * array_size];
    arrays = new T * [num_arrays];

    // Initialize the pointers in the arrays array
    for (size_t i = 0; i < num_arrays; ++i) {
        arrays[i] = data + i * array_size;
    }
}

template <typename T, size_t num_arrays, size_t array_size>
weaved_array<T, num_arrays, array_size>::~weaved_array() {
    delete[] data;
    delete[] arrays;
}

template <typename T, size_t num_arrays, size_t array_size>
T* weaved_array<T, num_arrays, array_size>::getArray(size_t index) const {
    if (index >= num_arrays) {
        throw std::out_of_range("Array index out of range");
    }
    return arrays[index];
}

template <typename T, size_t num_arrays, size_t array_size>
T weaved_array<T, num_arrays, array_size>::get(size_t array_index, size_t element_index) const {
    if (array_index >= num_arrays || element_index >= array_size) {
        throw std::out_of_range("Index out of range");
    }
    return arrays[array_index][element_index];
}

template <typename T, size_t num_arrays, size_t array_size>
void weaved_array<T, num_arrays, array_size>::set(size_t array_index, size_t element_index, const T& value) {
    if (array_index >= num_arrays || element_index >= array_size) {
        throw std::out_of_range("Index out of range");
    }
    arrays[array_index][element_index] = value;
}
