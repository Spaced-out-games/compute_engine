#pragma once
#include <immintrin.h>
#include <array>
#include <iostream>
#include <algorithm>

#define SIMD_VECTOR_SIZE 4





/* Abstract SIMD float vector type. The size of the vector varies based on what the CPU you are compiling for can handle.
For machines with AVX-512 support, this will store a __m128512 and call the appropriate operands.
For machines with only AVX2, this will store a __m128. SSE2? __m128. No support? Defaults to float.

Since this is meant to be an abstraction for use in parallelized systems, all operations are executed using the appropriate instruction for their vector size.
For high-end machines, pow() executes _mm512_pow_ps on the underlying __m128512, but on an old PC without SIMD support, pow() is just the standard C implementation, no SIMD tricks.

send a function pointer to the driver, and have it run your task using SIMD (if available) on as many cores as the CPU can handle.

Usage:
If you want to use SIMD_vecf, simply include the header in your project and create instances as so:

SIMD_vecf x(5.0f); // All elements are zero
SIMD_vecf y = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}; // specify each individually
SIMD_vecf mask = (y < 5); // Compares each element of y to 5. 1.0f if true, 0.0f otherwise.
y += 3.0; // Add 3.0 to each element of y
std::cout << y; // print y
y.inline_pow(mask + 1.0f); // For any element < 5, it squares it, otherwise, it stays the same.

On its own, this is pretty useless, however, the power becomes clear when you create peice-wise functions and pass a pointer to the function to the compute engine.

Performance tips:
Try to use the inline variants of every function if at all possible. Instead of returning the expression, it just changes the calling instance's data instead of returning an altered copy.
In a single call, this is nigh meaningless, but at scale, it's incredibly detrimental to the performance of your application.

Another suggested method is to partition your data in such a way so that it's as contiguous as possible. If you have two arrays, X, and Y, the last element of X should touch the first element of Y.
This makes cache happy and reduces the amount of reads / writes to disk we perform, which is often the largest bottleneck in computing - memory bandwidth

You should also stay away from branching. If you need conditional computation, use the select() method. It takes in the output if(true), and the output if(false), as well as a mask
*/
struct SIMD_vecf {
    __m128 data;


    /* --------------------------------CONSTRUCTORS------------------------------------*/

    // Basic constructor, doesn't initialize the data
    SIMD_vecf() {}

    // Initialize a SIMD_vecf from a float. Copies initial_data in every slot
    SIMD_vecf(float initial_data) : data(_mm_set1_ps(initial_data)) {}


    // Constructor to initialize with std::initializer_list
    SIMD_vecf(std::initializer_list<float> init_list) {
        float temp[8] = { 0.0f }; // Initialize to zeroes
        std::copy(init_list.begin(), init_list.end(), temp);
        data = _mm_loadu_ps(temp);
    }

    // Constructor to initialize with std::array
    SIMD_vecf(const std::array<float, 8>& arr) {
        data = _mm_loadu_ps(arr.data());
    }

    // Constructor to initialize with an array of 8 floats
    SIMD_vecf(const float* initial_data) {
        data = _mm_loadu_ps(initial_data);
    }


    // Gets a vector of ones
    SIMD_vecf ones() const {
        return SIMD_vecf(constants[1]);

    }

    // Gets a vector of zeroes
    SIMD_vecf zeroes() const {
        return SIMD_vecf(constants[5]);
    }

    /* -----------------------Arithmetic w/SIMD_vecf-------------------------- */

    // Overload the + operator for SIMD_vecf
    SIMD_vecf operator+(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_add_ps(data, other.data));
    }

    // Overload the - operator for SIMD_vecf
    SIMD_vecf operator-(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_sub_ps(data, other.data));
    }

    // Overload the / operator for SIMD_vecf
    SIMD_vecf operator/(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_div_ps(data, other.data));
    }

    // Overload the * operator for SIMD_vecf
    SIMD_vecf operator*(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_mul_ps(data, other.data));
    }

    SIMD_vecf operator%(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_fmod_ps(data, other.data));
    }

    /* -------------------- Arithmetic with standard 32-bit float -------------------- */
        // Overload the + operator for SIMD_vecf
    SIMD_vecf operator+(const float other) const {
        __m128 vector = _mm_set1_ps(other);

        return SIMD_vecf(_mm_add_ps(data, vector));
    }

    // Overload the - operator for SIMD_vecf
    SIMD_vecf operator-(const float other) const {
        __m128 vector = _mm_set1_ps(other);
        return SIMD_vecf(_mm_sub_ps(data, vector));
    }

    // Overload the / operator for SIMD_vecf
    SIMD_vecf operator/(const float other) const {
        __m128 vector = _mm_set1_ps(other);

        return SIMD_vecf(_mm_div_ps(data, vector));
    }

    // Overload the * operator for SIMD_vecf
    SIMD_vecf operator*(const float other) const {
        __m128 vector = _mm_set1_ps(other);
        return SIMD_vecf(_mm_mul_ps(data, vector));
    }

    SIMD_vecf operator%(const float& other) const {
        __m128 vector = _mm_set1_ps(other);
        return SIMD_vecf(_mm_fmod_ps(data, vector));
    }

    SIMD_vecf operator-() const {
        return SIMD_vecf(_mm_sub_ps(zeroes().data, data));
    }

    /* -------------------------- inline Arithmetic w/SIMD_vecf -------------------------- */

        // Overload the += operator for SIMD_vecf
    SIMD_vecf& operator+=(const SIMD_vecf& other) {
        data = _mm_add_ps(data, other.data);
        return *this;
    }

    // Overload the -= operator for SIMD_vecf
    SIMD_vecf& operator-=(const SIMD_vecf& other) {
        data = _mm_sub_ps(data, other.data);
        return *this;
    }

    // Overload the /= operator for SIMD_vecf
    SIMD_vecf& operator/=(const SIMD_vecf& other) {
        data = _mm_div_ps(data, other.data);
        return *this;
    }

    // Overload the *= operator for SIMD_vecf
    SIMD_vecf& operator*=(const SIMD_vecf& other) {
        data = _mm_mul_ps(data, other.data);
        return *this;

    }

    SIMD_vecf& operator%=(const SIMD_vecf& other) {
        data = _mm_fmod_ps(data, other.data);
        return *this;

    }


    /* -------------------------- inline Arithmetic w/32-bit float -------------------------- */

    // Overload the += operator for SIMD_vecf
    SIMD_vecf& operator+=(const float& other) {
        __m128 vector = _mm_set1_ps(other);
        data = _mm_add_ps(data, vector);
        return *this;

    }

    // Overload the -= operator for SIMD_vecf
    SIMD_vecf& operator-=(const float& other) {
        __m128 vector = _mm_set1_ps(other);
        data = _mm_sub_ps(data, vector);
        return *this;

    }

    // Overload the /= operator for SIMD_vecf
    SIMD_vecf& operator/=(const float& other) {
        __m128 vector = _mm_set1_ps(other);
        data = _mm_div_ps(data, vector);
        return *this;

    }

    // Overload the *= operator for SIMD_vecf
    SIMD_vecf& operator*=(const float& other) {
        __m128 vector = _mm_set1_ps(other);
        data = _mm_mul_ps(data, vector);
        return *this;

    }

    SIMD_vecf& operator%=(const float& other) {
        __m128 vector = _mm_set1_ps(other);
        data = _mm_fmod_ps(data, vector);
        return *this;
    }

    /* -------------------------------Exponents, logs, and powers------------------------------------*/

    // Returns this ^ Y
    SIMD_vecf pow(const SIMD_vecf& Y) {
        return SIMD_vecf(_mm_pow_ps(data, Y.data));
    }

    // this = this ^ Y
    void inline_pow(const SIMD_vecf& Y) {
        data = _mm_pow_ps(data, Y.data);
    }

    //  Constructs a vector where every element is y and returns this ^ y.
    SIMD_vecf pow(const float y) {
        __m128 Y = _mm_set1_ps(y);
        return SIMD_vecf(_mm_pow_ps(data, Y));
    }

    // Constructs a vector where every element is y and sets this = this ^ y.
    void inline_pow(const float y) {
        __m128 Y = _mm_set1_ps(y);
        data = _mm_pow_ps(data, Y);
    }

    // Computes the natural log of each element
    SIMD_vecf log() const {
        return SIMD_vecf(_mm_log_ps(data));
    }
    // this = log (this)
    void inline_log() {
        data = _mm_log_ps(data);
    }

    // Returns log2(this)
    SIMD_vecf log2() const {
        return SIMD_vecf(_mm_log2_ps(data));
    }
    // this = log2(this)
    void inline_log2() {
        data = _mm_log2_ps(data);
    }

    // Returns log10(this)
    SIMD_vecf log10() const {
        return SIMD_vecf(_mm_log2_ps(data));
    }
    // This = log10(this)
    void inline_log10() {
        data = _mm_log2_ps(data);
    }

    // Returns exp(this)
    SIMD_vecf exp()
    {
        return SIMD_vecf(_mm_exp_ps(data));
    }
    // this = exp(this)
    void inline_exp()
    {
        data = _mm_exp_ps(data);
    }

    // Returns exp2(this)
    SIMD_vecf exp2()
    {
        return SIMD_vecf(_mm_exp2_ps(data));
    }

    // this = exp2(this)
    void inline_exp2()
    {
        data = _mm_exp2_ps(data);
    }

    // Returns exp10(this)
    SIMD_vecf exp10()
    {
        return SIMD_vecf(_mm_exp10_ps(data));
    }

    // this = exp10(this)
    void inline_exp10()
    {
        data = _mm_exp10_ps(data);
    }

    /* ----------------------------------------Misc.--------------------------------------------------*/

    // Returns ceil (this)
    SIMD_vecf ceil() {
        return _mm_round_ps(data, _MM_FROUND_TO_POS_INF);
    }

    // this = ceil (this)
    void inline_ceil() {
        data = _mm_round_ps(data, _MM_FROUND_TO_POS_INF);
    }

    // Returns floor (this)
    SIMD_vecf floor() {
        return _mm_round_ps(data, _MM_FROUND_TO_NEG_INF);
    }

    // this = floor (this)
    void inline_floor() {
        data = _mm_round_ps(data, _MM_FROUND_TO_NEG_INF);
    }

    // Returns round(this)
    SIMD_vecf round() {
        return _mm_round_ps(data, _MM_FROUND_TO_NEAREST_INT);
    }

    // this = round(this)
    void inline_round() {
        data = _mm_round_ps(data, _MM_FROUND_TO_NEAREST_INT);
    }

    // returns this rounded towards zero
    SIMD_vecf truncate() {
        return _mm_round_ps(data, _MM_FROUND_TO_ZERO);
    }

    // Rounds this towards zero
    void inline_truncate() {
        data = _mm_round_ps(data, _MM_FROUND_TO_ZERO);
    }

    // returns abs(this)
    SIMD_vecf abs() {
        const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        return _mm_and_ps(data, mask);
    }

    // this = abs(this)
    void inline_abs() {
        const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
        data = _mm_and_ps(data, mask);
    }

    /* ------------------------------------------------Trig functions------------------------------------------------- */

    // Consider adding utility functions for degrees / radian conversion

    // Returns sin(this)
    SIMD_vecf sin()
    {
        return SIMD_vecf(_mm_sin_ps(data));
    }
    // this = sin(this)
    void inline_sin()
    {
        data = _mm_sin_ps(data);
    }

    // Returns asin(this)
    SIMD_vecf asin()
    {
        return SIMD_vecf(_mm_asin_ps(data));
    }
    // this = asin(this)
    void inline_asin()
    {
        data = _mm_asin_ps(data);
    }
    // LAZY CHECKPOINT

    SIMD_vecf sinh()
    {
        return SIMD_vecf(_mm_sinh_ps(data));
    }

    void inline_sinh()
    {
        data = _mm_sinh_ps(data);
    }


    SIMD_vecf asinh()
    {
        return SIMD_vecf(_mm_asinh_ps(data));
    }

    void inline_asinh()
    {
        data = _mm_asinh_ps(data);
    }

    SIMD_vecf cos()
    {
        return SIMD_vecf(_mm_cos_ps(data));
    }

    void inline_cos()
    {
        data = _mm_cos_ps(data);
    }

    SIMD_vecf acos()
    {
        return SIMD_vecf(_mm_acos_ps(data));
    }

    void inline_acos()
    {
        data = _mm_acos_ps(data);
    }

    SIMD_vecf cosh()
    {
        return SIMD_vecf(_mm_cosh_ps(data));
    }

    void inline_cosh()
    {
        data = _mm_cosh_ps(data);
    }

    SIMD_vecf acosh()
    {
        return SIMD_vecf(_mm_acosh_ps(data));
    }

    void inline_acosh()
    {
        data = _mm_acosh_ps(data);
    }

    SIMD_vecf tan()
    {
        return SIMD_vecf(_mm_tan_ps(data));
    }

    void inline_tan()
    {
        data = _mm_tan_ps(data);
    }

    SIMD_vecf atan()
    {
        return SIMD_vecf(_mm_atan_ps(data));
    }
    void inline_atan()
    {
        data = _mm_atan_ps(data);
    }

    SIMD_vecf tanh()
    {
        return SIMD_vecf(_mm_tanh_ps(data));
    }

    void inline_tanh()
    {
        data = _mm_tanh_ps(data);
    }
    SIMD_vecf atanh()
    {
        return SIMD_vecf(_mm_atanh_ps(data));
    }

    void inline_atanh()
    {
        data = _mm_atanh_ps(data);
    }

    SIMD_vecf atan2(const SIMD_vecf& y)
    {
        return SIMD_vecf(_mm_atan2_ps(data, y.data));
    }

    void inline_atan2(const SIMD_vecf& y)
    {
        data = _mm_atan2_ps(data, y.data);
    }

    SIMD_vecf atan2(const float& y)
    {
        __m128 Y = _mm_set1_ps(y);
        return SIMD_vecf(_mm_atan2_ps(data, Y));
    }

    void inline_atan2(const float& y)
    {
        __m128 Y = _mm_set1_ps(y);
        data = _mm_atan2_ps(data, Y);
    }



    /* -----------------------SISD arithmetic-------------------------- */

    // Overload the + operator for SIMD_vecf
    SIMD_vecf operator+(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        return SIMD_vecf(_mm_add_ps(data, mask));
    }

    // Overload the - operator for SIMD_vecf
    SIMD_vecf operator-(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        return SIMD_vecf(_mm_sub_ps(data, mask));
    }

    // Overload the / operator for SIMD_vecf
    SIMD_vecf operator/(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        return SIMD_vecf(_mm_div_ps(data, mask));
    }

    // Overload the * operator for SIMD_vecf
    SIMD_vecf operator*(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        return SIMD_vecf(_mm_mul_ps(data, mask));
    }


    /* -------------------------SIMD conditionals------------------------ */


    // TODO: Add alternatives that take in on_true and on_false options to select those, instead of 1.0 or 0.0
    // Overload the < operator for SIMD_vecf
    SIMD_vecf operator<(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_cmp_ps(data, other.data, _CMP_LT_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the <= operator for SIMD_vecf
    SIMD_vecf operator<=(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_cmp_ps(data, other.data, _CMP_LE_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the > operator for SIMD_vecf
    SIMD_vecf operator>(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_cmp_ps(data, other.data, _CMP_GT_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the >= operator for SIMD_vecf
    SIMD_vecf operator>=(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_cmp_ps(data, other.data, _CMP_GE_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the == operator for SIMD_vecf
    SIMD_vecf operator==(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_cmp_ps(data, other.data, _CMP_EQ_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the != operator for SIMD_vecf
    SIMD_vecf operator!=(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_cmp_ps(data, other.data, _CMP_NEQ_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    /* -------------------------SISD conditionals------------------------ */

    // Overload the < operator for SIMD_vecf
    SIMD_vecf operator<(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        __m128 cmp_result = _mm_cmp_ps(data, mask, _CMP_LT_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the <= operator for SIMD_vecf
    SIMD_vecf operator<=(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        __m128 cmp_result = _mm_cmp_ps(data, mask, _CMP_LE_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the > operator for SIMD_vecf
    SIMD_vecf operator>(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        __m128 cmp_result = _mm_cmp_ps(data, mask, _CMP_GT_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the >= operator for SIMD_vecf
    SIMD_vecf operator>=(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        __m128 cmp_result = _mm_cmp_ps(data, mask, _CMP_GE_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the == operator for SIMD_vecf
    SIMD_vecf operator==(const float& other) const {
        __m128 mask = _mm_set1_ps(other);
        __m128 cmp_result = _mm_cmp_ps(data, mask, _CMP_EQ_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the != operator for SIMD_vecf
    SIMD_vecf operator!=(const float& other) const {
        // Create the mask
        __m128 mask = _mm_set1_ps(other);
        __m128 cmp_result = _mm_cmp_ps(data, mask, _CMP_NEQ_OS);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    /* -------------------------Binary operations------------------------ */
    // Overload the bitwise AND operator for SIMD_vecf
    SIMD_vecf operator&(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_and_ps(data, other.data));
    }

    // Overload the bitwise OR operator for SIMD_vecf
    SIMD_vecf operator|(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_or_ps(data, other.data));
    }

    // Overload the bitwise XOR operator for SIMD_vecf
    SIMD_vecf operator^(const SIMD_vecf& other) const {
        return SIMD_vecf(_mm_xor_ps(data, other.data));
    }

    // Overload the bitwise NOT operator for SIMD_vecf
    SIMD_vecf operator~() const {
        __m128 all_ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
        return SIMD_vecf(_mm_xor_ps(data, all_ones));
    }

    // Overload the logical AND operator for SIMD_vecf
    SIMD_vecf operator&&(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_and_ps(
            _mm_cmp_ps(data, _mm_setzero_ps(), _CMP_NEQ_OQ),
            _mm_cmp_ps(other.data, _mm_setzero_ps(), _CMP_NEQ_OQ)
        );
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the logical OR operator for SIMD_vecf
    SIMD_vecf operator||(const SIMD_vecf& other) const {
        __m128 cmp_result = _mm_or_ps(
            _mm_cmp_ps(data, _mm_setzero_ps(), _CMP_NEQ_OQ),
            _mm_cmp_ps(other.data, _mm_setzero_ps(), _CMP_NEQ_OQ)
        );
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Overload the logical NOT operator for SIMD_vecf
    SIMD_vecf operator!() const {
        __m128 cmp_result = _mm_cmp_ps(data, _mm_setzero_ps(), _CMP_EQ_OQ);
        return SIMD_vecf(_mm_blendv_ps(_mm_setzero_ps(), _mm_set1_ps(1.0f), cmp_result));
    }

    // Get the square root
    SIMD_vecf sqrt() const {
        return SIMD_vecf(_mm_sqrt_ps(data));
    }
    // Get the square root
    void inline_sqrt() {
        data = _mm_sqrt_ps(data);
    }

    // Optimize this later. Just return data[index]
    float operator[](size_t index) const {
        alignas(32) float vals[8];
        _mm_storeu_ps(vals, data);
        return vals[index];
    }

    // returns this * multiplier + addend
    SIMD_vecf mul_add(const SIMD_vecf multiplier, const SIMD_vecf addend) const {
        return SIMD_vecf(_mm_fmadd_ps(data, multiplier.data, addend.data));
    }
    // this = this * multiplier + addend
    void inline_mul_add(const SIMD_vecf multiplier, const SIMD_vecf addend) {
        data = _mm_fmadd_ps(data, multiplier.data, addend.data);
    }

    // Overload the << operator for outputting SIMD_vecf values
    friend std::ostream& operator<<(std::ostream& os, const SIMD_vecf& SIMD_vecf) {
        for (int i = 0; i < SIMD_vecf.SIMD_vecf_size(); ++i) {
            os << SIMD_vecf[i] << " ";
        }
        return os;
    }

    // 8 packed floats per __m128
    static int SIMD_vecf_size() {
        return SIMD_VECTOR_SIZE;
    }

    // Destructor
    ~SIMD_vecf() = default;


private:
    // Private Constructor to initialize with __m128 data. Private for a consistent interface
    SIMD_vecf(__m128 initial_data) : data(initial_data) {}

    static const __m128 constants[6];

};

// Initialize the constants
const __m128 SIMD_vecf::constants[6] = {
    _mm_set1_ps(1.4426950408889634f), // log2(e)
    _mm_set1_ps(1.0f),                // 1.0
    _mm_set1_ps(0.5f),                // 0.5
    _mm_set1_ps(0.3333333333333333f), // 1/3
    _mm_set1_ps(0.25f),               // 0.25
    _mm_set1_ps(0.0f)                 // 0.0
};
