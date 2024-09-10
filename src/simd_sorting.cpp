#include "simd_sort_utils.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {
// https://stackoverflow.com/questions/35311711/whats-the-right-way-to-compute-integral-base-2-logarithms-at-compile-time
constexpr size_t cilog2(uint64_t val) {
  return (val != 0u) ? 1 + cilog2(val >> 1u) : -1;
}
} // namespace

template <typename T>
inline void __attribute__((always_inline)) simd_sort_block2(T *&input_ptr,
                                                            T *&output_ptr) {
  auto ptrs = std::array<T *, 2>{};
  ptrs[0] = input_ptr;
  ptrs[1] = output_ptr;
  // Apply 4x4 Sorting network.
  {
    auto *inptr = reinterpret_cast<block16_t *>(ptrs[0]);
    auto *const end = reinterpret_cast<block16_t *>(ptrs[0] + BLOCK_SIZE);
    while (inptr < end) {
      sort4x4(reinterpret_cast<T *>(inptr), reinterpret_cast<T *>(inptr));
      ++inptr;
    }
  }
}

template <typename T>
inline void __attribute__((always_inline)) simd_sort_block(T *&input_ptr,
                                                           T *&output_ptr) {
  auto ptrs = std::array<T *, 2>{};
  ptrs[0] = input_ptr;
  ptrs[1] = output_ptr;
  // Apply 4x4 Sorting network.
  {
    // auto *inptr = reinterpret_cast<block16_t *>(ptrs[0]);
    // auto *const end = reinterpret_cast<block16_t *>(ptrs[0] + BLOCK_SIZE);
    // while (inptr < end) {
    //   sort4x4(reinterpret_cast<T *>(inptr), reinterpret_cast<T *>(inptr));
    //   ++inptr;
    // }
    auto *inptr = reinterpret_cast<block4_t *>(ptrs[0]);
    auto *const end = reinterpret_cast<block4_t *>(ptrs[0] + BLOCK_SIZE);
    while (inptr < end) {
      auto data = load4(reinterpret_cast<T *>(inptr));
      data = sort_register(data);
      store4(data, reinterpret_cast<T *>(inptr));
      ++inptr;
    }
  }
  //   constexpr auto LOG_BLOCK_SIZE = cilog2(BLOCK_SIZE);
  //   constexpr auto STOP_LEVEL = LOG_BLOCK_SIZE - 2;
  //
  //   auto merge_level = [&]<typename MergeKernel>(size_t level,
  //                                                MergeKernel merge_kernel) {
  //     auto ptr_index = level & 1u;
  //     auto *input = ptrs[ptr_index];
  //     auto *output = ptrs[ptr_index ^ 1u];
  //     auto *const end = input + BLOCK_SIZE;
  //
  //     const auto input_length = 1u << level;         // = 2^level
  //     const auto output_length = input_length << 1u; // = input_length x 2
  //     while (input < end) {
  //       merge_kernel(input, input + input_length, output, input_length);
  //       input += output_length;
  //       output += output_length;
  //     }
  //   };
  //   merge_level(2, &merge4_eqlen<T>);
  //   merge_level(3, &merge8_eqlen<T>);
  // #pragma unroll
  //   for (auto level = size_t{4}; level < STOP_LEVEL; ++level) {
  //     merge_level(level, &merge16_eqlen<T>);
  //   }
  //
  //   auto input_length = 1u << STOP_LEVEL;
  //   auto ptr_index = STOP_LEVEL & 1u;
  //   auto *input = ptrs[ptr_index];
  //   auto *output = ptrs[ptr_index ^ 1u];
  //
  //   merge16_eqlen<T>(input, input + input_length, output, input_length);
  //   merge16_eqlen<T>(input + 2 * input_length, input + 3 * input_length,
  //                    output + 2 * input_length, input_length);
  //   input_length <<= 1u;
  //   // NOLINTNEXTLINE
  //   merge16_eqlen<T>(output, output + input_length, input, input_length);
  //   input_ptr = output;
  //   output_ptr = input;
}

int main() {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  std::uniform_int_distribution<> dis(0, 100);

  alignas(32) auto data = std::vector<uint64_t>(BLOCK_SIZE);
  alignas(32) auto cmp_data = std::vector<uint64_t>(BLOCK_SIZE);
  alignas(32) auto output = std::vector<uint64_t>(BLOCK_SIZE);

  auto *input_ptr = data.data();
  auto *output_ptr = output.data();

  constexpr auto TEST_RUNS = 1000;
  uint64_t total_time = 0;
  for (auto run = size_t{0}; run < TEST_RUNS + 2; ++run) {
    for (auto index = size_t{0}; index < BLOCK_SIZE; ++index) {
      const auto val = dis(gen);
      data[index] = val;
      cmp_data[index] = val;
    }
    auto start = high_resolution_clock::now();
    simd_sort_block(input_ptr, output_ptr);
    auto end = high_resolution_clock::now();
    if (run < 2) {
      continue;
    }
    total_time += duration_cast<std::chrono::microseconds>(end - start).count();
  }
  std::cout << "total time = " << total_time << std::endl;
  total_time /= TEST_RUNS;
  std::cout << "mean time = " << total_time << std::endl;

  std::cout << "====================================" << std::endl;

  total_time = 0;
  for (auto run = size_t{0}; run < TEST_RUNS + 2; ++run) {
    for (auto index = size_t{0}; index < BLOCK_SIZE; ++index) {
      const auto val = dis(gen);
      data[index] = val;
      cmp_data[index] = val;
    }
    auto start = high_resolution_clock::now();
    simd_sort_block2(input_ptr, output_ptr);
    auto end = high_resolution_clock::now();
    if (run < 2) {
      continue;
    }
    total_time += duration_cast<std::chrono::microseconds>(end - start).count();
  }
  std::cout << "total time = " << total_time << std::endl;
  total_time /= TEST_RUNS;
  std::cout << "mean time = " << total_time << std::endl;

  // for (auto index = size_t{0}; index < BLOCK_SIZE; ++index) {
  //   const auto val = dis(gen);
  //   data[index] = val;
  //   cmp_data[index] = val;
  // }
  // assert(!std::ranges::is_sorted(data));
  // simd_sort_block(input_ptr, output_ptr);
  // std::ranges::sort(cmp_data);
  // assert(data == cmp_data);
  // assert(std::ranges::is_sorted(data));
  // std::cout << "is sorted: " << std::ranges::is_sorted(data) << std::endl;
  return 0;
}
