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
#include <vector>

template <typename T> struct ChunkInfo {
  T *input;
  T *output;
  size_t size;
};

template <typename T>
void avxsort_aligned(T *&input_ptr, T *&output_ptr, size_t element_count) {
  if (element_count <= 0) {
    return;
  }
  auto *input = input_ptr;
  auto *output = output_ptr;

  auto chunk_count = element_count / BLOCK_SIZE;
  const auto remaining_items = element_count % BLOCK_SIZE;
  auto chunk_infos = std::vector<ChunkInfo<T>>{};
  chunk_infos.reserve(chunk_count + (remaining_items > 0));

  for (auto chunk_index = size_t{0}; chunk_index < chunk_count; ++chunk_index) {
    const auto offset = chunk_index * BLOCK_SIZE;
    chunk_infos.emplace_back(input + offset, output + offset, BLOCK_SIZE);
  }
  for (auto chunk_index = size_t{0}; chunk_index < chunk_count; ++chunk_index) {
    auto &chunk_info = chunk_infos[chunk_index];
    simd_sort_block(chunk_info.input, chunk_info.output);
    std::swap(chunk_info.input, chunk_info.output);
  }
  if (remaining_items) {
    auto &chunk_info = chunk_infos.back();
    // TODO(finn): Implement avxsort for remaining items.
  }
  chunk_count += remaining_items > 0;
  const auto log_n = static_cast<size_t>(std::ceil(std::log2(element_count)));
  constexpr auto LOG_BLOCK_SIZE = cilog2(BLOCK_SIZE);

  for (auto index = LOG_BLOCK_SIZE; index < log_n; ++index) {
    auto new_chunk_count = size_t{0};
    for (auto chunk_index = size_t{0}; chunk_index < chunk_count - 1;
         chunk_index += 2) {
      auto &a_info = chunk_infos[chunk_index];
      auto &b_info = chunk_infos[chunk_index + 1];
      auto *input_a = a_info.input;
      auto *input_b = b_info.input;
      auto *out = a_info.output;
      const auto a_size = a_info.size;
      const auto b_size = b_info.size;

      merge4_varlen(input_a, input_b, out, a_size, b_size);

      chunk_infos[new_chunk_count] = {out, input_a, a_size + b_size};
      ++new_chunk_count;
    }
    if (chunk_count % 2) {
      chunk_infos[new_chunk_count] = chunk_infos[chunk_count - 1];
      ++new_chunk_count;
    }
    chunk_count = new_chunk_count;
  }
  output_ptr = chunk_infos[0].input;
  input_ptr = chunk_infos[0].output;
}

int main() {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  std::uniform_int_distribution<> dis(0, 100);

  constexpr auto SCALE = 1000;
  constexpr auto NUM_ITEMS = SCALE * BLOCK_SIZE;

  std::cout << "num_items: " << NUM_ITEMS << std::endl;

  alignas(32) auto data = std::vector<uint64_t>(NUM_ITEMS);
  alignas(32) auto cmp_data = std::vector<uint64_t>(NUM_ITEMS);
  alignas(32) auto output = std::vector<uint64_t>(NUM_ITEMS);

  auto *input_ptr = data.data();
  auto *output_ptr = output.data();

  for (auto index = size_t{0}; index < BLOCK_SIZE; ++index) {
    const auto val = dis(gen);
    data[index] = val;
    cmp_data[index] = val;
  }
  assert(!std::ranges::is_sorted(data));
  auto start1 = high_resolution_clock::now();
  std::ranges::sort(cmp_data);
  auto end1 = high_resolution_clock::now();
  auto ms1 = duration_cast<milliseconds>(end1 - start1).count();
  std::cout << "normal sort took: " << ms1 << std::endl;

  auto start2 = high_resolution_clock::now();
  avxsort_aligned(input_ptr, output_ptr, NUM_ITEMS);
  auto end2 = high_resolution_clock::now();
  auto ms2 = duration_cast<milliseconds>(end2 - start2).count();
  std::cout << "SIMD sort took: " << ms2 << std::endl;

  std::cout << "SIMD improvement = "
            << (static_cast<double>(ms1) / static_cast<double>(ms2))
            << std::endl;

  auto &sorted_data = (output_ptr == output.data()) ? output : data;
  assert(sorted_data == cmp_data);
  assert(std::ranges::is_sorted(sorted_data));
  std::cout << "is sorted: " << std::ranges::is_sorted(sorted_data)
            << std::endl;

  return 0;
}
