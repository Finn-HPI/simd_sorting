#include "simd_sort_utils.hpp"
#include <algorithm>
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
void simd_sort(T *&input_ptr, T *&output_ptr, size_t element_count) {
  if (element_count <= 0) {
    return;
  }
  constexpr auto BLOCK_SIZE = block_size<T>();
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
    simd_sort_chunk<256>(chunk_info.input, chunk_info.output);
    std::swap(chunk_info.input, chunk_info.output);
  }
  if (remaining_items) {
    auto &chunk_info = chunk_infos.back();
    // TODO(finn): Implement avxsort for remaining items.
    assert(false && "Sort for remaining values not implemented.");
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

template <typename ElementType, int elements_per_register>
void sort_elements(ElementType *& /*input*/, ElementType *& /*output*/,
                   size_t /*num_elements*/) {
  assert(false &&
         "Sort is not available for the number of elements per register.");
}

int main() {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  std::uniform_int_distribution<uint64_t> dis(
      0, std::numeric_limits<uint64_t>::max());

  constexpr auto SCALE = 70;
  constexpr auto NUM_ITEMS = SCALE * block_size<uint64_t>();

  std::cout << "num_items: " << NUM_ITEMS << std::endl;

  alignas(32) auto data_simd = std::vector<uint64_t>(NUM_ITEMS);
  alignas(32) auto data_sort = std::vector<uint64_t>(NUM_ITEMS);
  alignas(32) auto data_qsort = std::vector<uint64_t>(NUM_ITEMS);

  alignas(32) auto output = std::vector<uint64_t>(NUM_ITEMS);

  auto *input_ptr = data_simd.data();
  auto *output_ptr = output.data();

  for (auto index = size_t{0}; index < NUM_ITEMS; ++index) {
    const auto val = dis(gen);
    data_simd[index] = val;
    data_sort[index] = val;
    data_qsort[index] = val;
  }

  auto start1 = high_resolution_clock::now();
  std::qsort(data_qsort.data(), data_qsort.size(), sizeof(uint64_t),
             [](const void *first, const void *second) {
               const auto arg1 = *static_cast<const uint64_t *>(first);
               const auto arg2 = *static_cast<const uint64_t *>(second);
               return static_cast<int>(arg1 > arg2) -
                      static_cast<int>(arg1 < arg2);
             });
  auto end1 = high_resolution_clock::now();
  auto ms1 = duration_cast<milliseconds>(end1 - start1).count();
  std::cout << "std::qsort took: " << ms1 << std::endl;

  auto start2 = high_resolution_clock::now();
  std::ranges::sort(data_sort);
  auto end2 = high_resolution_clock::now();
  auto ms2 = duration_cast<milliseconds>(end2 - start2).count();
  std::cout << "std::sort took: " << ms2 << std::endl;

  auto start3 = high_resolution_clock::now();
  simd_sort(input_ptr, output_ptr, NUM_ITEMS);
  auto end3 = high_resolution_clock::now();
  auto ms3 = duration_cast<milliseconds>(end3 - start3).count();
  std::cout << "avxsort took: " << ms3 << std::endl;

  auto baseline = (ms1 < ms2) ? ms1 : ms2;
  auto simd_x = static_cast<double>(baseline) / static_cast<double>(ms3);
  auto simd_improvement = (simd_x - 1.0) * 100;
  std::cout << "SIMD sort performed " << simd_improvement << "% (x" << simd_x
            << ") better compared to fastest baseline ("
            << ((ms1 > ms2) ? "std::sort" : "std::qsort") << ")." << std::endl;
  auto &sorted_data = (output_ptr == output.data()) ? output : data_simd;
  assert(std::ranges::is_sorted(sorted_data));
  assert(sorted_data == data_sort);
  return 0;
}
