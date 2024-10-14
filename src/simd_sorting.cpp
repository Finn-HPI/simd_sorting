#include "simd_sort_utils.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <random>

template <typename T> struct AlignedAllocater {
  using value_type = T;
  AlignedAllocater(std::size_t alignment) : alignment_(alignment) {}
  T *allocate(std::size_t n) {
    void *ptr = std::aligned_alloc(alignment_, n * sizeof(T));
    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<T *>(ptr);
  }
  void deallocate(T *ptr, std::size_t) noexcept { std::free(ptr); }

private:
  std::size_t alignment_;
};

template <typename T> struct BlockInfo {
  T *input;
  T *output;
  size_t size;
};

template <typename T>
void simd_sort(T *&input_ptr, T *&output_ptr, size_t element_count) {
  if (element_count <= 0) [[unlikely]] {
    return;
  }
  constexpr auto BLOCK_SIZE = block_size<T>();
  auto *input = input_ptr;
  auto *output = output_ptr;
  // We split our data into blocks of size BLOCK_SIZE and compute the bounds for
  // each block.
  auto block_count = element_count / BLOCK_SIZE;
  const auto remaining_items = element_count % BLOCK_SIZE;
  auto block_infos = std::vector<BlockInfo<T>>{};
  block_infos.reserve(block_count + (remaining_items > 0));
  for (auto block_index = size_t{0}; block_index < block_count; ++block_index) {
    const auto offset = block_index * BLOCK_SIZE;
    block_infos.emplace_back(input + offset, output + offset, BLOCK_SIZE);
  }
  // We then call our local sort routine for each block.
  for (auto block_index = size_t{0}; block_index < block_count; ++block_index) {
    auto &block_info = block_infos[block_index];
    simd_sort_block<4>(block_info.input, block_info.output);
    std::swap(block_info.input, block_info.output);
  }
  if (remaining_items) {
    auto &block_info = block_infos.back();
    // TODO(finn): Implement avxsort for remaining items.
    assert(false && "Sort for remaining values not implemented.");
  }
  block_count += remaining_items > 0;

  // Next we merge all these chunks iteratively to achieve a global sorting.
  const auto log_n = static_cast<size_t>(std::ceil(std::log2(element_count)));
  constexpr auto LOG_BLOCK_SIZE = cilog2(BLOCK_SIZE);

  for (auto index = LOG_BLOCK_SIZE; index < log_n; ++index) {
    auto updated_block_count = size_t{0};
    for (auto block_index = size_t{0}; block_index < block_count - 1;
         block_index += 2) {
      auto &a_info = block_infos[block_index];
      auto &b_info = block_infos[block_index + 1];
      auto *input_a = a_info.input;
      auto *input_b = b_info.input;
      auto *out = a_info.output;
      const auto a_size = a_info.size;
      const auto b_size = b_info.size;

      merge4_varlen(input_a, input_b, out, a_size, b_size);

      block_infos[updated_block_count] = {out, input_a, a_size + b_size};
      ++updated_block_count;
    }
    if (block_count % 2) {
      block_infos[updated_block_count] = block_infos[block_count - 1];
      ++updated_block_count;
    }
    block_count = updated_block_count;
  }
  output_ptr = block_infos[0].input;
  input_ptr = block_infos[0].output;
}

template <typename T>
std::vector<T, AlignedAllocater<T>> create_aligned_vector(size_t count,
                                                          size_t alignment) {
  return std::move(std::vector<T, AlignedAllocater<T>>(
      count, AlignedAllocater<T>(alignment)));
}

int main() {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  using KeyType = uint64_t;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  std::uniform_int_distribution<KeyType> dis(
      0, std::numeric_limits<KeyType>::max());

  constexpr auto SCALE = 200;
  constexpr auto ALIGNMENT = 32;
  constexpr auto NUM_ITEMS = SCALE * block_size<KeyType>();

  std::cout << "num_items: " << NUM_ITEMS << std::endl;

  auto data_simd = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);
  auto data_sort = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);
  auto data_qsort = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);

  auto output = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);

  auto *input_ptr = data_simd.data();
  auto *output_ptr = output.data();

  for (auto index = size_t{0}; index < NUM_ITEMS; ++index) {
    const auto val = dis(gen);
    data_simd[index] = val;
    data_sort[index] = val;
    data_qsort[index] = val;
  }
  if (!is_aligned(data_simd.data(), 32) || !is_aligned(output.data(), 32)) {
    std::cerr << "Data or Output is not aligend";
    return -1;
  } else {
    std::cout << "data is aligned" << std::endl;
  }
  std::ranges::sort(data_sort);
  simd_sort(input_ptr, output_ptr, NUM_ITEMS);
  auto &sorted_data = (output_ptr == output.data()) ? output : data_simd;
  std::cout << "is sorted: " << std::ranges::is_sorted(sorted_data)
            << std::endl;
  assert(std::ranges::is_sorted(sorted_data));
  assert(sorted_data == data_sort);
  return 0;
}
