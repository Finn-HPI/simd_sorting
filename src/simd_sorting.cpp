#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>

#include "simd_sort_utils.hpp"

template <typename T>
struct AlignedAllocater {
  using value_type = T;

  explicit AlignedAllocater(std::size_t alignment) : _alignment(alignment) {}

  T* allocate(std::size_t size) {
    void* ptr = std::aligned_alloc(_alignment, size * sizeof(T));
    if (!ptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t /*size*/) noexcept {
    std::free(ptr);
  }

 private:
  std::size_t _alignment;
};

template <typename T>
std::vector<T, AlignedAllocater<T>> create_aligned_vector(size_t count, size_t alignment) {
  return std::move(std::vector<T, AlignedAllocater<T>>(count, AlignedAllocater<T>(alignment)));
}

int main() {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;
  using KeyType = uint64_t;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  std::uniform_int_distribution<KeyType> dis(0, std::numeric_limits<KeyType>::max());

  constexpr auto SCALE = 200;
  constexpr auto ALIGNMENT = 32;
  constexpr auto NUM_ITEMS = SCALE * block_size<KeyType>();
  constexpr auto COUNT_PER_REGISTER = 4;  // 64-bit elements with AVX2.

  std::cout << "num_items: " << NUM_ITEMS << std::endl;

  auto data_simd = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);
  auto data_sort = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);
  auto data_qsort = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);

  auto output = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);

  using VecType = Vec<256, KeyType>;
  std::cout << "vec size: " << sizeof(VecType) << std::endl;

  auto* input_ptr = data_simd.data();
  auto* output_ptr = output.data();

  for (auto index = size_t{0}; index < NUM_ITEMS; ++index) {
    const auto val = dis(gen);
    data_simd[index] = val;
    data_sort[index] = val;
    data_qsort[index] = val;
  }

  if (!is_aligned(data_simd.data(), 32) || !is_aligned(output.data(), 32)) {
    std::cerr << "Data or Output is not aligend";
    return -1;
  }

  std::ranges::sort(data_sort);

  simd_sort<COUNT_PER_REGISTER>(input_ptr, output_ptr, NUM_ITEMS);

  auto& sorted_data = (output_ptr == output.data()) ? output : data_simd;
  std::cout << "is sorted: " << std::ranges::is_sorted(sorted_data) << std::endl;
  assert(std::ranges::is_sorted(sorted_data));
  assert(sorted_data == data_sort);
  return 0;
}
