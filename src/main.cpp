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
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    void* memory_start_address = std::aligned_alloc(_alignment, size * sizeof(T));
    if (!memory_start_address) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(memory_start_address);
  }

  void deallocate(T* pointer, std::size_t /*size*/) noexcept {
    std::free(pointer);
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
  using KeyType = double;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  std::uniform_real_distribution<KeyType> dis(0, std::numeric_limits<KeyType>::max());

  constexpr auto SCALE = 200;
  constexpr auto ALIGNMENT = 32;
  constexpr auto NUM_ITEMS = SCALE * block_size<KeyType>();  //+ block_size<KeyType>() / 2;
  constexpr auto COUNT_PER_REGISTER = 4;                     // 64-bit elements with AVX2.

  std::cout << "num_items: " << NUM_ITEMS << std::endl;

  auto data_simd = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);
  auto data_sort = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);
  auto data_qsort = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);

  auto output = create_aligned_vector<KeyType>(NUM_ITEMS, ALIGNMENT);

  using VecType = Vec<256, KeyType>;

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

  // ========== Measure std::qsort ================================

  auto start1 = high_resolution_clock::now();
  std::qsort(data_qsort.data(), data_qsort.size(), sizeof(KeyType), [](const void* first, const void* second) {
    const auto arg1 = *static_cast<const KeyType*>(first);
    const auto arg2 = *static_cast<const KeyType*>(second);
    return static_cast<int>(arg1 > arg2) - static_cast<int>(arg1 < arg2);
  });
  auto end1 = high_resolution_clock::now();
  auto ms1 = duration_cast<milliseconds>(end1 - start1).count();
  std::cout << "qsort: " << ms1 << std::endl;

  // ========== Measure std::ranges::sort ========================

  auto start2 = high_resolution_clock::now();
  std::ranges::sort(data_sort);
  auto end2 = high_resolution_clock::now();
  auto ms2 = duration_cast<milliseconds>(end2 - start2).count();
  std::cout << "sort: " << ms2 << std::endl;

  // ========== Measure simd_sort ================================

  auto start3 = high_resolution_clock::now();
  simd_sort<COUNT_PER_REGISTER>(input_ptr, output_ptr, NUM_ITEMS);
  auto end3 = high_resolution_clock::now();
  auto ms3 = duration_cast<milliseconds>(end3 - start3).count();

  auto& sorted_data = (output_ptr == output.data()) ? output : data_simd;
  assert(std::ranges::is_sorted(sorted_data));
  assert(sorted_data == data_sort);

  std::cout << "simd_sort: " << ms3 << std::endl;

  // ========= Evaluation ========================================

  auto simd_improvement_qsort = static_cast<double>(ms1) / static_cast<double>(ms3);
  auto simd_improvement_sort = static_cast<double>(ms2) / static_cast<double>(ms3);

  std::cout << "simd_sort is " << (simd_improvement_qsort - 1) * 100 << "% " << "faster than std::qsort (x"
            << simd_improvement_qsort << ")." << std::endl;
  std::cout << "simd_sort is " << (simd_improvement_sort - 1) * 100 << "% " << "faster than std::sort (x"
            << simd_improvement_sort << ")." << std::endl;

  return 0;
}
