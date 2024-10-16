#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>

#include "cxxopts.hpp"

#include "simd_local_sort.hpp"

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

template <size_t count_per_register, typename KeyType>
void benchmark(size_t num_items) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  std::uniform_real_distribution<KeyType> dis(0, std::numeric_limits<KeyType>::max());
  constexpr auto ALIGNMENT = 32;

  auto data_simd = create_aligned_vector<KeyType>(num_items, ALIGNMENT);
  auto data_sort = create_aligned_vector<KeyType>(num_items, ALIGNMENT);
  auto data_qsort = create_aligned_vector<KeyType>(num_items, ALIGNMENT);

  auto output = create_aligned_vector<KeyType>(num_items, ALIGNMENT);

  auto* input_ptr = data_simd.data();
  auto* output_ptr = output.data();

  for (auto index = size_t{0}; index < num_items; ++index) {
    const auto val = dis(gen);
    data_simd[index] = val;
    data_sort[index] = val;
    data_qsort[index] = val;
  }

  if (!is_aligned(data_simd.data(), 32) || !is_aligned(output.data(), 32)) {
    assert(false && "Data or Output is not aligend");
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
  simd_sort<count_per_register>(input_ptr, output_ptr, num_items);
  auto end3 = high_resolution_clock::now();
  auto ms3 = duration_cast<milliseconds>(end3 - start3).count();

  std::cout << "simd_sort: " << ms3 << std::endl;

  // ========= Evaluation ========================================

  auto simd_improvement_qsort = static_cast<double>(ms1) / static_cast<double>(ms3);
  auto simd_improvement_sort = static_cast<double>(ms2) / static_cast<double>(ms3);

  std::cout << "simd_sort is " << (simd_improvement_qsort - 1) * 100 << "% " << "faster than std::qsort (x"
            << simd_improvement_qsort << ")." << std::endl;
  std::cout << "simd_sort is " << (simd_improvement_sort - 1) * 100 << "% " << "faster than std::sort (x"
            << simd_improvement_sort << ")." << std::endl;

  auto& sorted_data = (output_ptr == output.data()) ? output : data_simd;
  assert(std::ranges::is_sorted(sorted_data));
  assert(sorted_data == data_sort);
  std::cout << "output is sorted = " << std::ranges::is_sorted(sorted_data)
            << " and is same as std::sort = " << (sorted_data == data_sort) << std::endl;
}

int main(int argc, char* argv[]) {
  using KeyType = double;
  cxxopts::Options options("SIMDSort", "A single-threaded simd_sort benchmark.");
  // clang-format off
  options.add_options()
  ("c,cpr", "element count per simd register", cxxopts::value<size_t>()) 
  ("s,scale", "num_items = scale * cache_block_size", cxxopts::value<size_t>()->default_value("200"))
  ("h,help", "Print usage")
  ;
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") != 0u) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const auto scale = result["scale"].as<size_t>();
  const auto num_items = scale * block_size<KeyType>();        // block_size<KeyType>() / 2;
  const auto count_per_register = result["cpr"].as<size_t>();  // 64-bit elements with AVX2.

  std::cout << "Scale: " << scale << " cpr: " << count_per_register << "num_items: " << num_items << std::endl;

  if (count_per_register == 4) {
    benchmark<4, KeyType>(num_items);
  } else if (count_per_register == 2) {
    benchmark<2, KeyType>(num_items);
  } else {
    assert(false && "benchmark not implemented");
  }

  return 0;
}
