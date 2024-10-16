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

  std::cout << "[Configuration] scale: " << scale << ", cpr: " << count_per_register << ", num_items: " << num_items
            << std::endl;
  std::cout << "L2_CACHE_SIZE: " << L2_CACHE_SIZE << std::endl;

  if (count_per_register == 4) {
    benchmark<4, KeyType>(num_items);
  } else if (count_per_register == 2) {
    benchmark<2, KeyType>(num_items);
  } else {
    assert(false && "benchmark not implemented");
  }

  return 0;
}
