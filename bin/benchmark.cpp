#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

#include <boost/align/aligned_allocator.hpp>

#include "cxxopts.hpp"

#include "simd_local_sort.hpp"

template <typename T>
auto get_uniform_distribution(T min, T max) {
  if constexpr (std::is_same_v<T, double>) {
    return std::uniform_real_distribution<double>(min, max);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return std::uniform_int_distribution<int64_t>(min, max);
  } else {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, int64_t>, "Unsupported type for uniform distribution");
  }
}

template <class T, std::size_t alignment = 1>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, alignment>>;

template <size_t count_per_register, typename KeyType>
void benchmark(const size_t scale, const size_t num_warumup_runs, const size_t num_runs, std::ofstream& out) {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  std::cout << "Running benchmark for scale: " << scale << " ..." << std::endl;

  std::random_device rnd;
  std::mt19937 gen(rnd());
  auto dis = get_uniform_distribution<KeyType>(0, std::numeric_limits<KeyType>::max());

  constexpr auto ALIGNMENT = 32;

  const auto base_num_items = 1'048'576;  // 2^20
  const auto num_items = base_num_items * scale;

  auto data = aligned_vector<KeyType, ALIGNMENT>(num_items);
  auto data_std_sort = aligned_vector<KeyType, ALIGNMENT>(num_items);
  auto data_simd_sort = aligned_vector<KeyType, ALIGNMENT>(num_items);
  auto output_simd_sort = aligned_vector<KeyType, ALIGNMENT>(num_items);

  for (auto& val : data) {
    val = dis(gen);
  }
  std::cout << "start execution" << std::endl;
  assert(!std::ranges::is_sorted(data));

  std::vector<uint64_t> runtimes_std_sort;
  std::vector<uint64_t> runtimes_simd_sort;
  runtimes_std_sort.reserve(num_runs);
  runtimes_simd_sort.reserve(num_runs);

  auto* input_ptr = data_simd_sort.data();
  auto* output_ptr = output_simd_sort.data();

  const auto num_total_runs = num_warumup_runs + num_runs;
  for (size_t run_index = 0; run_index < num_total_runs; ++run_index) {
    // data_std_sort.clear();
    for (auto index = size_t{0}; index < num_items; ++index) {
      data_std_sort[index] = data[index];
      data_simd_sort[index] = data[index];
    }

    for (volatile auto& elem : data_std_sort) {
      // Access element to bring it into cache
    }

    //////////////////////////////
    /// START TIMING std::sort ///
    //////////////////////////////

    auto start_std_sort = std::chrono::steady_clock::now();
    std::sort(data_std_sort.begin(), data_std_sort.end());
    auto end_std_sort = std::chrono::steady_clock::now();

    //NOLINTNEXTLINE
    asm volatile("" : : "r"(data_std_sort.data()) : "memory");  // Ensures std::sort is not optimized away

    /////////////////////////////
    /// END TIMING std::sort  ///
    /////////////////////////////

    const uint64_t runtime_std_sort =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_std_sort - start_std_sort).count();

    input_ptr = data_simd_sort.data();
    output_ptr = output_simd_sort.data();

    for (volatile auto& elem : data_simd_sort) {
      // Access element to bring it into cache
    }

    //////////////////////////////
    /// START TIMING simd_sort ///
    //////////////////////////////

    auto start_simd_sort = std::chrono::steady_clock::now();
    simd_sort<count_per_register>(input_ptr, output_ptr, num_items);
    auto end_simd_sort = std::chrono::steady_clock::now();

    /////////////////////////////
    /// END TIMING simd_sort  ///
    /////////////////////////////

    const uint64_t runtime_simd_sort =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_simd_sort - start_simd_sort).count();

    auto& sorted_data = (output_ptr == output_simd_sort.data()) ? output_simd_sort : data_simd_sort;

    assert(std::ranges::is_sorted(sorted_data));
    assert(sorted_data == data_std_sort);

    if (run_index < num_warumup_runs) {
      // Skip warm-up runs
      continue;
    }
    runtimes_std_sort.push_back(runtime_std_sort);
    runtimes_simd_sort.push_back(runtime_simd_sort);
  }

  const auto total_duration_std_sort = std::accumulate(runtimes_std_sort.begin(), runtimes_std_sort.end(), 0ul);
  const double avg_duration_std_sort = static_cast<double>(total_duration_std_sort) / runtimes_std_sort.size();

  const auto total_duration_simd_sort = std::accumulate(runtimes_simd_sort.begin(), runtimes_simd_sort.end(), 0ul);
  const double avg_duration_simd_sort = static_cast<double>(total_duration_simd_sort) / runtimes_simd_sort.size();
  const auto speed_up = static_cast<double>(avg_duration_std_sort) / static_cast<double>(avg_duration_simd_sort);
  out << scale << "," << avg_duration_std_sort << "," << avg_duration_simd_sort << "," << speed_up << std::endl;
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("SIMDSort", "A single-threaded simd_sort benchmark.");
  // clang-format off
  options.add_options()
  ("c,cpr", "element count per simd register", cxxopts::value<size_t>()->default_value("4")) 
  ("t,dt", "element data type", cxxopts::value<std::string>()->default_value("double"))
  ("w,warmup", "number of warmup runs", cxxopts::value<size_t>()->default_value("1"))
  ("r,runs", "number of runs", cxxopts::value<size_t>()->default_value("5"))
  ("o,output", "Output file name", cxxopts::value<std::string>()->default_value("benchmark.csv")) 
  ("h,help", "Print usage")
  ;
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") != 0u) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  const auto count_per_register = result["cpr"].as<size_t>();  // 64-bit elements with AVX2.
  const auto output_path = result["output"].as<std::string>();
  std::cout << "[Configuration] cpr: " << count_per_register << std::endl;
  std::cout << "L2_CACHE_SIZE: " << L2_CACHE_SIZE << std::endl;

  auto output_file = std::ofstream(output_path);
  if (!output_file.is_open()) {
    std::cerr << "Error: Could not open the file!" << std::endl;
    return 1;
  }
  const auto key_type = result["dt"].as<std::string>();
  const auto num_warumup_runs = result["warmup"].as<size_t>();
  const auto num_runs = result["runs"].as<size_t>();

  if (count_per_register == 4) {
    if (key_type == "double") {
      for (auto scale = size_t{1}; scale <= 256; scale *= 2) {
        benchmark<4, double>(scale, num_warumup_runs, num_runs, output_file);
      }
    } else if (key_type == "int64_t") {
      for (auto scale = size_t{1}; scale <= 256; scale *= 2) {
        benchmark<4, int64_t>(scale, num_warumup_runs, num_runs, output_file);
      }
    }
  } else if (count_per_register == 2) {
    if (key_type == "double") {
      for (auto scale = size_t{1}; scale <= 256; scale *= 2) {
        benchmark<2, double>(scale, num_warumup_runs, num_runs, output_file);
      }
    } else if (key_type == "int64_t") {
      for (auto scale = size_t{1}; scale <= 256; scale *= 2) {
        benchmark<2, int64_t>(scale, num_warumup_runs, num_runs, output_file);
      }
    }
  } else {
    assert(false && "benchmark not implemented");
  }

  return 0;
}
