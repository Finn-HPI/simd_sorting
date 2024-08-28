#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

constexpr auto L2_CACHE_SIZE = 256 * 1024;
constexpr auto BLOCK_SIZE = L2_CACHE_SIZE / (2 * sizeof(uint64_t));

#define LOWER_HAVES 0, 1, 4, 5
#define UPPER_HAVES 2, 3, 6, 7
#define INTERLEAVE_LOWERS 0, 4, 1, 5
#define INTERLEAVE_UPPERS 2, 6, 3, 7

using tuple_t = struct {
  uint32_t key;
  uint32_t rid;
};

template <typename T> using Vec __attribute__((vector_size(32))) = T;
using UnalignedVec64 __attribute__((aligned(1))) = Vec<uint64_t>;
template <typename T> using AlignedVec __attribute__((aligned(8))) = Vec<T>;

using vec64_t = Vec<uint64_t>;
using block16 = struct alignas(16) {};

template <typename T>
inline void __attribute((always_inline)) store_vec(Vec<T> data,
                                                   T *__restrict output) {
  auto *out_vec = reinterpret_cast<AlignedVec<T> *>(output);
  *out_vec = data;
}

template <typename T> inline Vec<T> reverse4(Vec<T> vec) {
  return __builtin_shufflevector(vec, vec, 3, 2, 1, 0);
}

template <typename T>
inline void bitonic4(Vec<T> in1, Vec<T> in2, Vec<T> &out1, Vec<T> &out2) {
  // NOLINTBEGIN
  // Level 1
  auto lo1 = __builtin_elementwise_min(in1, in2);
  auto hi1 = __builtin_elementwise_max(in1, in2);
  auto lo1_perm = __builtin_shufflevector(lo1, hi1, 0, 1, 4, 5);
  auto hi1_perm = __builtin_shufflevector(lo1, hi1, 2, 3, 6, 7);
  // Level 2
  auto lo2 = __builtin_elementwise_min(lo1_perm, hi1_perm);
  auto hi2 = __builtin_elementwise_max(lo1_perm, hi1_perm);
  auto lo2_perm = __builtin_shufflevector(lo2, hi2, 0, 4, 2, 6);
  auto hi2_perm = __builtin_shufflevector(lo2, hi2, 1, 5, 3, 7);
  // Level 3
  auto lo3 = __builtin_elementwise_min(lo2_perm, hi2_perm);
  auto hi3 = __builtin_elementwise_max(lo2_perm, hi2_perm);
  out1 = __builtin_shufflevector(lo3, hi3, 0, 4, 1, 5);
  out2 = __builtin_shufflevector(lo3, hi3, 2, 6, 3, 7);
  // NOLINTEND
}

template <typename T>
inline void bitonic4_merge(Vec<T> in1, Vec<T> in2, Vec<T> &out1, Vec<T> &out2) {
  in2 = reverse4(in2);
  bitonic4(in1, in2, out1, out2);
}

template <typename T>
inline void __attribute((always_inline)) x86_sort4x4(const T *data, T *output) {
  // Sorting Network
  constexpr auto TYPE_SIZE = sizeof(T);
  constexpr auto BYTE_OFFSET = 256 / (TYPE_SIZE * TYPE_SIZE);
  // NOLINTBEGIN
  using VecAligned = AlignedVec<T>;
  auto row_0 = *reinterpret_cast<const VecAligned *>(data);
  auto row_1 = *reinterpret_cast<const VecAligned *>(data + BYTE_OFFSET);
  auto row_2 = *reinterpret_cast<const VecAligned *>(data + BYTE_OFFSET * 2);
  auto row_3 = *reinterpret_cast<const VecAligned *>(data + BYTE_OFFSET * 3);

  auto temp_a = __builtin_elementwise_min(row_0, row_2);
  auto temp_b = __builtin_elementwise_max(row_0, row_2);
  auto temp_c = __builtin_elementwise_min(row_1, row_3);
  auto temp_d = __builtin_elementwise_max(row_1, row_3);
  auto temp_e = __builtin_elementwise_max(temp_a, temp_c);
  auto temp_f = __builtin_elementwise_min(temp_b, temp_d);

  row_0 = __builtin_elementwise_min(temp_a, temp_c);
  row_1 = __builtin_elementwise_min(temp_e, temp_f);
  row_2 = __builtin_elementwise_max(temp_e, temp_f);
  row_3 = __builtin_elementwise_max(temp_b, temp_d);
  // NOLINTEND

  // Transpose Matrix
  auto ab_lo = __builtin_shufflevector(row_0, row_1, INTERLEAVE_LOWERS);
  auto ab_hi = __builtin_shufflevector(row_0, row_1, INTERLEAVE_UPPERS);
  auto cd_lo = __builtin_shufflevector(row_2, row_3, INTERLEAVE_LOWERS);
  auto cd_hi = __builtin_shufflevector(row_2, row_3, INTERLEAVE_UPPERS);

  row_0 = __builtin_shufflevector(ab_lo, cd_lo, LOWER_HAVES);
  row_1 = __builtin_shufflevector(ab_lo, cd_lo, UPPER_HAVES);
  row_2 = __builtin_shufflevector(ab_hi, cd_hi, LOWER_HAVES);
  row_3 = __builtin_shufflevector(ab_hi, cd_hi, UPPER_HAVES);

  // Write to output
  store_vec(row_0, output);
  store_vec(row_1, output + 4);
  store_vec(row_2, output + 8);
  store_vec(row_3, output + 12);
}

inline void __attribute__((always_inline)) sort_chunk(uint64_t *inputptr,
                                                      uint64_t *outputptr) {
  auto ptrs = std::array<uint64_t *, 2>{};
  ptrs[0] = inputptr;
  ptrs[1] = outputptr;
  auto *inptr = reinterpret_cast<block16 *>(ptrs[0]);
  auto *const end = inptr + BLOCK_SIZE;
  while (inptr < end) {
    x86_sort4x4(reinterpret_cast<uint64_t *>(inptr),
                reinterpret_cast<uint64_t *>(inptr));
    ++inptr;
  }
}

int main() {

  std::random_device rnd;  // Obtain a random number from hardware
  std::mt19937 gen(rnd()); // Seed the generator
  std::uniform_int_distribution<> dis(0, 100); // Define the range

  alignas(64) auto data = std::vector<uint64_t>(BLOCK_SIZE);

  for (auto &value : data) {
    value = dis(gen);
  }

  for (size_t i = 0; i < data.size(); ++i) {
    std::cout << "data[" << i << "] = " << data[i] << std::endl;
  }

  // std::cout << "4x4 AVX Sort with compiler intrinsics!" << std::endl;
  // alignas(64) auto data =
  //     std::vector<uint64_t>{1, 17, 5, 8, 3, 2, 5, 9, 4, 32, 1, 11, 0, 5, 6,
  //     10};
  // x86_sort4x4(data.data(), data.data());
  // for (int idx = 0; auto &val : data) {
  //   if (idx > 0 && idx % 4 == 0) {
  //     std::cout << std::endl;
  //   }
  //   std::cout << val << " ";
  //   ++idx;
  // }

  // std::cout << std::endl << "Test bitonic4_merge" << std::endl;
  //
  // auto row_0 = *reinterpret_cast<const AlignedVec<uint64_t> *>(data.data());
  // auto row_1 = *reinterpret_cast<const AlignedVec<uint64_t> *>(data.data() +
  // 4);
  //
  // bitonic4_merge(row_0, row_1, row_0, row_1);
  //
  // store_vec(row_0, data.data());
  // store_vec(row_1, data.data() + 4);
  //
  // for (int idx = 0; auto &val : data) {
  //   if (idx > 0 && idx % 4 == 0) {
  //     std::cout << std::endl;
  //   }
  //   std::cout << val << " ";
  //   ++idx;
  // }

  return 0;
}
