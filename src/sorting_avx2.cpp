#include <array>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <random>

constexpr auto CHOOSE_BOTH_LOWER_HALVES = 0x20;
constexpr auto CHOOSE_BOTH_UPPER_HALVES = 0x31;

inline void __attribute((always_inline))
avx2_transpose_4x4(__m256d &row_0, __m256d &row_1, __m256d &row_2,
                   __m256d &row_3) {
  const auto ab_lo = _mm256_unpacklo_pd(row_0, row_1);
  const auto ab_hi = _mm256_unpackhi_pd(row_0, row_1);
  const auto cd_lo = _mm256_unpacklo_pd(row_2, row_3);
  const auto cd_hi = _mm256_unpackhi_pd(row_2, row_3);
  row_0 = _mm256_permute2f128_pd(ab_lo, cd_lo, CHOOSE_BOTH_LOWER_HALVES);
  row_1 = _mm256_permute2f128_pd(ab_lo, cd_lo, CHOOSE_BOTH_UPPER_HALVES);
  row_2 = _mm256_permute2f128_pd(ab_hi, cd_hi, CHOOSE_BOTH_LOWER_HALVES);
  row_3 = _mm256_permute2f128_pd(ab_hi, cd_hi, CHOOSE_BOTH_UPPER_HALVES);
}

inline void __attribute((always_inline))
simd_sort4x64(int64_t *__restrict data, int64_t *__restrict output) {
  auto row_0 = _mm256_load_pd(reinterpret_cast<double const *>(data));
  auto row_1 = _mm256_load_pd(reinterpret_cast<double const *>(data + 4));
  auto row_2 = _mm256_load_pd(reinterpret_cast<double const *>(data + 8));
  auto row_3 = _mm256_load_pd(reinterpret_cast<double const *>(data + 12));

  auto temp_a = _mm256_min_pd(row_0, row_2);   // line 1 top
  auto temp_b = _mm256_max_pd(row_0, row_2);   // line 1 bottom
  auto temp_c = _mm256_min_pd(row_1, row_3);   // line 2 top
  auto temp_d = _mm256_max_pd(row_1, row_3);   // line 2 bottom
  auto temp_e = _mm256_max_pd(temp_a, temp_c); // line 3 bottom
  auto temp_f = _mm256_min_pd(temp_b, temp_d); // line 4 top

  row_0 = _mm256_min_pd(temp_a, temp_c); // line 3 top
  row_1 = _mm256_min_pd(temp_e, temp_f); // line 5 top
  row_2 = _mm256_max_pd(temp_e, temp_f); // line 5 bottom
  row_3 = _mm256_max_pd(temp_b, temp_d); // line 4 bottom;

  avx2_transpose_4x4(row_0, row_1, row_2, row_3);

  _mm256_store_pd(reinterpret_cast<double *>(output), row_0);
  _mm256_store_pd(reinterpret_cast<double *>(output + 4), row_1);
  _mm256_store_pd(reinterpret_cast<double *>(output + 8), row_2);
  _mm256_store_pd(reinterpret_cast<double *>(output + 12), row_3);
}

inline void __attribute((always_inline))
avx512_transpose_8x8(__m512i &row_0, __m512i &row_1, __m512i &row_2,
                     __m512i &row_3, __m512i &row_4, __m512i &row_5,
                     __m512i &row_6, __m512i &row_7) {
  __m512i t0 = _mm512_unpacklo_epi64(row_0, row_1);
  __m512i t1 = _mm512_unpackhi_epi64(row_0, row_1);
  __m512i t2 = _mm512_unpacklo_epi64(row_2, row_3);
  __m512i t3 = _mm512_unpackhi_epi64(row_2, row_3);
  __m512i t4 = _mm512_unpacklo_epi64(row_4, row_5);
  __m512i t5 = _mm512_unpackhi_epi64(row_4, row_5);
  __m512i t6 = _mm512_unpacklo_epi64(row_6, row_7);
  __m512i t7 = _mm512_unpackhi_epi64(row_6, row_7);

  __m512i tt0 = _mm512_shuffle_i64x2(t0, t2, 0x44);
  __m512i tt1 = _mm512_shuffle_i64x2(t0, t2, 0xEE);
  __m512i tt2 = _mm512_shuffle_i64x2(t1, t3, 0x44);
  __m512i tt3 = _mm512_shuffle_i64x2(t1, t3, 0xEE);
  __m512i tt4 = _mm512_shuffle_i64x2(t4, t6, 0x44);
  __m512i tt5 = _mm512_shuffle_i64x2(t4, t6, 0xEE);
  __m512i tt6 = _mm512_shuffle_i64x2(t5, t7, 0x44);
  __m512i tt7 = _mm512_shuffle_i64x2(t5, t7, 0xEE);

  const auto indices_one = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
  const auto indices_two = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
  row_0 = _mm512_permutex2var_epi64(tt0, indices_one, tt4);
  row_1 = _mm512_permutex2var_epi64(tt2, indices_one, tt6);
  row_2 = _mm512_permutex2var_epi64(tt0, indices_two, tt4);
  row_3 = _mm512_permutex2var_epi64(tt2, indices_two, tt6);
  row_4 = _mm512_permutex2var_epi64(tt1, indices_one, tt5);
  row_5 = _mm512_permutex2var_epi64(tt3, indices_one, tt7);
  row_6 = _mm512_permutex2var_epi64(tt1, indices_two, tt5);
  row_7 = _mm512_permutex2var_epi64(tt3, indices_two, tt7);
}

inline void __attribute((always_inline))
simd_sort8x64(int64_t *__restrict data, int64_t *__restrict output) {
  auto row_a = _mm512_load_epi64(data);
  auto row_b = _mm512_load_epi64(data + 8);
  auto row_c = _mm512_load_epi64(data + 16);
  auto row_d = _mm512_load_epi64(data + 24);
  auto row_e = _mm512_load_epi64(data + 32);
  auto row_f = _mm512_load_epi64(data + 40);
  auto row_g = _mm512_load_epi64(data + 48);
  auto row_h = _mm512_load_epi64(data + 56);
  /* 1. level of sorting network */
  auto row_a2 = _mm512_min_epi64(row_a, row_c);
  auto row_c2 = _mm512_max_epi64(row_a, row_c);
  auto row_b2 = _mm512_min_epi64(row_b, row_d);
  auto row_d2 = _mm512_max_epi64(row_b, row_d);
  auto row_e2 = _mm512_min_epi64(row_e, row_g);
  auto row_g2 = _mm512_max_epi64(row_e, row_g);
  auto row_f2 = _mm512_min_epi64(row_f, row_h);
  auto row_h2 = _mm512_max_epi64(row_f, row_h);
  /* 2. level of sorting network */
  auto row_a3 = _mm512_min_epi64(row_a2, row_e2);
  auto row_e3 = _mm512_max_epi64(row_a2, row_e2);
  auto row_b3 = _mm512_min_epi64(row_b2, row_f2);
  auto row_f3 = _mm512_max_epi64(row_b2, row_f2);
  auto row_c3 = _mm512_min_epi64(row_c2, row_g2);
  auto row_g3 = _mm512_max_epi64(row_c2, row_g2);
  auto row_d3 = _mm512_min_epi64(row_d2, row_h2);
  auto row_h3 = _mm512_max_epi64(row_d2, row_h2);
  /* 3. level of sorting network */
  auto row_a4 = _mm512_min_epi64(row_a3, row_b3);
  auto row_b4 = _mm512_max_epi64(row_a3, row_b3);
  auto row_c4 = _mm512_min_epi64(row_c3, row_d3);
  auto row_d4 = _mm512_max_epi64(row_c3, row_d3);
  auto row_e4 = _mm512_min_epi64(row_e3, row_f3);
  auto row_f4 = _mm512_max_epi64(row_e3, row_f3);
  auto row_g4 = _mm512_min_epi64(row_g3, row_h3);
  auto row_h4 = _mm512_max_epi64(row_g3, row_h3);
  /* 4. level of sorting network */
  auto row_c5 = _mm512_min_epi64(row_c4, row_e4);
  auto row_e5 = _mm512_max_epi64(row_c4, row_e4);
  auto row_d5 = _mm512_min_epi64(row_d4, row_f4);
  auto row_f5 = _mm512_max_epi64(row_d4, row_f4);
  /* 5. level of sorting network */
  auto row_b5 = _mm512_min_epi64(row_b4, row_e5);
  auto row_e6 = _mm512_max_epi64(row_b4, row_e5);
  auto row_d6 = _mm512_min_epi64(row_d5, row_g4);
  auto row_g5 = _mm512_max_epi64(row_d5, row_g4);
  /* 6. level of sorting network */
  auto row_b6 = _mm512_min_epi64(row_b5, row_c5);
  auto row_c6 = _mm512_max_epi64(row_b5, row_c5);
  auto row_d7 = _mm512_min_epi64(row_d6, row_e6);
  auto row_e7 = _mm512_max_epi64(row_d6, row_e6);
  auto row_f6 = _mm512_min_epi64(row_f5, row_g5);
  auto row_g6 = _mm512_max_epi64(row_f5, row_g5);

  avx512_transpose_8x8(row_a4, row_b6, row_c6, row_d7, row_e7, row_f5, row_g6,
                       row_h4);

  _mm512_store_epi64(output, row_a4);
  _mm512_store_epi64(output + 8, row_b6);
  _mm512_store_epi64(output + 16, row_c6);
  _mm512_store_epi64(output + 24, row_d7);
  _mm512_store_epi64(output + 32, row_e7);
  _mm512_store_epi64(output + 40, row_f5);
  _mm512_store_epi64(output + 48, row_g6);
  _mm512_store_epi64(output + 56, row_h4);
}

constexpr auto DATA_SIZE = size_t{589824};
alignas(64) auto input = std::array<int64_t, DATA_SIZE>{};
alignas(64) auto output_avx2 = std::array<int64_t, DATA_SIZE>{};
alignas(64) auto output_avx512 = std::array<int64_t, DATA_SIZE>{};

int main() {
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::nanoseconds;

  unsigned int seed = 12345;
  std::mt19937 gen(seed); // mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int64_t> distrib(1, 1000000);

  for (auto i = size_t{0}; i < DATA_SIZE; ++i) {
    input[i] = distrib(gen);
  }

  // auto start = high_resolution_clock::now();
  // for (auto i = size_t{0}; i < DATA_SIZE; i += 64) {
  //   simd_sort8x64(input.data() + i, output.data() + i);
  // }
  // auto end = high_resolution_clock::now();

  auto start = high_resolution_clock::now();
  for (auto i = size_t{0}; i < DATA_SIZE; i += 16) {
    auto *start = input.data() + i;
    auto *check = output_avx2.data() + i;
    simd_sort4x64(&input[i], &output_avx2[i]);
  }
  auto end = high_resolution_clock::now();
  auto ns_int = duration_cast<nanoseconds>(end - start);
  std::cout << "took " << ns_int.count() << " ns." << std::endl;

  return 0;
}
