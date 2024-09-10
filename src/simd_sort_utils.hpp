#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

constexpr auto L2_CACHE_SIZE = 256 * 1024;
constexpr auto BLOCK_SIZE = L2_CACHE_SIZE / (2 * sizeof(uint64_t));

#define LOWER_HAVES 0, 1, 4, 5
#define UPPER_HAVES 2, 3, 6, 7
#define INTERLEAVE_LOWERS 0, 4, 1, 5
#define INTERLEAVE_UPPERS 2, 6, 3, 7

template <typename T> using Vec __attribute__((vector_size(32))) = T;
using UnalignedVec __attribute__((aligned(1))) = Vec<uint64_t>;
template <typename T> using AlignedVec __attribute__((aligned(32))) = Vec<T>;

using block4_t = struct alignas(32) {};
using block8_t = struct alignas(64) {};
using block16_t = struct alignas(128) {};

// Loading and Storing SIMD registers.

template <typename T>
inline __attribute((always_inline)) Vec<T> load4(T *addr) {
  return {addr[0], addr[1], addr[2], addr[3]};
}

template <typename T>
inline void __attribute((always_inline)) store4(Vec<T> data,
                                                T *__restrict output) {
  auto *out_vec = reinterpret_cast<UnalignedVec *>(output);
  *out_vec = data;
}

template <typename BlockType, typename ValueType>
inline __attribute((always_inline)) void
load8(Vec<ValueType> &reg_lo, Vec<ValueType> &reg_hi, BlockType *block_addr) {
  auto *addr = reinterpret_cast<ValueType *>(block_addr);
  reg_lo = load4(addr);
  reg_hi = load4(addr + 4);
}

template <typename T>
inline __attribute((always_inline)) void store8(Vec<T> &reg_lo, Vec<T> &reg_hi,
                                                block4_t *addr) {
  store4(reg_lo, reinterpret_cast<T *>(addr));
  store4(reg_hi, reinterpret_cast<T *>(addr + 1));
}

// Bitonic Merge Networks.

template <typename T> inline void reverse4(Vec<T> &vec) {
  vec = __builtin_shufflevector(vec, vec, 3, 2, 1, 0);
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
inline void bitonic8(Vec<T> in11, Vec<T> in12, Vec<T> in21, Vec<T> in22,
                     Vec<T> &out1, Vec<T> &out2, Vec<T> &out3, Vec<T> &out4) {
  // NOLINTBEGIN
  auto l11 = __builtin_elementwise_min(in11, in21);
  auto l12 = __builtin_elementwise_min(in12, in22);
  auto h11 = __builtin_elementwise_max(in11, in21);
  auto h12 = __builtin_elementwise_max(in12, in22);
  // NOLINTEND
  bitonic4(l11, l12, out1, out2);
  bitonic4(h11, h12, out3, out4);
}

template <typename T>
inline void bitonic4_merge(Vec<T> in1, Vec<T> in2, Vec<T> &out1, Vec<T> &out2) {
  reverse4(in2);
  bitonic4(in1, in2, out1, out2);
}

template <typename T>
inline void bitonic8_merge(Vec<T> in11, Vec<T> in12, Vec<T> in21, Vec<T> in22,
                           Vec<T> &out1, Vec<T> &out2, Vec<T> &out3,
                           Vec<T> &out4) {
  reverse4(in21);
  reverse4(in22);
  // NOLINTBEGIN
  auto l11 = __builtin_elementwise_min(in11, in22);
  auto l12 = __builtin_elementwise_min(in12, in21);
  auto h11 = __builtin_elementwise_max(in11, in22);
  auto h12 = __builtin_elementwise_max(in12, in21);
  // NOLINTEND
  bitonic4(l11, l12, out1, out2);
  bitonic4(h11, h12, out3, out4);
}

template <typename T>
inline void bitonic16_merge(Vec<T> in11, Vec<T> in12, Vec<T> in13, Vec<T> in14,
                            Vec<T> in21, Vec<T> in22, Vec<T> in23, Vec<T> in24,
                            Vec<T> &out1, Vec<T> &out2, Vec<T> &out3,
                            Vec<T> &out4, Vec<T> &out5, Vec<T> &out6,
                            Vec<T> &out7, Vec<T> &out8) {
  reverse4(in21);
  reverse4(in22);
  reverse4(in23);
  reverse4(in24);
  // NOLINTBEGIN
  auto l01 = __builtin_elementwise_min(in11, in24);
  auto l02 = __builtin_elementwise_min(in12, in23);
  auto l03 = __builtin_elementwise_min(in13, in22);
  auto l04 = __builtin_elementwise_min(in14, in21);
  auto h01 = __builtin_elementwise_max(in11, in24);
  auto h02 = __builtin_elementwise_max(in12, in23);
  auto h03 = __builtin_elementwise_max(in13, in22);
  auto h04 = __builtin_elementwise_max(in14, in21);
  // NOLINTEND
  bitonic8(l01, l02, l03, l04, out1, out2, out3, out4);
  bitonic8(h01, h02, h03, h04, out5, out6, out7, out8);
}

template <typename BlockType, typename T>
inline void __attribute__((always_inline))
choose_next_and_update_pointers(BlockType *&next, BlockType *&a_ptr,
                                BlockType *&b_ptr) {
  const int8_t cmp =
      *reinterpret_cast<T *>(a_ptr) < *reinterpret_cast<T *>(b_ptr);
  next = cmp ? a_ptr : b_ptr;
  a_ptr += cmp;
  b_ptr += !cmp;
}

template <typename T>
inline void __attribute((always_inline))
merge4_eqlen(T *const input_a, T *const input_b, T *const output,
             const size_t length) {
  auto *a_ptr = reinterpret_cast<block4_t *>(input_a);
  auto *b_ptr = reinterpret_cast<block4_t *>(input_b);
  auto *const a_end = reinterpret_cast<block4_t *>(input_a + length);
  auto *const b_end = reinterpret_cast<block4_t *>(input_b + length);

  auto *output_ptr = reinterpret_cast<block4_t *>(output);
  auto *next = b_ptr;
  auto output_lo = Vec<T>{};
  auto output_hi = Vec<T>{};
  auto a_loaded = load4<T>(reinterpret_cast<T *>(a_ptr));
  auto b_loaded = load4<T>(reinterpret_cast<T *>(b_ptr));
  ++a_ptr;
  ++b_ptr;
  bitonic4_merge(a_loaded, b_loaded, output_lo, output_hi);
  store4(output_lo, reinterpret_cast<uint64_t *>(output_ptr));
  ++output_ptr;
  // As long as both A and B are not empty, do fetch and 2x4 merge.
  while (a_ptr < a_end && b_ptr < b_end) {
    choose_next_and_update_pointers<block4_t, T>(next, a_ptr, b_ptr);
    a_loaded = output_hi;
    b_loaded = load4<T>(reinterpret_cast<T *>(next));
    bitonic4_merge(a_loaded, b_loaded, output_lo, output_hi);
    store4(output_lo, reinterpret_cast<T *>(output_ptr));
    ++output_ptr;
  }
  // If A not empty, merge remainder of A.
  while (a_ptr < a_end) {
    a_loaded = load4<T>(reinterpret_cast<T *>(a_ptr));
    b_loaded = output_hi;
    bitonic4_merge(a_loaded, b_loaded, output_lo, output_hi);
    store4(output_lo, reinterpret_cast<T *>(output_ptr));
    ++a_ptr;
    ++output_ptr;
  }
  // If A not empty, merge remainder of A.
  while (b_ptr < b_end) {
    a_loaded = output_hi;
    b_loaded = load4<T>(reinterpret_cast<T *>(b_ptr));
    bitonic4_merge(a_loaded, b_loaded, output_lo, output_hi);
    store4(output_lo, reinterpret_cast<T *>(output_ptr));
    ++b_ptr;
    ++output_ptr;
  }
  store4(output_hi, reinterpret_cast<T *>(output_ptr));
}

// Bitonic merge algorithms for equal sized A and B.

template <typename T>
inline void __attribute((always_inline))
merge8_eqlen(T *const input_a, T *const input_b, T *const output,
             const size_t length) {
  auto *a_ptr = reinterpret_cast<block8_t *>(input_a);
  auto *b_ptr = reinterpret_cast<block8_t *>(input_b);
  auto *const a_end = reinterpret_cast<block8_t *>(input_a + length);
  auto *const b_end = reinterpret_cast<block8_t *>(input_b + length);

  auto *output_ptr = reinterpret_cast<block8_t *>(output);
  auto *next = b_ptr;

  auto output_lo1 = Vec<T>{};
  auto output_lo2 = Vec<T>{};
  auto output_hi1 = Vec<T>{};
  auto output_hi2 = Vec<T>{};

  auto a_loaded1 = Vec<T>{};
  auto a_loaded2 = Vec<T>{};
  auto b_loaded1 = Vec<T>{};
  auto b_loaded2 = Vec<T>{};

  load8(a_loaded1, a_loaded2, a_ptr);
  load8(b_loaded1, b_loaded2, b_ptr);
  ++a_ptr;
  ++b_ptr;
  bitonic8_merge(a_loaded1, a_loaded2, b_loaded1, b_loaded2, output_lo1,
                 output_lo2, output_hi1, output_hi2);
  store8(output_lo1, output_lo2, reinterpret_cast<block4_t *>(output_ptr));
  ++output_ptr;
  while (a_ptr < a_end && b_ptr < b_end) {
    choose_next_and_update_pointers<block8_t, T>(next, a_ptr, b_ptr);
    a_loaded1 = output_hi1;
    a_loaded2 = output_hi2;
    load8(b_loaded1, b_loaded2, next);
    bitonic8_merge(a_loaded1, a_loaded2, b_loaded1, b_loaded2, output_lo1,
                   output_lo2, output_hi1, output_hi2);
    store8(output_lo1, output_lo2, reinterpret_cast<block4_t *>(output_ptr));
    ++output_ptr;
  }
  while (a_ptr < a_end) {
    load8(a_loaded1, a_loaded2, a_ptr);
    b_loaded1 = output_hi1;
    b_loaded2 = output_hi2;
    bitonic8_merge(a_loaded1, a_loaded2, b_loaded1, b_loaded2, output_lo1,
                   output_lo2, output_hi1, output_hi2);
    store8(output_lo1, output_lo2, reinterpret_cast<block4_t *>(output_ptr));
    ++output_ptr;
    ++a_ptr;
  }
  while (b_ptr < b_end) {
    a_loaded1 = output_hi1;
    a_loaded2 = output_hi2;
    load8(b_loaded1, b_loaded2, b_ptr);
    bitonic8_merge(a_loaded1, a_loaded2, b_loaded1, b_loaded2, output_lo1,
                   output_lo2, output_hi1, output_hi2);
    store8(output_lo1, output_lo2, reinterpret_cast<block4_t *>(output_ptr));
    ++output_ptr;
    ++b_ptr;
  }
  store8(output_hi1, output_hi2, reinterpret_cast<block4_t *>(output_ptr));
}

template <typename T>
inline void merge8_varlen(T *const input_a, T *const input_b, T *const output,
                          const size_t a_length, const size_t b_length) {}

template <typename T>
inline void __attribute((always_inline))
merge16_eqlen(T *const input_a, T *const input_b, T *const output,
              const size_t length) {

  auto *a_ptr = reinterpret_cast<block16_t *>(input_a);
  auto *b_ptr = reinterpret_cast<block16_t *>(input_b);
  auto *const a_end = reinterpret_cast<block16_t *>(input_a + length);
  auto *const b_end = reinterpret_cast<block16_t *>(input_b + length);

  auto *output_ptr = reinterpret_cast<block16_t *>(output);
  auto *next = b_ptr;

  auto reg_out1l1 = Vec<T>{};
  auto reg_out1l2 = Vec<T>{};
  auto reg_out1h1 = Vec<T>{};
  auto reg_out1h2 = Vec<T>{};

  auto reg_out2l1 = Vec<T>{};
  auto reg_out2l2 = Vec<T>{};
  auto reg_out2h1 = Vec<T>{};
  auto reg_out2h2 = Vec<T>{};

  auto reg_al1 = Vec<T>{};
  auto reg_al2 = Vec<T>{};
  auto reg_ah1 = Vec<T>{};
  auto reg_ah2 = Vec<T>{};

  auto reg_bl1 = Vec<T>{};
  auto reg_bl2 = Vec<T>{};
  auto reg_bh1 = Vec<T>{};
  auto reg_bh2 = Vec<T>{};

  load8(reg_al1, reg_al2, a_ptr);
  load8(reg_ah1, reg_ah2, reinterpret_cast<block8_t *>(a_ptr) + 1);
  ++a_ptr;

  load8(reg_bl1, reg_bl2, b_ptr);
  load8(reg_bh1, reg_bh2, reinterpret_cast<block8_t *>(b_ptr) + 1);
  ++b_ptr;

  bitonic16_merge(reg_al1, reg_al2, reg_ah1, reg_ah2, reg_bl1, reg_bl2, reg_bh1,
                  reg_bh2, reg_out1l1, reg_out1l2, reg_out1h1, reg_out1h2,
                  reg_out2l1, reg_out2l2, reg_out2h1, reg_out2h2);

  store8(reg_out1l1, reg_out1l2, reinterpret_cast<block4_t *>(output_ptr));
  store8(reg_out1h1, reg_out1h2, reinterpret_cast<block4_t *>(output_ptr) + 2);
  ++output_ptr;

  while (a_ptr < a_end && b_ptr < b_end) {
    choose_next_and_update_pointers<block16_t, T>(next, a_ptr, b_ptr);
    reg_al1 = reg_out2l1;
    reg_al2 = reg_out2l2;
    reg_ah1 = reg_out2h1;
    reg_ah2 = reg_out2h2;
    load8(reg_bl1, reg_bl2, reinterpret_cast<block8_t *>(next));
    load8(reg_bh1, reg_bh2, reinterpret_cast<block8_t *>(next) + 1);

    bitonic16_merge(reg_al1, reg_al2, reg_ah1, reg_ah2, reg_bl1, reg_bl2,
                    reg_bh1, reg_bh2, reg_out1l1, reg_out1l2, reg_out1h1,
                    reg_out1h2, reg_out2l1, reg_out2l2, reg_out2h1, reg_out2h2);
    store8(reg_out1l1, reg_out1l2, reinterpret_cast<block4_t *>(output_ptr));
    store8(reg_out1h1, reg_out1h2,
           reinterpret_cast<block4_t *>(output_ptr) + 2);
    ++output_ptr;
  }

  while (a_ptr < a_end) {
    reg_bl1 = reg_out2l1;
    reg_bl2 = reg_out2l2;
    reg_bh1 = reg_out2h1;
    reg_bh2 = reg_out2h2;
    load8(reg_al1, reg_al2, reinterpret_cast<block8_t *>(a_ptr));
    load8(reg_ah1, reg_ah2, reinterpret_cast<block8_t *>(a_ptr) + 1);
    ++a_ptr;
    bitonic16_merge(reg_al1, reg_al2, reg_ah1, reg_ah2, reg_bl1, reg_bl2,
                    reg_bh1, reg_bh2, reg_out1l1, reg_out1l2, reg_out1h1,
                    reg_out1h2, reg_out2l1, reg_out2l2, reg_out2h1, reg_out2h2);
    store8(reg_out1l1, reg_out1l2, reinterpret_cast<block4_t *>(output_ptr));
    store8(reg_out1h1, reg_out1h2,
           reinterpret_cast<block4_t *>(output_ptr) + 2);
    ++output_ptr;
  }

  while (b_ptr < b_end) {
    reg_al1 = reg_out2l1;
    reg_al2 = reg_out2l2;
    reg_ah1 = reg_out2h1;
    reg_ah2 = reg_out2h2;
    load8(reg_bl1, reg_bl2, reinterpret_cast<block8_t *>(b_ptr));
    load8(reg_bh1, reg_bh2, reinterpret_cast<block8_t *>(b_ptr) + 1);
    ++b_ptr;
    bitonic16_merge(reg_al1, reg_al2, reg_ah1, reg_ah2, reg_bl1, reg_bl2,
                    reg_bh1, reg_bh2, reg_out1l1, reg_out1l2, reg_out1h1,
                    reg_out1h2, reg_out2l1, reg_out2l2, reg_out2h1, reg_out2h2);
    store8(reg_out1l1, reg_out1l2, reinterpret_cast<block4_t *>(output_ptr));
    store8(reg_out1h1, reg_out1h2,
           reinterpret_cast<block4_t *>(output_ptr) + 2);
    ++output_ptr;
  }

  store8(reg_out2l1, reg_out2l2, reinterpret_cast<block4_t *>(output_ptr));
  store8(reg_out2h1, reg_out2h2, reinterpret_cast<block4_t *>(output_ptr) + 2);
}

// Bitonic merge algorithms for variable sized A and B.

template <typename T>
inline void __attribute((always_inline))
merge4_varlen(T *input_a, T *input_b, T *output, const size_t length_a,
              const size_t length_b) {
  const auto length_a4 = length_a & ~0x3u;
  const auto length_b4 = length_b & ~0x3u;

  const auto a_index = size_t{0};
  const auto b_index = size_t{0};

  auto a_i = size_t{0};
  auto b_i = size_t{0};

  auto &out = output;

  if (length_a4 > 4 && length_b4 > 4) {
    auto *a_ptr = reinterpret_cast<block4_t *>(input_a);
    auto *b_ptr = reinterpret_cast<block4_t *>(input_b);
    auto *const a_end = reinterpret_cast<block4_t *>(input_a + length_a) - 1;
    auto *const b_end = reinterpret_cast<block4_t *>(input_b + length_b) - 1;

    auto *output_ptr = reinterpret_cast<block4_t *>(out);
    auto *next = b_ptr;
    auto output_lo = Vec<T>{};
    auto output_hi = Vec<T>{};
    auto a_loaded = load4<T>(reinterpret_cast<T *>(a_ptr));
    auto b_loaded = load4<T>(reinterpret_cast<T *>(b_ptr));
    ++a_ptr;
    ++b_ptr;
    bitonic4_merge(a_loaded, b_loaded, output_lo, output_hi);
    store4(output_lo, reinterpret_cast<uint64_t *>(output_ptr));
    ++output_ptr;
    auto it = 0;
    // As long as both A and B are not empty, do fetch and 2x4 merge.
    while (a_ptr < a_end && b_ptr < b_end) {
      choose_next_and_update_pointers<block4_t, T>(next, a_ptr, b_ptr);
      a_loaded = output_hi;
      b_loaded = load4<T>(reinterpret_cast<T *>(next));
      bitonic4_merge(a_loaded, b_loaded, output_lo, output_hi);
      store4(output_lo, reinterpret_cast<T *>(output_ptr));
      ++output_ptr;
    }
    if (output_hi[3] <= *reinterpret_cast<T *>(a_ptr)) {
      --a_ptr;
      store4(output_hi, reinterpret_cast<T *>(a_ptr));
    } else {
      --b_ptr;
      store4(output_hi, reinterpret_cast<T *>(b_ptr));
    }

    a_i = reinterpret_cast<T *>(a_ptr) - input_a;
    b_i = reinterpret_cast<T *>(b_ptr) - input_b;

    input_a = reinterpret_cast<T *>(a_ptr);
    input_b = reinterpret_cast<T *>(b_ptr);
    out = reinterpret_cast<T *>(output_ptr);
  }
  // Serial Merge.
  while (a_i < length_a && b_i < length_b) {
    auto *next = input_b;
    const auto cmp = *input_a < *input_b;
    const auto cmp_neg = !cmp;
    a_i += cmp;
    b_i += cmp_neg;
    next = cmp ? input_a : input_b;
    *out = *next;
    ++out;
    input_a += cmp;
    input_b += cmp_neg;
  }
  while (a_i < length_a) {
    *out = *input_a;
    ++a_i;
    out++;
    ++input_a;
  }
  while (b_i < length_b) {
    *out = *input_b;
    ++b_i;
    out++;
    ++input_b;
  }
}

// Sorting network for 4x4 input.

template <typename T> void sort4x4(T *data, T *output) {
  constexpr auto TYPE_SIZE = sizeof(T);
  constexpr auto BYTE_OFFSET = 256 / (TYPE_SIZE * TYPE_SIZE);
  auto row_0 = load4(data);
  auto row_1 = load4(data + BYTE_OFFSET);
  auto row_2 = load4(data + BYTE_OFFSET * 2);
  auto row_3 = load4(data + BYTE_OFFSET * 3);

  // NOLINTBEGIN
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
  store4(row_0, output);
  store4(row_1, output + 4);
  store4(row_2, output + 8);
  store4(row_3, output + 12);
}

template <typename T, int... indices>
Vec<T> inline cmp_merge(Vec<T> in1, Vec<T> in2) {
  // NOLINTBEGIN
  auto min = __builtin_elementwise_min(in1, in2);
  auto max = __builtin_elementwise_max(in1, in2);
  // NOLINTEND
  return __builtin_shufflevector(min, max, indices...);
}

template <typename T> Vec<T> inline sort_register(Vec<T> reg) {
  reg = cmp_merge<T, 0, 5, 2, 7>(reg,
                                 __builtin_shufflevector(reg, reg, 1, 0, 3, 2));
  reg = cmp_merge<T, 0, 1, 6, 7>(reg,
                                 __builtin_shufflevector(reg, reg, 3, 2, 1, 0));
  reg = cmp_merge<T, 0, 5, 2, 7>(reg,
                                 __builtin_shufflevector(reg, reg, 1, 0, 3, 2));
  return reg;
}

// https://stackoverflow.com/questions/35311711/whats-the-right-way-to-compute-integral-base-2-logarithms-at-compile-time
constexpr size_t cilog2(uint64_t val) {
  return (val != 0u) ? 1 + cilog2(val >> 1u) : -1;
}

template <typename T>
inline void __attribute__((always_inline)) simd_sort_block(T *&input_ptr,
                                                           T *&output_ptr) {
  auto ptrs = std::array<T *, 2>{};
  ptrs[0] = input_ptr;
  ptrs[1] = output_ptr;
  // Apply 4x4 Sorting network.
  {
    auto *inptr = reinterpret_cast<block16_t *>(ptrs[0]);
    auto *const end = reinterpret_cast<block16_t *>(ptrs[0] + BLOCK_SIZE);
    while (inptr < end) {
      sort4x4(reinterpret_cast<T *>(inptr), reinterpret_cast<T *>(inptr));
      ++inptr;
    }
  }
  constexpr auto LOG_BLOCK_SIZE = cilog2(BLOCK_SIZE);
  constexpr auto STOP_LEVEL = LOG_BLOCK_SIZE - 2;

  auto merge_level = [&]<typename MergeKernel>(size_t level,
                                               MergeKernel merge_kernel) {
    auto ptr_index = level & 1u;
    auto *input = ptrs[ptr_index];
    auto *output = ptrs[ptr_index ^ 1u];
    auto *const end = input + BLOCK_SIZE;

    const auto input_length = 1u << level;         // = 2^level
    const auto output_length = input_length << 1u; // = input_length x 2
    while (input < end) {
      merge_kernel(input, input + input_length, output, input_length);
      input += output_length;
      output += output_length;
    }
  };
  merge_level(2, &merge4_eqlen<T>);
  merge_level(3, &merge8_eqlen<T>);
#pragma unroll
  for (auto level = size_t{4}; level < STOP_LEVEL; ++level) {
    merge_level(level, &merge16_eqlen<T>);
  }

  auto input_length = 1u << STOP_LEVEL;
  auto ptr_index = STOP_LEVEL & 1u;
  auto *input = ptrs[ptr_index];
  auto *output = ptrs[ptr_index ^ 1u];

  merge16_eqlen<T>(input, input + input_length, output, input_length);
  merge16_eqlen<T>(input + 2 * input_length, input + 3 * input_length,
                   output + 2 * input_length, input_length);
  input_length <<= 1u;
  // NOLINTNEXTLINE
  merge16_eqlen<T>(output, output + input_length, input, input_length);
  input_ptr = output;
  output_ptr = input;
}
