#pragma once

#include <cstddef>
#include <cstdint>

constexpr auto L2_CACHE_SIZE = 256 * 1024;

template <typename T>
constexpr size_t block_size() {
  return L2_CACHE_SIZE / (2 * sizeof(T));
}

#define LOWER_HALVES 0, 1, 4, 5
#define UPPER_HALVES 2, 3, 6, 7
#define INTERLEAVE_LOWERS 0, 4, 1, 5
#define INTERLEAVE_UPPERS 2, 6, 3, 7

template <size_t reg_size, typename T>
using Vec __attribute__((vector_size(reg_size))) = T;

// Loading and Storing SIMD registers.

template <typename VecType, typename T>
inline __attribute((always_inline)) VecType load_aligned(T* addr) {
  return *reinterpret_cast<VecType*>(addr);
}

template <typename VecType, typename T>
inline __attribute((always_inline)) VecType load_unaligned(T* addr) {
  using UnalignedVecType __attribute__((aligned(1))) = VecType;
  return *reinterpret_cast<UnalignedVecType*>(addr);
}

template <typename VecType, typename T>
inline void __attribute((always_inline)) store_aligned(VecType data, T* __restrict output) {
  auto* out_vec = reinterpret_cast<VecType*>(output);
  *out_vec = data;
}

template <typename VecType, typename T>
inline void __attribute((always_inline)) store_unaligned(VecType data, T* __restrict output) {
  using UnalignedVecType __attribute__((aligned(1))) = VecType;
  auto* out_vec = reinterpret_cast<UnalignedVecType*>(output);
  *out_vec = data;
}

// Struct for loading & storing  multiple vector registers.

template <size_t register_count, size_t elements_per_register, typename VecType>
struct MultiVec {
  template <typename T>
  inline void __attribute__((always_inline)) load(T* /*address*/) {
    static_assert(false, "Not implemented.");
  }

  template <typename T>
  inline void __attribute__((always_inline)) store(T* /*address*/) {
    static_assert(false, "Not implemented.");
  }

  inline VecType& __attribute__((always_inline)) first() {
    static_assert(false, "Not implemented.");
  }

  inline VecType& __attribute__((always_inline)) last() {
    static_assert(false, "Not implemented.");
  }
};

template <size_t elements_per_register, typename VecType>
struct MultiVec<1, elements_per_register, VecType> {
  VecType a;

  template <typename T>
  inline void __attribute__((always_inline)) load(T* address) {
    a = load_aligned<VecType>(address);
  }

  template <typename T>
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
  }

  inline VecType& __attribute__((always_inline)) first() {
    return a;
  }

  inline VecType& __attribute__((always_inline)) last() {
    return a;
  }
};

template <typename VecType, size_t elements_per_register>
struct MultiVec<2, elements_per_register, VecType> {
  VecType a;
  VecType b;

  template <typename T>
  inline void __attribute__((always_inline)) load(T* address) {
    a = load_aligned<VecType>(address);
    b = load_aligned<VecType>(address + elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
    store_aligned(b, address + elements_per_register);
  }

  inline VecType& __attribute__((always_inline)) first() {
    return a;
  }

  inline VecType& __attribute__((always_inline)) last() {
    return b;
  }
};

template <typename VecType, size_t elements_per_register>
struct MultiVec<4, elements_per_register, VecType> {
  VecType a;
  VecType b;
  VecType c;
  VecType d;

  template <typename T>
  inline void __attribute__((always_inline)) load(T* address) {
    a = load_aligned<VecType>(address);
    b = load_aligned<VecType>(address + elements_per_register);
    c = load_aligned<VecType>(address + 2 * elements_per_register);
    d = load_aligned<VecType>(address + 3 * elements_per_register);
  }

  template <typename T>
  inline void __attribute__((always_inline)) store(T* address) {
    store_aligned(a, address);
    store_aligned(b, address + elements_per_register);
    store_aligned(c, address + 2 * elements_per_register);
    store_aligned(d, address + 3 * elements_per_register);
  }

  inline VecType& __attribute__((always_inline)) first() {
    return a;
  }

  inline VecType& __attribute__((always_inline)) last() {
    return d;
  }
};

// Collection of utility functions.

template <typename T>
inline __attribute((always_inline)) bool is_aligned(T* addr, size_t byte_alignment) {
  return reinterpret_cast<std::uintptr_t>(addr) % byte_alignment == 0;
}

template <typename BlockType, typename T>
inline void __attribute__((always_inline)) choose_next_and_update_pointers(BlockType*& next, BlockType*& a_ptr,
                                                                           BlockType*& b_ptr) {
  const int8_t cmp = *reinterpret_cast<T*>(a_ptr) < *reinterpret_cast<T*>(b_ptr);
  next = cmp ? a_ptr : b_ptr;
  a_ptr += cmp;
  b_ptr += !cmp;
}

template <size_t kernel_size>
constexpr size_t get_alignment_bitmask() {
  return ~(kernel_size - 1);
}
