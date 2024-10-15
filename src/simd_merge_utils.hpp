#pragma once
#include <cassert>

#include "simd_basics.hpp"

template <size_t count_per_register, typename T>
struct TwoWayMerge {
  template <size_t kernel_size>
  static inline void __attribute__((always_inline)) merge_equal_length(T* const /*input_a*/, T* const /*input_b*/,
                                                                       T* const /*output*/, const size_t /*length*/) {
    assert(false && "Not implemented.");
  };

  template <size_t kernel_size>
  static inline void __attribute__((always_inline)) merge_variable_length(T* /*input_a*/, T* /*input_b*/, T* /*output*/,
                                                                          const size_t /*length_a*/,
                                                                          const size_t /*length_b*/) {
    assert(false && "Not implemented.");
  };
};

template <typename T>
struct TwoWayMerge<4, T> {
  static constexpr auto COUNT_PER_REGISTER = 4;
  static constexpr auto REGISTER_WIDTH = COUNT_PER_REGISTER * sizeof(T);
  using VecType = Vec<REGISTER_WIDTH, T>;

 private:
  // Begin merge network primitives.
  static inline void _reverse(VecType& vec) {
    vec = __builtin_shufflevector(vec, vec, 3, 2, 1, 0);
  }

  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
  static inline void _merge_network_2x4(VecType& input_a, VecType& input_b, VecType& out1, VecType& out2) {
    // Level 1
    auto lo1 = __builtin_elementwise_min(input_a, input_b);
    auto hi1 = __builtin_elementwise_max(input_a, input_b);
    auto lo1_perm = __builtin_shufflevector(lo1, hi1, LOWER_HALVES);
    auto hi1_perm = __builtin_shufflevector(lo1, hi1, UPPER_HALVES);
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
  }

  static inline void _merge_network_2x8(VecType& in11, VecType& in12, VecType& in21, VecType& in22, VecType& out1,
                                        VecType& out2, VecType& out3, VecType& out4) {
    // NOLINTBEGIN
    auto l11 = __builtin_elementwise_min(in11, in21);
    auto l12 = __builtin_elementwise_min(in12, in22);
    auto h11 = __builtin_elementwise_max(in11, in21);
    auto h12 = __builtin_elementwise_max(in12, in22);
    // NOLINTEND
    _merge_network_2x4(l11, l12, out1, out2);
    _merge_network_2x4(h11, h12, out3, out4);
  }

  template <size_t kernel_size, typename MulitVecType>
  struct BitonicMergeNetwork {
    static inline void merge(MulitVecType& /*in1*/, MulitVecType& /*in2*/, MulitVecType& /*out1*/,
                             MulitVecType& /*out2*/) {
      assert(false && "Not implemented.");
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<4, MultiVecType> {
    static inline void merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1, MultiVecType& out2) {
      _reverse(in2.a);
      _merge_network_2x4(in1.a, in2.a, out1.a, out2.a);
    }
  };

  template <typename MulitVecType>
  struct BitonicMergeNetwork<8, MulitVecType> {
    static inline void merge(MulitVecType& in1, MulitVecType& in2, MulitVecType& out1, MulitVecType& out2) {
      _reverse(in2.a);
      _reverse(in2.b);
      // NOLINTBEGIN
      auto l11 = __builtin_elementwise_min(in1.a, in2.b);
      auto l12 = __builtin_elementwise_min(in1.b, in2.a);
      auto h11 = __builtin_elementwise_max(in1.a, in2.b);
      auto h12 = __builtin_elementwise_max(in1.b, in2.a);
      // NOLINTEND
      _merge_network_2x4(l11, l12, out1.a, out1.b);
      _merge_network_2x4(h11, h12, out2.a, out2.b);
    }
  };

  template <typename MulitVecType>
  struct BitonicMergeNetwork<16, MulitVecType> {
    static inline void merge(MulitVecType& in1, MulitVecType& in2, MulitVecType& out1, MulitVecType& out2) {
      _reverse(in2.a);
      _reverse(in2.b);
      _reverse(in2.c);
      _reverse(in2.d);
      // NOLINTBEGIN
      auto l01 = __builtin_elementwise_min(in1.a, in2.d);
      auto l02 = __builtin_elementwise_min(in1.b, in2.c);
      auto l03 = __builtin_elementwise_min(in1.c, in2.b);
      auto l04 = __builtin_elementwise_min(in1.d, in2.a);
      auto h01 = __builtin_elementwise_max(in1.a, in2.d);
      auto h02 = __builtin_elementwise_max(in1.b, in2.c);
      auto h03 = __builtin_elementwise_max(in1.c, in2.b);
      auto h04 = __builtin_elementwise_max(in1.d, in2.a);
      // NOLINTEND
      _merge_network_2x8(l01, l02, l03, l04, out1.a, out1.b, out1.c, out1.d);
      _merge_network_2x8(h01, h02, h03, h04, out2.a, out2.b, out2.c, out2.d);
    }
  };

  // NOLINTEND(cppcoreguidelines-pro-type-vararg, hicpp-vararg)

  // End merge network primitives.

 public:
  template <size_t kernel_size>
  static inline void __attribute__((always_inline)) merge_equal_length(T* const a_address, T* const b_address,
                                                                       T* const output_address, const size_t length) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto REGISTER_COUNT = kernel_size / COUNT_PER_REGISTER;
    using MultiVecType = MultiVec<REGISTER_COUNT, COUNT_PER_REGISTER, VecType>;
    using BitonicMergeNetwork = BitonicMergeNetwork<kernel_size, MultiVecType>;

    auto* a_pointer = reinterpret_cast<block_t*>(a_address);
    auto* b_pointer = reinterpret_cast<block_t*>(b_address);
    auto* const a_end = reinterpret_cast<block_t*>(a_address + length);
    auto* const b_end = reinterpret_cast<block_t*>(b_address + length);

    auto* output_pointer = reinterpret_cast<block_t*>(output_address);
    auto* next_pointer = b_pointer;

    auto a_input = MultiVecType{};
    auto b_input = MultiVecType{};
    auto lower_merge_output = MultiVecType{};
    auto upper_merge_output = MultiVecType{};
    a_input.load(a_address);
    b_input.load(b_address);
    ++a_pointer;
    ++b_pointer;

    BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
    lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
    ++output_pointer;

    // As long as both A and B are not empty, do fetch and 2x4 merge.
    while (a_pointer < a_end && b_pointer < b_end) {
      choose_next_and_update_pointers<block_t, T>(next_pointer, a_pointer, b_pointer);
      a_input = upper_merge_output;
      b_input.load(reinterpret_cast<T*>(next_pointer));
      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++output_pointer;
    }
    // If A not empty, merge remainder of A.
    while (a_pointer < a_end) {
      a_input.load(reinterpret_cast<T*>(a_pointer));
      b_input = upper_merge_output;
      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++a_pointer;
      ++output_pointer;
    }
    // If B not empty, merge remainder of B.
    while (b_pointer < b_end) {
      a_input = upper_merge_output;
      b_input.load(reinterpret_cast<T*>(b_pointer));
      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++b_pointer;
      ++output_pointer;
    }
    upper_merge_output.store(reinterpret_cast<T*>(output_pointer));
  }

  template <size_t kernel_size>
  static inline void __attribute__((always_inline)) merge_variable_length(T* a_address, T* b_address, T* output_address,
                                                                          const size_t a_length,
                                                                          const size_t b_length) {
    using block_t = struct alignas(kernel_size * sizeof(T)) {};

    static constexpr auto REGISTER_COUNT = kernel_size / COUNT_PER_REGISTER;
    using MulitVecType = MultiVec<REGISTER_COUNT, COUNT_PER_REGISTER, VecType>;
    using BitonicMergeNetwork = BitonicMergeNetwork<kernel_size, MulitVecType>;

    constexpr auto ALIGNMENT_BIT_MASK = get_alignment_bitmask<kernel_size>();

    const auto a_rounded_length = a_length & ALIGNMENT_BIT_MASK;
    const auto b_rounded_length = b_length & ALIGNMENT_BIT_MASK;

    auto a_index = size_t{0};
    auto b_index = size_t{0};

    auto& out = output_address;

    if (a_rounded_length > kernel_size && b_rounded_length > kernel_size) {
      auto* a_pointer = reinterpret_cast<block_t*>(a_address);
      auto* b_pointer = reinterpret_cast<block_t*>(b_address);
      auto* const a_end = reinterpret_cast<block_t*>(a_address + a_length) - 1;
      auto* const b_end = reinterpret_cast<block_t*>(b_address + b_length) - 1;

      auto* output_pointer = reinterpret_cast<block_t*>(out);
      auto* next_pointer = b_pointer;

      auto a_input = MulitVecType{};
      auto b_input = MulitVecType{};
      auto lower_merge_output = MulitVecType{};
      auto upper_merge_output = MulitVecType{};
      a_input.load(reinterpret_cast<T*>(a_pointer));
      b_input.load(reinterpret_cast<T*>(b_pointer));
      ++a_pointer;
      ++b_pointer;

      BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
      lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
      ++output_pointer;

      // As long as both A and B are not empty, do fetch and 2x4 merge.
      while (a_pointer < a_end && b_pointer < b_end) {
        choose_next_and_update_pointers<block_t, T>(next_pointer, a_pointer, b_pointer);
        a_input = upper_merge_output;
        b_input.load(reinterpret_cast<T*>(next_pointer));
        BitonicMergeNetwork::merge(a_input, b_input, lower_merge_output, upper_merge_output);
        lower_merge_output.store(reinterpret_cast<T*>(output_pointer));
        ++output_pointer;
      }

      const auto last_element_from_merge_output = upper_merge_output.last()[COUNT_PER_REGISTER - 1];
      if (last_element_from_merge_output <= *reinterpret_cast<T*>(a_pointer)) {
        --a_pointer;
        upper_merge_output.store(reinterpret_cast<T*>(a_pointer));
      } else {
        --b_pointer;
        upper_merge_output.store(reinterpret_cast<T*>(b_pointer));
      }

      a_index = reinterpret_cast<T*>(a_pointer) - a_address;
      b_index = reinterpret_cast<T*>(b_pointer) - b_address;
      a_address = reinterpret_cast<T*>(a_pointer);
      b_address = reinterpret_cast<T*>(b_pointer);
      out = reinterpret_cast<T*>(output_pointer);
    }
    // Serial Merge.
    while (a_index < a_length && b_index < b_length) {
      auto* next = b_address;
      const auto cmp = *a_address < *b_address;
      const auto cmp_neg = !cmp;
      a_index += cmp;
      b_index += cmp_neg;
      next = cmp ? a_address : b_address;
      *out = *next;
      ++out;
      a_address += cmp;
      b_address += cmp_neg;
    }
    while (a_index < a_length) {
      *out = *a_address;
      ++a_index;
      out++;
      ++a_address;
    }
    while (b_index < b_length) {
      *out = *b_address;
      ++b_index;
      out++;
      ++b_address;
    }
  }
};
