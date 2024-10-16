#pragma once

#include "abstract_two_way_merge.hpp"
#include "simd_utils.hpp"

template <size_t count_per_register, typename T>
class TwoWayMerge : public AbstractTwoWayMerge<count_per_register, T, TwoWayMerge<count_per_register, T>> {
  friend struct AbstractTwoWayMerge<count_per_register, T, TwoWayMerge<count_per_register, T>>;
};

// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg, hicpp-vararg)

template <typename T>
class TwoWayMerge<2, T> : public AbstractTwoWayMerge<2, T, TwoWayMerge<2, T>> {
  friend class AbstractTwoWayMerge<2, T, TwoWayMerge<2, T>>;
  static constexpr auto COUNT_PER_REGISTER = 2;
  static constexpr auto REGISTER_SIZE = COUNT_PER_REGISTER * sizeof(T);
  using VecType = Vec<REGISTER_SIZE, T>;

  template <size_t kernel_size, typename MulitVecType>
  struct BitonicMergeNetwork {
    static inline void __attribute__((always_inline)) merge(MulitVecType& /*in1*/, MulitVecType& /*in2*/,
                                                            MulitVecType& /*out1*/, MulitVecType& /*out2*/) {
      static_assert(false, "Not implemented.");
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<2, MultiVecType> {
    static inline void __attribute__((always_inline)) merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1,
                                                            MultiVecType& out2) {
      _reverse(in2.a);
      _merge_network_2x2(in1.a, in2.a, out1.a, out2.a);
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<4, MultiVecType> {
    static inline void __attribute__((always_inline)) merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1,
                                                            MultiVecType& out2) {
      _reverse(in2.a);
      _reverse(in2.b);
      auto l11 = __builtin_elementwise_min(in1.a, in2.b);
      auto l12 = __builtin_elementwise_min(in1.b, in2.a);
      auto h11 = __builtin_elementwise_max(in1.a, in2.b);
      auto h12 = __builtin_elementwise_max(in1.b, in2.a);
      _merge_network_2x2(l11, l12, out1.a, out1.b);
      _merge_network_2x2(h11, h12, out2.a, out2.b);
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<8, MultiVecType> {
    static inline void __attribute__((always_inline)) merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1,
                                                            MultiVecType& out2) {
      _reverse(in2.a);
      _reverse(in2.b);
      _reverse(in2.c);
      _reverse(in2.d);
      auto l01 = __builtin_elementwise_min(in1.a, in2.d);
      auto l02 = __builtin_elementwise_min(in1.b, in2.c);
      auto l03 = __builtin_elementwise_min(in1.c, in2.b);
      auto l04 = __builtin_elementwise_min(in1.d, in2.a);
      auto h01 = __builtin_elementwise_max(in1.a, in2.d);
      auto h02 = __builtin_elementwise_max(in1.b, in2.c);
      auto h03 = __builtin_elementwise_max(in1.c, in2.b);
      auto h04 = __builtin_elementwise_max(in1.d, in2.a);
      _merge_network_2x4(l01, l02, l03, l04, out1.a, out1.b, out1.c, out1.d);
      _merge_network_2x4(h01, h02, h03, h04, out2.a, out2.b, out2.c, out2.d);
    }
  };

  // Begin merge network primitives.
  static inline void __attribute__((always_inline)) _reverse(VecType& vec) {
    vec = __builtin_shufflevector(vec, vec, 1, 0);
  }

  static inline void __attribute__((always_inline)) _merge_network_2x2(VecType& input_a, VecType& input_b,
                                                                       VecType& out1, VecType& out2) {
    // Level 1
    auto low1 = __builtin_elementwise_min(input_a, input_b);
    auto high1 = __builtin_elementwise_max(input_a, input_b);
    auto permutated_low1 = __builtin_shufflevector(low1, high1, 0, 2);
    auto permutated_high1 = __builtin_shufflevector(low1, high1, 1, 3);
    // Level 2
    auto low2 = __builtin_elementwise_min(permutated_low1, permutated_high1);
    auto high2 = __builtin_elementwise_max(permutated_low1, permutated_high1);
    out1 = __builtin_shufflevector(low2, high2, 0, 2);
    out2 = __builtin_shufflevector(low2, high2, 1, 3);
  }

  static inline void __attribute__((always_inline)) _merge_network_2x4(VecType& in11, VecType& in12, VecType& in21,
                                                                       VecType& in22, VecType& out1, VecType& out2,
                                                                       VecType& out3, VecType& out4) {
    auto l11 = __builtin_elementwise_min(in11, in21);
    auto h11 = __builtin_elementwise_max(in11, in21);
    auto l12 = __builtin_elementwise_min(in12, in22);
    auto h12 = __builtin_elementwise_max(in12, in22);
    _merge_network_2x2(l11, l12, out1, out2);
    _merge_network_2x2(h11, h12, out3, out4);
  }
};

template <typename T>
class TwoWayMerge<4, T> : public AbstractTwoWayMerge<4, T, TwoWayMerge<4, T>> {
  friend struct AbstractTwoWayMerge<4, T, TwoWayMerge<4, T>>;

  static constexpr auto COUNT_PER_REGISTER = 4;
  static constexpr auto REGISTER_SIZE = COUNT_PER_REGISTER * sizeof(T);
  using VecType = Vec<REGISTER_SIZE, T>;

  template <size_t kernel_size, typename MulitVecType>
  struct BitonicMergeNetwork {
    static inline void __attribute__((always_inline)) merge(MulitVecType& /*in1*/, MulitVecType& /*in2*/,
                                                            MulitVecType& /*out1*/, MulitVecType& /*out2*/) {
      static_assert(false, "Not implemented.");
    }
  };

  template <typename MultiVecType>
  struct BitonicMergeNetwork<4, MultiVecType> {
    static inline void __attribute__((always_inline)) merge(MultiVecType& in1, MultiVecType& in2, MultiVecType& out1,
                                                            MultiVecType& out2) {
      _reverse(in2.a);
      _merge_network_2x4(in1.a, in2.a, out1.a, out2.a);
    }
  };

  template <typename MulitVecType>
  struct BitonicMergeNetwork<8, MulitVecType> {
    static inline void __attribute__((always_inline)) merge(MulitVecType& in1, MulitVecType& in2, MulitVecType& out1,
                                                            MulitVecType& out2) {
      _reverse(in2.a);
      _reverse(in2.b);
      auto l11 = __builtin_elementwise_min(in1.a, in2.b);
      auto l12 = __builtin_elementwise_min(in1.b, in2.a);
      auto h11 = __builtin_elementwise_max(in1.a, in2.b);
      auto h12 = __builtin_elementwise_max(in1.b, in2.a);
      _merge_network_2x4(l11, l12, out1.a, out1.b);
      _merge_network_2x4(h11, h12, out2.a, out2.b);
    }
  };

  template <typename MulitVecType>
  struct BitonicMergeNetwork<16, MulitVecType> {
    static inline void __attribute__((always_inline)) merge(MulitVecType& in1, MulitVecType& in2, MulitVecType& out1,
                                                            MulitVecType& out2) {
      _reverse(in2.a);
      _reverse(in2.b);
      _reverse(in2.c);
      _reverse(in2.d);
      auto l01 = __builtin_elementwise_min(in1.a, in2.d);
      auto l02 = __builtin_elementwise_min(in1.b, in2.c);
      auto l03 = __builtin_elementwise_min(in1.c, in2.b);
      auto l04 = __builtin_elementwise_min(in1.d, in2.a);
      auto h01 = __builtin_elementwise_max(in1.a, in2.d);
      auto h02 = __builtin_elementwise_max(in1.b, in2.c);
      auto h03 = __builtin_elementwise_max(in1.c, in2.b);
      auto h04 = __builtin_elementwise_max(in1.d, in2.a);
      _merge_network_2x8(l01, l02, l03, l04, out1.a, out1.b, out1.c, out1.d);
      _merge_network_2x8(h01, h02, h03, h04, out2.a, out2.b, out2.c, out2.d);
    }
  };

  // Begin merge network primitives.
  static inline void __attribute__((always_inline)) _reverse(VecType& vec) {
    vec = __builtin_shufflevector(vec, vec, 3, 2, 1, 0);
  }

  static inline void __attribute__((always_inline)) _merge_network_2x4(VecType& input_a, VecType& input_b,
                                                                       VecType& out1, VecType& out2) {
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

  static inline void __attribute__((always_inline)) _merge_network_2x8(VecType& in11, VecType& in12, VecType& in21,
                                                                       VecType& in22, VecType& out1, VecType& out2,
                                                                       VecType& out3, VecType& out4) {
    // NOLINTBEGIN
    auto l11 = __builtin_elementwise_min(in11, in21);
    auto l12 = __builtin_elementwise_min(in12, in22);
    auto h11 = __builtin_elementwise_max(in11, in21);
    auto h12 = __builtin_elementwise_max(in12, in22);
    // NOLINTEND
    _merge_network_2x4(l11, l12, out1, out2);
    _merge_network_2x4(h11, h12, out3, out4);
  }
};

// NOLINTEND(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
