// BSD 2-Clause License
//
// Copyright 2019 James Melville
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// OF SUCH DAMAGE.

// Three-component combined Tausworthe "taus88" PRNG from:
//
// L'Ecuyer, P. (1996).
// Maximally equidistributed combined Tausworthe generators.
// Mathematics of Computation, 65(213), 203-213.

#ifndef TDOANN_TAUPRNG_H
#define TDOANN_TAUPRNG_H

#include <cmath>
#include <cstdint>
#include <limits>

namespace tdoann {
class tau_prng {
  uint64_t state0;
  uint64_t state1; // technically this needs to always be > 7
  uint64_t state2; // and this should be > 15

  // Numbers from Figure 1
  static const constexpr uint64_t MAGIC0{4294967294};
  static const constexpr uint64_t MAGIC1{4294967288};
  static const constexpr uint64_t MAGIC2{4294967280};

  // These are the first q1, q2, q3 from Example 3
  static const constexpr uint64_t QUICKTAUS_Q0{13};
  static const constexpr uint64_t QUICKTAUS_Q1{2};
  static const constexpr uint64_t QUICKTAUS_Q2{3};

  // These are the first s1, s2, s3 from Example 3
  static const constexpr uint64_t QUICKTAUS_S0{12};
  static const constexpr uint64_t QUICKTAUS_S1{4};
  static const constexpr uint64_t QUICKTAUS_S2{17};

  // From Figure 1
  static const constexpr uint64_t SHIFT0{19};
  static const constexpr uint64_t SHIFT1{25};
  static const constexpr uint64_t SHIFT2{11};

  static const constexpr uint64_t MASK{0xffffffff};

  static const constexpr uint64_t MIN_STATE1{8};
  static const constexpr uint64_t MIN_STATE2{16};

public:
  static constexpr double DINT_MAX =
      static_cast<double>((std::numeric_limits<int>::max)());

  tau_prng(uint64_t state0, uint64_t state1, uint64_t state2)
      : state0(state0), state1(state1 >= MIN_STATE1 ? state1 : MIN_STATE1),
        state2(state2 >= MIN_STATE2 ? state2 : MIN_STATE2) {}

  auto operator()() -> int32_t {
    state0 = (((state0 & MAGIC0) << QUICKTAUS_S0) & MASK) ^
             ((((state0 << QUICKTAUS_Q0) & MASK) ^ state0) >> SHIFT0);
    state1 = (((state1 & MAGIC1) << QUICKTAUS_S1) & MASK) ^
             ((((state1 << QUICKTAUS_Q1) & MASK) ^ state1) >> SHIFT1);
    state2 = (((state2 & MAGIC2) << QUICKTAUS_S2) & MASK) ^
             ((((state2 << QUICKTAUS_Q2) & MASK) ^ state2) >> SHIFT2);

    return static_cast<int32_t>(state0 ^ state1 ^ state2);
  }

  auto rand() -> double { return std::abs(operator()() / DINT_MAX); }
};
} // namespace tdoann
#endif // TDOANN_TAUPRNG_H
