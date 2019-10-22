//  rnndescent -- Nearest Neighbor Descent method for approximate nearest neighbors
//
//  Copyright (C) 2019 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

// Three-component combined Tausworthe "taus88" PRNG from L'Ecuyer 1996.

#ifndef rnndescent_TAUPRNG_H
#define rnndescent_TAUPRNG_H

#include <cmath>
#include <limits>
#include "Rcpp.h"

// based on code in the dqsample package
uint64_t random64() {
  return R::runif(0, 1) * std::numeric_limits<uint64_t>::max();
}

class tau_prng {
  uint64_t state0;
  uint64_t state1; // technically this needs to always be > 7
  uint64_t state2; // and this should be > 15

  static constexpr uint64_t MAGIC0 = static_cast<uint64_t>(4294967294);
  static constexpr uint64_t MAGIC1 = static_cast<uint64_t>(4294967288);
  static constexpr uint64_t MAGIC2 = static_cast<uint64_t>(4294967280);

public:

  static constexpr double DINT_MAX = static_cast<double>(std::numeric_limits<int>::max());

  tau_prng() {
    state0 = random64();
    state1 = random64();
    state2 = random64();
  }

  tau_prng(uint64_t state0, uint64_t state1, uint64_t state2)
    : state0(state0), state1(state1), state2(state2) {}

  int32_t operator()() {
    state0 = (((state0 & MAGIC0) << 12) & 0xffffffff) ^
      ((((state0 << 13) & 0xffffffff) ^ state0) >> 19);
    state1 = (((state1 & MAGIC1) << 4) & 0xffffffff) ^
      ((((state1 << 2) & 0xffffffff) ^ state1) >> 25);
    state2 = (((state2 & MAGIC2) << 17) & 0xffffffff) ^
      ((((state2 << 3) & 0xffffffff) ^ state2) >> 11);

    return state0 ^ state1 ^ state2;
  }

  double rand() {
    return std::abs(operator()() / DINT_MAX);
  }
};

struct TauRand {

  tau_prng prng;

  TauRand(): prng() {}
  TauRand(uint64_t state0, uint64_t state1, uint64_t state2) :
    prng(state0, state1, state2) {}

  // a random uniform value between 0 and 1
  double unif() {
    return prng.rand();
  }
};

#endif // rnndescent_TAUPRNG_H
