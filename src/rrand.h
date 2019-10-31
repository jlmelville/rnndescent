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

#ifndef RNND_RRAND_H
#define RNND_RRAND_H

#include "Rcpp.h"
#include "tauprng.h"

// based on code in the dqsample package
uint64_t random64() {
  return R::runif(0, 1) * std::numeric_limits<uint64_t>::max();
}

struct TauRand {

  tau_prng prng;

  TauRand(): prng(random64(), random64(), random64()) {}
  TauRand(uint64_t state0, uint64_t state1, uint64_t state2) :
    prng(state0, state1, state2) {}

  // a random uniform value between 0 and 1
  double unif() {
    return prng.rand();
  }
};

#endif // RNND_RRAND_H
