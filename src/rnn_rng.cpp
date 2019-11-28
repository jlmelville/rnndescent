//  rnndescent -- Nearest Neighbor Descent method for approximate nearest
//  neighbors
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

#include "Rcpp.h"
// [[Rcpp::depends(dqrng)]]
#include "rnn_rng.h"
#include <dqrng.h>

void set_seed() {
  dqrng::dqRNGkind("Xoroshiro128+");
  auto seed = Rcpp::IntegerVector::create(R::runif(0, 1) *
                                          (std::numeric_limits<int>::max)());
  dqrng::dqset_seed(seed);
}

// based on code in the dqsample package
uint64_t random64() {
  return R::runif(0, 1) * std::numeric_limits<uint64_t>::max();
}

double RRand::unif() { return R::runif(0, 1); }

TauRand::TauRand() : prng(random64(), random64(), random64()) {}
double TauRand::unif() { return prng.rand(); }
