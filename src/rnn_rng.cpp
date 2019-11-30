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

#include <limits>

#include "Rcpp.h"
// [[Rcpp::depends(dqrng)]]
#include "R_randgen.h"
#include "convert_seed.h"
#include "dqrng_generator.h"
#include <dqrng.h>

#include "rnn_rng.h"
#include "rnn_sample.h"

// Uses R API: Not thread safe
uint64_t pseed() {
  Rcpp::IntegerVector seed(2, dqrng::R_random_int);
  return dqrng::convert_seed<uint64_t>(seed);
}

dqrng::rng64_t parallel_rng(uint64_t seed) {
  return std::make_shared<dqrng::random_64bit_wrapper<pcg64>>();
}

// based on code in the dqsample package
uint64_t random64() {
  return R::runif(0, 1) * (std::numeric_limits<uint64_t>::max)();
}

double RRand::unif() { return R::runif(0, 1); }

TauRand::TauRand(uint64_t seed, uint64_t seed2) : prng(nullptr) {
  dqrng::rng64_t rng = parallel_rng(seed);
  rng->seed(seed, seed2);
  auto tau_seeds =
      sample<uint64_t>(rng, (std::numeric_limits<uint64_t>::max)(), 3, true);

  prng.reset(new tdoann::tau_prng(tau_seeds[0], tau_seeds[1], tau_seeds[2]));
}
double TauRand::unif() { return prng->rand(); }
