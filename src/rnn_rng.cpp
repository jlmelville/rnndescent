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

#include "R_randgen.h"
#include "Rcpp.h"
#include "convert_seed.h"
#include "dqrng_generator.h"
#include <dqrng.h>

#include "dqsample.h"

#include "rnn_rng.h"

// Uses R API: Not thread safe
auto pseed() -> uint64_t {
  Rcpp::IntegerVector seed(2, dqrng::R_random_int);
  return dqrng::convert_seed<uint64_t>(seed);
}

auto parallel_rng() -> dqrng::rng64_t {
  return std::make_shared<dqrng::random_64bit_wrapper<pcg64>>();
}

// based on code in the dqsample package
auto random64() -> uint64_t {
  return R::runif(0, 1) * (std::numeric_limits<uint64_t>::max)();
}

auto combine_seeds(uint32_t msw, uint32_t lsw) -> uint64_t {
  return (static_cast<uint64_t>(msw) << 32) | static_cast<uint64_t>(lsw);
}

auto RRand::unif() -> double { return R::runif(0, 1); }

TauRand::TauRand(uint64_t seed, uint64_t seed2) : prng(nullptr) {
  dqrng::rng64_t rng = parallel_rng();
  rng->seed(seed, seed2);

  // Stitch together 3 64-bit ints from 6 32-bit ones
  std::vector<uint32_t> tau_seeds32;
  dqsample::sample<uint32_t>(tau_seeds32, rng,
                             (std::numeric_limits<uint32_t>::max)(), 6, true);

  prng.reset(
      new tdoann::tau_prng(combine_seeds(tau_seeds32[0], tau_seeds32[1]),
                           combine_seeds(tau_seeds32[2], tau_seeds32[3]),
                           combine_seeds(tau_seeds32[4], tau_seeds32[5])));
}
auto TauRand::unif() -> double { return prng->rand(); }

ParallelRand::ParallelRand() : seed(0) {}
void ParallelRand::reseed() { seed = pseed(); }
auto ParallelRand::get_rand(uint64_t seed2) -> TauRand {
  return TauRand(seed, seed2);
}
