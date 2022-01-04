//  rnnr -- Code to generate random numbers in parallel
//
//  Copyright (C) 2022 James Melville
//
//  rnnr is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnnr is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnnr.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RNNR_SAMPLE_H
#define RNNR_SAMPLE_H

#include <limits>
#include <memory>
#include <vector>

#include "R_randgen.h"
#include "Rcpp.h"

#include "convert_seed.h"
#include "dqrng_generator.h"
#include <dqrng.h>

#include "dqsample.h"
#include "tdoann/tauprng.h"

// Uses R API: Not thread safe
inline auto pseed() -> uint64_t {
  Rcpp::IntegerVector seed(2, dqrng::R_random_int);
  return dqrng::convert_seed<uint64_t>(seed);
}

inline auto parallel_rng() -> dqrng::rng64_t {
  return std::make_shared<dqrng::random_64bit_wrapper<pcg64>>();
}

inline auto combine_seeds(uint32_t msw, uint32_t lsw) -> uint64_t {
  return (static_cast<uint64_t>(msw) << 32) | static_cast<uint64_t>(lsw);
}

// Uniform RNG

// Use R API for RNG
struct RRand {
  // a random uniform value between 0 and 1
  auto unif() -> double { return R::runif(0, 1); };
};

// Use Taus88 RNG
struct TauRand {
  std::unique_ptr<tdoann::tau_prng> prng;

  TauRand(uint64_t seed, uint64_t seed2) : prng(nullptr) {
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
  // a random uniform value between 0 and 1
  auto unif() -> double { return prng->rand(); }
};

struct ParallelRand {
  uint64_t seed;

  ParallelRand() : seed(0) {}
  void reseed() { seed = pseed(); };
  auto get_rand(uint64_t seed2) -> TauRand { return TauRand(seed, seed2); };
};

// Integer Sampler

template <typename Int> struct DQIntSampler {

  static auto get_seed() -> uint64_t { return pseed(); }

  uint64_t seed;
  uint64_t seed2;
  dqrng::rng64_t rng;

  DQIntSampler(uint64_t seed, uint64_t seed2) : rng(parallel_rng()) {
    rng->seed(seed, seed2);
  }

  auto sample(int n, int size, bool replace = false) -> std::vector<Int> {
    std::vector<Int> result;
    dqsample::sample<Int>(result, rng, n, size, replace);
    return result;
  }
};

#endif // RNNR_SAMPLE_H
