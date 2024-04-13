//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2022 James Melville
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

#ifndef RNNDESCENT_RANDOM_H
#define RNNDESCENT_RANDOM_H

#include <limits>
#include <memory>
#include <vector>

#include "R_randgen.h"
#include "Rcpp.h"

#include "convert_seed.h"
#include "dqrng_generator.h"
#include <dqrng.h>
#include <dqrng_distribution.h>
#include <pcg_random.hpp>

#include "dqsample.h"
#include "tdoann/random.h"
#include "tdoann/tauprng.h"

namespace rnndescent {

// Uses R API: Not thread safe
inline auto r_seed() -> uint64_t {
  Rcpp::IntegerVector seed(2, dqrng::R_random_int);
  return dqrng::convert_seed<uint64_t>(seed);
}

inline auto create_dqrng() -> dqrng::rng64_t {
  auto seed1 = r_seed();
  auto seed2 = r_seed();
  return dqrng::generator<pcg64>(seed1, seed2);
}

inline auto create_dqrng(uint64_t seed, uint64_t seed2) -> dqrng::rng64_t {
  return dqrng::generator<pcg64>(seed, seed2);
}

inline auto combine_seeds(uint32_t msw, uint32_t lsw) -> uint64_t {
  return (static_cast<uint64_t>(msw) << 32U) | static_cast<uint64_t>(lsw);
}

// Uniform RNG using R API: don't try and use this in anything that might
// end up in a thread
class RRand : public tdoann::RandomGenerator {
public:
  double unif() override { return R::runif(0, 1); }
};

// Use Taus88 RNG
struct TauRand : public tdoann::RandomGenerator {
  std::unique_ptr<tdoann::tau_prng> prng{nullptr};

  TauRand(uint64_t seed, uint64_t seed2) {
    dqrng::rng64_t rng = create_dqrng(seed, seed2);

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
  double unif() override { return prng->rand(); }
};

struct PcgRand : public tdoann::RandomGenerator {
  dqrng::uniform_distribution dist;
  pcg64 prng;

  PcgRand(uint64_t seed, uint64_t seed2) : dist(0.0, 1.0), prng(seed, seed2) {}

  double unif() override { return dist(prng); }
};

template <typename T>
class ParallelRNGAdapter : public tdoann::ParallelRandomProvider {
  uint64_t seed{0};

public:
  ParallelRNGAdapter() = default;

  void initialize() override { seed = r_seed(); }

  std::unique_ptr<tdoann::RandomGenerator>
  get_parallel_instance(uint64_t seed2) override {
    return std::make_unique<T>(seed, seed2);
  }
};

template <typename Int>
class DQIntSampler : public tdoann::RandomIntGenerator<Int> {
private:
  dqrng::rng64_t rng;

public:
  // Not thread safe
  DQIntSampler() : rng(create_dqrng()) {}

  DQIntSampler(uint64_t seed, uint64_t seed2)
      : rng(create_dqrng(seed, seed2)) {}

  // Generates a random integer in range [0, n)
  Int rand_int(Int n) override {
    std::vector<Int> result;
    dqsample::sample<Int>(result, rng, n, 1, false);
    return result[0];
  }

  // Generates n_ints random integers in range [0, max_val)
  std::vector<Int> sample(int max_val, int n_ints) override {
    std::vector<Int> result;
    dqsample::sample<Int>(result, rng, max_val, n_ints, false);
    return result;
  }
};

template <typename Int, template <typename> class RNG>
class ParallelIntRNGAdapter : public tdoann::ParallelRandomIntProvider<Int> {
private:
  uint64_t seed{0};

public:
  ParallelIntRNGAdapter() = default;

  // Not thread safe: call this outside the lambda
  void initialize() override { seed = r_seed(); }

  // call this inside a window to get a thread safe RNG: seed1 is intended to
  // be related to a system RNG (e.g. the R RNG) and seed2 is one of the window
  // parameters (e.g. the end of the window size) so that you get different
  // random numbers in each window, but they are related to the random number
  // seed
  std::unique_ptr<tdoann::RandomIntGenerator<Int>>
  get_parallel_instance(uint64_t seed2) override {
    return std::make_unique<RNG<Int>>(seed, seed2);
  }
};

} // namespace rnndescent

#endif // RNNDESCENT_RANDOM_H
