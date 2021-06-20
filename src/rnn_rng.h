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

#ifndef RNN_RNG_H
#define RNN_RNG_H

#include <memory>
#include <vector>

#include "dqrng_generator.h"
#include "tdoann/tauprng.h"

auto pseed() -> uint64_t;
auto parallel_rng() -> dqrng::rng64_t;
auto combine_seeds(uint32_t, uint32_t) -> uint64_t;

// Use R API for RNG
struct RRand {
  // a random uniform value between 0 and 1
  auto unif() -> double;
};

// Use Taus88 RNG
struct TauRand {
  std::unique_ptr<tdoann::tau_prng> prng;

  TauRand(uint64_t seed, uint64_t seed2);
  // a random uniform value between 0 and 1
  auto unif() -> double;
};

struct ParallelRand {
  uint64_t seed;

  ParallelRand();
  void reseed();
  auto get_rand(uint64_t seed2) -> TauRand;
};

#endif // RNN_RNG_H
