//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
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

#ifndef RNN_SAMPLE_H
#define RNN_SAMPLE_H

#include <dqrng.h>

#include "dqsample.h"

#include "rnn_rng.h"

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

#endif // RNN_SAMPLE_H
