//  rnndescent -- An R package for nearest neighbor descent
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

#ifndef RNN_CANDIDATEPRIORITY_H
#define RNN_CANDIDATEPRIORITY_H

#include "tdoann/candidatepriority.h"
#include "tdoann/typedefs.h"

#include "rnn_rng.h"

namespace rnnd {
namespace cp {
struct RandomSerial {
  RRand rand;
  RandomSerial() : rand() {}
  auto operator()(const NeighborHeap &, std::size_t) -> double {
    return rand.unif();
  }
  const constexpr static bool should_sort = false;
};

struct RandomParallel {
  TauRand rand;
  RandomParallel(uint64_t seed, std::size_t end) : rand(seed, end) {}

  auto operator()(const NeighborHeap &, std::size_t) -> double {
    return rand.unif();
  }
  const constexpr static bool should_sort = false;
};
} // namespace cp
} // namespace rnnd

namespace tdoann {
namespace cp {
template <> struct Factory<rnnd::cp::RandomParallel> {
  using Type = rnnd::cp::RandomParallel;
  uint64_t seed;
  Factory(uint64_t seed) : seed(seed) {}

  auto create() -> Type { return Type(seed, 42); }

  auto create(std::size_t, std::size_t end) -> Type { return Type(seed, end); }
  const constexpr static bool should_sort = Type::should_sort;
};
} // namespace cp
} // namespace tdoann

#endif // RNN_CANDIDATEPRIORITY_H
