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

struct CandidatePriorityRandomSerial {
  RRand rand;
  CandidatePriorityRandomSerial() : rand() {}
  double operator()(const NeighborHeap &, std::size_t) { return rand.unif(); }
  const constexpr static bool should_sort = false;
};

struct CandidatePriorityRandomParallel {
  TauRand rand;
  CandidatePriorityRandomParallel(uint64_t seed, std::size_t end)
      : rand(seed, end) {}

  double operator()(const NeighborHeap &, std::size_t) { return rand.unif(); }
  const constexpr static bool should_sort = false;
};

template <> struct CandidatePriorityFactory<CandidatePriorityRandomParallel> {
  using Type = CandidatePriorityRandomParallel;
  uint64_t seed;
  CandidatePriorityFactory(uint64_t seed) : seed(seed) {}

  Type create() { return Type(seed, 42); }

  Type create(std::size_t, std::size_t end) { return Type(seed, end); }
  const constexpr static bool should_sort = Type::should_sort;
};

#endif // RNN_CANDIDATEPRIORITY_H
