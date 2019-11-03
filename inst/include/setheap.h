// BSD 2-Clause License
//
// Copyright 2019 James Melville
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// OF SUCH DAMAGE.

#ifndef NND_SETHEAP_H
#define NND_SETHEAP_H

#include <unordered_set>

#include <boost/functional/hash.hpp>

#include "heap.h"

using pair = std::pair<std::size_t, std::size_t>;


// Checks for duplicates by storing a set of already-seen pairs. Takes up more
// memory but might be faster if lots of duplicate pairs are expected
template <typename WeightMeasure>
struct SetHeap
{
  NeighborHeap neighbor_heap;
  const WeightMeasure& weight_measure;
  std::unordered_set<std::pair<std::size_t, std::size_t>,
                     boost::hash<pair>> seen;

  SetHeap(const WeightMeasure& weight_measure,
          const std::size_t n_points,
          const std::size_t size)
    : neighbor_heap(n_points, size),
      weight_measure(weight_measure),
      seen()
    {}

  SetHeap(const SetHeap&) = default;
  ~SetHeap() = default;
  SetHeap& operator=(const SetHeap &) = default;

  std::size_t add_pair(
      std::size_t i,
      std::size_t j,
      bool flag)
  {
    if (i > j) {
      std::swap(i, j);
    }

    if (!seen.emplace(i, j).second) {
      return 0;
    }

    double d = weight_measure(i, j);

    std::size_t c = 0;
    if (d < neighbor_heap.distance(i, 0)) {
      c += neighbor_heap.unchecked_push(i, d, j, flag);
    }
    if (i != j && d < neighbor_heap.distance(j, 0)) {
      c += neighbor_heap.unchecked_push(j, d, i, flag);
    }

    return c;
  }

  std::size_t add_pair_asymm(
      std::size_t i,
      std::size_t j,
      bool flag
    )
  {
    pair p(i, j);
    if (!seen.emplace(p).second) {
      return 0;
    }

    double d = weight_measure(i, j);

    std::size_t c = 0;
    if (d < neighbor_heap.distance(i, 0)) {
      c += neighbor_heap.unchecked_push(i, d, j, flag);
    }

    return c;
  }
};

#endif // NND_SETHEAP_H
