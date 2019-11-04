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

#ifndef NND_ARRAYHEAP_H
#define NND_ARRAYHEAP_H

#include "heap.h"

// Checks for duplicates by iterating over the entire array of stored indexes
template <typename WeightMeasure>
struct ArrayHeap
{
  NeighborHeap neighbor_heap;
  WeightMeasure weight_measure;

  ArrayHeap(
    WeightMeasure& weight_measure,
    const std::size_t n_points,
    const std::size_t size) :
    neighbor_heap(n_points, size),
    weight_measure(weight_measure)
  {}

  ArrayHeap(const ArrayHeap&) = default;
  ~ArrayHeap() = default;
  ArrayHeap& operator=(const ArrayHeap &) = default;

  std::size_t add_pair(
      std::size_t i,
      std::size_t j,
      bool flag)
  {
    double d = weight_measure(i, j);

    std::size_t c = neighbor_heap.checked_push(i, d, j, flag);
    if (i != j) {
      c += neighbor_heap.checked_push(j, d, i, flag);
    }

    return c;
  }

  bool contains(std::size_t row, std::size_t index) const
  {
    const std::size_t n_nbrs = neighbor_heap.n_nbrs;
    const std::size_t rnnbrs = row * n_nbrs;
    for (std::size_t i = 0; i < n_nbrs; i++) {
      if (index == neighbor_heap.idx[rnnbrs + i]) {
        return true;
      }
    }
    return false;
  }
};

template <typename Rand>
struct RandomWeight
{
  Rand& rand;

  RandomWeight(Rand& rand) : rand(rand) { }

  double operator()(
      std::size_t i,
      std::size_t j)
  {
    return rand.unif();
  }
};

template <typename Rand>
using RandomHeap = ArrayHeap<RandomWeight<Rand>>;

#endif // NND_ARRAYHEAP_H
