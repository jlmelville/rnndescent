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

#ifndef NND_NN_BRUTE_FORCE_H
#define NND_NN_BRUTE_FORCE_H

#include "heap.h"

template <typename Distance,
          typename Progress>
void nnbf(
    SimpleNeighborHeap& neighbor_heap,
    Distance& distance,
    Progress& progress)
{
  const std::size_t n_points = neighbor_heap.n_points;
  const std::size_t n_nbrs = neighbor_heap.n_nbrs;
  for (std::size_t i = 0; i < n_points; i++) {
    const std::size_t i0 = i * n_nbrs;
    for (std::size_t j = i; j < n_points; j++) {
      double weight = distance(i, j);
      if (weight < neighbor_heap.distance(i0)) {
        neighbor_heap.unchecked_push(i, weight, j);
      }
      if (i != j && weight < neighbor_heap.distance(j * n_nbrs)) {
        neighbor_heap.unchecked_push(j, weight, i);
      }
    }
    progress.increment();
    if (progress.check_interrupt()) {
      break;
    };
  }

  neighbor_heap.deheap_sort();
}

#endif // NND_NN_BRUTE_FORCE_H
