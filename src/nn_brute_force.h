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

#ifndef RNND_NN_BRUTE_FORCE_H
#define RNND_NN_BRUTE_FORCE_H

#include <Rcpp.h>

#include "heap.h"

template <typename Distance,
          typename Progress>
void nnbf(
    ArrayHeap<Distance>& heap,
    Progress& progress,
    bool verbose = false)
{
  auto& neighbor_heap = heap.neighbor_heap;
  const std::size_t n_points = neighbor_heap.n_points;
  const std::size_t n_nbrs = neighbor_heap.n_nbrs;
  for (std::size_t i = 0; i < n_points; i++) {
    const std::size_t i0 = i * n_nbrs;
    for (std::size_t j = i; j < n_points; j++) {
      double weight = heap.weight_measure(i, j);
      if (weight < neighbor_heap.distance(i0)) {
        neighbor_heap.unchecked_push(i, weight, j, true);
      }
      if (i != j && weight < neighbor_heap.distance(j * n_nbrs)) {
        neighbor_heap.unchecked_push(j, weight, i, true);
      }
    }
    progress.check_interrupt();
  }

  neighbor_heap.deheap_sort();
}

#endif // RNND_NN_BRUTE_FORCE_H
