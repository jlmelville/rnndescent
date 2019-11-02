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

#ifndef RNND_RNN_BRUTE_FORCE_PARALLEL_H
#define RNND_RNN_BRUTE_FORCE_PARALLEL_H

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include "rnn_parallel.h"
#include "heap.h"

template<typename Distance>
struct BruteForceWorker : public RcppParallel::Worker {

  SimpleNeighborHeap neighbor_heap;
  Distance distance;
  const std::size_t n_points;
  const std::size_t n_nbrs;

  BruteForceWorker(
    SimpleNeighborHeap& neighbor_heap,
    Distance& distance
  ) :
    neighbor_heap(neighbor_heap),
    distance(distance),
    n_points(neighbor_heap.n_points),
    n_nbrs(neighbor_heap.n_nbrs)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      const std::size_t i0 = i * n_nbrs;
      for (std::size_t j = 0; j < n_points; j++) {
        double weight = distance(i, j);
        if (weight >= neighbor_heap.distance(i0)) {
          continue;
        }
        neighbor_heap.unchecked_push(i, weight, j);
      }
    }
  }
};

template <typename Distance,
          typename Progress>
void nnbf_parallel(
    SimpleNeighborHeap& neighbor_heap,
    Distance& distance,
    Progress& progress,
    std::size_t grain_size = 1
    )
{
  BruteForceWorker<Distance> worker(neighbor_heap, distance);

  batch_parallel_for(worker, progress, worker.n_points, 64, grain_size);

  neighbor_heap = worker.neighbor_heap;
  neighbor_heap.deheap_sort();
}

#endif // RNND_RNN_BRUTE_FORCE_PARALLEL_H