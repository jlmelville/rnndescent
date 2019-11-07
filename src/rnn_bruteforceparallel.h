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

#ifndef RNN_BRUTEFORCEPARALLEL_H
#define RNN_BRUTEFORCEPARALLEL_H

#include <Rcpp.h>
// [[Rcpp::depends(RcppParallel)]]
#include "heap.h"
#include "rnn_parallel.h"
#include <RcppParallel.h>

template <typename Distance>
struct BruteForceWorker : public RcppParallel::Worker {

  SimpleNeighborHeap neighbor_heap;
  Distance distance;
  const std::size_t n_points;
  const std::size_t n_nbrs;

  BruteForceWorker(SimpleNeighborHeap &neighbor_heap, Distance &distance)
      : neighbor_heap(neighbor_heap), distance(distance),
        n_points(neighbor_heap.n_points), n_nbrs(neighbor_heap.n_nbrs) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for (std::size_t j = 0; j < n_points; j++) {
        double d = distance(i, j);
        if (neighbor_heap.accepts(i, d)) {
          neighbor_heap.unchecked_push(i, d, j);
        }
      }
    }
  }
};

template <typename Distance, typename Progress>
void nnbf_parallel(SimpleNeighborHeap &neighbor_heap, Distance &distance,
                   Progress &progress, std::size_t grain_size = 1) {
  BruteForceWorker<Distance> worker(neighbor_heap, distance);

  batch_parallel_for(worker, progress, worker.n_points, 64, grain_size);

  neighbor_heap = worker.neighbor_heap;
  neighbor_heap.deheap_sort();
}

#endif // RNND_BRUTEFORCEPARALLEL_H
