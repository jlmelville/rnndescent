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
#include "heap.h"

template<typename Distance>
struct BruteForceWorker : public RcppParallel::Worker {

  ArrayHeap<Distance> heap;
  std::size_t n_points;

  BruteForceWorker(
    ArrayHeap<Distance>& heap
  ) :
    heap(heap),
    n_points(heap.neighbor_heap.n_points)
  {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for (std::size_t j = 0; j < n_points; j++) {
        heap.add_pair_asymm(i, j, true);
      }
    }
  }
};

template <typename Distance,
          typename Progress>
void nnbf_parallel(
    ArrayHeap<Distance>& heap,
    Progress& progress,
    std::size_t grain_size = 1,
    bool verbose = false
    )
{
  BruteForceWorker<Distance> worker(heap);
  RcppParallel::parallelFor(0, worker.n_points, worker, grain_size);
  heap = worker.heap;
  heap.neighbor_heap.deheap_sort();
}

#endif // RNND_RNN_BRUTE_FORCE_PARALLEL_H
