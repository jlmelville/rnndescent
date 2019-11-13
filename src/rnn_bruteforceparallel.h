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
#include "rnn_parallel.h"
#include "tdoann/heap.h"
#include <RcppParallel.h>

template <typename Distance>
struct BruteForceWorker : public RcppParallel::Worker {

  tdoann::SimpleNeighborHeap &neighbor_heap;
  Distance &distance;
  const std::size_t n_ref_points;

  BruteForceWorker(tdoann::SimpleNeighborHeap &neighbor_heap,
                   Distance &distance, std::size_t n_ref_points)
      : neighbor_heap(neighbor_heap), distance(distance),
        n_ref_points(n_ref_points) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t query = begin; query < end; query++) {
      for (std::size_t ref = 0; ref < n_ref_points; ref++) {
        double d = distance(ref, query);
        if (neighbor_heap.accepts(query, d)) {
          neighbor_heap.unchecked_push(query, d, ref);
        }
      }
    }
  }
};

template <typename Distance, typename Progress>
void nnbf_parallel(tdoann::SimpleNeighborHeap &neighbor_heap,
                   Distance &distance, Progress &progress,
                   std::size_t block_size = 64, std::size_t grain_size = 1) {

  nnbf_parallel_query(neighbor_heap, distance, neighbor_heap.n_points, progress,
                      block_size, grain_size);
}

template <typename Distance, typename Progress>
void nnbf_parallel_query(tdoann::SimpleNeighborHeap &neighbor_heap,
                         Distance &distance, const std::size_t n_ref_points,
                         Progress &progress, std::size_t block_size = 64,
                         std::size_t grain_size = 1) {
  BruteForceWorker<Distance> worker(neighbor_heap, distance, n_ref_points);

  batch_parallel_for(worker, progress, neighbor_heap.n_points, block_size,
                     grain_size);

  neighbor_heap.deheap_sort();
}

#endif // RNND_BRUTEFORCEPARALLEL_H
