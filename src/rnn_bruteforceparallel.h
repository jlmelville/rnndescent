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

#include "rnn_parallel.h"
#include "tdoann/bruteforce.h"
#include "tdoann/heap.h"
#include "tdoann/progress.h"
#include <Rcpp.h>

template <typename Distance>
struct BruteForceWorker : public BatchParallelWorker {

  SimpleNeighborHeap &neighbor_heap;
  Distance &distance;
  const std::size_t n_ref_points;
  tdoann::NullProgress progress;

  BruteForceWorker(SimpleNeighborHeap &neighbor_heap, Distance &distance,
                   std::size_t n_ref_points)
      : neighbor_heap(neighbor_heap), distance(distance),
        n_ref_points(n_ref_points), progress() {}

  void operator()(std::size_t begin, std::size_t end) {
    nnbf_query_window(neighbor_heap, distance, n_ref_points, progress, begin,
                      end);
  }
};

template <typename Distance, typename Progress>
void nnbf_parallel(SimpleNeighborHeap &neighbor_heap, Distance &distance,
                   Progress &progress, std::size_t block_size = 64,
                   std::size_t grain_size = 1) {

  nnbf_parallel_query(neighbor_heap, distance, neighbor_heap.n_points, progress,
                      block_size, grain_size);
}

template <typename Distance, typename Progress>
void nnbf_parallel_query(SimpleNeighborHeap &neighbor_heap, Distance &distance,
                         const std::size_t n_ref_points, Progress &progress,
                         std::size_t block_size = 64,
                         std::size_t grain_size = 1) {
  BruteForceWorker<Distance> worker(neighbor_heap, distance, n_ref_points);

  batch_parallel_for(worker, progress, neighbor_heap.n_points, block_size,
                     grain_size);

  neighbor_heap.deheap_sort();
}

#endif // RNND_BRUTEFORCEPARALLEL_H
