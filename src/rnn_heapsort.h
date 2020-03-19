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

#ifndef RNN_HEAPSORT_H
#define RNN_HEAPSORT_H

#include "tdoann/heap.h"
#include "tdoann/parallel.h"
#include "tdoann/progress.h"
#include "tdoann/typedefs.h"

template <typename NbrHeap = SimpleNeighborHeap>
struct HeapSortWorker : public tdoann::BatchParallelWorker {
  NbrHeap &heap;
  HeapSortWorker(NbrHeap &heap) : heap(heap) {}

  void operator()(std::size_t begin, std::size_t end) {
    for (auto i = begin; i < end; i++) {
      heap.deheap_sort(i);
    }
  }
};

template <typename NbrHeap = SimpleNeighborHeap,
          typename Parallel = tdoann::NoParallel>
void sort_heap_parallel(NbrHeap &neighbor_heap, std::size_t n_threads,
                        std::size_t block_size, std::size_t grain_size) {
  tdoann::NullProgress progress;
  HeapSortWorker<NbrHeap> sort_worker(neighbor_heap);
  tdoann::batch_parallel_for<decltype(progress), decltype(sort_worker),
                             Parallel>(sort_worker, progress,
                                       neighbor_heap.n_points, n_threads,
                                       block_size, grain_size);
}

#endif // RNN_HEAPSORT_H
