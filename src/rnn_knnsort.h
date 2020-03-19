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

#ifndef RNN_KNNSORT_H
#define RNN_KNNSORT_H

#include "tdoann/heap.h"
#include "tdoann/nngraph.h"
#include "tdoann/parallel.h"
#include "tdoann/progress.h"

#include "rnn_heapsort.h"
#include "rnn_vectoheap.h"

template <typename HeapAdd, typename Progress = tdoann::NullProgress,
          typename NbrHeap = SimpleNeighborHeap,
          typename Parallel = tdoann::NoParallel>
void sort_knn_graph_parallel(tdoann::NNGraph &nn_graph, std::size_t n_threads,
                             std::size_t block_size, std::size_t grain_size) {
  NbrHeap heap(nn_graph.n_points, nn_graph.n_nbrs);
  graph_to_heap_parallel<HeapAdd, Progress>(heap, nn_graph, n_threads,
                                            block_size, grain_size);
  sort_heap_parallel(heap, n_threads, block_size, grain_size);

  tdoann::heap_to_graph(heap, nn_graph);
}

template <typename HeapAdd, typename Progress = tdoann::NullProgress,
          typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph(tdoann::NNGraph &nn_graph) {
  NbrHeap heap(nn_graph.n_points, nn_graph.n_nbrs);
  graph_to_heap_serial<HeapAdd, Progress>(heap, nn_graph, 1000);

  heap.deheap_sort();

  tdoann::heap_to_graph(heap, nn_graph);
}

#endif // RNN_KNNSORT_H
