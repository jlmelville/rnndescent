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

#include "rnn_heapsort.h"
#include "rnn_nngraph.h"
#include "rnn_vectoheap.h"

template <typename NbrHeap = SimpleNeighborHeap>
void heap_to_graph(const NbrHeap &heap, NNGraph &nn_graph) {
  for (std::size_t c = 0; c < nn_graph.n_points; c++) {
    std::size_t cnnbrs = c * nn_graph.n_nbrs;
    for (std::size_t r = 0; r < nn_graph.n_nbrs; r++) {
      std::size_t rc = cnnbrs + r;
      nn_graph.idx[rc] = static_cast<int>(heap.idx[rc]) + 1;
      nn_graph.dist[rc] = static_cast<double>(heap.dist[rc]);
    }
  }
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph_parallel(NNGraph &nn_graph, std::size_t n_threads,
                             std::size_t block_size, std::size_t grain_size,
                             int max_idx = (std::numeric_limits<int>::max)()) {
  NbrHeap heap(nn_graph.n_points, nn_graph.n_nbrs);
  graph_to_heap_parallel<HeapAdd>(heap, nn_graph, n_threads, block_size,
                                  grain_size, max_idx);
  sort_heap_parallel(heap, n_threads, block_size, grain_size);

  heap_to_graph(heap, nn_graph);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph(NNGraph &nn_graph) {
  NbrHeap heap(nn_graph.n_points, nn_graph.n_nbrs);
  graph_to_heap_serial<HeapAdd>(heap, nn_graph, 1000);

  heap.deheap_sort();

  heap_to_graph(heap, nn_graph);
}

#endif // RNN_KNNSORT_H
