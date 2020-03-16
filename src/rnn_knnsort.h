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
#include "rnn_vectoheap.h"

template <typename NbrHeap = SimpleNeighborHeap>
void heap_to_vec(const NbrHeap &heap, std::vector<int> &nn_idx,
                 std::size_t n_points, std::vector<double> &nn_dist) {
  std::size_t n_nbrs = nn_idx.size() / n_points;

  for (std::size_t c = 0; c < n_points; c++) {
    std::size_t cnnbrs = c * n_nbrs;
    for (std::size_t r = 0; r < n_nbrs; r++) {
      std::size_t rc = cnnbrs + r;
      nn_idx[rc] = static_cast<int>(heap.idx[rc]) + 1;
      nn_dist[rc] = static_cast<double>(heap.dist[rc]);
    }
  }
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph_parallel(std::vector<int> &nn_idx, std::size_t n_points,
                             std::vector<double> &nn_dist,
                             std::size_t n_threads, std::size_t block_size,
                             std::size_t grain_size,
                             int max_idx = (std::numeric_limits<int>::max)()) {
  std::size_t n_nbrs = nn_idx.size() / n_points;

  NbrHeap heap(n_points, n_nbrs);
  vec_to_heap_parallelt<HeapAdd>(heap, nn_idx, n_points, nn_dist, n_threads,
                                 block_size, grain_size, max_idx);
  sort_heap_parallel(heap, n_threads, block_size, grain_size);

  heap_to_vec(heap, nn_idx, n_points, nn_dist);
}

template <typename HeapAdd, typename NbrHeap = SimpleNeighborHeap>
void sort_knn_graph(std::vector<int> &nn_idx, std::size_t n_points,
                    std::vector<double> &nn_dist) {
  std::size_t n_nbrs = nn_idx.size() / n_points;

  NbrHeap heap(n_points, n_nbrs);
  vec_to_heap_serialt<HeapAdd>(heap, nn_idx, n_points, nn_dist, 1000);

  heap.deheap_sort();

  heap_to_vec(heap, nn_idx, n_points, nn_dist);
}

#endif // RNN_KNNSORT_H
