// BSD 2-Clause License
//
// Copyright 2019 James Melville
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// OF SUCH DAMAGE.

#ifndef TDOANN_NNGRAPH_H
#define TDOANN_NNGRAPH_H

#include <vector>

#include "typedefs.h"

namespace tdoann {

struct NNGraph {
  std::vector<int> idx;
  std::vector<double> dist;

  std::size_t n_points;
  std::size_t n_nbrs;

  NNGraph(const std::vector<int> &idx, const std::vector<double> &dist,
          std::size_t n_points)
      : idx(idx), dist(dist), n_points(n_points),
        n_nbrs(idx.size() / n_points) {}
};

template <typename NbrHeap = SimpleNeighborHeap>
void heap_to_graph(const NbrHeap &heap, tdoann::NNGraph &nn_graph) {
  for (std::size_t c = 0; c < nn_graph.n_points; c++) {
    std::size_t cnnbrs = c * nn_graph.n_nbrs;
    for (std::size_t r = 0; r < nn_graph.n_nbrs; r++) {
      std::size_t rc = cnnbrs + r;
      nn_graph.idx[rc] = static_cast<int>(heap.idx[rc]);
      nn_graph.dist[rc] = static_cast<double>(heap.dist[rc]);
    }
  }
}

} // namespace tdoann

#endif // TDOANN_NNGRAPH_H
