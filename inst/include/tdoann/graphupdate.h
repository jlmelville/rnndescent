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

#ifndef TDOANN_GRAPHUPDATE_H
#define TDOANN_GRAPHUPDATE_H

#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "heap.h"

namespace tdoann {

template <typename Idx> struct GraphCache {
private:
  std::vector<std::unordered_set<Idx>> seen;

public:
  GraphCache(std::size_t n_points, std::size_t n_nbrs,
             const std::vector<Idx> &idx_data)
      : seen(n_points) {
    for (Idx i = 0, innbrs = 0; i < n_points; i++, innbrs += n_nbrs) {
      for (std::size_t j = 0, idx_ij = innbrs; j < n_nbrs; j++, idx_ij++) {
        auto idx_p = idx_data[idx_ij];
        if (i > idx_p) {
          seen[idx_p].emplace(i);
        } else {
          seen[i].emplace(idx_p);
        }
      }
    }
  }

  // Static factory function
  template <typename DistOut>
  static GraphCache<Idx> from_heap(const NNDHeap<DistOut, Idx> &heap) {
    return GraphCache<Idx>(heap.n_points, heap.n_nbrs, heap.idx);
  }

  auto contains(const Idx &idx_p, const Idx &idx_q) const -> bool {
    return seen[idx_p].find(idx_q) != seen[idx_p].end();
  }

  auto insert(Idx idx_p, Idx idx_q) -> bool {
    return !seen[idx_p].emplace(idx_q).second;
  }
};

} // namespace tdoann

#endif // TDOANN_GRAPHUPDATE_H
