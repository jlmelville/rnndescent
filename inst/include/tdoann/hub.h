// BSD 2-Clause License
//
// Copyright 2021 James Melville
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

#ifndef TDOANN_HUB_H
#define TDOANN_HUB_H

#include <algorithm>
#include <vector>

namespace tdoann {

template <typename T>
auto reverse_nbr_counts_impl(const std::vector<T> &forward_nbrs,
                             std::size_t n_points) -> std::vector<std::size_t> {
  const std::size_t n_nbrs = forward_nbrs.size() / n_points;

  std::vector<std::size_t> counts(n_points);
  std::size_t innbrs;
  std::size_t inbr;

  for (std::size_t i = 0; i < n_points; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      inbr = forward_nbrs[innbrs + j];
      if (inbr == i) {
        continue;
      }
      ++counts[inbr];
    }
  }
  return counts;
}

// treat the knn graph as directed where each point is the "head" of the
// directed edge, and each neighbor is the "tail". The head and tail nodes may
// be entirely disjoint, e.g. coming from query and reference nodes,
// respectively.
template <typename T>
auto reverse_nbr_counts_impl(const std::vector<T> &forward_nbrs,
                             std::size_t n_head_points,
                             std::size_t n_tail_points)
    -> std::vector<std::size_t> {
  const std::size_t n_nbrs = forward_nbrs.size() / n_head_points;

  std::vector<std::size_t> counts(n_tail_points);
  std::size_t innbrs;

  for (std::size_t i = 0; i < n_head_points; i++) {
    innbrs = i * n_nbrs;
    for (std::size_t j = 0; j < n_nbrs; j++) {
      ++counts[forward_nbrs[innbrs + j]];
    }
  }
  return counts;
}

template <typename T>
auto reverse_nbr_counts(const std::vector<T> &forward_nbrs,
                        std::size_t n_points, bool include_self)
    -> std::vector<std::size_t> {

  // include_self: either the index into the nbrs and the nbrs are from two
  // disjoint set of nodes or it's the same nodes and we don't mind counting
  // loops
  if (include_self) {
    auto n_tail_points =
        *std::max_element(forward_nbrs.begin(), forward_nbrs.end()) + 1;
    return reverse_nbr_counts_impl(forward_nbrs, n_points, n_tail_points);
  } else {
    return reverse_nbr_counts_impl(forward_nbrs, n_points);
  }
}
} // namespace tdoann
#endif // TDOANN_HUB_H
