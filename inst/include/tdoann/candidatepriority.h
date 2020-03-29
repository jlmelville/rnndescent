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

#ifndef TDOANN_CANDIDATEPRIORITY_H
#define TDOANN_CANDIDATEPRIORITY_H

#include "heap.h"
#include "typedefs.h"

namespace tdoann {

// Explore neighbors in increasing distance order (i.e. closest neighbors are
// processed first)
struct CandidatePriorityLowDistance {
  auto operator()(const NeighborHeap &current_graph, std::size_t ij) -> double {
    return current_graph.dist[ij];
  }
  const constexpr static bool should_sort = true;
};

// Explore neighbors in decreases distance order (i.e. furthest neighbors are
// processed first)
struct CandidatePriorityHighDistance {
  auto operator()(const NeighborHeap &current_graph, std::size_t ij) -> double {
    return -current_graph.dist[ij];
  }
  const constexpr static bool should_sort = true;
};

template <typename CandidatePriority> struct CandidatePriorityFactory {
  using Type = CandidatePriority;
  auto create() -> Type { return Type(); }
  // Window size in parallel computations.
  auto create(std::size_t, std::size_t) -> Type { return Type(); }
  const constexpr static bool should_sort = Type::should_sort;
};

} // namespace tdoann
#endif // TDOANN_CANDIDATEPRIORITY_H
