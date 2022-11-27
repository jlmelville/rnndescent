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

#ifndef TDOANN_NBRQUEUE_H
#define TDOANN_NBRQUEUE_H

#include <queue>
#include <utility>
#include <vector>

namespace tdoann {

// A priority queue that stores neighbors, where a smaller distance gives
// a higher priority
template <typename DistOut, typename Idx> class NbrQueue {
private:
  using Nbr = std::pair<DistOut, Idx>;

  // std::priority_queue is a max heap, so we need to implement the comparison
  // as "greater than" to get the smallest distance first
  struct NbrCompare {
    auto operator()(const Nbr &left, const Nbr &right) -> bool {
      return left.first > right.first;
    }
  };
  std::priority_queue<Nbr, std::vector<Nbr>, NbrCompare> queue;

public:
  NbrQueue() : queue() {}

  auto pop() -> Nbr {
    auto result = queue.top();
    queue.pop();
    return result;
  }

  template <typename... Args> void emplace(Args... args) {
    queue.emplace(args...);
  }
  auto empty() const -> bool { return queue.empty(); }
};

} // namespace tdoann

#endif // TDOANN_NBRQUEUE_H
