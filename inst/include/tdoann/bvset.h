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

#ifndef TDOANN_BVSET_H
#define TDOANN_BVSET_H

#include "bitvec.h"

namespace tdoann {

inline auto create_set(std::size_t n_points) -> BitVec {
  const std::size_t n_bitsets = bitvec_size(n_points);
  return BitVec(n_bitsets);
}

template <typename T> void mark_visited(BitVec &table, T candidate) {
  auto res = std::ldiv(candidate, BITVEC_BIT_WIDTH);
  table[res.quot].set(res.rem);
}

template <typename T>
auto has_been_and_mark_visited(BitVec &table, T candidate) -> bool {
  auto res = std::ldiv(candidate, BITVEC_BIT_WIDTH);
  auto &chunk = table[res.quot];
  auto is_visited = chunk.test(res.rem);
  chunk.set(res.rem);
  return is_visited;
}

} // namespace tdoann

#endif // TDOANN_BVSET_H
