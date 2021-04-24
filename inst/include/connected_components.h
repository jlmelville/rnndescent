// BSD 2-Clause License
//
// Copyright 2020 James Melville
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

// Translated from the Python source code of:
//   scipy.sparse.csgraph.connected_components
//   Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
//   License: BSD, (C) 2012

#ifndef RNND_CONNECTED_COMPONENTS_H
#define RNND_CONNECTED_COMPONENTS_H

#include <iostream>
#include <utility>

namespace tdoann {

auto check_size(const std::vector<int>& v,
                std::size_t i,
                const std::string &vec_name,
                const std::string &idx_name) -> bool {
  bool size_ok = i < v.size();
  if (!size_ok) {
    std::cout << idx_name << " " << i << " too big for " << vec_name << " " << v.size() << std::endl;
  }
  return size_ok;
}

auto connected_components_undirected(std::size_t N,
                                     const std::vector<int> &indices1,
                                     const std::vector<int> &indptr1,
                                     const std::vector<int> &indices2,
                                     const std::vector<int> &indptr2)
-> std::pair<unsigned int, std::vector<int>> {
  constexpr int VOID = -1;
  constexpr int END = -2;
  std::vector<int> labels(N, VOID);
  std::vector<int> SS(labels);
  unsigned int label = 0;
  auto SS_head = END;
  for (std::size_t v = 0; v < N; ++v) {
    auto vv = v;
    if (!check_size(labels, vv, "labels", "vv")) break;
    if (labels[vv] == VOID) {
      SS_head = vv;
      if (!check_size(SS, vv, "SS", "vv")) break;
      SS[vv] = END;
      while (SS_head != END) {
        vv = SS_head;
        if (!check_size(SS, vv, "SS", "vv")) break;
        SS_head = SS[vv];
        if (!check_size(labels, vv, "labels", "vv")) break;
        labels[vv] = label;
        if (!check_size(indptr1, vv, "indptr1", "vv")) break;
        if (!check_size(indptr1, vv + 1, "indptr1", "vv+1")) break;
        for (auto jj = indptr1[vv]; jj < indptr1[vv + 1]; ++jj) {
          if (!check_size(indices1, jj, "indices1", "jj")) break;
          auto ww = indices1[jj];
          if (!check_size(SS, ww, "SS", "ww")) break;
          if (SS[ww] == VOID) {
            SS[ww] = SS_head;
            SS_head = ww;
          }
        }
        if (!check_size(indptr2, vv, "indptr2", "vv")) break;
        if (!check_size(indptr2, vv + 1, "indptr2", "vv+1")) break;
        for (auto jj = indptr2[vv]; jj < indptr2[vv + 1]; ++jj) {
          if (!check_size(indices2, jj, "indices2", "jj")) break;
          auto ww = indices2[jj];
          if (!check_size(SS, ww, "SS", "ww")) break;
          if (SS[ww] == VOID) {
            SS[ww] = SS_head;
            SS_head = ww;
          }
        }
      }
      ++label;
    }
  }
  return {label, labels};
}
} // namespace tdoann

#endif // RNND_CONNECTED_COMPONENTS_H
