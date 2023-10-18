// BSD 2-Clause License
//
// Copyright 2023 James Melville
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

#ifndef TDOANN_DISTANCETEMPLATE_H
#define TDOANN_DISTANCETEMPLATE_H

#include <cstdint>
#include <vector>

#include "bitvec.h"
#include "distance.h"

namespace tdoann {

// template for distance functors

template <typename T>
auto do_nothing(const std::vector<T> &vec, std::size_t /* ndim */)
    -> std::vector<T> {
  return vec;
}

template <typename In, typename Out, typename It, Out (*dfun)(It, It, It),
          std::vector<In> (*initfun)(const std::vector<In> &vec,
                                     std::size_t ndim) = do_nothing,
          typename Idx = uint32_t>
struct SelfDistance {
  SelfDistance(const std::vector<In> &data, std::size_t ndim)
      : x(initfun(data, ndim)), ndim(ndim), nx(data.size() / ndim), ny(nx) {}

  inline auto operator()(Idx i, Idx j) const -> Out {
    const std::size_t di = ndim * i;
    return dfun(x.begin() + di, x.begin() + di + ndim, x.begin() + ndim * j);
  }

  const std::vector<In> x;
  std::size_t ndim;
  Idx nx;
  Idx ny;

  using Input = In;
  using Output = Out;
  using Index = Idx;
};

template <typename In, typename Out, typename It, Out (*dfun)(It, It, It),
          std::vector<In> (*initfun)(const std::vector<In> &vec,
                                     std::size_t ndim) = do_nothing,
          typename Idx = uint32_t>
struct QueryDistance {
  QueryDistance(const std::vector<In> &x, const std::vector<In> &y,
                std::size_t ndim)
      : x(initfun(x, ndim)), y(initfun(y, ndim)), ndim(ndim),
        nx(x.size() / ndim), ny(y.size() / ndim) {}

  inline auto operator()(Idx i, Idx j) const -> Out {
    const std::size_t di = ndim * i;
    return dfun(x.begin() + di, x.begin() + di + ndim, y.begin() + ndim * j);
  }

  const std::vector<In> x;
  const std::vector<In> y;
  std::size_t ndim;
  Idx nx;
  Idx ny;

  using Input = In;
  using Output = Out;
  using Index = Idx;
};

template <typename In, typename Out, typename Idx = uint32_t>
struct BHammingSelf {
  const BitVec bitvec;
  std::size_t vec_len; // size of the bitvec
  std::size_t ndim;
  Idx nx;
  Idx ny;

  BHammingSelf(const std::vector<In> &data, std::size_t ndim)
      : bitvec(to_bitvec(data, ndim)), vec_len(bitvec_size(ndim)), ndim(ndim),
        nx(data.size() / ndim), ny(nx) {}

  auto operator()(Idx i, Idx j) const -> Out {
    return bhamming_impl<Out>(bitvec, i, bitvec, j, vec_len);
  }

  using Input = In;
  using Output = Out;
  using Index = Idx;
};

template <typename In, typename Out, typename Idx = uint32_t>
struct BHammingQuery {
  const BitVec bx;
  const BitVec by;
  std::size_t vec_len;
  std::size_t ndim;
  Idx nx;
  Idx ny;

  BHammingQuery(const std::vector<In> &x, const std::vector<In> &y,
                std::size_t ndim)
      : bx(to_bitvec(x, ndim)), by(to_bitvec(y, ndim)),
        vec_len(bitvec_size(ndim)), ndim(ndim), nx(x.size() / ndim),
        ny(y.size() / ndim) {}

  auto operator()(Idx i, Idx j) const -> Out {
    return bhamming_impl<Out>(bx, i, by, j, vec_len);
  }

  using Input = In;
  using Output = Out;
  using Index = Idx;
};
} // namespace tdoann

#endif // TDOANN_DISTANCETEMPLATE_H
