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

#ifndef TDOANN_DISTANCE_H
#define TDOANN_DISTANCE_H

#include <vector>

#include "bitvec.h"

namespace tdoann {
template <typename In, typename Out, typename Idx = uint32_t> struct Euclidean {
  Euclidean(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim), nx(data.size() / ndim),
        ny(data.size() / ndim) {}

  Euclidean(const std::vector<In> &x, const std::vector<In> &y,
            std::size_t ndim)
      : x(x), y(y), ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  auto operator()(Idx i, Idx j) const -> Out {
    Out sum = 0.0;
    std::size_t di = ndim * i;
    std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      Out diff = x[di + d] - y[dj + d];
      sum += diff * diff;
    }

    return std::sqrt(sum);
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

template <typename In, typename Out, typename Idx = uint32_t> struct L2Sqr {
  L2Sqr(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim), nx(data.size() / ndim),
        ny(data.size() / ndim) {}
  L2Sqr(const std::vector<In> &x, const std::vector<In> &y, std::size_t ndim)
      : x(x), y(y), ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  auto operator()(Idx i, Idx j) const -> Out {
    Out sum = 0.0;
    std::size_t di = ndim * i;
    std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      Out diff = x[di + d] - y[dj + d];
      sum += diff * diff;
    }

    return sum;
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

// relies on NRVO to avoid a copy
template <typename T>
auto normalize(const std::vector<T> &vec, std::size_t ndim) -> std::vector<T> {
  std::vector<T> normalized(vec.size());
  std::size_t npoints = vec.size() / ndim;
  for (std::size_t i = 0; i < npoints; i++) {
    std::size_t di = ndim * i;
    T norm = 0.0;

    for (std::size_t d = 0; d < ndim; d++) {
      auto val = vec[di + d];
      norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-30;
    for (std::size_t d = 0; d < ndim; d++) {
      normalized[di + d] = vec[di + d] / norm;
    }
  }
  return normalized;
}

template <typename In, typename Out, typename Idx = uint32_t>
auto cosine_impl(const std::vector<In> &x, Idx i, const std::vector<In> &y,
                 Idx j, std::size_t ndim) -> Out {
  std::size_t di = ndim * i;
  std::size_t dj = ndim * j;

  Out sum = 0.0;
  for (std::size_t d = 0; d < ndim; d++) {
    sum += x[di + d] * y[dj + d];
  }

  return 1.0 - sum;
}

template <typename In, typename Out, typename Idx = uint32_t>
struct CosineSelf {
  const std::vector<In> x;
  std::size_t ndim;
  Idx nx;
  Idx ny;

  CosineSelf(const std::vector<In> &data, std::size_t ndim)
      : x(normalize(data, ndim)), ndim(ndim), nx(data.size() / ndim), ny(nx) {}

  auto operator()(Idx i, Idx j) const -> Out {
    return cosine_impl<In, Out, Idx>(x, i, x, j, ndim);
  }

  using Input = In;
  using Output = Out;
  using Index = Idx;
};

template <typename In, typename Out, typename Idx = uint32_t>
struct CosineQuery {
  const std::vector<In> x_;
  const std::vector<In> y_;
  std::size_t ndim;
  Idx nx;
  Idx ny;

  CosineQuery(const std::vector<In> &x, const std::vector<In> &y,
              std::size_t ndim)
      : x_(normalize(x, ndim)), y_(normalize(y, ndim)), ndim(ndim),
        nx(x.size() / ndim), ny(y.size() / ndim) {}

  auto operator()(Idx i, Idx j) const -> Out {
    return cosine_impl<In, Out, Idx>(x_, i, y_, j, ndim);
  }

  using Input = In;
  using Output = Out;
  using Index = Idx;
};

template <typename In, typename Out, typename Idx = uint32_t> struct Manhattan {
  Manhattan(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim), nx(data.size() / ndim),
        ny(data.size() / ndim) {}
  Manhattan(const std::vector<In> &x, const std::vector<In> &y,
            std::size_t ndim)
      : x(x), y(y), ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  auto operator()(Idx i, Idx j) const -> Out {
    Out sum = 0.0;
    std::size_t di = ndim * i;
    std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += std::abs(x[di + d] - y[dj + d]);
    }

    return sum;
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

template <typename Out, typename Idx = uint32_t>
auto bhamming_impl(const BitVec &x, Idx i, const BitVec &y, Idx j,
				   std::size_t len) -> Out {
  Out sum = 0;
  std::size_t di = len * i;
  std::size_t dj = len * j;

  for (std::size_t d = 0; d < len; d++) {
    sum += (x[di + d] ^ y[dj + d]).count();
  }

  return sum;
}

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

  BHammingQuery(const std::vector<In> &x, const std::vector<In> &y, std::size_t ndim)
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

template <typename In, typename Out, typename Idx = uint32_t>
struct Hamming {

  Hamming(const std::vector<In> &data, std::size_t ndim)
	: x(data), y(data), ndim(ndim), nx(data.size() / ndim),
	  ny(data.size() / ndim) {}

  Hamming(const std::vector<In> &x, const std::vector<In> &y, std::size_t ndim)
	: x(x), y(y), ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  auto operator()(Idx i, Idx j) const -> Out {
    Out sum = 0.0;
    std::size_t di = ndim * i;
    std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += (x[di + d] != y[dj + d]);
    }

    return sum;
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

} // namespace tdoann
#endif // TDOANN_DISTANCE_H
