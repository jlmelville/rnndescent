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

#include <bitset>
#include <cmath>
#include <vector>

namespace tdoann {
template <typename In, typename Out> struct Euclidean {
  Euclidean(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim), nx(data.size() / ndim),
        ny(data.size() / ndim) {}

  Euclidean(const std::vector<In> &x, const std::vector<In> &y,
            std::size_t ndim)
      : x(x), y(y), ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
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
  std::size_t nx;
  std::size_t ny;

  using Input = In;
};

template <typename In, typename Out> struct L2Sqr {
  L2Sqr(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim), nx(data.size() / ndim),
        ny(data.size() / ndim) {}
  L2Sqr(const std::vector<In> &x, const std::vector<In> &y, std::size_t ndim)
      : x(x), y(y), ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
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
  std::size_t nx;
  std::size_t ny;

  using Input = In;
};

// relies on NRVO to avoid a copy
template <typename T>
std::vector<T> normalize(const std::vector<T> &vec, std::size_t ndim) {
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

template <typename In, typename Out>
Out cosine_impl(const std::vector<In> &x, std::size_t i,
                const std::vector<In> &y, std::size_t j, std::size_t ndim) {
  std::size_t di = ndim * i;
  std::size_t dj = ndim * j;

  Out sum = 0.0;
  for (std::size_t d = 0; d < ndim; d++) {
    sum += x[di + d] * y[dj + d];
  }

  return 1.0 - sum;
}

template <typename In, typename Out> struct CosineSelf {
  const std::vector<In> x;
  std::size_t ndim;
  std::size_t nx;
  std::size_t ny;

  CosineSelf(const std::vector<In> &data, std::size_t ndim)
      : x(normalize(data, ndim)), ndim(ndim), nx(data.size() / ndim), ny(nx) {}

  Out operator()(std::size_t i, std::size_t j) const {
    return cosine_impl<In, Out>(x, i, x, j, ndim);
  }

  using Input = In;
};

template <typename In, typename Out> struct CosineQuery {
  const std::vector<In> x_;
  const std::vector<In> y_;
  std::size_t ndim;
  std::size_t nx;
  std::size_t ny;

  CosineQuery(const std::vector<In> &x, const std::vector<In> &y,
              std::size_t ndim)
      : x_(normalize(x, ndim)), y_(normalize(y, ndim)), ndim(ndim),
        nx(x.size() / ndim), ny(y.size() / ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
    return cosine_impl<In, Out>(x_, i, y_, j, ndim);
  }

  using Input = In;
};

template <typename In, typename Out> struct Manhattan {
  Manhattan(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim), nx(data.size() / ndim),
        ny(data.size() / ndim) {}
  Manhattan(const std::vector<In> &x, const std::vector<In> &y,
            std::size_t ndim)
      : x(x), y(y), ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
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
  std::size_t nx;
  std::size_t ny;

  using Input = In;
};

template <int n> using BitSet = std::bitset<n>;
using BitVec = std::vector<BitSet<64>>;

// Instead of storing each bit as an element, we will pack them
// into a series of 64-bit bitsets. Possibly compilers are smart enough
// to use built in integer popcount routines for the bitset count()
// method. Relies on NRVO to avoid copying return value
template <typename T>
BitVec to_bitvec(const std::vector<T> &vec, std::size_t ndim) {
  BitSet<64> bits;
  std::size_t bit_count = 0;
  std::size_t vd_count = 0;

  BitVec bitvec;

  for (std::size_t i = 0; i < vec.size(); i++) {
    if (bit_count == bits.size() || vd_count == ndim) {
      // filled up current bitset
      bitvec.push_back(bits);
      bit_count = 0;
      bits.reset();

      if (vd_count == ndim) {
        // end of item
        vd_count = 0;
      }
    }
    bits[bit_count] = vec[i];

    ++vd_count;
    ++bit_count;
  }
  if (bit_count > 0) {
    bitvec.push_back(bits);
  }

  return bitvec;
}

template <typename Out>
Out hamming_impl(const BitVec &x, std::size_t i, const BitVec &y, std::size_t j,
                 std::size_t ndim) {
  Out sum = 0;
  std::size_t di = ndim * i;
  std::size_t dj = ndim * j;

  for (std::size_t d = 0; d < ndim; d++) {
    sum += (x[di + d] ^ y[dj + d]).count();
  }

  return sum;
}

template <typename In, typename Out> struct HammingSelf {
  const BitVec bitvec;
  std::size_t ndim_;
  std::size_t ndim;
  std::size_t nx;
  std::size_t ny;

  HammingSelf(const std::vector<In> &data, std::size_t ndim)
      : bitvec(to_bitvec(data, ndim)),
        ndim_(
            std::ceil(ndim / static_cast<float>(BitVec::value_type{}.size()))),
        ndim(ndim), nx(data.size() / ndim), ny(nx) {}

  Out operator()(std::size_t i, std::size_t j) const {
    return hamming_impl<Out>(bitvec, i, bitvec, j, ndim_);
  }

  using Input = In;
};

template <typename In, typename Out> struct HammingQuery {
  const BitVec bx;
  const BitVec by;
  std::size_t ndim_;
  std::size_t ndim;
  std::size_t nx;
  std::size_t ny;

  HammingQuery(const std::vector<In> &x, const std::vector<In> &y,
               std::size_t ndim)
      : bx(to_bitvec(x, ndim)), by(to_bitvec(y, ndim)),
        ndim_(
            std::ceil(ndim / static_cast<float>(BitVec::value_type{}.size()))),
        ndim(ndim), nx(x.size() / ndim), ny(y.size() / ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
    return hamming_impl<Out>(bx, i, by, j, ndim_);
  }

  using Input = In;
};

} // namespace tdoann
#endif // TDOANN_DISTANCE_H
