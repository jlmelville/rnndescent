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
#include <memory>
#include <vector>

template <typename In, typename Out> struct Euclidean {
  Euclidean(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim) {}

  Euclidean(const std::vector<In> &x, const std::vector<In> &y,
            std::size_t ndim)
      : x(x), y(y), ndim(ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      const Out diff = x[di + d] - y[dj + d];
      sum += diff * diff;
    }

    return std::sqrt(sum);
  }

  const std::vector<In> &x;
  const std::vector<In> &y;
  const std::size_t ndim;

  typedef In in_type;
};

template <typename In, typename Out> struct L2 {
  L2(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim) {}
  L2(const std::vector<In> &x, const std::vector<In> &y, std::size_t ndim)
      : x(x), y(y), ndim(ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      const Out diff = x[di + d] - y[dj + d];
      sum += diff * diff;
    }

    return sum;
  }

  const std::vector<In> &x;
  const std::vector<In> &y;
  const std::size_t ndim;

  typedef In in_type;
};

template <typename T>
void normalize(const std::vector<T> &vec, std::size_t ndim,
               std::vector<T> &normalized) {
  const std::size_t npoints = vec.size() / ndim;
  for (std::size_t i = 0; i < npoints; i++) {
    const std::size_t di = ndim * i;
    T norm = 0.0;

    for (std::size_t d = 0; d < ndim; d++) {
      const auto val = vec[di + d];
      norm += val * val;
    }
    norm = std::sqrt(norm) + 1e-30;
    for (std::size_t d = 0; d < ndim; d++) {
      normalized[di + d] = vec[di + d] / norm;
    }
  }
}

template <typename In, typename Out> struct CosineN {
  CosineN(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim) {}
  CosineN(const std::vector<In> &x, const std::vector<In> &y, std::size_t ndim)
      : x(x), y(y), ndim(ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    Out sum = 0.0;
    for (std::size_t d = 0; d < ndim; d++) {
      sum += x[di + d] * y[dj + d];
    }

    return 1.0 - sum;
  }

  const std::vector<In> x;
  const std::vector<In> y;
  const std::size_t ndim;

  typedef In in_type;
};

// normalize data on input
template <typename In, typename Out> struct Cosine {
  Cosine(const std::vector<In> &data, std::size_t ndim) : cosine_norm(nullptr) {
    std::vector<In> xn(data.size());
    normalize(data, ndim, xn);
    const auto &yn = xn;
    cosine_norm.reset(new CosineN<In, Out>(xn, yn, ndim));
  }

  Cosine(const std::vector<In> &x, const std::vector<In> &y, std::size_t ndim)
      : cosine_norm(nullptr) {
    std::vector<In> xn(x.size());
    normalize(x, ndim, xn);

    std::vector<In> yn(y.size());
    normalize(y, ndim, yn);
    cosine_norm.reset(new CosineN<In, Out>(xn, yn, ndim));
  }

  Out operator()(std::size_t i, std::size_t j) const {
    return (*cosine_norm)(i, j);
  }

  std::unique_ptr<CosineN<In, Out>> cosine_norm;

  typedef In in_type;
};

template <typename In, typename Out> struct Manhattan {
  Manhattan(const std::vector<In> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim) {}
  Manhattan(const std::vector<In> &x, const std::vector<In> &y,
            std::size_t ndim)
      : x(x), y(y), ndim(ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
    Out sum = 0.0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += std::abs(x[di + d] - y[dj + d]);
    }

    return sum;
  }

  const std::vector<In> &x;
  const std::vector<In> &y;
  const std::size_t ndim;

  typedef In in_type;
};

template <typename T>
void to_bitset(const std::vector<T> &vec, std::size_t ndim,
               std::vector<std::bitset<64>> &bitvec) {
  std::bitset<64> bits;
  std::size_t bit_count = 0;
  std::size_t vd_count = 0;

  for (std::size_t i = 0; i < vec.size(); i++) {
    if (bit_count == 64 || vd_count == ndim) {
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
}

template <typename Out> struct HammingB {
  HammingB(const std::vector<std::bitset<64>> &data, std::size_t ndim)
      : x(data), y(data), ndim(ndim) {}
  HammingB(const std::vector<std::bitset<64>> &x,
           const std::vector<std::bitset<64>> &y, std::size_t ndim)
      : x(x), y(y), ndim(ndim) {}

  Out operator()(std::size_t i, std::size_t j) const {
    Out sum = 0;
    const std::size_t di = ndim * i;
    const std::size_t dj = ndim * j;

    for (std::size_t d = 0; d < ndim; d++) {
      sum += (x[di + d] ^ y[dj + d]).count();
    }

    return sum;
  }

  const std::vector<std::bitset<64>> x;
  const std::vector<std::bitset<64>> y;
  const std::size_t ndim;
};

template <typename In, typename Out> struct Hamming {
  // Instead of storing each bit as an element, we will pack them
  // into a series of 64-bit bitsets. Possibly compilers are smart enough
  // to use built in integer popcount routines for the bitset count()
  // method.
  Hamming(const std::vector<In> &vdata, const std::size_t vndim)
      : hammingb(nullptr) {
    std::vector<std::bitset<64>> x;
    to_bitset(vdata, vndim, x);
    const auto &y = x;
    const auto ndim = std::ceil(vndim / 64.0);
    hammingb.reset(new HammingB<Out>(x, y, ndim));
  }

  Hamming(const std::vector<In> &x, const std::vector<In> &y,
          const std::size_t vndim)
      : hammingb(nullptr) {
    std::vector<std::bitset<64>> bx;
    to_bitset(x, vndim, bx);

    std::vector<std::bitset<64>> by;
    to_bitset(y, vndim, by);

    const auto ndim = std::ceil(vndim / 64.0);
    hammingb.reset(new HammingB<Out>(std::move(bx), std::move(by), ndim));
  }

  Out operator()(std::size_t i, std::size_t j) const {
    return (*hammingb)(i, j);
  }

  std::unique_ptr<HammingB<Out>> hammingb;

  typedef In in_type;
};

#endif // TDOANN_DISTANCE_H
