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

#ifndef TDOANN_DISTANCEBASE_H
#define TDOANN_DISTANCEBASE_H

#include <cstdint>
#include <memory>
#include <vector>

#include "bitvec.h"
#include "distance.h"

// Pointer-based polymorphic classes for calculating distances. This is all
// boilerplate: the actual calculations take place in functions defined in
// distance.h

namespace tdoann {

template <typename Out, typename Idx = uint32_t> class BaseDistance {
public:
  using Output = Out;
  using Index = Idx;
  virtual ~BaseDistance() = default;
  virtual Out calculate(Idx i, Idx j) const = 0;
  virtual std::size_t get_nx() const = 0;
  virtual std::size_t get_ny() const = 0;
};

template <typename T> struct DistanceTraits;

template <typename Out, typename Idx>
struct DistanceTraits<std::unique_ptr<tdoann::BaseDistance<Out, Idx>>> {
  using Output = Out;
  using Index = Idx;
};

// Mixin for shared Self data
// The QueryDataMixin with xdata(data), ydata(data) is entirely sufficient
// if you don't need to preprocess the data during creation
template <typename In> class SelfDataMixin {
public:
  explicit SelfDataMixin(const std::vector<In> &data, std::size_t ndim)
      : x(data), nx(data.size() / ndim), ndim(ndim) {}

  std::size_t get_nx() const { return nx; }

  // For self-distance, nx == ny
  std::size_t get_ny() const { return nx; }

protected:
  const std::vector<In> x;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In> class QueryDataMixin {
public:
  explicit QueryDataMixin(const std::vector<In> &xdata,
                          const std::vector<In> &ydata, std::size_t ndim)
      : x(xdata), y(ydata), nx(xdata.size() / ndim), ny(ydata.size() / ndim),
        ndim(ndim) {}

  std::size_t get_nx() const { return nx; }

  std::size_t get_ny() const { return ny; }

protected:
  const std::vector<In> x;
  const std::vector<In> y;
  std::size_t nx;
  std::size_t ny;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class L2SqrDistance : public BaseDistance<Out, Idx>, public QueryDataMixin<In> {
public:
  using QueryDataMixin<In>::QueryDataMixin;

  L2SqrDistance(const std::vector<In> &data, std::size_t ndim)
      : QueryDataMixin<In>(data, data, ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return l2sqr<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                      this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class EuclideanDistance : public BaseDistance<Out, Idx>,
                          public QueryDataMixin<In> {
public:
  using QueryDataMixin<In>::QueryDataMixin;

  EuclideanDistance(const std::vector<In> &data, std::size_t ndim)
      : QueryDataMixin<In>(data, data, ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return euclidean<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx>
class ManhattanDistance : public BaseDistance<Out, Idx>,
                          public QueryDataMixin<In> {
public:
  using QueryDataMixin<In>::QueryDataMixin;
  ManhattanDistance(const std::vector<In> &data, std::size_t ndim)
      : QueryDataMixin<In>(data, data, ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::manhattan<Out>(this->x.begin() + di,
                                  this->x.begin() + di + this->ndim,
                                  this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CosineSelfDistance : public BaseDistance<Out, Idx>,
                           public SelfDataMixin<In> {
public:
  CosineSelfDistance(const std::vector<In> &data, std::size_t ndim)
      : SelfDataMixin<In>(normalize(data, ndim), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::inner_product<Out>(this->x.begin() + di,
                                      this->x.begin() + di + this->ndim,
                                      this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CosineQueryDistance : public BaseDistance<Out, Idx>,
                            public QueryDataMixin<In> {
public:
  CosineQueryDistance(const std::vector<In> &xdata,
                      const std::vector<In> &ydata, std::size_t ndim)
      : QueryDataMixin<In>(normalize(xdata, ndim), normalize(ydata, ndim),
                           ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::inner_product<Out>(this->x.begin() + di,
                                      this->x.begin() + di + this->ndim,
                                      this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationSelfDistance : public BaseDistance<Out, Idx>,
                                public SelfDataMixin<In> {
public:
  CorrelationSelfDistance(const std::vector<In> &data, std::size_t ndim)
      : SelfDataMixin<In>(normalize_center(data, ndim), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::inner_product<Out>(this->x.begin() + di,
                                      this->x.begin() + di + this->ndim,
                                      this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationQueryDistance : public BaseDistance<Out, Idx>,
                                 public QueryDataMixin<In> {
public:
  CorrelationQueryDistance(const std::vector<In> &xdata,
                           const std::vector<In> &ydata, std::size_t ndim)
      : QueryDataMixin<In>(normalize_center(xdata, ndim),
                           normalize_center(ydata, ndim), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::inner_product<Out>(this->x.begin() + di,
                                      this->x.begin() + di + this->ndim,
                                      this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class HammingDistance : public BaseDistance<Out, Idx>,
                        public QueryDataMixin<In> {
public:
  using QueryDataMixin<In>::QueryDataMixin;
  HammingDistance(const std::vector<In> &data, std::size_t ndim)
      : QueryDataMixin<In>(data, data, ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::hamming<Out>(this->x.begin() + di,
                                this->x.begin() + di + this->ndim,
                                this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename Out, typename Idx = uint32_t>
class BHammingSelfDistance : public BaseDistance<Out, Idx> {
public:
  BHammingSelfDistance(const std::vector<uint8_t> &data, std::size_t ndim)
      : bitvec(to_bitvec(data, ndim)), vec_len(bitvec_size(ndim)), ndim(ndim),
        nx(data.size() / ndim), ny(nx) {}

  Out calculate(Idx i, Idx j) const override {
    return bhamming_impl<Out>(bitvec, i, bitvec, j, vec_len);
  }
  std::size_t get_nx() const override { return nx; }

  std::size_t get_ny() const override { return nx; }

private:
  const BitVec bitvec;
  std::size_t vec_len;
  std::size_t ndim;
  std::size_t nx;
  std::size_t ny;
};

template <typename Out, typename Idx = uint32_t>
class BHammingQueryDistance : public BaseDistance<Out, Idx> {
public:
  BHammingQueryDistance(const std::vector<uint8_t> &x,
                        const std::vector<uint8_t> &y, std::size_t ndim)
      : bx(to_bitvec(x, ndim)), by(to_bitvec(y, ndim)),
        vec_len(bitvec_size(ndim)), ndim(ndim), nx(x.size() / ndim),
        ny(y.size() / ndim) {}

  Out calculate(Idx i, Idx j) const override {
    return bhamming_impl<Out>(bx, i, by, j, vec_len);
  }

  std::size_t get_nx() const override { return nx; }

  std::size_t get_ny() const override { return ny; }

private:
  const BitVec bx;
  const BitVec by;
  std::size_t vec_len;
  std::size_t ndim;
  std::size_t nx;
  std::size_t ny;
};
} // namespace tdoann

#endif // TDOANN_DISTANCEBASE_H
