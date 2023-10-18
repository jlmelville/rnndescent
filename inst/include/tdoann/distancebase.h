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

template <typename In> class SelfDataMixin {
public:
  template <typename VecIn>
  SelfDataMixin(VecIn &&data, std::size_t ndim)
      : x(std::forward<VecIn>(data)), nx(x.size() / ndim), ndim(ndim) {}

  std::size_t get_nx() const { return nx; }

  // For self-distance, nx == ny
  std::size_t get_ny() const { return nx; }

protected:
  mutable std::vector<In> x;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In> class QueryDataMixin {
public:
  template <typename VecIn>
  QueryDataMixin(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : x(std::forward<VecIn>(xdata)), y(std::forward<VecIn>(ydata)),
        nx(x.size() / ndim), ny(y.size() / ndim), ndim(ndim) {}

  std::size_t get_nx() const { return nx; }

  std::size_t get_ny() const { return ny; }

protected:
  // these are not const because for data which needs to be transformed
  // (e.g. preprocessing versions of cosine and correlation) we need to do the
  // operations in-place to avoid unnecessary allocations, which in turn
  // requires work to be done in the constructor body on the x, y fields.
  mutable std::vector<In> x;
  mutable std::vector<In> y;
  std::size_t nx;
  std::size_t ny;
  std::size_t ndim;
};

template <typename In, typename Out, typename Idx = uint32_t>
class L2SqrSelfDistance : public BaseDistance<Out, Idx>,
                          public SelfDataMixin<In> {
public:
  template <typename VecIn>
  L2SqrSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return l2sqr<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                      this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class L2SqrQueryDistance : public BaseDistance<Out, Idx>,
                           public QueryDataMixin<In> {
public:
  template <typename VecIn>
  L2SqrQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return l2sqr<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                      this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class EuclideanSelfDistance : public BaseDistance<Out, Idx>,
                              public SelfDataMixin<In> {
public:
  template <typename VecIn>
  EuclideanSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return euclidean<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class EuclideanQueryDistance : public BaseDistance<Out, Idx>,
                               public QueryDataMixin<In> {
public:
  template <typename VecIn>
  EuclideanQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return euclidean<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class ManhattanSelfDistance : public BaseDistance<Out, Idx>,
                              public SelfDataMixin<In> {
public:
  template <typename VecIn>
  ManhattanSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return manhattan<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class ManhattanQueryDistance : public BaseDistance<Out, Idx>,
                               public QueryDataMixin<In> {
public:
  template <typename VecIn>
  ManhattanQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return manhattan<Out>(this->x.begin() + di,
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
  template <typename VecIn>
  CosineSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return cosine<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                       this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CosineQueryDistance : public BaseDistance<Out, Idx>,
                            public QueryDataMixin<In> {
public:
  template <typename VecIn>
  CosineQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return cosine<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                       this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CosinePreprocessSelfDistance : public BaseDistance<Out, Idx>,
                                     public SelfDataMixin<In> {
public:
  template <typename VecIn>
  CosinePreprocessSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);
  }

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::inner_product<Out>(this->x.begin() + di,
                                      this->x.begin() + di + this->ndim,
                                      this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CosinePreprocessQueryDistance : public BaseDistance<Out, Idx>,
                                      public QueryDataMixin<In> {
public:
  template <typename VecIn>
  CosinePreprocessQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->y), this->ndim);
  }

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
  template <typename VecIn>
  CorrelationSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return correlation<Out>(this->x.begin() + di,
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
  template <typename VecIn>
  CorrelationQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return correlation<Out>(this->x.begin() + di,
                            this->x.begin() + di + this->ndim,
                            this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationPreprocessSelfDistance : public BaseDistance<Out, Idx>,
                                          public SelfDataMixin<In> {
public:
  template <typename VecIn>
  CorrelationPreprocessSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    mean_center(const_cast<std::vector<In> &>(this->x), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);
  }

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return tdoann::inner_product<Out>(this->x.begin() + di,
                                      this->x.begin() + di + this->ndim,
                                      this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationPreprocessQueryDistance : public BaseDistance<Out, Idx>,
                                           public QueryDataMixin<In> {
public:
  template <typename VecIn>
  CorrelationPreprocessQueryDistance(VecIn &&xdata, VecIn &&ydata,
                                     std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    mean_center(const_cast<std::vector<In> &>(this->x), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);

    mean_center(const_cast<std::vector<In> &>(this->y), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->y), this->ndim);
  }

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
class HammingSelfDistance : public BaseDistance<Out, Idx>,
                            public SelfDataMixin<In> {
public:
  template <typename VecIn>
  HammingSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In>(std::forward<VecIn>(data), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return hamming<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                        this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return SelfDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return SelfDataMixin<In>::get_ny(); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class HammingQueryDistance : public BaseDistance<Out, Idx>,
                             public QueryDataMixin<In> {
public:
  template <typename VecIn>
  HammingQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In>(std::forward<VecIn>(xdata),
                           std::forward<VecIn>(ydata), ndim) {}

  Out calculate(Idx i, Idx j) const override {
    const std::size_t di = this->ndim * i;
    return hamming<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                        this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return QueryDataMixin<In>::get_nx(); }

  std::size_t get_ny() const override { return QueryDataMixin<In>::get_ny(); }
};

template <typename Out, typename Idx = uint32_t>
class BHammingSelfDistance : public BaseDistance<Out, Idx> {
public:
  template <typename VecIn>
  BHammingSelfDistance(VecIn &&data, std::size_t ndim)
      : vec_len(num_blocks_needed(ndim)), nx(data.size() / ndim), ny(nx),
        bdata(to_bitvec(std::forward<VecIn>(data), ndim)) {}

  Out calculate(Idx i, Idx j) const override {
    return bhamming_impl<Out>(bdata, i, bdata, j, vec_len);
  }
  std::size_t get_nx() const override { return nx; }

  std::size_t get_ny() const override { return nx; }

private:
  std::size_t vec_len;
  std::size_t nx;
  std::size_t ny;
  const BitVec bdata;
};

template <typename Out, typename Idx = uint32_t>
class BHammingQueryDistance : public BaseDistance<Out, Idx> {
public:
  template <typename VecIn>
  BHammingQueryDistance(VecIn &&x, VecIn &&y, std::size_t ndim)
      : vec_len(num_blocks_needed(ndim)), nx(x.size() / ndim),
        ny(y.size() / ndim), bx(to_bitvec(std::forward<VecIn>(x), ndim)),
        by(to_bitvec(std::forward<VecIn>(y), ndim)) {}

  Out calculate(Idx i, Idx j) const override {
    return bhamming_impl<Out>(bx, i, by, j, vec_len);
  }

  std::size_t get_nx() const override { return nx; }

  std::size_t get_ny() const override { return ny; }

private:
  std::size_t vec_len;
  std::size_t nx;
  std::size_t ny;
  const BitVec bx;
  const BitVec by;
};
} // namespace tdoann

#endif // TDOANN_DISTANCEBASE_H
