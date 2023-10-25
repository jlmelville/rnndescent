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
  virtual Out calculate(const Idx &i, const Idx &j) const = 0;
  virtual std::size_t get_nx() const = 0;
  virtual std::size_t get_ny() const = 0;

  // needed for RP Tree calculations
  // https://github.com/lmcinnes/pynndescent/blob/db258cea34cce7e11e90a460c1f8a0bd8b69f1c1/pynndescent/pynndescent_.py#L764
  // angular metrics currently are:
  // "cosine", "dot", "correlation", "dice", "jaccard", "hellinger", "hamming",
  // other metrics are considered to be euclidean. By default metrics are not
  // considered angular so you only have to override this if the metric is
  // angular
  virtual bool is_angular() const { return false; }
};

// Distance calculators which can return an iterator pointing to a contiguous
// region of memory holding the ith data point. This is most of them, although
// be aware of calculators which pre-process the data, as they will return the
// pre-processed vectors. An example of a calculator which can't implement this
// interface is the BHamming binary Hamming calculator. This interface is needed
// only for something like RPTrees where the distance between items and a
// hyperplane is needed.
template <typename In, typename Out, typename Idx = uint32_t>
class VectorDistance : public BaseDistance<Out, Idx> {
public:
  using Iterator = typename std::vector<In>::const_iterator;

  virtual ~VectorDistance() = default;

  // return iterator pointing at the ith data point
  virtual auto get_x(Idx i) const -> Iterator = 0;
  virtual auto get_y(Idx i) const -> Iterator = 0;
};

// these traits are used to extract the types for the template parameters of
// the distance calculators, e.g.:
// auto distance_ptr = create_self_vector_distance(data, metric);
// using Idx = typename tdoann::DistanceTraits<decltype(distance_ptr)>::Index;
template <typename T> struct DistanceTraits;

template <typename Out, typename Idx>
struct DistanceTraits<std::unique_ptr<BaseDistance<Out, Idx>>> {
  using Output = Out;
  using Index = Idx;
};

template <typename In, typename Out, typename Idx>
struct DistanceTraits<std::unique_ptr<VectorDistance<In, Out, Idx>>> {
  using Input = In;
  using Output = Out;
  using Index = Idx;
};

// These mixins hold vector and dimension data. Due to the joys of inheritance
// chains and resolution of virtual methods, classes using this mixin save very
// little on boilerplate (in fact they might even be more verbose).

// SelfDataMixin is for distances calculated within a single dataset.
template <typename In, typename Idx> class SelfDataMixin {
public:
  using Iterator = typename std::vector<In>::const_iterator;

  template <typename VecIn>
  SelfDataMixin(VecIn &&data, std::size_t ndim)
      : x(std::forward<VecIn>(data)), nx(x.size() / ndim), ndim(ndim) {}

  std::size_t get_nx() const { return nx; }
  // For self-distance, nx == ny
  std::size_t get_ny() const { return get_nx(); }
  Iterator get_x(Idx i) const { return x.begin() + ndim * i; }
  // For self-distance get_y(i) == get_x(i)
  Iterator get_y(Idx i) const { return get_x(i); }

protected:
  mutable std::vector<In> x;
  std::size_t nx;
  std::size_t ndim;
};

template <typename In, typename Idx> class QueryDataMixin {
public:
  using Iterator = typename std::vector<In>::const_iterator;

  template <typename VecIn>
  QueryDataMixin(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : x(std::forward<VecIn>(xdata)), y(std::forward<VecIn>(ydata)),
        nx(x.size() / ndim), ny(y.size() / ndim), ndim(ndim) {}

  std::size_t get_nx() const { return nx; }
  std::size_t get_ny() const { return ny; }
  Iterator get_x(Idx i) const { return x.begin() + i * ndim; }
  Iterator get_y(Idx i) const { return y.begin() + i * ndim; }

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
class L2SqrSelfDistance : public VectorDistance<In, Out, Idx>,
                          public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  L2SqrSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return l2sqr<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                      this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class L2SqrQueryDistance : public VectorDistance<In, Out, Idx>,
                           public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  L2SqrQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return l2sqr<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                      this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class EuclideanSelfDistance : public VectorDistance<In, Out, Idx>,
                              public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  EuclideanSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return euclidean<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class EuclideanQueryDistance : public VectorDistance<In, Out, Idx>,
                               public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  EuclideanQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return euclidean<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class ManhattanSelfDistance : public VectorDistance<In, Out, Idx>,
                              public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  ManhattanSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return manhattan<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->x.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class ManhattanQueryDistance : public VectorDistance<In, Out, Idx>,
                               public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  ManhattanQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return manhattan<Out>(this->x.begin() + di,
                          this->x.begin() + di + this->ndim,
                          this->y.begin() + this->ndim * j);
  }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CosineSelfDistance : public VectorDistance<In, Out, Idx>,
                           public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CosineSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return cosine<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                       this->x.begin() + this->ndim * j);
  }

  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CosineQueryDistance : public VectorDistance<In, Out, Idx>,
                            public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CosineQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return cosine<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                       this->y.begin() + this->ndim * j);
  }

  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CosinePreprocessSelfDistance : public VectorDistance<In, Out, Idx>,
                                     public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CosinePreprocessSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return inner_product<Out>(this->x.begin() + di,
                              this->x.begin() + di + this->ndim,
                              this->x.begin() + this->ndim * j);
  }

  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CosinePreprocessQueryDistance : public VectorDistance<In, Out, Idx>,
                                      public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CosinePreprocessQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->y), this->ndim);
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return inner_product<Out>(this->x.begin() + di,
                              this->x.begin() + di + this->ndim,
                              this->y.begin() + this->ndim * j);
  }
  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationSelfDistance : public VectorDistance<In, Out, Idx>,
                                public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CorrelationSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return correlation<Out>(this->x.begin() + di,
                            this->x.begin() + di + this->ndim,
                            this->x.begin() + this->ndim * j);
  }

  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationQueryDistance : public VectorDistance<In, Out, Idx>,
                                 public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CorrelationQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return correlation<Out>(this->x.begin() + di,
                            this->x.begin() + di + this->ndim,
                            this->y.begin() + this->ndim * j);
  }
  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationPreprocessSelfDistance : public VectorDistance<In, Out, Idx>,
                                          public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CorrelationPreprocessSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    mean_center(const_cast<std::vector<In> &>(this->x), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return inner_product<Out>(this->x.begin() + di,
                              this->x.begin() + di + this->ndim,
                              this->x.begin() + this->ndim * j);
  }
  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

// pre-calculates data for faster perf in calculate
template <typename In, typename Out, typename Idx = uint32_t>
class CorrelationPreprocessQueryDistance : public VectorDistance<In, Out, Idx>,
                                           public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  CorrelationPreprocessQueryDistance(VecIn &&xdata, VecIn &&ydata,
                                     std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {
    // must cast away the constness temporarily to do in-place initialization
    mean_center(const_cast<std::vector<In> &>(this->x), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->x), this->ndim);

    mean_center(const_cast<std::vector<In> &>(this->y), this->ndim);
    normalize(const_cast<std::vector<In> &>(this->y), this->ndim);
  }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return inner_product<Out>(this->x.begin() + di,
                              this->x.begin() + di + this->ndim,
                              this->y.begin() + this->ndim * j);
  }
  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class HammingSelfDistance : public VectorDistance<In, Out, Idx>,
                            public SelfDataMixin<In, Idx> {
public:
  using Mixin = SelfDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  HammingSelfDistance(VecIn &&data, std::size_t ndim)
      : SelfDataMixin<In, Idx>(std::forward<VecIn>(data), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return hamming<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                        this->x.begin() + this->ndim * j);
  }
  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename In, typename Out, typename Idx = uint32_t>
class HammingQueryDistance : public VectorDistance<In, Out, Idx>,
                             public QueryDataMixin<In, Idx> {
public:
  using Mixin = QueryDataMixin<In, Idx>;
  using Iterator = typename Mixin::Iterator;

  template <typename VecIn>
  HammingQueryDistance(VecIn &&xdata, VecIn &&ydata, std::size_t ndim)
      : QueryDataMixin<In, Idx>(std::forward<VecIn>(xdata),
                                std::forward<VecIn>(ydata), ndim) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return hamming<Out>(this->x.begin() + di, this->x.begin() + di + this->ndim,
                        this->y.begin() + this->ndim * j);
  }
  bool is_angular() const override { return true; }

  std::size_t get_nx() const override { return Mixin::get_nx(); }
  std::size_t get_ny() const override { return Mixin::get_ny(); }
  Iterator get_x(Idx i) const override { return Mixin::get_x(i); }
  Iterator get_y(Idx i) const override { return Mixin::get_y(i); }
};

template <typename Out, typename Idx = uint32_t>
class BHammingSelfDistance : public BaseDistance<Out, Idx> {
public:
  template <typename VecIn>
  BHammingSelfDistance(VecIn &&data, std::size_t ndim)
      : vec_len(num_blocks_needed(ndim)), nx(data.size() / ndim), ny(nx),
        bdata(to_bitvec(std::forward<VecIn>(data), ndim)) {}

  Out calculate(const Idx &i, const Idx &j) const override {
    return bhamming_impl<Out>(bdata, i, bdata, j, vec_len);
  }
  bool is_angular() const override { return true; }

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

  Out calculate(const Idx &i, const Idx &j) const override {
    return bhamming_impl<Out>(bx, i, by, j, vec_len);
  }

  bool is_angular() const override { return true; }

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
