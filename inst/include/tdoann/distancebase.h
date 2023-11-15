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

template <typename In> using DataIt = typename std::vector<In>::const_iterator;
template <typename In, typename Out>
using DistanceFunc = Out (*)(DataIt<In>, DataIt<In>, DataIt<In>);
template <typename In>
using PreprocessFunc = void (*)(std::vector<In> &, std::size_t);

template <typename In, typename Out, typename Idx>
class SelfDistanceCalculator : public VectorDistance<In, Out, Idx> {
public:
  using Iterator = typename std::vector<In>::const_iterator;
  using DistanceFunc = Out (*)(Iterator, Iterator, Iterator);

  virtual ~SelfDistanceCalculator() = default;

  template <typename VecIn>
  SelfDistanceCalculator(VecIn &&data, std::size_t ndim,
                         DistanceFunc distance_func,
                         PreprocessFunc<In> preprocess_func = nullptr)
      : x(std::move(data)), nx(x.size() / ndim), ndim(ndim),
        distance_func(distance_func) {
    if (preprocess_func) {
      preprocess_func(x, ndim);
    }
  }

  std::size_t get_nx() const override { return nx; }
  // For self-distance, nx == ny
  std::size_t get_ny() const override { return get_nx(); }
  Iterator get_x(Idx i) const override { return x.begin() + ndim * i; }
  // For self-distance get_y(i) == get_x(i)
  Iterator get_y(Idx i) const override { return get_x(i); }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return distance_func(this->x.begin() + di,
                         this->x.begin() + di + this->ndim,
                         this->x.begin() + this->ndim * j);
  }

protected:
  std::vector<In> x;
  std::size_t nx;
  std::size_t ndim;
  DistanceFunc distance_func;
};

template <typename In, typename Out, typename Idx>
class QueryDistanceCalculator : public VectorDistance<In, Out, Idx> {
public:
  using Iterator = typename std::vector<In>::const_iterator;
  using DistanceFunc = Out (*)(Iterator, Iterator, Iterator);

  virtual ~QueryDistanceCalculator() = default;

  template <typename VecIn>
  QueryDistanceCalculator(VecIn &&xdata, VecIn &&ydata, std::size_t ndim,
                          DistanceFunc distance_func,
                          PreprocessFunc<In> preprocess_func = nullptr)
      : x(std::forward<VecIn>(xdata)), y(std::forward<VecIn>(ydata)),
        nx(x.size() / ndim), ny(y.size() / ndim), ndim(ndim),
        distance_func(distance_func) {
    if (preprocess_func) {
      preprocess_func(x, ndim);
      preprocess_func(y, ndim);
    }
  }

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }
  Iterator get_x(Idx i) const override { return x.begin() + i * ndim; }
  Iterator get_y(Idx i) const override { return y.begin() + i * ndim; }

  Out calculate(const Idx &i, const Idx &j) const override {
    const std::size_t di = this->ndim * i;
    return distance_func(this->x.begin() + di,
                         this->x.begin() + di + this->ndim,
                         this->y.begin() + this->ndim * j);
  }

protected:
  std::vector<In> x;
  std::vector<In> y;
  std::size_t nx;
  std::size_t ny;
  std::size_t ndim;
  DistanceFunc distance_func;
};

} // namespace tdoann

#endif // TDOANN_DISTANCEBASE_H
