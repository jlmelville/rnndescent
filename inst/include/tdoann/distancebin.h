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

#ifndef TDOANN_DISTANCEBIN_H
#define TDOANN_DISTANCEBIN_H

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "bitvec.h"
#include "distancebase.h"

// specialized dense binary versions which pack the data into bitsets. These
// can be a lot faster than either the standard dense or sparse versions.

namespace tdoann {

template <typename Out, typename Idx = uint32_t>
Out bdice(const BitVec &x, Idx i, const BitVec &y, Idx j, std::size_t len,
          std::size_t /* ndim */) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_true_true = 0;
  std::size_t num_not_equal = 0;
  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_true_true += (xi & yj).count();
    num_not_equal += (xi ^ yj).count();
  }

  if (num_not_equal == 0) {
    return Out{};
  } else {
    return static_cast<Out>(static_cast<double>(num_not_equal) /
                            (2 * num_true_true + num_not_equal));
  }
}

template <typename Out, typename Idx = uint32_t>
Out bhamming(const BitVec &x, const Idx i, const BitVec &y, Idx j,
             std::size_t len, std::size_t ndim) {
  Out sum = 0;
  std::size_t di = len * i;
  std::size_t dj = len * j;

  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    sum += (x[di] ^ y[dj]).count();
  }

  return static_cast<Out>(static_cast<double>(sum) / ndim);
}

template <typename Out, typename Idx = uint32_t>
Out bjaccard(const BitVec &x, Idx i, const BitVec &y, Idx j, std::size_t len,
             std::size_t /* ndim */) {
  std::size_t intersection = 0;
  std::size_t union_count = 0;
  std::size_t di = len * i;
  std::size_t dj = len * j;

  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto x_bitset = x[di];
    auto y_bitset = y[dj];
    intersection += (x_bitset & y_bitset).count();
    union_count += (x_bitset | y_bitset).count();
  }

  if (union_count == 0) {
    return Out(0);
  } else {
    return static_cast<Out>(static_cast<double>(union_count - intersection) /
                            union_count);
  }
}

template <typename Out, typename Idx = uint32_t>
Out bkulsinski(const BitVec &x, Idx i, const BitVec &y, Idx j, std::size_t len,
               std::size_t ndim) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_true_true = 0;
  std::size_t num_not_equal = 0;
  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_true_true += (xi & yj).count();
    num_not_equal += (xi ^ yj).count();
  }

  if (num_not_equal == 0) {
    return Out(0);
  } else {
    return static_cast<Out>(
        static_cast<double>(num_not_equal - num_true_true + ndim) /
        (num_not_equal + ndim));
  }
}

template <typename Out, typename Idx = uint32_t>
Out bmatching(const BitVec &x, Idx i, const BitVec &y, Idx j, std::size_t len,
              std::size_t ndim) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_not_equal = 0;
  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_not_equal += (xi ^ yj).count();
  }

  return static_cast<Out>(static_cast<double>(num_not_equal) / ndim);
}

template <typename Out, typename Idx = uint32_t>
Out brogers_tanimoto(const BitVec &x, Idx i, const BitVec &y, Idx j,
                     std::size_t len, std::size_t ndim) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_not_equal = 0;
  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_not_equal += (xi ^ yj).count();
  }

  return static_cast<Out>((2.0 * num_not_equal) / (ndim + num_not_equal));
}

template <typename Out, typename Idx = uint32_t>
Out brussell_rao(const BitVec &x, Idx i, const BitVec &y, Idx j,
                 std::size_t len, std::size_t ndim) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_true_true = 0;
  std::size_t num_x_true = 0;
  std::size_t num_y_true = 0;

  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_true_true += (xi & yj).count();
    num_x_true += xi.count();
    num_y_true += yj.count();
  }

  if (num_true_true == num_x_true && num_true_true == num_y_true) {
    return Out(0);
  } else {
    return static_cast<Out>(ndim - num_true_true) / static_cast<Out>(ndim);
  }
}

template <typename Out, typename Idx = uint32_t>
Out bsokal_michener(const BitVec &x, Idx i, const BitVec &y, Idx j,
                    std::size_t len, std::size_t ndim) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_not_equal = 0;

  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_not_equal += (xi ^ yj).count();
  }

  return static_cast<Out>(static_cast<double>(num_not_equal + num_not_equal) /
                          (ndim + num_not_equal));
}

template <typename Out, typename Idx = uint32_t>
Out bsokal_sneath(const BitVec &x, Idx i, const BitVec &y, Idx j,
                  std::size_t len, std::size_t ndim) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_true_true = 0;
  std::size_t num_not_equal = 0;

  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_true_true += (xi & yj).count();
    num_not_equal += (xi ^ yj).count();
  }

  if (num_not_equal == 0) {
    return Out(0);
  } else {
    return static_cast<Out>(num_not_equal) /
           static_cast<Out>(0.5 * num_true_true + num_not_equal);
  }
}

template <typename Out, typename Idx = uint32_t>
Out byule(const BitVec &x, Idx i, const BitVec &y, Idx j, std::size_t len,
          std::size_t ndim) {
  std::size_t di = len * i;
  std::size_t dj = len * j;

  std::size_t num_true_true = 0;
  std::size_t num_true_false = 0;
  std::size_t num_false_true = 0;
  std::size_t num_false_false = 0;

  for (std::size_t d = 0; d < len; ++d, ++di, ++dj) {
    auto xi = x[di];
    auto yj = y[dj];
    num_true_true += (xi & yj).count();
    num_true_false += (xi & ~yj).count();
    num_false_true += (~xi & yj).count();
  }

  num_false_false = ndim - num_true_true - num_true_false - num_false_true;

  if (num_true_false == 0 || num_false_true == 0) {
    return Out(0);
  } else {
    return static_cast<Out>(2.0 * num_true_false * num_false_true) /
           static_cast<Out>(num_true_true * num_false_false +
                            num_true_false * num_false_true);
  }
}

template <typename Out, typename Idx>
using BinaryDistanceFunc = Out (*)(const BitVec &, Idx, const BitVec &, Idx,
                                   std::size_t, std::size_t);

template <typename Out, typename Idx>
class BinarySelfDistanceCalculator : public BaseDistance<Out, Idx> {
public:
  template <typename VecIn>
  BinarySelfDistanceCalculator(VecIn &&data, std::size_t ndim,
                               BinaryDistanceFunc<Out, Idx> distance)
      : vec_len(num_blocks_needed(ndim)), nx(data.size() / ndim),
        bdata(to_bitvec(std::forward<VecIn>(data), ndim)),
        distance_func(distance), ndim(ndim) {}

  virtual ~BinarySelfDistanceCalculator() = default;

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return nx; }

  Out calculate(const Idx &i, const Idx &j) const override {
    return distance_func(this->bdata, i, this->bdata, j, this->vec_len,
                         this->ndim);
  }

protected:
  std::size_t vec_len;
  std::size_t nx;
  BitVec bdata;
  BinaryDistanceFunc<Out, Idx> distance_func;
  std::size_t ndim;
};

template <typename Out, typename Idx>
class BinaryQueryDistanceCalculator : public BaseDistance<Out, Idx> {
public:
  template <typename VecIn>
  BinaryQueryDistanceCalculator(VecIn &&x, VecIn &&y, std::size_t ndim,
                                BinaryDistanceFunc<Out, Idx> distance)
      : vec_len(num_blocks_needed(ndim)), nx(x.size() / ndim),
        ny(y.size() / ndim), bx(to_bitvec(std::forward<VecIn>(x), ndim)),
        by(to_bitvec(std::forward<VecIn>(y), ndim)), distance_func(distance),
        ndim(ndim) {}

  virtual ~BinaryQueryDistanceCalculator() = default;

  std::size_t get_nx() const override { return nx; }
  std::size_t get_ny() const override { return ny; }

  Out calculate(const Idx &i, const Idx &j) const override {
    return distance_func(this->bx, i, this->by, j, this->vec_len, this->ndim);
  }

protected:
  std::size_t vec_len;
  std::size_t nx;
  std::size_t ny;
  BitVec bx;
  BitVec by;
  BinaryDistanceFunc<Out, Idx> distance_func;
  std::size_t ndim;
};

} // namespace tdoann

#endif // TDOANN_DISTANCEBIN_H
