//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_DISTANCE_BATCH_H
#define RNN_DISTANCE_BATCH_H

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <utility>

#include "tdoann/distance.h"

namespace rnndescent {

enum class DenseBatchMetric {
  raw_dot,
  squared_euclidean,
  euclidean,
  inner_product,
  dot,
  alternative_dot,
  manhattan
};

template <typename Out, typename Idx> struct DenseDistanceFour {
  using Targets = std::array<Idx, 4>;
  using Distances = std::array<Out, 4>;
  std::function<Distances(Idx, const Targets &)> evaluate;

  template <DenseBatchMetric Metric>
  static auto bind(const float *data, std::size_t dimension)
      -> DenseDistanceFour {
    return {[data, dimension](Idx p, const Targets &q) {
      return run<Metric>(data, dimension, p, q);
    }};
  }

  template <DenseBatchMetric Metric>
  static auto run(const float *data, std::size_t dimension, Idx p,
                  const Targets &q) -> Distances {
    return run_from<Metric>(data + p * dimension, data, dimension, q);
  }

  template <DenseBatchMetric Metric>
  static auto run_from(const float *x, const float *data, std::size_t dimension,
                       const Targets &q) -> Distances {
    if constexpr (Metric == DenseBatchMetric::squared_euclidean ||
                  Metric == DenseBatchMetric::euclidean ||
                  Metric == DenseBatchMetric::raw_dot) {
      return blocked_four<Metric>(
          x,
          {data + q[0] * dimension, data + q[1] * dimension,
           data + q[2] * dimension, data + q[3] * dimension},
          dimension);
    }
    const auto *y0 = data + q[0] * dimension;
    const auto *y1 = data + q[1] * dimension;
    const auto *y2 = data + q[2] * dimension;
    const auto *y3 = data + q[3] * dimension;
    Out sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    // Share the anchor load and keep each candidate's feature order.
    for (std::size_t j = 0; j < dimension; ++j) {
      const auto anchor = x[j];
      if constexpr (Metric == DenseBatchMetric::manhattan) {
        sum0 += std::abs(anchor - y0[j]);
        sum1 += std::abs(anchor - y1[j]);
        sum2 += std::abs(anchor - y2[j]);
        sum3 += std::abs(anchor - y3[j]);
      } else {
        sum0 += anchor * y0[j];
        sum1 += anchor * y1[j];
        sum2 += anchor * y2[j];
        sum3 += anchor * y3[j];
      }
    }
    return {finish<Metric>(sum0), finish<Metric>(sum1), finish<Metric>(sum2),
            finish<Metric>(sum3)};
  }

  template <DenseBatchMetric Metric>
  static auto blocked_four(const float *x,
                           const std::array<const float *, 4> &rows,
                           std::size_t dimension) -> Distances {
    static_assert(Metric == DenseBatchMetric::raw_dot ||
                  Metric == DenseBatchMetric::squared_euclidean ||
                  Metric == DenseBatchMetric::euclidean);
    constexpr auto width = tdoann::feature_sum_width;
    std::array<std::array<Out, width>, 4> sums{};
    const auto block_end = dimension - dimension % width;
    // Each lane accumulates one residue class of feature indices.
    for (std::size_t j = 0; j < block_end; j += width) {
      if constexpr (Metric == DenseBatchMetric::raw_dot) {
        // Update targets in pairs so each pair shares the anchor block.
        // Fixed lane indices preserve the eight independent partial sums.
        constexpr auto lanes = std::make_index_sequence<width>{};
        for (std::size_t target = 0; target < rows.size(); target += 2) {
          sums[target] =
              add_dot_block(sums[target], x + j, rows[target] + j, lanes);
          sums[target + 1] = add_dot_block(sums[target + 1], x + j,
                                           rows[target + 1] + j, lanes);
        }
      } else {
        for (std::size_t target = 0; target < rows.size(); ++target) {
          for (std::size_t lane = 0; lane < width; ++lane) {
            const auto delta = x[j + lane] - rows[target][j + lane];
            const auto square = delta * delta;
            sums[target][lane] += square;
          }
        }
      }
    }
    Distances result;
    for (std::size_t target = 0; target < rows.size(); ++target) {
      const auto &lanes = sums[target];
      Out sum = std::accumulate(lanes.begin(), lanes.end(), Out{0});
      for (auto j = block_end; j < dimension; ++j) {
        if constexpr (Metric == DenseBatchMetric::raw_dot) {
          sum += x[j] * rows[target][j];
        } else {
          const auto delta = x[j] - rows[target][j];
          const auto square = delta * delta;
          sum += square;
        }
      }
      result[target] = finish<Metric>(sum);
    }
    return result;
  }

private:
  template <std::size_t... Lane>
  static auto add_dot_block(const std::array<Out, sizeof...(Lane)> &sums,
                            const float *x, const float *y,
                            std::index_sequence<Lane...>)
      -> std::array<Out, sizeof...(Lane)> {
    return {{(sums[Lane] + x[Lane] * y[Lane])...}};
  }

  template <DenseBatchMetric Metric> static auto finish(Out sum) -> Out {
    if constexpr (Metric == DenseBatchMetric::euclidean) {
      return std::sqrt(sum);
    } else if constexpr (Metric == DenseBatchMetric::inner_product) {
      return tdoann::finish_inner_product(sum);
    } else if constexpr (Metric == DenseBatchMetric::dot) {
      return tdoann::finish_dot(sum);
    } else if constexpr (Metric == DenseBatchMetric::alternative_dot) {
      return tdoann::finish_alternative_dot(sum);
    } else {
      return sum;
    }
  }
};

} // namespace rnndescent

#endif // RNN_DISTANCE_BATCH_H
