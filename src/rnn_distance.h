//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2019 James Melville
//
//  This file is part of rnndescent
//
//  rnndescent is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  rnndescent is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with rnndescent.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RNN_DISTANCE_H
#define RNN_DISTANCE_H

#include <memory>
#include <type_traits>

#include <Rcpp.h>

#include "tdoann/distancebase.h"

#include "rnn_util.h"

template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_query_distance(const Rcpp::NumericMatrix &reference,
                      const Rcpp::NumericMatrix &query,
                      const std::string &metric) {
  using In = RNN_DEFAULT_DIST;
  using Out = RNN_DEFAULT_DIST;

  const auto ndim = reference.nrow();

  if (metric == "bhamming") {
    auto ref_bvec = r_to_vec<uint8_t>(reference);
    auto query_bvec = r_to_vec<uint8_t>(query);
    return std::make_unique<tdoann::BHammingQueryDistance<Out, Idx>>(
        std::move(ref_bvec), std::move(query_bvec), ndim);
  }

  auto ref_vec = r_to_vec<In>(reference);
  auto query_vec = r_to_vec<In>(query);

  if (metric == "euclidean") {
    return std::make_unique<tdoann::EuclideanQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }
  if (metric == "l2sqr") {
    return std::make_unique<tdoann::L2SqrQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }
  if (metric == "manhattan") {
    return std::make_unique<tdoann::ManhattanQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }
  if (metric == "cosine") {
    return std::make_unique<tdoann::CosineQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }
  if (metric == "cosine-cache") {
    return std::make_unique<tdoann::CosineCacheQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }
  if (metric == "correlation") {
    return std::make_unique<tdoann::CorrelationQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }
  if (metric == "correlation-cache") {
    return std::make_unique<
        tdoann::CorrelationCacheQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }
  if (metric == "hamming") {
    return std::make_unique<tdoann::HammingQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }

  Rcpp::stop("Bad metric");
}

template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_self_distance(const Rcpp::NumericMatrix &data,
                     const std::string &metric) {
  using In = RNN_DEFAULT_DIST;
  using Out = RNN_DEFAULT_DIST;

  const auto ndim = data.nrow();
  // handle binary first
  if (metric == "bhamming") {
    auto data_bvec = r_to_vec<uint8_t>(data);
    return std::make_unique<tdoann::BHammingSelfDistance<Out, Idx>>(
        std::move(data_bvec), ndim);
  }

  auto data_vec = r_to_vec<In>(data);
  if (metric == "euclidean") {
    return std::make_unique<tdoann::EuclideanSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  if (metric == "l2sqr") {
    return std::make_unique<tdoann::L2SqrSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  if (metric == "manhattan") {
    return std::make_unique<tdoann::ManhattanSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  if (metric == "cosine") {
    return std::make_unique<tdoann::CosineSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  if (metric == "cosine-cache") {
    return std::make_unique<tdoann::CosineCacheSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  if (metric == "correlation") {
    return std::make_unique<tdoann::CorrelationSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  if (metric == "correlation-cache") {
    return std::make_unique<tdoann::CorrelationCacheSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  if (metric == "hamming") {
    return std::make_unique<tdoann::HammingSelfDistance<In, Out, Idx>>(
        std::move(data_vec), ndim);
  }
  Rcpp::stop("Bad metric");
}

#endif // RNN_DISTANCE_H
