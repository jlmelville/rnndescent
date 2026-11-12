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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include <Rcpp.h>

#include "tdoann/distancebase.h"
#include "tdoann/distancebin.h"
#include "tdoann/sparse.h"

#include "rnn_util.h"

template <typename In, typename Out>
const std::unordered_map<std::string, tdoann::DistanceFunc<In, Out>> &
get_metric_map() {
  using InIt = tdoann::DataIt<In>;
  static const std::unordered_map<std::string, tdoann::DistanceFunc<In, Out>>
      metric_map = {
          {"braycurtis", tdoann::bray_curtis<Out, InIt>},
          {"canberra", tdoann::canberra<Out, InIt>},
          {"chebyshev", tdoann::chebyshev<Out, InIt>},
          {"correlation", tdoann::correlation<Out, InIt>},
          {"correlation-preprocess", tdoann::inner_product<Out, InIt>},
          {"cosine", tdoann::cosine<Out, InIt>},
          {"alternative-cosine", tdoann::alternative_cosine<Out, InIt>},
          {"cosine-preprocess", tdoann::inner_product<Out, InIt>},
          {"dot", tdoann::dot<Out, InIt>},
          {"alternative-dot", tdoann::alternative_dot<Out, InIt>},
          {"dice", tdoann::dice<Out, InIt>},
          {"euclidean", tdoann::euclidean<Out, InIt>},
          {"hamming", tdoann::hamming<Out, InIt>},
          {"haversine", tdoann::haversine<Out, InIt>},
          {"hellinger", tdoann::hellinger<Out, InIt>},
          {"alternative-hellinger", tdoann::alternative_hellinger<Out, InIt>},
          {"jaccard", tdoann::jaccard<Out, InIt>},
          {"alternative-jaccard", tdoann::alternative_jaccard<Out, InIt>},
          {"jensenshannon", tdoann::jensen_shannon_divergence<Out, InIt>},
          {"kulsinski", tdoann::kulsinski<Out, InIt>},
          {"manhattan", tdoann::manhattan<Out, InIt>},
          {"matching", tdoann::matching<Out, InIt>},
          {"rogerstanimoto", tdoann::rogers_tanimoto<Out, InIt>},
          {"russellrao", tdoann::russell_rao<Out, InIt>},
          {"sokalmichener", tdoann::sokal_michener<Out, InIt>},
          {"sokalsneath", tdoann::sokal_sneath<Out, InIt>},
          {"spearmanr", tdoann::spearmanr<Out, InIt>},
          {"sqeuclidean", tdoann::squared_euclidean<Out, InIt>},
          {"symmetrickl", tdoann::symmetric_kl_divergence<Out, InIt>},
          {"trueangular", tdoann::true_angular<Out, InIt>},
          {"tsss", tdoann::tsss<Out, InIt>},
          {"yule", tdoann::yule<Out, InIt>}};
  return metric_map;
}

template <typename In>
const std::unordered_map<std::string, tdoann::PreprocessFunc<In>> &
get_preprocess_map() {
  static const std::unordered_map<std::string, tdoann::PreprocessFunc<In>> map =
      {{"cosine-preprocess", tdoann::normalize<In>},
       {"correlation-preprocess", tdoann::mean_center_and_normalize<In>},
       {"dot", tdoann::normalize<In>},
       {"alternative-dot", tdoann::normalize<In>}};
  return map;
}

template <typename Out, typename Idx>
const std::unordered_map<std::string, tdoann::BinaryDistanceFunc<Out, Idx>> &
get_binary_metric_map() {
  static const std::unordered_map<std::string,
                                  tdoann::BinaryDistanceFunc<Out, Idx>>
      metric_map = {{"dice", tdoann::bdice<Out, Idx>},
                    {"hamming", tdoann::bhamming<Out, Idx>},
                    {"jaccard", tdoann::bjaccard<Out, Idx>},
                    {"kulsinski", tdoann::bkulsinski<Out, Idx>},
                    {"matching", tdoann::bmatching<Out, Idx>},
                    {"rogerstanimoto", tdoann::brogers_tanimoto<Out, Idx>},
                    {"russellrao", tdoann::brussell_rao<Out, Idx>},
                    {"sokalmichener", tdoann::bsokal_michener<Out, Idx>},
                    {"sokalsneath", tdoann::bsokal_sneath<Out, Idx>},
                    {"yule", tdoann::byule<Out, Idx>}};
  return metric_map;
}

template <typename In, typename Out>
const std::unordered_map<std::string, tdoann::SparseDistanceFunc<In, Out>> &
get_sparse_metric_map() {
  using InIt = tdoann::DataIt<In>;
  static const std::unordered_map<std::string,
                                  tdoann::SparseDistanceFunc<In, Out>>
      metric_map = {
          {"braycurtis", tdoann::sparse_bray_curtis<Out, InIt>},
          {"canberra", tdoann::sparse_canberra<Out, InIt>},
          {"chebyshev", tdoann::sparse_chebyshev<Out, InIt>},
          {"correlation", tdoann::sparse_correlation<Out, InIt>},
          {"cosine", tdoann::sparse_cosine<Out, InIt>},
          {"alternative-cosine", tdoann::sparse_alternative_cosine<Out, InIt>},
          {"dice", tdoann::sparse_dice<Out, InIt>},
          {"dot", tdoann::sparse_dot<Out, InIt>},
          {"alternative-dot", tdoann::sparse_alternative_dot<Out, InIt>},
          {"euclidean", tdoann::sparse_euclidean<Out, InIt>},
          {"hamming", tdoann::sparse_hamming<Out, InIt>},
          {"jaccard", tdoann::sparse_jaccard<Out, InIt>},
          {"alternative-jaccard",
           tdoann::sparse_alternative_jaccard<Out, InIt>},
          {"hellinger", tdoann::sparse_hellinger<Out, InIt>},
          {"alternative-hellinger",
           tdoann::sparse_alternative_hellinger<Out, InIt>},
          {"jensenshannon",
           tdoann::sparse_jensen_shannon_divergence<Out, InIt>},
          {"kulsinski", tdoann::sparse_kulsinski<Out, InIt>},
          {"manhattan", tdoann::sparse_manhattan<Out, InIt>},
          {"matching", tdoann::sparse_matching<Out, InIt>},
          {"rogerstanimoto", tdoann::sparse_rogers_tanimoto<Out, InIt>},
          {"russellrao", tdoann::sparse_russell_rao<Out, InIt>},
          {"sokalmichener", tdoann::sparse_sokal_michener<Out, InIt>},
          {"sokalsneath", tdoann::sparse_sokal_sneath<Out, InIt>},
          {"spearmanr", tdoann::sparse_spearmanr<Out, InIt>},
          {"sqeuclidean", tdoann::sparse_squared_euclidean<Out, InIt>},
          {"symmetrickl", tdoann::sparse_symmetric_kl_divergence<Out, InIt>},
          {"trueangular", tdoann::sparse_true_angular<Out, InIt>},
          {"tsss", tdoann::sparse_tsss<Out, InIt>},
          {"yule", tdoann::sparse_yule<Out, InIt>}};
  return metric_map;
}

template <typename In, typename Out>
const std::unordered_map<std::string, tdoann::SparsePreprocessFunc<In>> &
get_sparse_preprocess_map() {
  static const std::unordered_map<std::string, tdoann::SparsePreprocessFunc<In>>
      map = {{"dot", tdoann::sparse_normalize<In, Out>},
             {"alternative-dot", tdoann::sparse_normalize<In, Out>}};
  return map;
}

inline bool is_probability_metric(const std::string &metric) {
  return metric == "hellinger" || metric == "alternative-hellinger" ||
         metric == "jensenshannon" || metric == "symmetrickl";
}

// Check original values while copying so float conversion cannot hide
// negatives.
template <typename In, typename Input>
std::vector<In>
r_to_vec_for_probability(const Input &data, const std::string &metric,
                         const char *name = "data", bool check_finite = false) {
  std::vector<In> result(data.size());
  auto output = result.begin();
  for (const double value : data) {
    if (check_finite && !std::isfinite(value)) {
      Rcpp::stop(std::string(name) +
                 " must contain only finite numeric feature values");
    }
    if (value < 0.0) {
      Rcpp::stop("metric '" + metric +
                 "' requires non-negative feature values in '" + name + "'");
    }
    *output++ = static_cast<In>(value);
  }
  return result;
}

inline void validate_haversine_ndim(std::size_t ndim,
                                    const std::string &metric) {
  if (metric == "haversine" && ndim != 2) {
    Rcpp::stop("haversine is only defined for 2 dimensional data");
  }
}

template <typename In>
inline void validate_haversine_radians(const std::vector<In> &data,
                                       std::size_t ndim,
                                       const std::string &metric) {
  if (metric != "haversine") {
    return;
  }

  const In pi = static_cast<In>(std::acos(-1.0));
  const In max_lat = pi / static_cast<In>(2);
  const In max_lon = static_cast<In>(2) * pi;
  const In tol = static_cast<In>(1.0e-7);

  for (std::size_t offset = 0; offset + 1 < data.size(); offset += ndim) {
    const In lat = data[offset];
    const In lon = data[offset + 1];
    if (std::abs(lat) > max_lat + tol || std::abs(lon) > max_lon + tol) {
      Rcpp::stop("haversine expects radians; latitude must be in [-pi/2, pi/2] "
                 "and longitude in [-2*pi, 2*pi]");
    }
  }
}

template <typename In, typename Out>
std::pair<tdoann::DistanceFunc<In, Out>, tdoann::PreprocessFunc<In>>
get_dense_distance_funcs(const std::string &metric) {
  const auto &metric_map = get_metric_map<In, Out>();
  if (metric_map.count(metric) == 0) {
    Rcpp::stop("Unknown metric '" + metric + "' for dense data");
  }
  auto distance_func = metric_map.at(metric);

  tdoann::PreprocessFunc<In> preprocess_func = nullptr;
  const auto &preprocess_map = get_preprocess_map<In>();
  if (preprocess_map.count(metric) > 0) {
    preprocess_func = preprocess_map.at(metric);
  }

  return {distance_func, preprocess_func};
}

template <typename In, typename Out>
std::pair<tdoann::SparseDistanceFunc<In, Out>, tdoann::SparsePreprocessFunc<In>>
get_sparse_distance_funcs(const std::string &metric) {
  const auto &metric_map = get_sparse_metric_map<In, Out>();
  if (metric_map.count(metric) == 0) {
    Rcpp::stop("Unknown metric '" + metric + "' for sparse data");
  }
  auto distance_func = metric_map.at(metric);

  tdoann::SparsePreprocessFunc<In> preprocess_func = nullptr;
  const auto &preprocess_map = get_sparse_preprocess_map<In, Out>();
  if (preprocess_map.count(metric) > 0) {
    preprocess_func = preprocess_map.at(metric);
  }

  return {distance_func, preprocess_func};
}

// Using Traits to return a pointer to BaseDistance or VectorDistance
// Functions can return a BaseDistance<Out, Idx> or VectorDistance<In, Out, Idx>
// depending on the template. Different number of template parameters means
// we must use variadic template arguments.
template <typename... Args> struct FactoryTraits;

template <typename Out, typename Idx>
struct FactoryTraits<tdoann::BaseDistance<Out, Idx>> {
  using type = tdoann::BaseDistance<Out, Idx>;
  using input_type = RNN_DEFAULT_IN;
  using output_type = Out;
  using index_type = Idx;
};

template <typename In, typename Out, typename Idx>
struct FactoryTraits<tdoann::VectorDistance<In, Out, Idx>> {
  using type = tdoann::VectorDistance<In, Out, Idx>;
  using input_type = In;
  using output_type = Out;
  using index_type = Idx;
};

template <typename In, typename Out, typename Idx>
struct FactoryTraits<tdoann::SelfDistanceCalculator<In, Out, Idx>> {
  using type = tdoann::SelfDistanceCalculator<In, Out, Idx>;
  using input_type = In;
  using output_type = Out;
  using index_type = Idx;
};

template <typename In, typename Out, typename Idx>
struct FactoryTraits<tdoann::SparseVectorDistance<In, Out, Idx>> {
  using type = tdoann::SparseVectorDistance<In, Out, Idx>;
  using input_type = In;
  using output_type = Out;
  using index_type = Idx;
};

// Forest projections still read the original coordinates via get_x/get_y.
template <typename In, typename Out, typename Idx>
class SpearmanQueryDistance final
    : public tdoann::QueryDistanceCalculator<In, Out, Idx> {
  using Base = tdoann::QueryDistanceCalculator<In, Out, Idx>;
  using Iterator = typename Base::Iterator;

  struct RankedRow {
    std::once_flag ready;
    std::vector<Out> values;
  };

  mutable std::vector<RankedRow> reference_ranks;
  mutable std::vector<RankedRow> query_ranks;

  auto ranks(RankedRow &row, Iterator begin) const -> const std::vector<Out> & {
    std::call_once(row.ready, [&] {
      row.values = tdoann::rankdata<Out>(begin, begin + this->ndim);
    });
    return row.values;
  }

public:
  SpearmanQueryDistance(std::vector<In> &&reference, std::vector<In> &&query,
                        std::size_t dimension)
      : Base(std::move(reference), std::move(query), dimension,
             tdoann::spearmanr<Out, Iterator>),
        reference_ranks(this->get_nx()), query_ranks(this->get_ny()) {}

  Out calculate(const Idx &reference, const Idx &query) const override {
    const auto &x_rank =
        ranks(reference_ranks[reference], this->get_x(reference));
    const auto &y_rank = ranks(query_ranks[query], this->get_y(query));
    return tdoann::correlation<Out>(x_rank.cbegin(), x_rank.cend(),
                                    y_rank.cbegin());
  }
};

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_query_distance_impl(
    std::vector<typename FactoryTraits<Args...>::input_type> ref_vec,
    std::vector<typename FactoryTraits<Args...>::input_type> query_vec,
    std::size_t ndim, const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;
  using Out = typename FactoryTraits<Args...>::output_type;
  using Idx = typename FactoryTraits<Args...>::index_type;

  validate_haversine_ndim(ndim, metric);
  validate_haversine_radians(ref_vec, ndim, metric);
  validate_haversine_radians(query_vec, ndim, metric);

  if (metric == "spearmanr") {
    return std::make_unique<SpearmanQueryDistance<In, Out, Idx>>(
        std::move(ref_vec), std::move(query_vec), ndim);
  }

  auto [distance_func, preprocess_func] =
      get_dense_distance_funcs<In, Out>(metric);

  return std::make_unique<tdoann::QueryDistanceCalculator<In, Out, Idx>>(
      std::move(ref_vec), std::move(query_vec), ndim, distance_func,
      preprocess_func);
}

template <typename In>
auto copy_finite_reference(const Rcpp::NumericMatrix &reference)
    -> std::vector<In> {
  const auto size = reference.size();
  std::vector<In> result(size);
  const double *input = reference.begin();
  for (R_xlen_t i = 0; i < size; ++i) {
    const double value = input[i];
    if (!std::isfinite(value)) {
      Rcpp::stop("reference must contain only finite numeric feature values");
    }
    result[i] = static_cast<In>(value);
  }
  return result;
}

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_query_distance_impl(const Rcpp::NumericMatrix &reference,
                           const Rcpp::NumericMatrix &query,
                           const std::string &metric,
                           bool check_reference_during_copy = false) {
  using In = typename FactoryTraits<Args...>::input_type;

  const auto ndim = reference.nrow();
  validate_haversine_ndim(ndim, metric);
  validate_haversine_ndim(query.nrow(), metric);
  const bool probability_metric = is_probability_metric(metric);
  auto ref_vec =
      probability_metric
          ? r_to_vec_for_probability<In>(reference, metric, "reference")
          : (check_reference_during_copy ? copy_finite_reference<In>(reference)
                                         : r_to_vec<In>(reference));
  auto query_vec = probability_metric
                       ? r_to_vec_for_probability<In>(query, metric, "query")
                       : r_to_vec<In>(query);

  return create_query_distance_impl<Args...>(
      std::move(ref_vec), std::move(query_vec), ndim, metric);
}

template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_query_distance(const Rcpp::NumericMatrix &reference,
                      const Rcpp::NumericMatrix &query,
                      const std::string &metric) {
  using Out = RNN_DEFAULT_DIST;
  // potential specializations here
  return create_query_distance_impl<tdoann::BaseDistance<Out, Idx>>(
      reference, query, metric);
}

inline auto pack_logical_matrix(const Rcpp::LogicalMatrix &matrix)
    -> tdoann::BitVec {
  constexpr std::size_t word_bits = tdoann::BITVEC_BIT_WIDTH;
  const auto n_features = static_cast<std::size_t>(matrix.nrow());
  const auto n_observations = static_cast<std::size_t>(matrix.ncol());
  tdoann::BitVec packed;
  packed.reserve(n_observations * tdoann::num_blocks_needed(n_features));
  auto values = matrix.begin();
  bool missing = false;
  for (std::size_t i = 0; i < n_observations; ++i) {
    for (std::size_t offset = 0; offset < n_features; offset += word_bits) {
      const auto count = std::min(word_bits, n_features - offset);
      std::uint64_t word = 0;
      for (std::size_t bit = 0; bit < count; ++bit) {
        const auto value = *values++;
        missing |= value == NA_LOGICAL;
        word |= static_cast<std::uint64_t>(value != 0) << bit;
      }
      packed.emplace_back(word);
    }
  }
  if (missing) {
    Rcpp::stop("Logical feature values must not be NA");
  }
  return packed;
}

template <typename Out, typename Idx>
auto get_binary_query_distance_func(const std::string &metric)
    -> tdoann::BinaryDistanceFunc<Out, Idx> {
  if (metric == "alternative-jaccard") {
    return tdoann::balternative_jaccard<Out, Idx>;
  }
  const auto &metrics = get_binary_metric_map<Out, Idx>();
  const auto found = metrics.find(metric);
  return found == metrics.end() ? nullptr : found->second;
}

template <typename In = RNN_DEFAULT_IN, typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_query_distance(const Rcpp::LogicalMatrix &reference,
                      const Rcpp::LogicalMatrix &query,
                      const std::string &metric) {
  using Out = RNN_DEFAULT_DIST;
  const auto ndim = reference.nrow();
  const auto evaluate = get_binary_query_distance_func<Out, Idx>(metric);

  if (metric == "alternative-jaccard") {
    auto ref_bits = pack_logical_matrix(reference);
    auto query_bits = pack_logical_matrix(query);
    return std::make_unique<tdoann::BinaryQueryDistanceCalculator<Out, Idx>>(
        std::move(ref_bits), std::move(query_bits), reference.ncol(),
        query.ncol(), ndim, evaluate);
  }

  if (evaluate) {
    std::vector<uint8_t> ref_bvec = r_to_binvec(reference);
    std::vector<uint8_t> query_bvec = r_to_binvec(query);
    return std::make_unique<tdoann::BinaryQueryDistanceCalculator<Out, Idx>>(
        std::move(ref_bvec), std::move(query_bvec), ndim, evaluate);
  }

  auto ref_vec = r_to_vec<In>(reference);
  auto query_vec = r_to_vec<In>(query);
  return create_query_distance_impl<tdoann::BaseDistance<Out, Idx>>(
      std::move(ref_vec), std::move(query_vec), ndim, metric);
}

using PreparedQuery =
    std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, RNN_DEFAULT_IDX>>;

inline SEXP prepared_query_tag() {
  return Rf_install("rnndescent:prepared_query");
}

inline PreparedQuery &prepared_query_slot(SEXP handle) {
  if (TYPEOF(handle) != EXTPTRSXP ||
      R_ExternalPtrTag(handle) != prepared_query_tag()) {
    Rcpp::stop("Invalid prepared query handle");
  }
  return *Rcpp::XPtr<PreparedQuery>(handle);
}

inline PreparedQuery take_prepared_query(SEXP handle) {
  return handle == R_NilValue ? nullptr
                              : std::move(prepared_query_slot(handle));
}

template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>
create_query_vector_distance(const Rcpp::NumericMatrix &reference,
                             const Rcpp::NumericMatrix &query,
                             const std::string &metric,
                             bool check_reference_during_copy = false) {
  return create_query_distance_impl<
      tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>(
      reference, query, metric, check_reference_during_copy);
}

template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>
create_query_vector_distance(const Rcpp::LogicalMatrix &reference,
                             const Rcpp::LogicalMatrix &query,
                             const std::string &metric) {
  using In = RNN_DEFAULT_IN;

  const auto ndim = reference.nrow();
  auto ref_vec = r_to_vec<In>(reference);
  auto query_vec = r_to_vec<In>(query);

  return create_query_distance_impl<
      tdoann::VectorDistance<In, RNN_DEFAULT_DIST, Idx>>(
      std::move(ref_vec), std::move(query_vec), ndim, metric);
}

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_self_distance_impl(
    std::vector<typename FactoryTraits<Args...>::input_type> data_vec,
    std::size_t ndim, const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;
  using Out = typename FactoryTraits<Args...>::output_type;
  using Idx = typename FactoryTraits<Args...>::index_type;

  validate_haversine_ndim(ndim, metric);
  validate_haversine_radians(data_vec, ndim, metric);

  auto [distance_func, preprocess_func] =
      get_dense_distance_funcs<In, Out>(metric);
  return std::make_unique<tdoann::SelfDistanceCalculator<In, Out, Idx>>(
      std::move(data_vec), ndim, distance_func, preprocess_func);
}

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_self_distance_impl(const Rcpp::NumericMatrix &data,
                          const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;

  const auto ndim = data.nrow();
  auto data_vec = is_probability_metric(metric)
                      ? r_to_vec_for_probability<In>(data, metric)
                      : r_to_vec<In>(data);

  return create_self_distance_impl<Args...>(std::move(data_vec), ndim, metric);
}

// Factory function to return a BaseDistance
template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_self_distance(const Rcpp::NumericMatrix &data,
                     const std::string &metric) {
  using Out = RNN_DEFAULT_DIST;
  // potential specializations here
  return create_self_distance_impl<tdoann::BaseDistance<Out, Idx>>(data,
                                                                   metric);
}

template <typename In = RNN_DEFAULT_IN, typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_self_distance(const Rcpp::LogicalMatrix &data,
                     const std::string &metric) {
  using Out = RNN_DEFAULT_DIST;
  const auto ndim = data.nrow();

  if (metric == "alternative-jaccard") {
    return std::make_unique<tdoann::BinarySelfDistanceCalculator<Out, Idx>>(
        pack_logical_matrix(data), data.ncol(), ndim,
        tdoann::balternative_jaccard<Out, Idx>);
  }

  const auto &metric_map = get_binary_metric_map<Out, Idx>();
  if (metric_map.count(metric) > 0) {
    std::vector<uint8_t> data_bvec = r_to_binvec(data);
    return std::make_unique<tdoann::BinarySelfDistanceCalculator<Out, Idx>>(
        std::move(data_bvec), ndim, metric_map.at(metric));
  }

  auto data_vec = r_to_vec<In>(data);
  return create_self_distance_impl<tdoann::BaseDistance<Out, Idx>>(
      std::move(data_vec), ndim, metric);
}

template <typename In = RNN_DEFAULT_IN, typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::SelfDistanceCalculator<In, RNN_DEFAULT_DIST, Idx>>
create_self_distance(std::vector<In> data_vec, std::size_t ndim,
                     const std::string &metric) {
  return create_self_distance_impl<
      tdoann::SelfDistanceCalculator<In, RNN_DEFAULT_DIST, Idx>>(
      std::move(data_vec), ndim, metric);
}

template <typename In = RNN_DEFAULT_IN, typename Idx = RNN_DEFAULT_IDX>
tdoann::SelfDistanceCalculator<In, RNN_DEFAULT_DIST, Idx>
create_self_distance_calculator(std::vector<In> data_vec, std::size_t ndim,
                                const std::string &metric) {
  using Out = RNN_DEFAULT_DIST;

  validate_haversine_ndim(ndim, metric);
  validate_haversine_radians(data_vec, ndim, metric);

  auto [distance_func, preprocess_func] =
      get_dense_distance_funcs<In, Out>(metric);
  return tdoann::SelfDistanceCalculator<In, Out, Idx>(
      std::move(data_vec), ndim, distance_func, preprocess_func);
}

// Sparse distances

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_sparse_query_distance_impl(
    std::vector<std::size_t> ref_ind, std::vector<std::size_t> ref_ptr,
    std::vector<typename FactoryTraits<Args...>::input_type> ref_data,
    std::vector<std::size_t> query_ind, std::vector<std::size_t> query_ptr,
    std::vector<typename FactoryTraits<Args...>::input_type> query_data,
    std::size_t ndim, const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;
  using Out = typename FactoryTraits<Args...>::output_type;
  using Idx = typename FactoryTraits<Args...>::index_type;

  auto [distance_func, preprocess_func] =
      get_sparse_distance_funcs<In, Out>(metric);

  return std::make_unique<tdoann::SparseQueryDistanceCalculator<In, Out, Idx>>(
      std::move(ref_ind), std::move(ref_ptr), std::move(ref_data),
      std::move(query_ind), std::move(query_ptr), std::move(query_data), ndim,
      distance_func, preprocess_func);
}

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_sparse_query_distance_impl(const Rcpp::IntegerVector &ref_ind,
                                  const Rcpp::IntegerVector &ref_ptr,
                                  const Rcpp::NumericVector &ref_data,
                                  const Rcpp::IntegerVector &query_ind,
                                  const Rcpp::IntegerVector &query_ptr,
                                  const Rcpp::NumericVector &query_data,
                                  std::size_t ndim, const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;

  const bool probability_metric = is_probability_metric(metric);
  auto ref_ind_cpp = r_to_vec<std::size_t>(ref_ind);
  auto ref_ptr_cpp = r_to_vec<std::size_t>(ref_ptr);
  auto ref_data_cpp =
      probability_metric
          ? r_to_vec_for_probability<In>(ref_data, metric, "reference")
          : r_to_vec<In>(ref_data);

  auto query_ind_cpp = r_to_vec<std::size_t>(query_ind);
  auto query_ptr_cpp = r_to_vec<std::size_t>(query_ptr);
  auto query_data_cpp =
      probability_metric
          ? r_to_vec_for_probability<In>(query_data, metric, "query")
          : r_to_vec<In>(query_data);

  return create_sparse_query_distance_impl<
      typename FactoryTraits<Args...>::type>(
      std::move(ref_ind_cpp), std::move(ref_ptr_cpp), std::move(ref_data_cpp),
      std::move(query_ind_cpp), std::move(query_ptr_cpp),
      std::move(query_data_cpp), ndim, metric);
}

// Factory function to return a BaseDistance
template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_sparse_query_distance(const Rcpp::IntegerVector &ref_ind,
                             const Rcpp::IntegerVector &ref_ptr,
                             const Rcpp::NumericVector &ref_data,
                             const Rcpp::IntegerVector &query_ind,
                             const Rcpp::IntegerVector &query_ptr,
                             const Rcpp::NumericVector &query_data,
                             std::size_t ndim, const std::string &metric) {
  return create_sparse_query_distance_impl<
      tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>(ref_ind, ref_ptr, ref_data,
                                                   query_ind, query_ptr,
                                                   query_data, ndim, metric);
}

// Factory function to return a sparse VectorDistance
template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<
    tdoann::SparseVectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>
create_sparse_query_vector_distance(
    const Rcpp::IntegerVector &ref_ind, const Rcpp::IntegerVector &ref_ptr,
    const Rcpp::NumericVector &ref_data, const Rcpp::IntegerVector &query_ind,
    const Rcpp::IntegerVector &query_ptr, const Rcpp::NumericVector &query_data,
    std::size_t ndim, const std::string &metric) {

  return create_sparse_query_distance_impl<
      tdoann::SparseVectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>(
      ref_ind, ref_ptr, ref_data, query_ind, query_ptr, query_data, ndim,
      metric);
}

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_sparse_self_distance_impl(
    std::vector<std::size_t> ind_vec, std::vector<std::size_t> ptr_vec,
    std::vector<typename FactoryTraits<Args...>::input_type> data_vec,
    std::size_t ndim, const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;
  using Out = typename FactoryTraits<Args...>::output_type;
  using Idx = typename FactoryTraits<Args...>::index_type;

  auto [distance_func, preprocess_func] =
      get_sparse_distance_funcs<In, Out>(metric);

  return std::make_unique<tdoann::SparseSelfDistanceCalculator<In, Out, Idx>>(
      std::move(ind_vec), std::move(ptr_vec), std::move(data_vec), ndim,
      distance_func, preprocess_func);
}

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_sparse_self_distance_impl(const Rcpp::IntegerVector &ind,
                                 const Rcpp::IntegerVector &ptr,
                                 const Rcpp::NumericVector &data,
                                 std::size_t ndim, const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;

  auto ind_vec = r_to_vec<std::size_t>(ind);
  auto ptr_vec = r_to_vec<std::size_t>(ptr);
  auto data_vec = is_probability_metric(metric)
                      ? r_to_vec_for_probability<In>(data, metric)
                      : r_to_vec<In>(data);

  return create_sparse_self_distance_impl<
      typename FactoryTraits<Args...>::type>(std::move(ind_vec),
                                             std::move(ptr_vec),
                                             std::move(data_vec), ndim, metric);
}

// Factory function to return a BaseDistance
template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_sparse_self_distance(const Rcpp::IntegerVector &ind,
                            const Rcpp::IntegerVector &ptr,
                            const Rcpp::NumericVector &data, std::size_t ndim,
                            const std::string &metric) {
  // handle special case if needed here
  return create_sparse_self_distance_impl<
      tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>(ind, ptr, data, ndim,
                                                   metric);
}

template <typename In = RNN_DEFAULT_DIST, typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_sparse_self_distance(std::vector<std::size_t> ind_vec,
                            std::vector<std::size_t> ptr_vec,
                            std::vector<In> data_vec, std::size_t ndim,
                            const std::string &metric) {
  return create_sparse_self_distance_impl<
      tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>(
      std::move(ind_vec), std::move(ptr_vec), std::move(data_vec), ndim,
      metric);
}

#endif // RNN_DISTANCE_H
