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

template <typename In>
const std::unordered_map<std::string, tdoann::SparsePreprocessFunc<In>> &
get_sparse_preprocess_map() {
  static const std::unordered_map<std::string, tdoann::SparsePreprocessFunc<In>>
      map = {{"dot", tdoann::sparse_normalize<In>},
             {"alternative-dot", tdoann::sparse_normalize<In>}};
  return map;
}

template <typename In, typename Out>
std::pair<tdoann::DistanceFunc<In, Out>, tdoann::PreprocessFunc<In>>
get_dense_distance_funcs(const std::string &metric) {
  const auto &metric_map = get_metric_map<In, Out>();
  if (metric_map.count(metric) == 0) {
    Rcpp::stop("Bad metric");
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
    Rcpp::stop("Bad metric");
  }
  auto distance_func = metric_map.at(metric);

  tdoann::SparsePreprocessFunc<In> preprocess_func = nullptr;
  const auto &preprocess_map = get_sparse_preprocess_map<In>();
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
struct FactoryTraits<tdoann::SparseVectorDistance<In, Out, Idx>> {
  using type = tdoann::SparseVectorDistance<In, Out, Idx>;
  using input_type = In;
  using output_type = Out;
  using index_type = Idx;
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

  auto [distance_func, preprocess_func] =
      get_dense_distance_funcs<In, Out>(metric);

  return std::make_unique<tdoann::QueryDistanceCalculator<In, Out, Idx>>(
      std::move(ref_vec), std::move(query_vec), ndim, distance_func,
      preprocess_func);
}

template <typename... Args>
std::unique_ptr<typename FactoryTraits<Args...>::type>
create_query_distance_impl(const Rcpp::NumericMatrix &reference,
                           const Rcpp::NumericMatrix &query,
                           const std::string &metric) {
  using In = typename FactoryTraits<Args...>::input_type;

  const auto ndim = reference.nrow();
  auto ref_vec = r_to_vec<In>(reference);
  auto query_vec = r_to_vec<In>(query);

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

template <typename In = RNN_DEFAULT_IN, typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::BaseDistance<RNN_DEFAULT_DIST, Idx>>
create_query_distance(const Rcpp::LogicalMatrix &reference,
                      const Rcpp::LogicalMatrix &query,
                      const std::string &metric) {
  using Out = RNN_DEFAULT_DIST;
  const auto ndim = reference.nrow();

  const auto &metric_map = get_binary_metric_map<Out, Idx>();
  if (metric_map.count(metric) > 0) {
    std::vector<uint8_t> ref_bvec = r_to_binvec(reference);
    std::vector<uint8_t> query_bvec = r_to_binvec(query);
    return std::make_unique<tdoann::BinaryQueryDistanceCalculator<Out, Idx>>(
        std::move(ref_bvec), std::move(query_bvec), ndim,
        metric_map.at(metric));
  }

  auto ref_vec = r_to_vec<In>(reference);
  auto query_vec = r_to_vec<In>(query);
  return create_query_distance_impl<tdoann::BaseDistance<Out, Idx>>(
      std::move(ref_vec), std::move(query_vec), ndim, metric);
}

template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>
create_query_vector_distance(const Rcpp::NumericMatrix &reference,
                             const Rcpp::NumericMatrix &query,
                             const std::string &metric) {
  return create_query_distance_impl<
      tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>(
      reference, query, metric);
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
  auto data_vec = r_to_vec<In>(data);

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
std::unique_ptr<tdoann::VectorDistance<In, RNN_DEFAULT_DIST, Idx>>
create_self_distance(std::vector<In> data_vec, std::size_t ndim,
                     const std::string &metric) {
  return create_self_distance_impl<
      tdoann::VectorDistance<In, RNN_DEFAULT_DIST, Idx>>(std::move(data_vec),
                                                         ndim, metric);
}

// Factory function to return a VectorDistance
template <typename Idx = RNN_DEFAULT_IDX>
std::unique_ptr<tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>
create_self_vector_distance(const Rcpp::NumericMatrix &data,
                            const std::string &metric) {
  return create_self_distance_impl<
      tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST, Idx>>(data,
                                                                     metric);
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

  auto ref_ind_cpp = r_to_vec<std::size_t>(ref_ind);
  auto ref_ptr_cpp = r_to_vec<std::size_t>(ref_ptr);
  auto ref_data_cpp = r_to_vec<In>(ref_data);

  auto query_ind_cpp = r_to_vec<std::size_t>(query_ind);
  auto query_ptr_cpp = r_to_vec<std::size_t>(query_ptr);
  auto query_data_cpp = r_to_vec<In>(query_data);

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
  auto data_vec = r_to_vec<In>(data);

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
