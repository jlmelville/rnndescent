//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_PREPARED_INDEX_H
#define RNN_PREPARED_INDEX_H

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <Rcpp.h>

#include "rnn_distance.h"
#include "tdoann/nngraph.h"
#include "tdoann/rptree.h"
#include "tdoann/rptreeimplicit.h"
#include "tdoann/rptreesparse.h"

namespace rnndescent {

template <typename In>
void rank_dense_observations(std::vector<In> &data, std::size_t dimension) {
  for (auto begin = data.begin(); begin != data.end(); begin += dimension) {
    const auto ranks = tdoann::rankdata<In>(begin, begin + dimension);
    std::copy(ranks.cbegin(), ranks.cend(), begin);
  }
}

struct PreparedDenseIndex {
  using In = RNN_DEFAULT_IN;
  using Out = RNN_DEFAULT_DIST;
  using Idx = RNN_DEFAULT_IDX;
  using Graph = tdoann::SparseNNGraph<Out, Idx>;
  using Tree = tdoann::SearchTree<In, Idx>;

  std::vector<In> reference;
  Graph graph;
  std::vector<Tree> forest;
  std::size_t ndim;
  std::string metric;
  tdoann::DistanceFunc<In, Out> calculate;
  tdoann::PreprocessFunc<In> preprocess;
};

inline auto prepared_index_tag() -> SEXP {
  return Rf_install("rnndescent:prepared_dense_index");
}

inline auto prepared_sparse_index_tag() -> SEXP {
  return Rf_install("rnndescent:prepared_sparse_index");
}

inline auto prepared_logical_index_tag() -> SEXP {
  return Rf_install("rnndescent:prepared_logical_index");
}

template <typename Index>
auto prepared_index(SEXP handle, SEXP tag) -> const Index & {
  if (TYPEOF(handle) != EXTPTRSXP || R_ExternalPtrTag(handle) != tag ||
      R_ExternalPtrAddr(handle) == nullptr) {
    Rcpp::stop("Prepared index is invalid. Call rnnd_prepare() on the original "
               "index in this R session");
  }
  return *Rcpp::XPtr<Index>(handle);
}

struct PreparedSparseIndex {
  using In = RNN_DEFAULT_IN;
  using Out = RNN_DEFAULT_DIST;
  using Idx = RNN_DEFAULT_IDX;
  using Distance = tdoann::SparseVectorDistance<In, Out, Idx>;
  using Graph = tdoann::SparseNNGraph<Out, Idx>;
  using Forest = std::variant<std::vector<tdoann::SparseSearchTree<In, Idx>>,
                              std::vector<tdoann::SearchTreeImplicit<Idx>>>;

  std::unique_ptr<Distance> reference;
  Graph graph;
  Forest forest;
  std::size_t ndim;
  std::string metric;
  tdoann::SparseDistanceFunc<In, Out> calculate;
};

struct PreparedLogicalIndex {
  using Out = RNN_DEFAULT_DIST;
  using Idx = RNN_DEFAULT_IDX;
  using Graph = tdoann::SparseNNGraph<Out, Idx>;

  tdoann::BitVec reference;
  Graph graph;
  std::vector<tdoann::SearchTreeImplicit<Idx>> forest;
  std::size_t ndim;
  tdoann::BinaryDistanceFunc<Out, Idx> calculate;
};

// Check the offsets and feature bounds before using them to traverse storage.
inline void check_prepared_sparse_storage(const Rcpp::IntegerVector &ind,
                                          const Rcpp::IntegerVector &ptr,
                                          R_xlen_t data_size, int n_features) {
  if (ptr.size() == 0 || ptr[0] != 0 || ptr[ptr.size() - 1] != data_size ||
      ind.size() != data_size || !std::is_sorted(ptr.begin(), ptr.end()) ||
      std::any_of(ind.begin(), ind.end(),
                  [n_features](int i) { return i < 0 || i >= n_features; })) {
    Rcpp::stop("Invalid sparse storage for prepared search");
  }
}

class PreparedIndexQuery final
    : public tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                    RNN_DEFAULT_IDX> {
  using In = PreparedDenseIndex::In;
  using Out = PreparedDenseIndex::Out;
  using Idx = PreparedDenseIndex::Idx;

  const PreparedDenseIndex &index;
  std::vector<In> query;
  std::vector<In> query_ranks;

public:
  PreparedIndexQuery(const PreparedDenseIndex &index,
                     const Rcpp::NumericMatrix &query_matrix)
      : index(index), query(is_probability_metric(index.metric)
                                ? r_to_vec_for_probability<In>(
                                      query_matrix, index.metric, "query")
                                : r_to_vec<In>(query_matrix)) {
    validate_haversine_radians(query, index.ndim, index.metric);
    if (index.preprocess) {
      index.preprocess(query, index.ndim);
    } else if (index.metric == "spearmanr") {
      // Explicit forest projections use the original query coordinates.
      query_ranks = query;
      rank_dense_observations(query_ranks, index.ndim);
    }
  }

  std::size_t get_nx() const override { return index.graph.n_points; }
  std::size_t get_ny() const override { return query.size() / index.ndim; }
  Iterator get_x(Idx row) const override {
    return index.reference.cbegin() + row * index.ndim;
  }
  Iterator get_y(Idx row) const override {
    return query.cbegin() + row * index.ndim;
  }
  Out calculate(const Idx &ref, const Idx &qry) const override {
    auto begin = get_x(ref);
    const auto values = query_ranks.empty()
                            ? get_y(qry)
                            : query_ranks.cbegin() + qry * index.ndim;
    return index.calculate(begin, begin + index.ndim, values);
  }
};

class PreparedSparseQuery final : public PreparedSparseIndex::Distance {
  const PreparedSparseIndex &index;
  std::unique_ptr<PreparedSparseIndex::Distance> query;

public:
  PreparedSparseQuery(const PreparedSparseIndex &index,
                      const Rcpp::IntegerVector &ind,
                      const Rcpp::IntegerVector &ptr,
                      const Rcpp::NumericVector &data)
      : index(index),
        query(create_sparse_self_distance_impl<PreparedSparseIndex::Distance>(
            ind, ptr, data, index.ndim, index.metric)) {}

  std::size_t get_nx() const override { return index.reference->get_nx(); }
  std::size_t get_ny() const override { return query->get_nx(); }
  SparseObs get_x(RNN_DEFAULT_IDX row) const override {
    return index.reference->get_x(row);
  }
  SparseObs get_y(RNN_DEFAULT_IDX row) const override {
    return query->get_x(row);
  }
  RNN_DEFAULT_DIST calculate(const RNN_DEFAULT_IDX &ref,
                             const RNN_DEFAULT_IDX &qry) const override {
    auto [ref_ind, ref_size, ref_data] = get_x(ref);
    auto [qry_ind, qry_size, qry_data] = get_y(qry);
    return index.calculate(ref_ind, ref_size, ref_data, qry_ind, qry_size,
                           qry_data, index.ndim);
  }
};

class PreparedLogicalQuery final
    : public tdoann::BaseDistance<PreparedLogicalIndex::Out,
                                  PreparedLogicalIndex::Idx> {
  using Out = PreparedLogicalIndex::Out;
  using Idx = PreparedLogicalIndex::Idx;

  const PreparedLogicalIndex &index;
  tdoann::BitVec query;
  std::size_t n_query;
  std::size_t vec_len;

public:
  PreparedLogicalQuery(const PreparedLogicalIndex &index,
                       const Rcpp::LogicalMatrix &query_matrix)
      : index(index), query(pack_logical_query_matrix(query_matrix)),
        n_query(query_matrix.ncol()),
        vec_len(tdoann::num_blocks_needed(index.ndim)) {}

  std::size_t get_nx() const override { return index.graph.n_points; }
  std::size_t get_ny() const override { return n_query; }
  Out calculate(const Idx &ref, const Idx &qry) const override {
    return index.calculate(index.reference, ref, query, qry, vec_len,
                           index.ndim);
  }
};

} // namespace rnndescent

#endif // RNN_PREPARED_INDEX_H
