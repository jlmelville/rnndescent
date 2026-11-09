//  rnndescent -- An R package for nearest neighbor descent
//
//  Copyright (C) 2026 James Melville
//  This file is part of rnndescent and is distributed under the terms of the
//  GNU General Public License, version 3 or later.

#ifndef RNN_PREPARED_INDEX_H
#define RNN_PREPARED_INDEX_H

#include <cstddef>
#include <string>
#include <vector>

#include <Rcpp.h>

#include "rnn_distance.h"
#include "tdoann/nngraph.h"
#include "tdoann/rptree.h"

namespace rnndescent {

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
};

inline auto prepared_index_tag() -> SEXP {
  return Rf_install("rnndescent:prepared_dense_index");
}

inline auto prepared_dense_index(SEXP handle) -> const PreparedDenseIndex & {
  if (TYPEOF(handle) != EXTPTRSXP ||
      R_ExternalPtrTag(handle) != prepared_index_tag() ||
      R_ExternalPtrAddr(handle) == nullptr) {
    Rcpp::stop("Prepared index is invalid. Call rnnd_prepare() on the original "
               "index in this R session");
  }
  return *Rcpp::XPtr<PreparedDenseIndex>(handle);
}

class PreparedIndexQuery final
    : public tdoann::VectorDistance<RNN_DEFAULT_IN, RNN_DEFAULT_DIST,
                                    RNN_DEFAULT_IDX> {
  using In = PreparedDenseIndex::In;
  using Out = PreparedDenseIndex::Out;
  using Idx = PreparedDenseIndex::Idx;

  const PreparedDenseIndex &index;
  std::vector<In> query;

public:
  PreparedIndexQuery(const PreparedDenseIndex &index,
                     const Rcpp::NumericMatrix &query_matrix)
      : index(index), query(r_to_vec<In>(query_matrix)) {}

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
    return index.calculate(begin, begin + index.ndim, get_y(qry));
  }
};

} // namespace rnndescent

#endif // RNN_PREPARED_INDEX_H
