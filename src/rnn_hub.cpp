
#include <algorithm>
#include <vector>

#include "tdoann/hub.h"

#include <Rcpp.h>

#include "rnn_heaptor.h"
#include "rnn_parallel.h"
#include "rnn_rtoheap.h"

using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector reverse_nbr_size_impl(IntegerMatrix nn_idx, std::size_t k,
                                    std::size_t len,
                                    bool include_self = false) {
  const std::size_t nr = nn_idx.nrow();
  const std::size_t nc = nn_idx.ncol();

  if (nc < k) {
    stop("Not enough columns in index matrix");
  }

  auto data = as<std::vector<std::size_t>>(nn_idx);

  std::vector<std::size_t> n_reverse(len);

  for (std::size_t i = 0; i < nr; i++) {
    for (std::size_t j = 0; j < k; j++) {
      std::size_t inbr = data[nr * j + i] - 1;
      if (inbr == static_cast<std::size_t>(-1)) {
        continue;
      }
      if (inbr == i && !include_self) {
        continue;
      }
      ++n_reverse[inbr];
    }
  }
  return IntegerVector(n_reverse.begin(), n_reverse.end());
}

// [[Rcpp::export]]
List reverse_knn_impl(IntegerMatrix idx, NumericMatrix dist,
                      std::size_t n_neighbors) {

  auto nn_heap =
      r_to_heap_missing_ok<tdoann::HeapAddSymmetric,
                           tdoann::NNHeap<float, std::size_t>>(idx, dist);

  auto reversed = tdoann::reverse_heap(nn_heap, n_neighbors, nn_heap.n_nbrs);

  return heap_to_r(reversed);
}

// [[Rcpp::export]]
List deg_adj_graph_impl(IntegerMatrix idx, NumericMatrix dist,
                        std::size_t n_rev_nbrs, std::size_t n_adj_nbrs) {
  auto nn_heap =
      r_to_heap_missing_ok<tdoann::HeapAddSymmetric,
                           tdoann::NNHeap<float, std::size_t>>(idx, dist);
  auto kog = tdoann::deg_adj_graph(nn_heap, n_rev_nbrs, n_adj_nbrs);
  return heap_to_r(kog);
}

// [[Rcpp::export]]
List ko_adj_graph_impl(IntegerMatrix idx, NumericMatrix dist,
                       std::size_t n_rev_nbrs, std::size_t n_adj_nbrs) {
  auto nn_heap =
      r_to_heap_missing_ok<tdoann::HeapAddSymmetric,
                           tdoann::NNHeap<float, std::size_t>>(idx, dist);
  auto dag = tdoann::ko_adj_graph(nn_heap, n_rev_nbrs, n_adj_nbrs);
  return heap_to_r(dag);
}
