
#include <algorithm>
#include <vector>

#include "connected_components.h"
#include "tdoann/hub.h"

#include <Rcpp.h>

#include "rnn_distance.h"
#include "rnn_heaptor.h"
#include "rnn_macros.h"
#include "rnn_parallel.h"
#include "rnn_rng.h"
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

// [[Rcpp::export]]
List mutualize_graph_impl(IntegerMatrix idx, NumericMatrix dist,
                          std::size_t n_nbrs) {
  auto nn_heap =
      r_to_heap_missing_ok<tdoann::HeapAddSymmetric,
                           tdoann::NNHeap<float, std::size_t>>(idx, dist);
  auto mheap = tdoann::mutualize_heap(nn_heap, n_nbrs);
  return heap_to_r(mheap);
}

// [[Rcpp::export]]
List partial_mutualize_graph_impl(IntegerMatrix idx, NumericMatrix dist,
                                  std::size_t n_nbrs) {
  auto nn_heap =
      r_to_heap_missing_ok<tdoann::HeapAddSymmetric,
                           tdoann::NNHeap<float, std::size_t>>(idx, dist);
  auto pmheap = tdoann::partial_mutualize_heap(nn_heap, n_nbrs);
  return heap_to_r(pmheap);
}

// [[Rcpp::export]]
List connected_components_undirected(std::size_t N,
                                     const std::vector<int> &indices1,
                                     const std::vector<int> &indptr1,
                                     const std::vector<int> &indices2,
                                     const std::vector<int> &indptr2) {

  std::pair<unsigned int, std::vector<int>> result =
      tdoann::connected_components_undirected(N, indices1, indptr1, indices2,
                                              indptr2);

  return List::create(_["n_components"] = result.first,
                      _["labels"] = result.second);
}

#define DIVERSIFY_IMPL()                                                       \
  return diversify_impl<Distance>(data, idx, dist, prune_probability);

#define DIVERSIFY_SP_IMPL()                                                    \
  return diversify_sp_impl<Distance>(data, graph_list, prune_probability);

#define DIVERSIFY_ALWAYS_SP_IMPL()                                             \
  return diversify_sp_impl<Distance>(data, graph_list);

template <typename Distance>
List diversify_impl(NumericMatrix data, IntegerMatrix idx, NumericMatrix dist,
                    double prune_probability) {
  auto distance = r_to_dist<Distance>(data);
  auto graph = r_to_graph<Distance>(idx, dist);

  RRand rand;
  auto diversified =
      tdoann::remove_long_edges(graph, distance, rand, prune_probability);

  return graph_to_r(diversified, true);
}

template <typename Distance>
List diversify_sp_impl(NumericMatrix data, List graph_list,
                       double prune_probability) {
  auto distance = r_to_dist<Distance>(data);
  auto graph = r_to_sparse_graph<Distance>(graph_list);

  RRand rand;
  auto diversified =
      tdoann::remove_long_edges_sp(graph, distance, rand, prune_probability);

  return sparse_graph_to_r(diversified);
}

template <typename Distance>
List diversify_sp_impl(NumericMatrix data, List graph_list) {
  auto distance = r_to_dist<Distance>(data);
  auto graph = r_to_sparse_graph<Distance>(graph_list);

  auto diversified = tdoann::remove_long_edges_sp(graph, distance);

  return sparse_graph_to_r(diversified);
}

// [[Rcpp::export]]
List diversify_cpp(NumericMatrix data, IntegerMatrix idx, NumericMatrix dist,
                   const std::string &metric = "euclidean",
                   double prune_probability = 1.0){
    DISPATCH_ON_DISTANCES(DIVERSIFY_IMPL)}

// [[Rcpp::export]]
List diversify_sp_cpp(NumericMatrix data, List graph_list,
                      const std::string &metric = "euclidean",
                      double prune_probability = 1.0){
    DISPATCH_ON_DISTANCES(DIVERSIFY_SP_IMPL)}

// [[Rcpp::export]]
List diversify_always_sp_cpp(NumericMatrix data, List graph_list,
                             const std::string &metric = "euclidean"){
    DISPATCH_ON_DISTANCES(DIVERSIFY_ALWAYS_SP_IMPL)}


struct Dummy {
  using Output = double;
  using Index = std::size_t;
};

// [[Rcpp::export]]
List r2spg(Rcpp::IntegerMatrix idx, Rcpp::NumericMatrix dist) {
  auto spg = r_to_sparse_graph<Dummy>(idx, dist);

  return List::create(_("p") = spg.row_ptr, _("j") = spg.col_idx,
                      _("x") = spg.dist);
}

// [[Rcpp::export]]
List merge_graph_lists_cpp(Rcpp::List gl1, Rcpp::List gl2) {
  auto g1 = r_to_sparse_graph<Dummy>(gl1);
  auto g2 = r_to_sparse_graph<Dummy>(gl2);

  auto g_merge = tdoann::merge_graphs(g1, g2);

  return sparse_graph_to_r(g_merge);
}

// [[Rcpp::export]]
List degree_prune_cpp(Rcpp::List graph_list, std::size_t max_degree) {
  auto graph = r_to_sparse_graph<Dummy>(graph_list);
  auto pruned = tdoann::degree_prune(graph, max_degree);
  return sparse_graph_to_r(pruned);
}
