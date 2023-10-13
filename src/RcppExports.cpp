// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true> &Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false> &Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// rnn_brute_force
List rnn_brute_force(const NumericMatrix &data, uint32_t nnbrs,
                     const std::string &metric, std::size_t n_threads,
                     bool verbose);
RcppExport SEXP _rnndescent_rnn_brute_force(SEXP dataSEXP, SEXP nnbrsSEXP,
                                            SEXP metricSEXP, SEXP n_threadsSEXP,
                                            SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type data(dataSEXP);
  Rcpp::traits::input_parameter<uint32_t>::type nnbrs(nnbrsSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen =
      Rcpp::wrap(rnn_brute_force(data, nnbrs, metric, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// rnn_brute_force_query
List rnn_brute_force_query(const NumericMatrix &reference,
                           const NumericMatrix &query, uint32_t nnbrs,
                           const std::string &metric, std::size_t n_threads,
                           bool verbose);
RcppExport SEXP _rnndescent_rnn_brute_force_query(
    SEXP referenceSEXP, SEXP querySEXP, SEXP nnbrsSEXP, SEXP metricSEXP,
    SEXP n_threadsSEXP, SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type reference(
      referenceSEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type query(querySEXP);
  Rcpp::traits::input_parameter<uint32_t>::type nnbrs(nnbrsSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen = Rcpp::wrap(rnn_brute_force_query(
      reference, query, nnbrs, metric, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// euclidean_distance
double euclidean_distance(NumericVector x, NumericVector y);
RcppExport SEXP _rnndescent_euclidean_distance(SEXP xSEXP, SEXP ySEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<NumericVector>::type x(xSEXP);
  Rcpp::traits::input_parameter<NumericVector>::type y(ySEXP);
  rcpp_result_gen = Rcpp::wrap(euclidean_distance(x, y));
  return rcpp_result_gen;
  END_RCPP
}
// l2sqr_distance
double l2sqr_distance(NumericVector x, NumericVector y);
RcppExport SEXP _rnndescent_l2sqr_distance(SEXP xSEXP, SEXP ySEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<NumericVector>::type x(xSEXP);
  Rcpp::traits::input_parameter<NumericVector>::type y(ySEXP);
  rcpp_result_gen = Rcpp::wrap(l2sqr_distance(x, y));
  return rcpp_result_gen;
  END_RCPP
}
// cosine_distance
double cosine_distance(NumericVector x, NumericVector y);
RcppExport SEXP _rnndescent_cosine_distance(SEXP xSEXP, SEXP ySEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<NumericVector>::type x(xSEXP);
  Rcpp::traits::input_parameter<NumericVector>::type y(ySEXP);
  rcpp_result_gen = Rcpp::wrap(cosine_distance(x, y));
  return rcpp_result_gen;
  END_RCPP
}
// manhattan_distance
double manhattan_distance(NumericVector x, NumericVector y);
RcppExport SEXP _rnndescent_manhattan_distance(SEXP xSEXP, SEXP ySEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<NumericVector>::type x(xSEXP);
  Rcpp::traits::input_parameter<NumericVector>::type y(ySEXP);
  rcpp_result_gen = Rcpp::wrap(manhattan_distance(x, y));
  return rcpp_result_gen;
  END_RCPP
}
// hamming_distance
double hamming_distance(IntegerVector x, IntegerVector y);
RcppExport SEXP _rnndescent_hamming_distance(SEXP xSEXP, SEXP ySEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<IntegerVector>::type x(xSEXP);
  Rcpp::traits::input_parameter<IntegerVector>::type y(ySEXP);
  rcpp_result_gen = Rcpp::wrap(hamming_distance(x, y));
  return rcpp_result_gen;
  END_RCPP
}
// correlation_distance
double correlation_distance(NumericVector x, NumericVector y);
RcppExport SEXP _rnndescent_correlation_distance(SEXP xSEXP, SEXP ySEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<NumericVector>::type x(xSEXP);
  Rcpp::traits::input_parameter<NumericVector>::type y(ySEXP);
  rcpp_result_gen = Rcpp::wrap(correlation_distance(x, y));
  return rcpp_result_gen;
  END_RCPP
}
// reverse_nbr_size_impl
IntegerVector reverse_nbr_size_impl(const IntegerMatrix &nn_idx,
                                    std::size_t nnbrs, std::size_t len,
                                    bool include_self);
RcppExport SEXP _rnndescent_reverse_nbr_size_impl(SEXP nn_idxSEXP,
                                                  SEXP nnbrsSEXP, SEXP lenSEXP,
                                                  SEXP include_selfSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type nn_idx(nn_idxSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type nnbrs(nnbrsSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type len(lenSEXP);
  Rcpp::traits::input_parameter<bool>::type include_self(include_selfSEXP);
  rcpp_result_gen =
      Rcpp::wrap(reverse_nbr_size_impl(nn_idx, nnbrs, len, include_self));
  return rcpp_result_gen;
  END_RCPP
}
// shared_nbr_size_impl
IntegerVector shared_nbr_size_impl(IntegerMatrix nn_idx,
                                   std::size_t n_neighbors);
RcppExport SEXP _rnndescent_shared_nbr_size_impl(SEXP nn_idxSEXP,
                                                 SEXP n_neighborsSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<IntegerMatrix>::type nn_idx(nn_idxSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_neighbors(n_neighborsSEXP);
  rcpp_result_gen = Rcpp::wrap(shared_nbr_size_impl(nn_idx, n_neighbors));
  return rcpp_result_gen;
  END_RCPP
}
// rnn_idx_to_graph_self
List rnn_idx_to_graph_self(const NumericMatrix &data, const IntegerMatrix &idx,
                           const std::string &metric, std::size_t n_threads,
                           bool verbose);
RcppExport SEXP _rnndescent_rnn_idx_to_graph_self(SEXP dataSEXP, SEXP idxSEXP,
                                                  SEXP metricSEXP,
                                                  SEXP n_threadsSEXP,
                                                  SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type data(dataSEXP);
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type idx(idxSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen =
      Rcpp::wrap(rnn_idx_to_graph_self(data, idx, metric, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// rnn_idx_to_graph_query
List rnn_idx_to_graph_query(const NumericMatrix &reference,
                            const NumericMatrix &query,
                            const IntegerMatrix &idx, const std::string &metric,
                            std::size_t n_threads, bool verbose);
RcppExport SEXP _rnndescent_rnn_idx_to_graph_query(SEXP referenceSEXP,
                                                   SEXP querySEXP, SEXP idxSEXP,
                                                   SEXP metricSEXP,
                                                   SEXP n_threadsSEXP,
                                                   SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type reference(
      referenceSEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type query(querySEXP);
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type idx(idxSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen = Rcpp::wrap(rnn_idx_to_graph_query(
      reference, query, idx, metric, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// local_scaled_nbrs
List local_scaled_nbrs(const IntegerMatrix &idx, const NumericMatrix &dist,
                       std::size_t n_scaled_nbrs, std::size_t k_begin,
                       std::size_t k_end, bool ret_scales,
                       std::size_t n_threads);
RcppExport SEXP _rnndescent_local_scaled_nbrs(SEXP idxSEXP, SEXP distSEXP,
                                              SEXP n_scaled_nbrsSEXP,
                                              SEXP k_beginSEXP, SEXP k_endSEXP,
                                              SEXP ret_scalesSEXP,
                                              SEXP n_threadsSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type idx(idxSEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type dist(distSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_scaled_nbrs(
      n_scaled_nbrsSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type k_begin(k_beginSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type k_end(k_endSEXP);
  Rcpp::traits::input_parameter<bool>::type ret_scales(ret_scalesSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  rcpp_result_gen = Rcpp::wrap(local_scaled_nbrs(
      idx, dist, n_scaled_nbrs, k_begin, k_end, ret_scales, n_threads));
  return rcpp_result_gen;
  END_RCPP
}
// merge_nn
List merge_nn(const IntegerMatrix &nn_idx1, const NumericMatrix &nn_dist1,
              const IntegerMatrix &nn_idx2, const NumericMatrix &nn_dist2,
              bool is_query, std::size_t n_threads, bool verbose);
RcppExport SEXP _rnndescent_merge_nn(SEXP nn_idx1SEXP, SEXP nn_dist1SEXP,
                                     SEXP nn_idx2SEXP, SEXP nn_dist2SEXP,
                                     SEXP is_querySEXP, SEXP n_threadsSEXP,
                                     SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type nn_idx1(
      nn_idx1SEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type nn_dist1(
      nn_dist1SEXP);
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type nn_idx2(
      nn_idx2SEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type nn_dist2(
      nn_dist2SEXP);
  Rcpp::traits::input_parameter<bool>::type is_query(is_querySEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen = Rcpp::wrap(merge_nn(nn_idx1, nn_dist1, nn_idx2, nn_dist2,
                                        is_query, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// merge_nn_all
List merge_nn_all(const List &nn_graphs, bool is_query, std::size_t n_threads,
                  bool verbose);
RcppExport SEXP _rnndescent_merge_nn_all(SEXP nn_graphsSEXP, SEXP is_querySEXP,
                                         SEXP n_threadsSEXP, SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const List &>::type nn_graphs(nn_graphsSEXP);
  Rcpp::traits::input_parameter<bool>::type is_query(is_querySEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen =
      Rcpp::wrap(merge_nn_all(nn_graphs, is_query, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// nn_descent
List nn_descent(const NumericMatrix &data, const IntegerMatrix &nn_idx,
                const NumericMatrix &nn_dist, const std::string &metric,
                std::size_t max_candidates, std::size_t n_iters, double delta,
                bool low_memory, std::size_t n_threads, bool verbose,
                const std::string &progress_type);
RcppExport SEXP _rnndescent_nn_descent(SEXP dataSEXP, SEXP nn_idxSEXP,
                                       SEXP nn_distSEXP, SEXP metricSEXP,
                                       SEXP max_candidatesSEXP,
                                       SEXP n_itersSEXP, SEXP deltaSEXP,
                                       SEXP low_memorySEXP, SEXP n_threadsSEXP,
                                       SEXP verboseSEXP,
                                       SEXP progress_typeSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type data(dataSEXP);
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type nn_idx(nn_idxSEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type nn_dist(
      nn_distSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type max_candidates(
      max_candidatesSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_iters(n_itersSEXP);
  Rcpp::traits::input_parameter<double>::type delta(deltaSEXP);
  Rcpp::traits::input_parameter<bool>::type low_memory(low_memorySEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type progress_type(
      progress_typeSEXP);
  rcpp_result_gen = Rcpp::wrap(
      nn_descent(data, nn_idx, nn_dist, metric, max_candidates, n_iters, delta,
                 low_memory, n_threads, verbose, progress_type));
  return rcpp_result_gen;
  END_RCPP
}
// diversify_cpp
List diversify_cpp(const NumericMatrix &data, const List &graph_list,
                   const std::string &metric, double prune_probability,
                   std::size_t n_threads);
RcppExport SEXP _rnndescent_diversify_cpp(SEXP dataSEXP, SEXP graph_listSEXP,
                                          SEXP metricSEXP,
                                          SEXP prune_probabilitySEXP,
                                          SEXP n_threadsSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type data(dataSEXP);
  Rcpp::traits::input_parameter<const List &>::type graph_list(graph_listSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<double>::type prune_probability(
      prune_probabilitySEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  rcpp_result_gen = Rcpp::wrap(
      diversify_cpp(data, graph_list, metric, prune_probability, n_threads));
  return rcpp_result_gen;
  END_RCPP
}
// merge_graph_lists_cpp
List merge_graph_lists_cpp(const List &graph_list1, const List &graph_list2);
RcppExport SEXP _rnndescent_merge_graph_lists_cpp(SEXP graph_list1SEXP,
                                                  SEXP graph_list2SEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const List &>::type graph_list1(
      graph_list1SEXP);
  Rcpp::traits::input_parameter<const List &>::type graph_list2(
      graph_list2SEXP);
  rcpp_result_gen = Rcpp::wrap(merge_graph_lists_cpp(graph_list1, graph_list2));
  return rcpp_result_gen;
  END_RCPP
}
// degree_prune_cpp
List degree_prune_cpp(const List &graph_list, std::size_t max_degree,
                      std::size_t n_threads);
RcppExport SEXP _rnndescent_degree_prune_cpp(SEXP graph_listSEXP,
                                             SEXP max_degreeSEXP,
                                             SEXP n_threadsSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const List &>::type graph_list(graph_listSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type max_degree(max_degreeSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  rcpp_result_gen =
      Rcpp::wrap(degree_prune_cpp(graph_list, max_degree, n_threads));
  return rcpp_result_gen;
  END_RCPP
}
// random_knn_cpp
List random_knn_cpp(const NumericMatrix &data, uint32_t nnbrs,
                    const std::string &metric, bool order_by_distance,
                    std::size_t n_threads, bool verbose);
RcppExport SEXP _rnndescent_random_knn_cpp(SEXP dataSEXP, SEXP nnbrsSEXP,
                                           SEXP metricSEXP,
                                           SEXP order_by_distanceSEXP,
                                           SEXP n_threadsSEXP,
                                           SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type data(dataSEXP);
  Rcpp::traits::input_parameter<uint32_t>::type nnbrs(nnbrsSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<bool>::type order_by_distance(
      order_by_distanceSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen = Rcpp::wrap(random_knn_cpp(
      data, nnbrs, metric, order_by_distance, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// random_knn_query_cpp
List random_knn_query_cpp(const NumericMatrix &reference,
                          const NumericMatrix &query, uint32_t nnbrs,
                          const std::string &metric, bool order_by_distance,
                          std::size_t n_threads, bool verbose);
RcppExport SEXP _rnndescent_random_knn_query_cpp(
    SEXP referenceSEXP, SEXP querySEXP, SEXP nnbrsSEXP, SEXP metricSEXP,
    SEXP order_by_distanceSEXP, SEXP n_threadsSEXP, SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type reference(
      referenceSEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type query(querySEXP);
  Rcpp::traits::input_parameter<uint32_t>::type nnbrs(nnbrsSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<bool>::type order_by_distance(
      order_by_distanceSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen = Rcpp::wrap(random_knn_query_cpp(
      reference, query, nnbrs, metric, order_by_distance, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// nn_query
List nn_query(const NumericMatrix &reference, const List &reference_graph_list,
              const NumericMatrix &query, const IntegerMatrix &nn_idx,
              const NumericMatrix &nn_dist, const std::string &metric,
              double epsilon, std::size_t n_threads, bool verbose);
RcppExport SEXP _rnndescent_nn_query(SEXP referenceSEXP,
                                     SEXP reference_graph_listSEXP,
                                     SEXP querySEXP, SEXP nn_idxSEXP,
                                     SEXP nn_distSEXP, SEXP metricSEXP,
                                     SEXP epsilonSEXP, SEXP n_threadsSEXP,
                                     SEXP verboseSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const NumericMatrix &>::type reference(
      referenceSEXP);
  Rcpp::traits::input_parameter<const List &>::type reference_graph_list(
      reference_graph_listSEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type query(querySEXP);
  Rcpp::traits::input_parameter<const IntegerMatrix &>::type nn_idx(nn_idxSEXP);
  Rcpp::traits::input_parameter<const NumericMatrix &>::type nn_dist(
      nn_distSEXP);
  Rcpp::traits::input_parameter<const std::string &>::type metric(metricSEXP);
  Rcpp::traits::input_parameter<double>::type epsilon(epsilonSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
  rcpp_result_gen =
      Rcpp::wrap(nn_query(reference, reference_graph_list, query, nn_idx,
                          nn_dist, metric, epsilon, n_threads, verbose));
  return rcpp_result_gen;
  END_RCPP
}
// sort_graph
List sort_graph(const List &graph_list, std::size_t n_threads);
RcppExport SEXP _rnndescent_sort_graph(SEXP graph_listSEXP,
                                       SEXP n_threadsSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<const List &>::type graph_list(graph_listSEXP);
  Rcpp::traits::input_parameter<std::size_t>::type n_threads(n_threadsSEXP);
  rcpp_result_gen = Rcpp::wrap(sort_graph(graph_list, n_threads));
  return rcpp_result_gen;
  END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rnndescent_rnn_brute_force", (DL_FUNC)&_rnndescent_rnn_brute_force, 5},
    {"_rnndescent_rnn_brute_force_query",
     (DL_FUNC)&_rnndescent_rnn_brute_force_query, 6},
    {"_rnndescent_euclidean_distance", (DL_FUNC)&_rnndescent_euclidean_distance,
     2},
    {"_rnndescent_l2sqr_distance", (DL_FUNC)&_rnndescent_l2sqr_distance, 2},
    {"_rnndescent_cosine_distance", (DL_FUNC)&_rnndescent_cosine_distance, 2},
    {"_rnndescent_manhattan_distance", (DL_FUNC)&_rnndescent_manhattan_distance,
     2},
    {"_rnndescent_hamming_distance", (DL_FUNC)&_rnndescent_hamming_distance, 2},
    {"_rnndescent_correlation_distance",
     (DL_FUNC)&_rnndescent_correlation_distance, 2},
    {"_rnndescent_reverse_nbr_size_impl",
     (DL_FUNC)&_rnndescent_reverse_nbr_size_impl, 4},
    {"_rnndescent_shared_nbr_size_impl",
     (DL_FUNC)&_rnndescent_shared_nbr_size_impl, 2},
    {"_rnndescent_rnn_idx_to_graph_self",
     (DL_FUNC)&_rnndescent_rnn_idx_to_graph_self, 5},
    {"_rnndescent_rnn_idx_to_graph_query",
     (DL_FUNC)&_rnndescent_rnn_idx_to_graph_query, 6},
    {"_rnndescent_local_scaled_nbrs", (DL_FUNC)&_rnndescent_local_scaled_nbrs,
     7},
    {"_rnndescent_merge_nn", (DL_FUNC)&_rnndescent_merge_nn, 7},
    {"_rnndescent_merge_nn_all", (DL_FUNC)&_rnndescent_merge_nn_all, 4},
    {"_rnndescent_nn_descent", (DL_FUNC)&_rnndescent_nn_descent, 11},
    {"_rnndescent_diversify_cpp", (DL_FUNC)&_rnndescent_diversify_cpp, 5},
    {"_rnndescent_merge_graph_lists_cpp",
     (DL_FUNC)&_rnndescent_merge_graph_lists_cpp, 2},
    {"_rnndescent_degree_prune_cpp", (DL_FUNC)&_rnndescent_degree_prune_cpp, 3},
    {"_rnndescent_random_knn_cpp", (DL_FUNC)&_rnndescent_random_knn_cpp, 6},
    {"_rnndescent_random_knn_query_cpp",
     (DL_FUNC)&_rnndescent_random_knn_query_cpp, 7},
    {"_rnndescent_nn_query", (DL_FUNC)&_rnndescent_nn_query, 9},
    {"_rnndescent_sort_graph", (DL_FUNC)&_rnndescent_sort_graph, 2},
    {NULL, NULL, 0}};

RcppExport void R_init_rnndescent(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
