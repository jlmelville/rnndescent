#' @useDynLib rnndescent, .registration = TRUE
# Suppress R CMD check note "Namespace in Imports field not imported from"
#' @importFrom dqrng dqset.seed
#' @importFrom Rcpp sourceCpp
#' @docType package
#' @name rnndescent-package
#' @keywords internal
#' @details
#'
#' The rnndescent package provides functions to create approximate nearest
#' neighbors using the Nearest Neighbor Descent (Dong and co-workers, 2010) and
#' Random Partition Tree (Dasgupta and Freund, 2008) methods. In comparison to
#' other packages, it offers more metrics and can be used with sparse matrices.
#' For querying new data, it uses graph diversification methods (Harwood and
#' Drummond, 2016) and back-tracking (Iwasakai and Miyazaki, 2018) to improve
#' the search performance. The package also provides functions to diagnose
#' hubness in nearest neighbor results (Radovanovic and co-workers, 2010).
#'
#' This library is based heavily on the 'PyNNDescent' Python library.
#'
#' General resources:
#'
#'   * Website for the 'rnndescent' package: <https://github.com/jlmelville/rnndescent>
#'   * Documentation for the 'rnndescent' package: <https://jlmelville.github.io/rnndescent/>
#'   * Website of the 'PyNNDescent' package: <https://github.com/lmcinnes/pynndescent>
#'
#' The following functions provide the main interface to the package, with
#' useful defaults:
#'
#'   * Find the approximate nearest neighbors: [rnnd_knn()]
#'   * Create a search index and query new neighbors: [rnnd_build()] and [rnnd_prepare()].
#'   * Query new neighbors (or refine an existing knn graph): [rnnd_query()].
#'
#' Some diagnostic and helper functions to help explore the the structure of the
#' graphs and how well the approximation is working:
#'
#'   * Find exact nearest neighbors: [brute_force_knn()], [brute_force_knn_query()].
#'   * Merging graphs: [merge_knn()].
#'   * Hubness: [k_occur()].
#'   * Overlap/accuracy of two neighbor graphs: [neighbor_overlap()].
#'
#' Some lower-level functions are also available if you want more control than
#' the `rnnd_*` functions provide:
#'
#'   * Find approximate nearest neighbors: [rpf_knn()], [nnd_knn()].
#'   * Generating random neighbors: [random_knn()], [random_knn_query()].
#'   * Building an index: [rpf_build()], [rpf_filter()].
#'   * Querying an index for new data: [rpf_knn_query()], [prepare_search_graph()],
#'      [graph_knn_query()].
#'
#' @references
#' Dasgupta, S., & Freund, Y. (2008, May).
#' Random projection trees and low dimensional manifolds.
#' In *Proceedings of the fortieth annual ACM symposium on Theory of computing*
#' (pp. 537-546).
#' \doi{10.1145/1374376.1374452}.
#'
#' Dong, W., Moses, C., & Li, K. (2011, March).
#' Efficient k-nearest neighbor graph construction for generic similarity measures.
#' In *Proceedings of the 20th international conference on World Wide Web*
#' (pp. 577-586).
#' ACM.
#' \doi{10.1145/1963405.1963487}.
#'
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
#' (pp. 5713-5722).
#'
#' Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010).
#' Hubs in space: Popular nearest neighbors in high-dimensional data.
#' *Journal of Machine Learning Research*, *11*, 2487-2531.
#' <https://www.jmlr.org/papers/v11/radovanovic10a.html>
#'
#' Iwasaki, M., & Miyazaki, D. (2018).
#' Optimization of indexing based on k-nearest neighbor graph for proximity search in high-dimensional data.
#' *arXiv preprint* *arXiv:1810.07355*.
#' <https://arxiv.org/abs/1810.07355>
"_PACKAGE"
