#' @useDynLib rnndescent, .registration = TRUE
# Suppress R CMD check note "Namespace in Imports field not imported from"
#' @importFrom dqrng dqset.seed
#' @docType package
#' @name rnndescent-package
#' @keywords internal
#' @details
#'
#' The rnndescent package provides functions to create approximate nearest
#' neighbors using the Nearest Neighbor Descent and Random Partition Tree
#' methods. In comparison to other packages, it offers more metrics and can be
#' used with sparse matrices. For querying new data, it uses graph
#' diversification methods to improve the search performance. The package also
#' provides functions to diagnose hubness in nearest neighbor results.
#'
#' General resources:
#'
#'   * Website for the rnndescent package: <https://github.com/jlmelville/rnndescent>
#'   * Website of the PyNNDescent package: <https://github.com/lmcinnes/pynndescent>
#'
#' Resources on specific topics:
#'
#'   * Create exact nearest neighbors: [brute_force_knn()], [brute_force_knn_query()]
#'   * Create approximate nearest neighbors: [rpf_knn()], [nnd_knn()]
#'   * Querying new data: [prepare_search_graph()], [graph_knn_query()]
#'   * Diagnostics and hubness: [k_occur()]
#'
#' @references
#' Dasgupta, S., & Freund, Y. (2008, May).
#' Random projection trees and low dimensional manifolds.
#' In *Proceedings of the fortieth annual ACM symposium on Theory of computing*
#' (pp. 537-546).
#' <https://doi.org/10.1145/1374376.1374452>.
#'
#' Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010).
#' Hubs in space: Popular nearest neighbors in high-dimensional data.
#' *Journal of Machine Learning Research*, *11*, 2487-2531.
#' <https://www.jmlr.org/papers/v11/radovanovic10a.html>
#'
#' Dong, W., Moses, C., & Li, K. (2011, March).
#' Efficient k-nearest neighbor graph construction for generic similarity measures.
#' In *Proceedings of the 20th international conference on World Wide Web*
#' (pp. 577-586).
#' ACM.
#' <https://doi.org/10.1145/1963405.1963487>.
#'
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
#' (pp. 5713-5722).
"_PACKAGE"
