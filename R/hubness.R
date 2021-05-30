# Hubness -----------------------------------------------------------------

#' k-Occurrence
#'
#' Calculates the the k-occurrence for a given nearest neighbor matrix
#'
#' The k-occurrence of an object is the number of times it occurs among the
#' k-nearest neighbors of objects in a dataset. This can take values between 0
#' and the size of the dataset. The larger the k-occurrence for an object, the
#' more "popular" it is. Very large values of the k-occurrence (much larger than
#' k) indicates that an object is a "hub" and also implies the existence of
#' "anti-hubs": objects that never appear as k-nearest neighbors of other
#' objects.
#'
#' The presence of hubs can reduce the accuracy of nearest-neighbor descent
#' and other approximate nearest neighbor algorithms in terms of retrieving the
#' exact k-nearest neighbors. However the appearance of hubs can still be
#' detected in these approximate results, so calculating the k-occurrences for
#' the output of nearest neighbor descent is a useful diagnostic step.
#'
#' @param idx integer matrix containing the nearest neighbor indices, integers
#'   labeled starting at 1. Note that the integer labels do \emph{not} have to
#'   refer to the rows of \code{idx}, for example if the nearest neighbor result
#'   is from querying one set of objects with respect to another (for instance
#'   from running \code{\link{graph_knn_query}}). You may also pass a nearest
#'   neighbor graph object (e.g. the output of running \code{\link{nnd_knn}}),
#'   and the indices will be extracted from it.
#' @param k The number of closest neighbors to use. Must be between 1 and the
#'   number of columns in \code{idx}. By default, all columns of \code{idx} are
#'   used.
#' @param include_self logical indicating whether the label \code{i} in
#'   \code{idx} is considered to be a valid neighbor when found in row \code{i}.
#'   By default this is \code{TRUE}. This can be set to \code{FALSE} when the
#'   the labels in \code{idx} refer to the row indices of \code{idx}, as in the
#'   case of results from \code{\link{nnd_knn}}. In this case you may not want
#'   to consider the trivial case of an object being a neighbor of itself. In
#'   all other cases leave this set to \code{TRUE}.
#' @return a vector of length \code{max(idx)}, containing the number of times an
#'   object in \code{idx} was found in the nearest neighbor list of the objects
#'   represented by the row indices of \code{idx}.
#' @examples
#' iris_nbrs <- brute_force_knn(iris, k = 15)
#' iris_ko <- k_occur(iris_nbrs$idx)
#' # items 42 and 107 are not in 15 nearest neighbors of any other members of
#' # iris
#' # for convenience you can also pass iris_nbrs directly:
#' # iris_ko <- k_occur(iris_nbrs)
#' which(iris_ko == 1) # they are only their own nearest neighbor
#' max(iris_ko) # most "popular" item appears on 29 15-nearest neighbor lists
#' which(iris_ko == max(iris_ko)) # it's iris item 64
#' # with k = 15, a maximum k-occurrence = 29 ~= 1.9 * k, which is not a cause
#' # for concern
#' @references
#' Radovanovic, M., Nanopoulos, A., & Ivanovic, M. (2010).
#' Hubs in space: Popular nearest neighbors in high-dimensional data.
#' \emph{Journal of Machine Learning Research}, \emph{11}, 2487-2531.
#' \url{https://www.jmlr.org/papers/v11/radovanovic10a.html}
#'
#' Bratic, B., Houle, M. E., Kurbalija, V., Oria, V., & Radovanovic, M. (2019).
#' The Influence of Hubness on NN-Descent.
#' \emph{International Journal on Artificial Intelligence Tools}, \emph{28}(06), 1960002.
#' \url{https://doi.org/10.1142/S0218213019600029}
#' @export
k_occur <- function(idx,
                    k = NULL,
                    include_self = TRUE) {
  if (is.list(idx)) {
    idx <- idx$idx
  }
  stopifnot(methods::is(idx, "matrix"))
  if (is.null(k)) {
    k <- ncol(idx)
  }
  nc <- ncol(idx)
  stopifnot(k >= 1)
  stopifnot(k <= nc)
  len <- max(idx)
  reverse_nbr_size_impl(idx, k, len, include_self)
}

reverse_knn <- function(idx, dist = NULL, k = NULL) {
  cg_res <- check_graph(idx = idx, dist = dist, k = k)
  Matrix::t(Matrix::drop0(graph_to_sparse_r(cg_res)))
}

reverse_knn_sp <- function(graph) {
  stopifnot(methods::is(graph, "sparseMatrix"))
  Matrix::t(Matrix::drop0(graph))
}

mutualize_knn <- function(idx, dist = NULL, k = NULL) {
  cg_res <- check_graph(idx = idx, dist = dist, k = k)
  spg <- graph_to_cs(cg_res)
  rspg <- reverse_knn(cg_res)
  Matrix::drop0(spg + rspg) / 2
}

partial_mutualize_knn <- function(idx, dist = NULL, k = NULL) {
  cg_res <- check_graph(idx = idx, dist = dist, k = k)
  partial_mutualize_graph_impl(idx = cg_res$idx, dist = cg_res$dist, n_nbrs = cg_res$k)
}

ko_adj_graph <- function(idx, dist = NULL, rev_k = NULL, fwd_k = NULL) {
  if (is.null(dist) && is.list(idx)) {
    dist <- idx$dist
  }
  if (is.list(idx)) {
    idx <- idx$idx
  }
  if (is.null(rev_k)) {
    rev_k <- ncol(idx)
  }
  if (is.null(fwd_k)) {
    fwd_k <- ncol(idx)
  }

  stopifnot(methods::is(idx, "matrix"))
  stopifnot(methods::is(dist, "matrix"))
  stopifnot(dim(idx) == dim(dist))
  stopifnot(rev_k > 0)
  stopifnot(fwd_k > 0)

  ko_adj_graph_impl(idx = idx, dist = dist, n_rev_nbrs = rev_k, n_adj_nbrs = fwd_k)
}

deg_adj_graph <- function(idx, dist = NULL, rev_k = NULL, fwd_k = NULL) {
  if (is.null(dist) && is.list(idx)) {
    dist <- idx$dist
  }
  if (is.list(idx)) {
    idx <- idx$idx
  }
  if (is.null(rev_k)) {
    rev_k <- ncol(idx)
  }
  if (is.null(fwd_k)) {
    fwd_k <- ncol(idx)
  }

  stopifnot(methods::is(idx, "matrix"))
  stopifnot(methods::is(dist, "matrix"))
  stopifnot(dim(idx) == dim(dist))
  stopifnot(rev_k > 0)
  stopifnot(fwd_k > 0)

  deg_adj_graph_impl(idx = idx, dist = dist, n_rev_nbrs = rev_k, n_adj_nbrs = fwd_k)
}

unreachable <- function(idx, k = NULL) {
  if (is.list(idx)) {
    idx <- idx$idx
  }
  ko <- k_occur(idx, k = k, include_self = FALSE)
  sum(ko == 0) / length(ko)
}

reachable <- function(idx, k = NULL) {
  if (is.list(idx)) {
    idx <- idx$idx
  }
  ko <- k_occur(idx, k = k, include_self = FALSE)
  1 - (sum(ko == 0) / length(ko))
}

graph_components <- function(graph, n_nbrs = NULL, n_ref = NULL) {
  connected_components(graph_to_sparse_r(graph, n_nbrs = n_nbrs))$n_components
}

connected_components <- function(X_csr) {
  X_t_csr <- Matrix::t(X_csr)
  connected_components_undirected(nrow(X_csr), X_csr@j, X_csr@p, X_t_csr@j, X_t_csr@p)
}
