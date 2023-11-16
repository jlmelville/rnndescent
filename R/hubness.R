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
#'   labeled starting at 1. Note that the integer labels do *not* have to
#'   refer to the rows of `idx`, for example if the nearest neighbor result
#'   is from querying one set of objects with respect to another (for instance
#'   from running [graph_knn_query()]). You may also pass a nearest
#'   neighbor graph object (e.g. the output of running [nnd_knn()]),
#'   and the indices will be extracted from it, or a sparse matrix in the same
#'   format as that returned by [prepare_search_graph()].
#' @param k The number of closest neighbors to use. Must be between 1 and the
#'   number of columns in `idx`. By default, all columns of `idx` are
#'   used. Ignored if `idx` is sparse.
#' @param include_self logical indicating whether the label `i` in
#'   `idx` is considered to be a valid neighbor when found in row `i`.
#'   By default this is `TRUE`. This can be set to `FALSE` when the
#'   the labels in `idx` refer to the row indices of `idx`, as in the
#'   case of results from [nnd_knn()]. In this case you may not want
#'   to consider the trivial case of an object being a neighbor of itself. In
#'   all other cases leave this set to `TRUE`.
#' @return a vector of length `max(idx)`, containing the number of times an
#'   object in `idx` was found in the nearest neighbor list of the objects
#'   represented by the row indices of `idx`.
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
#' *Journal of Machine Learning Research*, *11*, 2487-2531.
#' <https://www.jmlr.org/papers/v11/radovanovic10a.html>
#'
#' Bratic, B., Houle, M. E., Kurbalija, V., Oria, V., & Radovanovic, M. (2019).
#' The Influence of Hubness on NN-Descent.
#' *International Journal on Artificial Intelligence Tools*, *28*(06), 1960002.
#' <https://doi.org/10.1142/S0218213019600029>
#' @export
k_occur <- function(idx,
                    k = NULL,
                    include_self = TRUE) {
  if (methods::is(idx, "sparseMatrix")) {
    return(as.vector(table(Matrix::t(idx)@i)) - ifelse(include_self, 0, 1))
  }
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
  rnn_reverse_nbr_size(idx, k, len, include_self)
}

