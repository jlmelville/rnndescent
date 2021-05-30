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
