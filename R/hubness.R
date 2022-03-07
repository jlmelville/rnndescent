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
  reverse_nbr_size_impl(idx, k, len, include_self)
}


# Local Scaling -----------------------------------------------------------

#' Locally Scaled Nearest Neighbors
#'
#' Find a subset of nearest neighbors with the smallest generalized scaled
#' distance (Zelnik-Manor and Perona, 2004) as a selection criterion. This a
#' means to reduce the influence of hub points, as suggested by Schnitzer and
#' co-workers (2012).
#'
#' Local scaling is carried out by dividing by the distance to the k-th nearest
#' neighbor, to take into account the difference in local distance statistics of
#' each neighborhood. This function supports choosing the average distance over
#' a range of k, as used by Jegou and co-workers (2007). Note that the scaled
#' distances are used to select k neighbors from a larger list of candidate
#' neighbors, but the returned neighbor data uses the original unscaled
#' distances.
#'
#' @param nn Nearest neighbor data in the dense list format. The `k` scaled
#' neighbors are chosen from the candidates provided here, so the size of the
#' neighborhoods in `nn` should be at least `k`.
#' @param k size of the desired scaled neighborhood.
#' @param k_scale neighbor in `nn` to use to scale the distances. May be a
#' single value (between 1 and the size of the neighborhood) or a vector of
#' two values indicating the inclusive range of neighbors to use. In the latter
#' case, the average distance to the neighbors in the range will be used to
#' scale distances.
#' @param ret_scales If `TRUE` then return a vector of the local scales (the
#' average distance based on `k_scale` for each observation in `nn`).
#' @param n_threads number of threads to use for parallel processing.
#' @return the scaled `k` nearest neighbors in dense list format. The distances
#' returned are the unscaled distances.
#' @examples
#' set.seed(42)
#' # 100 x 10 Gaussian data: exhibits mild hubness
#' m <- matrix(rnorm(1000), nrow = 100, ncol = 10)
#'
#' # Find 50 nearest neighbors to give a reasonable set of candidates
#' nn50 <- brute_force_knn(m, k = 50)
#'
#' # Using the 15-nearest neighbors, maximum k-occurrence > 15 indicates any
#' # observations showing up more than expected
#' nn15_hubness <- max(k_occur(nn50, k = 15))
#'
#' # Find 15 locally scaled nearest neighbors from the 50 NN
#' # use average distance to neighbors 5-7 to represent the local distance
#' # statistics
#' lnn15 <- local_scale_nn(nn50, k = 15, k_scale = c(5, 7))
#'
#' lnn15_hubness <- max(k_occur(lnn15))
#'
#' # hubness has been reduced
#' lnn15_hubness < nn15_hubness # TRUE
#' @references
#' Jegou, H., Harzallah, H., & Schmid, C. (2007, June).
#' A contextual dissimilarity measure for accurate and efficient image search.
#' In *2007 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1-8).
#' IEEE.
#'
#' Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
#' Local and global scaling reduce hubs in space.
#' *Journal of Machine Learning Research*, *13*(10).
#'
#' Zelnik-Manor, L., & Perona, P. (2004).
#' Self-tuning spectral clustering.
#' In *Advances in neural information processing systems*, *17*.
#' @export
local_scale_nn <- function(nn,
                           k = 15,
                           k_scale = 2,
                           ret_scales = FALSE,
                           n_threads = 0) {
  if (!is_dense_nn(nn)) {
    stop("Bad neighbor format for nn")
  }
  if (length(k_scale) < 1 || length(k_scale) > 2) {
    stop("k_scale must be a single value or vector of (begin, end)")
  }
  k_begin <- k_scale[1]
  if (k_begin < 1) {
    stop("k_scale must be >= 1")
  }
  max_k <- ncol(nn$idx)
  if (k_begin > max_k) {
    stop("k_scale must be <= neighborhood size of nn")
  }

  # PaCMAP/TriMap use 4th to 6th "true" neighbors (i.e. skip the first nearest
  # neighbor if that is the observation itself)
  if (length(k_scale) == 2) {
    k_end <- k_scale[2]
  } else {
    k_end <- k_begin
  }
  if (k_end < k_begin) {
    stop("k_scale end point of must be <= start")
  }
  if (k_end > max_k) {
    stop("k_scale end point must be <= neighborhood size of nn")
  }

  if (k < 1) {
    stop("k must be >= 1")
  }
  if (k > max_k) {
    stop("k must be <= neighborhood size of nn")
  }

  local_scaled_nbrs(
    nn$idx,
    nn$dist,
    n_scaled_nbrs = k,
    k_begin = k_begin,
    k_end = k_end,
    ret_scales = ret_scales
  )
}
