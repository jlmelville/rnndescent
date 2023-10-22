#' Find Nearest Neighbors and Distances Using Random Projection Trees
#'
#' Find approximate nearest neighbors using the the Random Projection Trees
#' method of Dasgupta and Freund (2008).
#'
#' @param data Matrix of `n` items to generate neighbors for, with observations
#'   in the rows and features in the columns. Optionally, input can be passed
#'   with observations in the columns, by setting `obs = "C"`, which should be
#'   more efficient.
#' @param k Number of nearest neighbors to return. Optional if `init` is
#'   specified.
#' @param metric Type of distance calculation to use. One of:
#'   - `"euclidean"`.
#'   - `"l2sqr"` (squared Euclidean).
#'   - `"cosine"`.
#'   - `"cosine-preprocess"`: cosine with preprocessing: this trades memory for a
#'   potential speed up during the distance calculation.It should give the
#'   same results as `cosine`, give or take minor numerical changes. Be aware
#'   that the distance between two identical items may not always give exactly
#'   zero with this method.
#'   - `"manhattan"`.
#'   - `"correlation"` (1 minus the Pearson correlation).
#'   - `"correlation-preprocess"`: `correlation` with preprocessing. This trades
#'   memory for a potential speed up during the distance calculation. It should
#'   give the same results as `correlation`, give or take minor numerical
#'   changes. Be aware that the distance between two identical items may not
#'   always give exactly zero with this method.
#'   - `"hamming"`.
#'   - `"bhamming"` (hamming on binary data with bitset internal memory
#'   optimization).
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_trees The number of trees to use in the RP forest. A larger number
#'   will give more accurate results at the cost of a longer computation time.
#'   The default of `NULL` means that the number is chosen based on the number
#'   of observations in `data`.
#' @param leaf_size The maximum number of items that can appear in a leaf. The
#'   default of `NULL` means that the number of leaves is chosen based on the
#'   number of requested neighbors `k`.
#' @param include_self If `TRUE` (the default) then an item is considered to
#'   be a neighbor of itself. Hence the first nearest neighbor in the results
#'   will be the item itself. This is a convention that many nearest neighbor
#'   methods and software adopt, so if you want to use the resulting knn graph
#'   from this function in downstream applications or compare with other
#'   methods, you should probably keep this set to `TRUE`. However, if you are
#'   planning on using the result of this as initialization to another nearest
#'   neighbor method (e.g. [nnd_knn()]), then set this to `FALSE`.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the approximate nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
#'
#' `k` neighbors per observation are not guaranteed to be found. Missing data
#' is represented with an index of `0` and a distance of `NA`.
#' @examples
#' # Find 4 (approximate) nearest neighbors using Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- rp_tree_knn(iris, k = 4, metric = "euclidean", leaf_size = 3)
#'
#' # If you want to initialize another method (e.g. nearest neighbor descent)
#' # with the result of the RP forest, then it's more efficient to skip
#' # evaluating whether an item is a neighbor of itself by setting
#' # `include_self = FALSE`:
#' iris_rp <- rp_tree_knn(iris, k = 4, n_trees = 3, include_self = FALSE)
#' # Use it with e.g. `nnd_knn` -- this should be better than a random start
#' iris_nnd <- nnd_knn(iris, k = 4, init = iris_rp)
#'
#' @references
#' Dasgupta, S., & Freund, Y. (2008, May).
#' Random projection trees and low dimensional manifolds.
#' In *Proceedings of the fortieth annual ACM symposium on Theory of computing*
#' (pp. 537-546).
#' <https://doi.org/10.1145/1374376.1374452>.
#' @export
rp_tree_knn <- function(data,
                        k,
                        metric = "euclidean",
                        use_alt_metric = TRUE,
                        n_trees = NULL,
                        leaf_size = NULL,
                        include_self = TRUE,
                        n_threads = 0,
                        verbose = FALSE,
                        obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))
  n_obs <- switch(obs,
                  R = nrow,
                  C = ncol,
                  stop("Unknown obs type")
  )

  if (is.null(n_trees)) {
    n_trees <- 5 + as.integer(round(nrow(data) ^ 0.25))
    n_trees <- min(32, n_trees)
  }
  if (is.null(leaf_size)) {
    leaf_size <- max(10, k)
  }

  data <- x2m(data)
  check_k(k, n_obs(data))

  if (use_alt_metric) {
    actual_metric <- find_alt_metric(metric)
  } else {
    actual_metric <- metric
  }

  tsmessage(
    thread_msg(
      "Calculating rp tree k-nearest neighbors with k = ",
      k,
      n_threads = n_threads
    )
  )

  if (obs == "R") {
    data <- t(data)
  }

  res <-
    rp_tree_knn_cpp(data,
                    k,
                    actual_metric,
                    n_trees = n_trees,
                    leaf_size = leaf_size,
                    include_self = include_self,
                    n_threads = n_threads,
                    verbose = verbose
    )

  if (use_alt_metric) {
    res$dist <- apply_alt_metric_correction(metric, res$dist)
  }
  tsmessage("Finished")
  res
}

