#' Find nearest neighbors using a random projection forest
#'
#' Returns the approximate k-nearest neighbor graph of a dataset by searching
#' multiple random projection trees, a variant of k-d trees originated by
#' Dasgupta and Freund (2008).
#'
#' @param data Matrix of `n` items to generate neighbors for, with observations
#'   in the rows and features in the columns. Optionally, input can be passed
#'   with observations in the columns, by setting `obs = "C"`, which should be
#'   more efficient. Possible formats are [base::data.frame()], [base::matrix()]
#'   or [Matrix::sparseMatrix()]. Sparse matrices should be in `dgCMatrix`
#'   format. Dataframes will be converted to `numerical` matrix format
#'   internally, so if your data columns are `logical` and intended to be used
#'   with the specialized binary `metric`s, you should convert it to a logical
#'   matrix first (otherwise you will get the slower dense numerical version).
#' @param k Number of nearest neighbors to return. Optional if `init` is
#'   specified.
#' @param metric Type of distance calculation to use. One of:
#'   - `"braycurtis"`
#'   - `"canberra"`
#'   - `"chebyshev"`
#'   - `"correlation"` (1 minus the Pearson correlation)
#'   - `"cosine"`
#'   - `"dice"`
#'   - `"euclidean"`
#'   - `"hamming"`
#'   - `"hellinger"`
#'   - `"jaccard"`
#'   - `"jensenshannon"`
#'   - `"kulsinski"`
#'   - `"sqeuclidean"` (squared Euclidean)
#'   - `"manhattan"`
#'   - `"rogerstanimoto"`
#'   - `"russellrao"`
#'   - `"sokalmichener"`
#'   - `"sokalsneath"`
#'   - `"spearmanr"` (1 minus the Spearman rank correlation)
#'   - `"symmetrickl"` (symmetric Kullback-Leibler divergence)
#'   - `"tsss"` (Triangle Area Similarity-Sector Area Similarity or TS-SS
#'   metric)
#'   - `"yule"`
#'
#'   For non-sparse data, the following variants are available with
#'   preprocessing: this trades memory for a potential speed up during the
#'   distance calculation. Some minor numerical differences should be expected
#'   compared to the non-preprocessed versions:
#'   - `"cosine-preprocess"`: `cosine` with preprocessing.
#'   - `"correlation-preprocess"`: `correlation` with preprocessing.
#'
#'   For non-sparse binary data passed as a `logical` matrix, the following
#'   metrics have specialized variants which should be substantially faster than
#'   the non-binary variants (in other cases the logical data will be treated as
#'   a dense numeric vector of 0s and 1s):
#'   - `"dice"`
#'   - `"hamming"`
#'   - `"jaccard"`
#'   - `"kulsinski"`
#'   - `"matching"`
#'   - `"rogerstanimoto"`
#'   - `"russellrao"`
#'   - `"sokalmichener"`
#'   - `"sokalsneath"`
#'   - `"yule"`
#'
#'   Note that if `margin = "explicit"`, the metric is only used to determine
#'   whether an "angular" or "Euclidean" distance is used to measure the
#'   distance between split points in the tree.
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
#' @param max_tree_depth The maximum depth of the tree to build (default = 200).
#'   If the maximum tree depth is exceeded then the leaf size of a tree may
#'   exceed `leaf_size` which can result in a large number of neighbor distances
#'   being calculated. If `verbose = TRUE` a message will be logged to indicate
#'   that the leaf size is large. However, increasing the `max_tree_depth` may
#'   not help: it may be that there is something unusual about the distribution
#'   of your data set under your chose `metric` that makes a tree-based
#'   initialization inappropriate.
#' @param include_self If `TRUE` (the default) then an item is considered to
#'   be a neighbor of itself. Hence the first nearest neighbor in the results
#'   will be the item itself. This is a convention that many nearest neighbor
#'   methods and software adopt, so if you want to use the resulting knn graph
#'   from this function in downstream applications or compare with other
#'   methods, you should probably keep this set to `TRUE`. However, if you are
#'   planning on using the result of this as initialization to another nearest
#'   neighbor method (e.g. [nnd_knn()]), then set this to `FALSE`.
#' @param ret_forest If `TRUE` also return a search forest which can be used
#'   for future querying (via [rpf_knn_query()]) and filtering
#'   (via [rpf_filter()]). By default this is `FALSE`. Setting this to `TRUE`
#'   will change the output list to be nested (see the `Value` section below).
#' @param margin A character string specifying the method used to  assign points
#'   to one side of the hyperplane or the other. Possible values are:
#'   - `"explicit"` categorizes all distance metrics as either Euclidean or
#'   Angular (Euclidean after normalization), explicitly calculates a hyperplane
#'   and offset, and then calculates the margin based on the dot product with
#'   the hyperplane.
#'   - `"implicit"` calculates the distance from a point to each of the
#'   points defining the normal vector. The margin is calculated by comparing the
#'   two distances: the point is assigned to the side of the hyperplane that
#'   the normal vector point with the closest distance belongs to.
#'   - `"auto"` (the default) picks the margin method depending on whether a
#'   binary-specific `metric` such as `"bhammming"` is chosen, in which case
#'   `"implicit"` is used, and `"explicit"` otherwise: binary-specific metrics
#'   involve storing the data in a way that isn't very efficient for the
#'   `"explicit"` method and the binary-specific metric is usually a lot faster
#'   than the generic equivalent such that the cost of two distance calculations
#'   for the margin method is still faster.
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
#'   * `forest` (if `ret_forest = TRUE`) the RP forest that generated the
#'   neighbor graph, which can be used to query new data.
#'
#' `k` neighbors per observation are not guaranteed to be found. Missing data
#' is represented with an index of `0` and a distance of `NA`.
#' @examples
#' # Find 4 (approximate) nearest neighbors using Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- rpf_knn(iris, k = 4, metric = "euclidean", leaf_size = 3)
#'
#' # If you want to initialize another method (e.g. nearest neighbor descent)
#' # with the result of the RP forest, then it's more efficient to skip
#' # evaluating whether an item is a neighbor of itself by setting
#' # `include_self = FALSE`:
#' iris_rp <- rpf_knn(iris, k = 4, n_trees = 3, include_self = FALSE)
#' # for future querying you may want to also return the RP forest:
#' iris_rpf <- rpf_knn(iris,
#'   k = 4, n_trees = 3, include_self = FALSE,
#'   ret_forest = TRUE
#' )
#' @references
#' Dasgupta, S., & Freund, Y. (2008, May).
#' Random projection trees and low dimensional manifolds.
#' In *Proceedings of the fortieth annual ACM symposium on Theory of computing*
#' (pp. 537-546).
#' \doi{10.1145/1374376.1374452}.
#' @seealso [rpf_filter()], [nnd_knn()]
#' @export
rpf_knn <- function(data,
                    k,
                    metric = "euclidean",
                    use_alt_metric = TRUE,
                    n_trees = NULL,
                    leaf_size = NULL,
                    max_tree_depth = 200,
                    include_self = TRUE,
                    ret_forest = FALSE,
                    margin = "auto",
                    n_threads = 0,
                    verbose = FALSE,
                    obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))
  n_obs <- switch(obs,
    R = nrow,
    C = ncol,
    stop("Unknown obs type")
  )
  data <- x2m(data)
  check_k(k, n_obs(data))

  actual_metric <- get_actual_metric(use_alt_metric, metric, data, verbose)

  if (obs == "R") {
    data <- Matrix::t(data)
  }

  res <- rpf_knn_impl(
    data,
    k = k,
    metric = metric,
    use_alt_metric = use_alt_metric,
    actual_metric = actual_metric,
    n_trees = n_trees,
    leaf_size = leaf_size,
    max_tree_depth = max_tree_depth,
    include_self = include_self,
    ret_forest = ret_forest,
    margin = margin,
    n_threads = n_threads,
    verbose = verbose,
    unzero = TRUE
  )

  if (use_alt_metric) {
    res$dist <-
      apply_alt_metric_correction(metric, res$dist, is_sparse(data))
  }
  tsmessage("Finished")
  res
}

#' Create a random projection forest nearest neighbor index
#'
#' Builds a "forest" of Random Projection Trees (Dasgupta and Freund, 2008),
#' which can later be searched to find approximate nearest neighbors.
#'
#' @param data Matrix of `n` items to generate the index for, with observations
#'   in the rows and features in the columns. Optionally, input can be passed
#'   with observations in the columns, by setting `obs = "C"`, which should be
#'   more efficient. Possible formats are [base::data.frame()], [base::matrix()]
#'   or [Matrix::sparseMatrix()]. Sparse matrices should be in `dgCMatrix`
#'   format. Dataframes will be converted to `numerical` matrix format
#'   internally, so if your data columns are `logical` and intended to be used
#'   with the specialized binary `metric`s, you should convert it to a logical
#'   matrix first (otherwise you will get the slower dense numerical version).
#' @param metric Type of distance calculation to use. One of:
#'   - `"braycurtis"`
#'   - `"canberra"`
#'   - `"chebyshev"`
#'   - `"correlation"` (1 minus the Pearson correlation)
#'   - `"cosine"`
#'   - `"dice"`
#'   - `"euclidean"`
#'   - `"hamming"`
#'   - `"hellinger"`
#'   - `"jaccard"`
#'   - `"jensenshannon"`
#'   - `"kulsinski"`
#'   - `"sqeuclidean"` (squared Euclidean)
#'   - `"manhattan"`
#'   - `"rogerstanimoto"`
#'   - `"russellrao"`
#'   - `"sokalmichener"`
#'   - `"sokalsneath"`
#'   - `"spearmanr"` (1 minus the Spearman rank correlation)
#'   - `"symmetrickl"` (symmetric Kullback-Leibler divergence)
#'   - `"tsss"` (Triangle Area Similarity-Sector Area Similarity or TS-SS
#'   metric)
#'   - `"yule"`
#'
#'   For non-sparse data, the following variants are available with
#'   preprocessing: this trades memory for a potential speed up during the
#'   distance calculation. Some minor numerical differences should be expected
#'   compared to the non-preprocessed versions:
#'   - `"cosine-preprocess"`: `cosine` with preprocessing.
#'   - `"correlation-preprocess"`: `correlation` with preprocessing.
#'
#'   For non-sparse binary data passed as a `logical` matrix, the following
#'   metrics have specialized variants which should be substantially faster than
#'   the non-binary variants (in other cases the logical data will be treated as
#'   a dense numeric vector of 0s and 1s):
#'   - `"dice"`
#'   - `"hamming"`
#'   - `"jaccard"`
#'   - `"kulsinski"`
#'   - `"matching"`
#'   - `"rogerstanimoto"`
#'   - `"russellrao"`
#'   - `"sokalmichener"`
#'   - `"sokalsneath"`
#'   - `"yule"`
#'
#'   Note that if `margin = "explicit"`, the metric is only used to determine
#'   whether an "angular" or "Euclidean" distance is used to measure the
#'   distance between split points in the tree.
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`). Probably the only reason to set this to `FALSE` is
#'   if you suspect that some sort of numeric issue is occurring with your data
#'   in the alternative code path. Only applies if the implicit `margin` method
#'   is used.
#' @param n_trees The number of trees to use in the RP forest. A larger number
#'   will give more accurate results at the cost of a longer computation time.
#'   The default of `NULL` means that the number is chosen based on the number
#'   of observations in `data`.
#' @param leaf_size The maximum number of items that can appear in a leaf. This
#'   value should be chosen to match the expected number of neighbors you will
#'   want to retrieve when running queries (e.g. if you want find 50 nearest
#'   neighbors set `leaf_size = 50`) and should not be set to a value smaller
#'   than `10`.
#' @param max_tree_depth The maximum depth of the tree to build (default = 200).
#'   If the maximum tree depth is exceeded then the leaf size of a tree may
#'   exceed `leaf_size` which can result in a large number of neighbor distances
#'   being calculated. If `verbose = TRUE` a message will be logged to indicate
#'   that the leaf size is large. However, increasing the `max_tree_depth` may
#'   not help: it may be that there is something unusual about the distribution
#'   of your data set under your chose `metric` that makes a tree-based
#'   initialization inappropriate.
#' @param margin A character string specifying the method used to  assign points
#'   to one side of the hyperplane or the other. Possible values are:
#'   - `"explicit"` categorizes all distance metrics as either Euclidean or
#'   Angular (Euclidean after normalization), explicitly calculates a hyperplane
#'   and offset, and then calculates the margin based on the dot product with
#'   the hyperplane.
#'   - `"implicit"` calculates the distance from a point to each of the
#'   points defining the normal vector. The margin is calculated by comparing the
#'   two distances: the point is assigned to the side of the hyperplane that
#'   the normal vector point with the closest distance belongs to.
#'   - `"auto"` (the default) picks the margin method depending on whether a
#'   binary-specific `metric` such as `"bhammming"` is chosen, in which case
#'   `"implicit"` is used, and `"explicit"` otherwise: binary-specific metrics
#'   involve storing the data in a way that isn't very efficient for the
#'   `"explicit"` method and the binary-specific metric is usually a lot faster
#'   than the generic equivalent such that the cost of two distance calculations
#'   for the margin method is still faster.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return a forest of random projection trees as a list. Each tree in the
#' forest is a further list, but is not intended to be examined or manipulated
#' by the user. As a normal R data type, it can be safely serialized and
#' deserialized with [base::saveRDS()] and [base::readRDS()]. To use it for
#' querying pass it as the `forest` parameter of [rpf_knn_query()]. The forest
#' does not store any of the `data` passed into build the tree, so if you
#' are going to search the forest, you will also need to store the `data` used
#' to build it and provide it during the search.
#' @references
#' Dasgupta, S., & Freund, Y. (2008, May).
#' Random projection trees and low dimensional manifolds.
#' In *Proceedings of the fortieth annual ACM symposium on Theory of computing*
#' (pp. 537-546).
#' \doi{10.1145/1374376.1374452}.
#' @seealso [rpf_knn_query()]
#' @examples
#' # Build a forest of 10 trees from the odd rows
#' iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
#' iris_odd_forest <- rpf_build(iris_odd, n_trees = 10)
#'
#' iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
#' iris_even_nn <- rpf_knn_query(
#'   query = iris_even, reference = iris_odd,
#'   forest = iris_odd_forest, k = 15
#' )
#' @export
rpf_build <- function(data,
                      metric = "euclidean",
                      use_alt_metric = TRUE,
                      n_trees = NULL,
                      leaf_size = 10,
                      max_tree_depth = 200,
                      margin = "auto",
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
    n_trees <- 5 + as.integer(round(nrow(data)^0.25))
    n_trees <- min(32, n_trees)
  }

  data <- x2m(data)

  margin <- find_margin_method(margin, metric, data)

  actual_metric <- get_actual_metric(use_alt_metric, metric, data, verbose)

  tsmessage(
    thread_msg(
      "Building RP forest with n_trees = ",
      n_trees,
      " max leaf size = ",
      leaf_size,
      " margin = '", margin, "'",
      n_threads = n_threads
    )
  )

  if (obs == "R") {
    data <- Matrix::t(data)
  }

  if (margin == "implicit") {
    if (is_sparse(data)) {
      forest <- rnn_sparse_rp_forest_implicit_build(
        ind = data@i,
        ptr = data@p,
        data = data@x,
        ndim = nrow(data),
        metric = actual_metric,
        n_trees = n_trees,
        leaf_size = leaf_size,
        max_tree_depth = max_tree_depth,
        n_threads = n_threads,
        verbose = verbose
      )
    } else if (is.logical(data)) {
      forest <- rnn_logical_rp_forest_implicit_build(
        data,
        actual_metric,
        n_trees = n_trees,
        leaf_size = leaf_size,
        max_tree_depth = max_tree_depth,
        n_threads = n_threads,
        verbose = verbose
      )
    } else {
      forest <- rnn_rp_forest_implicit_build(
        data,
        actual_metric,
        n_trees = n_trees,
        leaf_size = leaf_size,
        max_tree_depth = max_tree_depth,
        n_threads = n_threads,
        verbose = verbose
      )
    }
  } else {
    # explicit margin
    if (is_sparse(data)) {
      forest <- rnn_sparse_rp_forest_build(
        ind = data@i,
        ptr = data@p,
        data = data@x,
        ndim = nrow(data),
        metric = actual_metric,
        n_trees = n_trees,
        leaf_size = leaf_size,
        max_tree_depth = max_tree_depth,
        n_threads = n_threads,
        verbose = verbose
      )
    }
    # no logical option here as explicit margin must do real-value maths
    # on hyperplanes rather than calculating distances between points in the
    # dataset
    else {
      forest <- rnn_rp_forest_build(
        data,
        actual_metric,
        n_trees = n_trees,
        leaf_size = leaf_size,
        max_tree_depth = max_tree_depth,
        n_threads = n_threads,
        verbose = verbose
      )
    }
  }
  forest <-
    set_forest_data(forest, use_alt_metric, metric, is_sparse(data))

  tsmessage("Finished")
  forest
}

#' Query a random projection forest index for nearest neighbors
#'
#' Run queries against a "forest" of Random Projection Trees (Dasgupta and
#' Freund, 2008), to return nearest neighbors taken from the reference data used
#' to build the forest.
#'
#' @param query Matrix of `n` query items, with observations in the rows and
#'   features in the columns. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `reference` data must be passed in the same orientation as
#'   `query`. Possible formats are [base::data.frame()], [base::matrix()]
#'   or [Matrix::sparseMatrix()]. Sparse matrices should be in `dgCMatrix`
#'   format. Dataframes will be converted to `numerical` matrix format
#'   internally, so if your data columns are `logical` and intended to be used
#'   with the specialized binary `metric`s, you should convert it to a logical
#'   matrix first (otherwise you will get the slower dense numerical version).
#' @param reference Matrix of `m` reference items, with observations in the rows
#'   and features in the columns. The nearest neighbors to the queries are
#'   calculated from this data and should be the same data used to build the
#'   `forest`. Optionally, the data may be passed with the observations in the
#'   columns, by setting `obs = "C"`, which should be more efficient. The
#'   `query` data must be passed in the same format and orientation as
#'   `reference`. Possible formats are [base::data.frame()], [base::matrix()] or
#'   [Matrix::sparseMatrix()]. Sparse matrices should be in `dgCMatrix` format.
#' @param forest A random partition forest, created by [rpf_build()],
#'   representing partitions of the data in `reference`.
#' @param k Number of nearest neighbors to return. You are unlikely to get good
#'   results if you choose a value substantially larger than the value of
#'   `leaf_size` used to build the `forest`.
#' @param cache if `TRUE` (the default) then candidate indices found in the
#'   leaves of the forest are cached to avoid recalculating the same distance
#'   repeatedly. This incurs an extra memory cost which scales with `n_threads`.
#'   Set this to `FALSE` to disable distance caching.
#' @param n_threads Number of threads to use. Note that the parallelism in the
#'   search is done over the observations in `query` not the trees in the
#'   `forest`. Thus a single observation will not see any speed-up from
#'   increasing `n_threads`.
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
#' @references
#' Dasgupta, S., & Freund, Y. (2008, May).
#' Random projection trees and low dimensional manifolds.
#' In *Proceedings of the fortieth annual ACM symposium on Theory of computing*
#' (pp. 537-546).
#' \doi{10.1145/1374376.1374452}.
#' @seealso [rpf_build()]
#' @examples
#' # Build a forest of 10 trees from the odd rows
#' iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
#' iris_odd_forest <- rpf_build(iris_odd, n_trees = 10)
#'
#' iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
#' iris_even_nn <- rpf_knn_query(
#'   query = iris_even, reference = iris_odd,
#'   forest = iris_odd_forest, k = 15
#' )
#' @export
rpf_knn_query <- function(query,
                          reference,
                          forest,
                          k,
                          cache = TRUE,
                          n_threads = 0,
                          verbose = FALSE,
                          obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))
  n_obs <- switch(obs,
    R = nrow,
    C = ncol,
    stop("Unknown obs type")
  )

  check_sparse(reference, query)

  reference <- x2m(reference)
  query <- x2m(query)
  check_k(k, n_obs(reference))

  if (!is.list(forest)) {
    stop("Bad forest format: not a list")
  }
  if (is.null(forest$margin)) {
    stop("Bad forest format: no 'margin' specified")
  }
  if (is_sparse(reference) && !forest$sparse) {
    stop("Incompatible sparse forest used with dense input data")
  }
  if (!is_sparse(reference) && forest$sparse) {
    stop("Incompatible dense forest used with sparse input data")
  }

  if (obs == "R") {
    reference <- Matrix::t(reference)
    query <- Matrix::t(query)
  }

  tsmessage(thread_msg("Querying rp forest for k = ",
    k, ifelse(cache, " with caching", ""),
    n_threads = n_threads
  ))

  metric <- forest$actual_metric
  if (is_sparse(reference)) {
    res <-
      rnn_sparse_rp_forest_search(
        ref_ind = reference@i,
        ref_ptr = reference@p,
        ref_data = reference@x,
        query_ind = query@i,
        query_ptr = query@p,
        query_data = query@x,
        ndim = nrow(reference),
        search_forest = forest,
        n_nbrs = k,
        metric = metric,
        cache = cache,
        n_threads = n_threads,
        verbose = verbose
      )
  } else if (is.logical(reference)) {
    res <-
      rnn_logical_rp_forest_search(
        reference = reference,
        query = query,
        search_forest = forest,
        n_nbrs = k,
        metric = metric,
        cache = cache,
        n_threads = n_threads,
        verbose = verbose
      )
  } else {
    res <-
      rnn_rp_forest_search(
        reference = reference,
        query = query,
        search_forest = forest,
        n_nbrs = k,
        metric = metric,
        cache = cache,
        n_threads = n_threads,
        verbose = verbose
      )
  }

  if (forest$use_alt_metric) {
    res$dist <-
      apply_alt_metric_correction(forest$original_metric, res$dist, is_sparse(reference))
  }
  tsmessage("Finished")
  res
}

#' Keep the best trees in a random projection forest
#'
#' Reduce the size of a random projection forest, by scoring each tree against
#' a k-nearest neighbors graph. Only the top N trees will be retained which
#' allows for a faster querying.
#'
#' Trees are scored based on how well each leaf reflects the neighbors as
#' specified in the nearest neighbor data. It's best to use as accurate nearest
#' neighbor data as you can and it does not need to come directly from
#' searching the `forest`: for example, the nearest neighbor data from running
#' [nnd_knn()] to optimize the neighbor data output from an RP Forest is a
#' good choice.
#'
#' Rather than rely on an RP Forest solely for approximate nearest neighbor
#' querying, it is probably more cost-effective to use a small number of trees
#' to initialize the neighbor list for use in a graph search via
#' [graph_knn_query()].
#'
#' @param nn Nearest neighbor data in the dense list format. This should be
#'   derived from the same data that was used to build the `forest`.
#' @param forest A random partition forest, e.g. created by [rpf_build()],
#'   representing partitions of the same underlying data reflected in `nn`.
#'   As a convenient, this parameter is ignored if the `nn` list contains a
#'   `forest` entry, e.g. from running [rpf_knn()] or [nnd_knn()] with
#'   `ret_forest = TRUE`, and the forest value will be extracted from `nn`.
#' @param n_trees The number of trees to retain. By default only the
#'   best-scoring tree is retained.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @return A forest with the best scoring `n_trees` trees.
#' @seealso [rpf_build()]
#' @examples
#' # Build a knn with a forest of 10 trees using the odd rows
#' iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
#' # also return the forest with the knn
#' rfknn <- rpf_knn(iris_odd, k = 15, n_trees = 10, ret_forest = TRUE)
#'
#' # keep the best 2 trees:
#' iris_odd_filtered_forest <- rpf_filter(rfknn)
#'
#' # get some new data to search
#' iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
#'
#' # search with the filtered forest
#' iris_even_nn <- rpf_knn_query(
#'   query = iris_even, reference = iris_odd,
#'   forest = iris_odd_filtered_forest, k = 15
#' )
#' @export
rpf_filter <-
  function(nn,
           forest = NULL,
           n_trees = 1,
           n_threads = 0,
           verbose = FALSE) {
    if (is.null(forest)) {
      if (is.null(nn$forest)) {
        stop("Must provide 'forest' parameter")
      }
      forest <- nn$forest
    }

    n_unfiltered_trees <- length(forest)
    if (n_trees < 1 || n_trees > n_unfiltered_trees) {
      stop("n_trees must be between 1 and ", n_unfiltered_trees)
    }
    tsmessage(thread_msg("Keeping ", n_trees, " best search trees",
      n_threads = n_threads
    ))
    filtered_forest <- rnn_score_forest(
      nn$idx,
      search_forest = forest,
      n_trees = n_trees,
      n_threads = n_threads,
      verbose = verbose
    )

    filtered_forest <-
      set_forest_data(
        filtered_forest,
        forest$use_alt_metric,
        forest$original_metric,
        forest$sparse
      )

    filtered_forest
  }
