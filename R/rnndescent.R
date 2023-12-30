# API ---------------------------------------------------------------------

#' Build approximate nearest neighbors index and neighbor graph
#'
#' This function builds an approximate nearest neighbors graph with convenient
#' defaults, then prepares the index for querying new data, for later use with
#' [rnnd_query()]. For more control over the process, please see the other
#' functions in the package.
#'
#' The process of k-nearest neighbor graph construction using Random Projection
#' Forests (Dasgupta and Freund, 2008) for initialization and Nearest Neighbor
#' Descent (Dong and co-workers, 2011) for refinement. Index preparation, uses
#' the graph diversification method of Harwood and Drummond (2016).
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
#' @param k Number of nearest neighbors to build the index for. You can specify
#'   a different number when running `rnnd_query`, but the index is calibrated
#'   using this value so it's recommended to set `k` according to the likely
#'   number of neighbors you will want to retrieve. Optional if `init` is
#'   specified, in which case `k` can be inferred from the `init` data. If you
#'   do both, then the specified version of `k` will take precedence.
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param init Name of the initialization strategy or initial `data` neighbor
#'   graph to optimize. One of:
#'   - `"rand"` random initialization (the default).
#'   - `"tree"` use the random projection tree method of Dasgupta and Freund
#'   (2008).
#'   - a pre-calculated neighbor graph. A list containing:
#'       - `idx` an `n` by `k` matrix containing the nearest neighbor indices.
#'       - `dist` (optional) an `n` by `k` matrix containing the nearest
#'       neighbor distances. If the input distances are omitted, they will be
#'       calculated for you.'
#'
#'   If `k` and `init` are specified as arguments to this function, and the
#'   number of neighbors provided in `init` is not equal to `k` then:
#'
#'   * if `k` is smaller, only the `k` closest values in `init` are retained.
#'   * if `k` is larger, then random neighbors will be chosen to fill `init` to
#'   the size of `k`. Note that there is no checking if any of the random
#'   neighbors are duplicates of what is already in `init` so effectively fewer
#'   than `k` neighbors may be chosen for some observations under these
#'   circumstances.
#' @param n_trees The number of trees to use in the RP forest. A larger number
#'   will give more accurate results at the cost of a longer computation time.
#'   The default of `NULL` means that the number is chosen based on the number
#'   of observations in `data`. Only used if `init = "tree"`.
#' @param leaf_size The maximum number of items that can appear in a leaf. This
#'   value should be chosen to match the expected number of neighbors you will
#'   want to retrieve when running queries (e.g. if you want find 50 nearest
#'   neighbors set `leaf_size = 50`) and should not be set to a value smaller
#'   than `10`. Only used if `init = "tree"`.
#' @param max_tree_depth The maximum depth of the tree to build (default = 200).
#'   If the maximum tree depth is exceeded then the leaf size of a tree may
#'   exceed `leaf_size` which can result in a large number of neighbor distances
#'   being calculated. If `verbose = TRUE` a message will be logged to indicate
#'   that the leaf size is large. However, increasing the `max_tree_depth` may
#'   not help: it may be that there is something unusual about the distribution
#'   of your data set under your chose `metric` that makes a tree-based
#'   initialization inappropriate. Only used if `init = "tree"`.
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
#'
#'   Only used if `init = "tree"`.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#'   By default, this will be chosen based on the number of observations in
#'   `data`.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to `k` to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper. By default, this
#'   is set to `k` or `60`, whichever is smaller.
#' @param delta The minimum relative change in the neighbor graph allowed before
#'   early stopping. Should be a value between 0 and 1. The smaller the value,
#'   the smaller the amount of progress between iterations is allowed. Default
#'   value of `0.001` means that at least 0.1% of the neighbor graph must
#'   be updated at each iteration.
#' @param low_memory If `TRUE`, use a lower memory, but more
#'   computationally expensive approach to index construction. If set to
#'   `FALSE`, you should see a noticeable speed improvement, especially when
#'   using a smaller number of threads, so this is worth trying if you have the
#'   memory to spare.
#' @param n_search_trees, the number of trees to keep in the search forest as
#'   part of index preparation. The default is `1`.
#' @param diversify_prob the degree of diversification of the search graph
#'   by removing unnecessary edges through occlusion pruning. This should take a
#'   value between `0` (no diversification) and `1` (remove as many edges as
#'   possible) and is treated as the probability of a neighbor being removed if
#'   it is found to be an "occlusion". If item `p` and `q`, two members of the
#'   neighbor list of item `i`, are closer to each other than they are to `i`,
#'   then the nearer neighbor `p` is said to "occlude" `q`. It is likely that
#'   `q` will be in the neighbor list of `p` so there is no need to retain it in
#'   the neighbor list of `i`. You may also set this to `NULL` to skip any
#'   occlusion pruning. Note that occlusion pruning is carried out twice, once
#'   to the forward neighbors, and once to the reverse neighbors.
#' @param pruning_degree_multiplier How strongly to truncate the final neighbor
#'   list for each item. The neighbor list of each item will be truncated to
#'   retain only the closest `d` neighbors, where
#'   `d = k * pruning_degree_multiplier`, and `k` is the
#'   original number of neighbors per item in `graph`. Roughly, values
#'   larger than `1` will keep all the nearest neighbors of an item, plus
#'   the given fraction of reverse neighbors (if they exist). For example,
#'   setting this to `1.5` will keep all the forward neighbors and then
#'   half as many of the reverse neighbors, although exactly which neighbors are
#'   retained is also dependent on any occlusion pruning that occurs. Set this
#'   to `NULL` to skip this step.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param progress Determines the type of progress information logged during the
#'   nearest neighbor descent stage when `verbose = TRUE`. Options are:
#'   * `"bar"`: a simple text progress bar.
#'   * `"dist"`: the sum of the distances in the approximate knn graph at the
#'     end of each iteration.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the approximate nearest neighbor index, a list containing:
#'    * `graph` the k-nearest neighbor graph, a list containing:
#'        * `idx` an n by k matrix containing the nearest neighbor indices.
#'        * `dist` an n by k matrix containing the nearest neighbor distances.
#'    * Other list items are intended only for internal use by other functions
#'    such as [rnnd_query()].
#' @seealso [rnnd_query()]
#' @examples
#' iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
#' iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
#'
#' # Find 4 (approximate) nearest neighbors using Euclidean distance
#' iris_even_index <- rnnd_build(iris_even, k = 4)
#' iris_odd_nbrs <- rnnd_query(index = iris_even_index, query = iris_odd, k = 4)
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
#' @export
rnnd_build <- function(data,
                       k = 30,
                       metric = "euclidean",
                       use_alt_metric = TRUE,
                       init = "tree",
                       n_trees = NULL,
                       leaf_size = NULL,
                       max_tree_depth = 200,
                       margin = "auto",
                       n_iters = NULL,
                       delta = 0.001,
                       max_candidates = NULL,
                       low_memory = TRUE,
                       n_search_trees = 1,
                       pruning_degree_multiplier = 1.5,
                       diversify_prob = 1.0,
                       n_threads = 0,
                       verbose = FALSE,
                       progress = "bar",
                       obs = "R") {
  data <- x2m(data)
  if (obs == "R") {
    data <- Matrix::t(data)
  }

  index <- nnd_knn(
    data,
    k = k,
    metric = metric,
    use_alt_metric = use_alt_metric,
    init = init,
    init_args = list(
      n_trees = n_trees,
      leaf_size = leaf_size,
      max_tree_depth = max_tree_depth,
      margin = margin
    ),
    ret_forest = TRUE,
    n_iters = n_iters,
    delta = delta,
    max_candidates = max_candidates,
    low_memory = low_memory,
    n_threads = n_threads,
    verbose = verbose,
    progress = progress,
    obs = "C"
  )

  index <- list(
    graph = list(idx = index$idx, dist = index$dist),
    forest = index$forest
  )

  index$prep <- list(
    pruning_degree_multiplier = pruning_degree_multiplier,
    diversify_prob = diversify_prob,
    n_search_trees = n_search_trees,
    is_prepared = FALSE
  )

  index$data <- data
  index$original_metric <- metric
  index$use_alt_metric <- use_alt_metric

  if (!is.null(index$forest)) {
    index$search_forest <-
      rpf_filter(
        nn = index$graph,
        forest = index$forest,
        n_trees = index$prep$n_search_trees,
        n_threads = n_threads,
        verbose = verbose
      )
  }

  search_graph <- prepare_search_graph(
    data = index$data,
    graph = index$graph,
    metric = index$original_metric,
    use_alt_metric = index$use_alt_metric,
    diversify_prob = index$prep$diversify_prob,
    pruning_degree_multiplier = index$prep$pruning_degree_multiplier,
    n_threads = n_threads,
    verbose = verbose,
    obs = "C"
  )
  index$prep$is_prepared <- TRUE
  index$search_graph <- search_graph

  # RP Forests can be large so after preparation we delete this
  index$forest <- NULL
  index
}

#' Query an index for approximate nearest neighbors
#'
#' Takes a nearest neighbor index produced by [rnnd_build()] and uses it to
#' find the nearest neighbors of a query set of observations, using a
#' back-tracking search with the search size determined by the method of
#' Iwasaki and Miyazaki (2018). For further control over the search effort, the
#' total number of distance calculations can also be bounded, similar to the
#' method of Harwood and Drummond (2016).
#'
#' @param index A nearest neighbor index produced by [rnnd_build()].
#' @param query Matrix of `n` query items, with observations in the rows and
#'   features in the columns. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. Possible formats are [base::data.frame()], [base::matrix()]
#'   or [Matrix::sparseMatrix()]. Sparse matrices should be in `dgCMatrix`
#'   format. Dataframes will be converted to `numerical` matrix format
#'   internally, so if your data columns are `logical` and intended to be used
#'   with the specialized binary `metric`s, you should convert it to a logical
#'   matrix first (otherwise you will get the slower dense numerical version).
#'   Sparse and non-sparse data cannot be mixed, so if the data used to build
#'   index was sparse, the `query` data must also be sparse. and vice versa.
#' @param k Number of nearest neighbors to return.
#' @param epsilon Controls trade-off between accuracy and search cost, as
#'   described by Iwasaki and Miyazaki (2018). Setting `epsilon` to a positive
#'   value specifies a distance tolerance on whether to explore the neighbors of
#'   candidate points. The larger the value, the more neighbors will be
#'   searched. A value of 0.1 allows query-candidate distances to be 10% larger
#'   than the current most-distant neighbor of the query point, 0.2 means 20%,
#'   and so on. Suggested values are between 0-0.5, although this value is
#'   highly dependent on the distribution of distances in the dataset (higher
#'   dimensional data should choose a smaller cutoff). Too large a value of
#'   `epsilon` will result in the query search approaching brute force
#'   comparison. Use this parameter in conjunction with `max_search_fraction` to
#'   prevent excessive run time. Default is 0.1. If you set `verbose = TRUE`,
#'   statistics of the number of distance calculations will be logged which can
#'   help you tune `epsilon`.
#' @param max_search_fraction Maximum fraction of the reference data to search.
#'  This is a value between 0 (search none of the reference data) and 1 (search
#'  all of the data if necessary). This works in conjunction with `epsilon` and
#'  will terminate the search early if the specified fraction of the reference
#'  data has been searched. Default is 1.
#' @param n_threads Number of threads to use.
#' @param init An optional matrix of `k` initial nearest neighbors for each
#'  query point.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the approximate nearest neighbor index, a list containing:
#'  * `idx` an n by k matrix containing the nearest neighbor indices.
#'  * `dist` an n by k matrix containing the nearest neighbor distances.
#' @seealso [rnnd_query()]
#' @examples
#' iris_even <- iris[seq_len(nrow(iris)) %% 2 == 0, ]
#' iris_odd <- iris[seq_len(nrow(iris)) %% 2 == 1, ]
#'
#' iris_even_index <- rnnd_build(iris_even, k = 4)
#' iris_odd_nbrs <- rnnd_query(index = iris_even_index, query = iris_odd, k = 4)
#'
#' @references
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
#' (pp. 5713-5722).
#'
#' Iwasaki, M., & Miyazaki, D. (2018).
#' Optimization of indexing based on k-nearest neighbor graph for proximity search in high-dimensional data.
#' *arXiv preprint* *arXiv:1810.07355*.
#' <https://arxiv.org/abs/1810.07355>
#' @export
rnnd_query <-
  function(index,
           query,
           k = 30,
           epsilon = 0.1,
           max_search_fraction = 1,
           init = NULL,
           n_threads = 0,
           verbose = FALSE,
           obs = "R") {
    if (is.null(init) && !is.null(index$search_forest)) {
      init <- index$search_forest
    }

    query <- x2m(query)
    if (obs == "R") {
      query <- Matrix::t(query)
    }
    res <- graph_knn_query(
      query = query,
      reference = index$data,
      reference_graph = index$search_graph,
      k = k,
      metric = index$original_metric,
      init = init,
      epsilon = epsilon,
      max_search_fraction = max_search_fraction,
      use_alt_metric = index$use_alt_metric,
      n_threads = n_threads,
      verbose = verbose,
      obs = "C"
    )
    res
  }


#' Find approximate nearest neighbors
#'
#' This function builds an approximate nearest neighbors graph of the provided
#' data using convenient defaults. It does not return an index for later
#' querying, to speed the graph construction and reduce the size and complexity
#' of the return value.
#'
#' The process of k-nearest neighbor graph construction using Random Projection
#' Forests (Dasgupta and Freund, 2008) for initialization and Nearest Neighbor
#' Descent (Dong and co-workers, 2011) for refinement. If you are sure you will
#' not want to query new data then compared to [rnnd_build()] this function has
#' the advantage of not storing the index, which can be very large.
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param init Name of the initialization strategy or initial `data` neighbor
#'   graph to optimize. One of:
#'   - `"rand"` random initialization (the default).
#'   - `"tree"` use the random projection tree method of Dasgupta and Freund
#'   (2008).
#'   - a pre-calculated neighbor graph. A list containing:
#'       - `idx` an `n` by `k` matrix containing the nearest neighbor indices.
#'       - `dist` (optional) an `n` by `k` matrix containing the nearest
#'       neighbor distances. If the input distances are omitted, they will be
#'       calculated for you.'
#'
#'   If `k` and `init` are specified as arguments to this function, and the
#'   number of neighbors provided in `init` is not equal to `k` then:
#'
#'   * if `k` is smaller, only the `k` closest values in `init` are retained.
#'   * if `k` is larger, then random neighbors will be chosen to fill `init` to
#'   the size of `k`. Note that there is no checking if any of the random
#'   neighbors are duplicates of what is already in `init` so effectively fewer
#'   than `k` neighbors may be chosen for some observations under these
#'   circumstances.
#' @param n_trees The number of trees to use in the RP forest. A larger number
#'   will give more accurate results at the cost of a longer computation time.
#'   The default of `NULL` means that the number is chosen based on the number
#'   of observations in `data`. Only used if `init = "tree"`.
#' @param leaf_size The maximum number of items that can appear in a leaf. This
#'   value should be chosen to match the expected number of neighbors you will
#'   want to retrieve when running queries (e.g. if you want find 50 nearest
#'   neighbors set `leaf_size = 50`) and should not be set to a value smaller
#'   than `10`. Only used if `init = "tree"`.
#' @param max_tree_depth The maximum depth of the tree to build (default = 200).
#'   If the maximum tree depth is exceeded then the leaf size of a tree may
#'   exceed `leaf_size` which can result in a large number of neighbor distances
#'   being calculated. If `verbose = TRUE` a message will be logged to indicate
#'   that the leaf size is large. However, increasing the `max_tree_depth` may
#'   not help: it may be that there is something unusual about the distribution
#'   of your data set under your chose `metric` that makes a tree-based
#'   initialization inappropriate. Only used if `init = "tree"`.
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
#'
#'   Only used if `init = "tree"`.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#'   By default, this will be chosen based on the number of observations in
#'   `data`.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to `k` to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper. By default, this
#'   is set to `k` or `60`, whichever is smaller.
#' @param delta The minimum relative change in the neighbor graph allowed before
#'   early stopping. Should be a value between 0 and 1. The smaller the value,
#'   the smaller the amount of progress between iterations is allowed. Default
#'   value of `0.001` means that at least 0.1% of the neighbor graph must
#'   be updated at each iteration.
#' @param low_memory If `TRUE`, use a lower memory, but more
#'   computationally expensive approach to index construction. If set to
#'   `FALSE`, you should see a noticeable speed improvement, especially when
#'   using a smaller number of threads, so this is worth trying if you have the
#'   memory to spare.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param progress Determines the type of progress information logged during the
#'   nearest neighbor descent stage when `verbose = TRUE`. Options are:
#'   * `"bar"`: a simple text progress bar.
#'   * `"dist"`: the sum of the distances in the approximate knn graph at the
#'     end of each iteration.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the approximate nearest neighbor index, a list containing:
#'    * `idx` an n by k matrix containing the nearest neighbor indices.
#'    * `dist` an n by k matrix containing the nearest neighbor distances.
#' @seealso [rnnd_build()], [rnnd_query()]
#' @examples
#'
#' # Find 4 (approximate) nearest neighbors using Euclidean distance
#' iris_knn <- rnnd_knn(iris, k = 4)
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
#' @export
rnnd_knn <- function(data,
                     k = 30,
                     metric = "euclidean",
                     use_alt_metric = TRUE,
                     init = "tree",
                     n_trees = NULL,
                     leaf_size = NULL,
                     max_tree_depth = 200,
                     margin = "auto",
                     n_iters = NULL,
                     delta = 0.001,
                     max_candidates = NULL,
                     low_memory = TRUE,
                     n_threads = 0,
                     verbose = FALSE,
                     progress = "bar",
                     obs = "R") {
  data <- x2m(data)
  if (obs == "R") {
    data <- Matrix::t(data)
  }

  nnd_knn(
    data,
    k = k,
    metric = metric,
    use_alt_metric = use_alt_metric,
    init = init,
    init_args = list(
      n_trees = n_trees,
      leaf_size = leaf_size,
      max_tree_depth = max_tree_depth,
      margin = margin
    ),
    ret_forest = FALSE,
    n_iters = n_iters,
    delta = delta,
    max_candidates = max_candidates,
    low_memory = low_memory,
    n_threads = n_threads,
    verbose = verbose,
    progress = progress,
    obs = "C"
  )
}


# kNN Construction --------------------------------------------------------

#' Find exact nearest neighbors by brute force
#'
#' Returns the exact nearest neighbors of a dataset. A brute force search is
#' carried out: all possible pairs of points are compared, and the nearest
#' neighbors are returned.
#'
#' This method is accurate but scales poorly with dataset size, so use with
#' caution with larger datasets. Having the exact neighbors as a ground truth to
#' compare with approximate results is useful for benchmarking and determining
#' parameter settings of the approximate methods.
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
#' @param k Number of nearest neighbors to return.
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
#' @examples
#' # Find the 4 nearest neighbors using Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "euclidean")
#'
#' # Manhattan (l1) distance
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "manhattan")
#'
#' # Multi-threading: you can choose the number of threads to use: in real
#' # usage, you will want to set n_threads to at least 2
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "manhattan", n_threads = 1)
#'
#' # Use verbose flag to see information about progress
#' iris_nn <- brute_force_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#' @export
brute_force_knn <- function(data,
                            k,
                            metric = "euclidean",
                            use_alt_metric = TRUE,
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

  tsmessage(
    thread_msg(
      "Calculating brute force k-nearest neighbors with k = ",
      k,
      n_threads = n_threads
    )
  )
  if (obs == "R") {
    data <- Matrix::t(data)
  }
  if (is_sparse(data)) {
    res <-
      rnn_sparse_brute_force(
        ind = data@i,
        ptr = data@p,
        data = data@x,
        ndim = nrow(data),
        nnbrs = k,
        metric = actual_metric,
        n_threads = n_threads,
        verbose = verbose
      )
  } else if (is.logical(data)) {
    res <-
      rnn_logical_brute_force(data,
        k,
        actual_metric,
        n_threads = n_threads,
        verbose = verbose
      )
  } else {
    res <-
      rnn_brute_force(data,
        k,
        actual_metric,
        n_threads = n_threads,
        verbose = verbose
      )
  }

  res$idx <- res$idx + 1

  if (use_alt_metric) {
    res$dist <-
      apply_alt_metric_correction(metric, res$dist, is_sparse(data))
  }
  tsmessage("Finished")
  res
}

#' Find nearest neighbors by random selection
#'
#' Create a neighbor graph by randomly selecting neighbors. This is not a useful
#' nearest neighbor method on its own, but can be used with other methods which
#' require initialization, such as [nnd_knn()].
#'
#' @param data Matrix of `n` items to generate random neighbors for, with
#'   observations in the rows and features in the columns. Optionally, input can
#'   be passed with observations in the columns, by setting `obs = "C"`, which
#'   should be more efficient. Possible formats are [base::data.frame()],
#'   [base::matrix()] or [Matrix::sparseMatrix()]. Sparse matrices should be in
#'   `dgCMatrix` format. Dataframes will be converted to `numerical` matrix
#'   format internally, so if your data columns are `logical` and intended to be
#'   used with the specialized binary `metric`s, you should convert it to a
#'   logical matrix first (otherwise you will get the slower dense numerical
#'   version).
#' @param k Number of nearest neighbors to return.
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param order_by_distance If `TRUE` (the default), then results for each
#'   item are returned by increasing distance. If you don't need the results
#'   sorted, e.g. you are going to pass the results as initialization to another
#'   routine like [nnd_knn()], set this to `FALSE` to save a small amount of
#'   computational time.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return a random neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
#' @examples
#' # Find 4 random neighbors and calculate their Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- random_knn(iris, k = 4, metric = "euclidean")
#'
#' # Manhattan (l1) distance
#' iris_nn <- random_knn(iris, k = 4, metric = "manhattan")
#'
#' # Multi-threading: you can choose the number of threads to use: in real
#' # usage, you will want to set n_threads to at least 2
#' iris_nn <- random_knn(iris, k = 4, metric = "manhattan", n_threads = 1)
#'
#' # Use verbose flag to see information about progress
#' iris_nn <- random_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#'
#' # These results can be improved by nearest neighbors descent. You don't need
#' # to specify k here because this is worked out from the initial input
#' iris_nn <- nnd_knn(iris, init = iris_nn, metric = "euclidean", verbose = TRUE)
#' @export
random_knn <-
  function(data,
           k,
           metric = "euclidean",
           use_alt_metric = TRUE,
           order_by_distance = TRUE,
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
    res <- random_knn_impl(
      reference = data,
      k = k,
      actual_metric = actual_metric,
      order_by_distance = order_by_distance,
      n_threads = n_threads,
      verbose = verbose
    )
    res$idx <- res$idx + 1

    if (use_alt_metric) {
      res$dist <-
        apply_alt_metric_correction(metric, res$dist, is_sparse(data))
    }
    tsmessage("Finished")
    res
  }

#' Find nearest neighbors using nearest neighbor descent
#'
#' Uses the Nearest Neighbor Descent method due to Dong and co-workers (2011)
#' to optimize an approximate nearest neighbor graph.
#'
#' If no initial graph is provided, a random graph is generated, or you may also
#' specify the use of a graph generated from a forest of random projection
#' trees, using the method of Dasgupta and Freund (2008).
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
#' @param init Name of the initialization strategy or initial `data` neighbor
#'   graph to optimize. One of:
#'   - `"rand"` random initialization (the default).
#'   - `"tree"` use the random projection tree method of Dasgupta and Freund
#'   (2008).
#'   - a pre-calculated neighbor graph. A list containing:
#'       - `idx` an `n` by `k` matrix containing the nearest neighbor indices.
#'       - `dist` (optional) an `n` by `k` matrix containing the nearest
#'       neighbor distances. If the input distances are omitted, they will be
#'       calculated for you.'
#'
#'   If `k` and `init` are specified as arguments to this function, and the
#'   number of neighbors provided in `init` is not equal to `k` then:
#'
#'   * if `k` is smaller, only the `k` closest values in `init` are retained.
#'   * if `k` is larger, then random neighbors will be chosen to fill `init` to
#'   the size of `k`. Note that there is no checking if any of the random
#'   neighbors are duplicates of what is already in `init` so effectively fewer
#'   than `k` neighbors may be chosen for some observations under these
#'   circumstances.
#' @param init_args a list containing arguments to pass to the random partition
#'   forest initialization. See [rpf_knn()] for possible arguments. To avoid
#'   inconsistences with the tree calculation and subsequent nearest neighbor
#'   descent optimization, if you attempt to provide a `metric` or
#'   `use_alt_metric` option in this list it will be ignored.
#' @param n_iters Number of iterations of nearest neighbor descent to carry out.
#'   By default, this will be chosen based on the number of observations in
#'   `data`.
#' @param max_candidates Maximum number of candidate neighbors to try for each
#'   item in each iteration. Use relative to `k` to emulate the "rho"
#'   sampling parameter in the nearest neighbor descent paper. By default, this
#'   is set to `k` or `60`, whichever is smaller.
#' @param delta The minimum relative change in the neighbor graph allowed before
#'   early stopping. Should be a value between 0 and 1. The smaller the value,
#'   the smaller the amount of progress between iterations is allowed. Default
#'   value of `0.001` means that at least 0.1% of the neighbor graph must
#'   be updated at each iteration.
#' @param low_memory If `TRUE`, use a lower memory, but more
#'   computationally expensive approach to index construction. If set to
#'   `FALSE`, you should see a noticeable speed improvement, especially
#'   when using a smaller number of threads, so this is worth trying if you have
#'   the memory to spare.
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param ret_forest If `TRUE` and `init = "tree"` then the RP forest used to
#'   initialize the nearest neighbors will be returned with the nearest neighbor
#'   data. See the `Value` section for details. The returned forest can be used
#'   as part of initializing the search for new data: see [rpf_knn_query()] and
#'   [rpf_filter()] for more details.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param progress Determines the type of progress information logged if
#'   `verbose = TRUE`. Options are:
#'   * `"bar"`: a simple text progress bar.
#'   * `"dist"`: the sum of the distances in the approximate knn graph at the
#'     end of each iteration.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return the approximate nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
#'   * `forest` (if `init = "tree"` and `ret_forest = TRUE` only): the RP forest
#'      used to initialize the neighbor data.
#' @examples
#' # Find 4 (approximate) nearest neighbors using Euclidean distance
#' # If you pass a data frame, non-numeric columns are removed
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean")
#'
#' # Manhattan (l1) distance
#' iris_nn <- nnd_knn(iris, k = 4, metric = "manhattan")
#'
#' # Multi-threading: you can choose the number of threads to use: in real
#' # usage, you will want to set n_threads to at least 2
#' iris_nn <- nnd_knn(iris, k = 4, metric = "manhattan", n_threads = 1)
#'
#' # Use verbose flag to see information about progress
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#'
#' # Nearest neighbor descent uses random initialization, but you can pass any
#' # approximation using the init argument (as long as the metrics used to
#' # calculate the initialization are compatible with the metric options used
#' # by nnd_knn).
#' iris_nn <- random_knn(iris, k = 4, metric = "euclidean")
#' iris_nn <- nnd_knn(iris, init = iris_nn, metric = "euclidean", verbose = TRUE)
#'
#' # Number of iterations controls how much optimization is attempted. A smaller
#' # value will run faster but give poorer results
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 2)
#'
#' # You can also control the amount of work done within an iteration by
#' # setting max_candidates
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", max_candidates = 50)
#'
#' # Optimization may also stop early if not much progress is being made. This
#' # convergence criterion can be controlled via delta. A larger value will
#' # stop progress earlier. The verbose flag will provide some information if
#' # convergence is occurring before all iterations are carried out.
#' set.seed(1337)
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 5, delta = 0.5)
#'
#' # To ensure that descent only stops if no improvements are made, set delta = 0
#' set.seed(1337)
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", n_iters = 5, delta = 0)
#'
#' # A faster version of the algorithm is available that avoids repeated
#' # distance calculations at the cost of using more RAM. Set low_memory to
#' # FALSE to try it.
#' set.seed(1337)
#' iris_nn <- nnd_knn(iris, k = 4, metric = "euclidean", low_memory = FALSE)
#'
#' # Using init = "tree" is usually more efficient than random initialization.
#' # arguments to the tree initialization method can be passed via the init_args
#' # list
#' set.seed(1337)
#' iris_nn <- nnd_knn(iris, k = 4, init = "tree", init_args = list(n_trees = 5))
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
#' @export
nnd_knn <- function(data,
                    k = NULL,
                    metric = "euclidean",
                    init = "rand",
                    init_args = NULL,
                    n_iters = NULL,
                    max_candidates = NULL,
                    delta = 0.001,
                    low_memory = TRUE,
                    use_alt_metric = TRUE,
                    n_threads = 0,
                    verbose = FALSE,
                    progress = "bar",
                    obs = "R",
                    ret_forest = FALSE) {
  stopifnot(tolower(progress) %in% c("bar", "dist"))
  obs <- match.arg(toupper(obs), c("C", "R"))

  actual_metric <-
    get_actual_metric(use_alt_metric, metric, data, verbose)

  data <- x2m(data)
  if (obs == "R") {
    data <- Matrix::t(data)
  }

  # data must be column-oriented at this point
  if (is.character(init)) {
    if (is.null(k)) {
      stop("Must provide k")
    }
    check_k(k, ncol(data))
    init <- match.arg(tolower(init), c("rand", "tree"))

    # if the use provided tree init args that could clash with nnd args
    # ignore them
    if (init == "tree" && is.list(init_args)) {
      if ("use_alt_metric" %in% init_args) {
        init_args$use_alt_metric <- use_alt_metric
      }
      if ("metric" %in% init_args) {
        init_args$metric <- metric
      }
    }
    tsmessage("Initializing neighbors using '", init, "' method")
    init <- switch(init,
      "rand" = random_knn_impl(
        reference = data,
        k = k,
        actual_metric = actual_metric,
        order_by_distance = FALSE,
        n_threads = n_threads,
        verbose = verbose
      ),
      "tree" = do.call(rpf_knn_impl, lmerge(
        # defaults we have no reason to change should match rpf_knn or rpf_build
        list(
          data,
          k = k,
          metric = metric,
          use_alt_metric = use_alt_metric,
          actual_metric = actual_metric,
          n_trees = NULL,
          leaf_size = NULL,
          max_tree_depth = 200,
          include_self = FALSE, # this is changed from default on purpose
          ret_forest = ret_forest,
          margin = "auto",
          unzero = FALSE,
          n_threads = n_threads,
          verbose = verbose
        ),
        init_args
      )),
      stop("Unknown initialization option '", init, "'")
    )
    # FIXME: can we just turn off unzero in tree and random return?
    init$idx <- init$idx + 1
    if (any(init$idx == 0)) {
      tsmessage(
        "Warning: initialization failed to find ",
        k,
        " neighbors for all points"
      )
    }
  } else {
    tsmessage("Initializing from user-supplied graph")
    # user-supplied input may need to be transformed to the actual metric
    if (use_alt_metric &&
      !is.null(init) && is.list(init) && !is.null(init$dist)) {
      tsmessage(
        "Applying metric correction to initial distances from '",
        metric, "' to '", actual_metric, "'"
      )
      init$dist <-
        apply_alt_metric_uncorrection(metric, init$dist, is_sparse(data))
    }

    if (is.null(k)) {
      k <- ncol(init$idx)
    }
  }
  forest <- NULL
  if (ret_forest && is.list(init) && !is.null(init$forest)) {
    forest <- init$forest
    init$forest <- NULL
  }

  init <-
    prepare_init_graph(
      init,
      k,
      data = data,
      metric = actual_metric,
      n_threads = n_threads,
      verbose = verbose
    )

  if (is.null(max_candidates)) {
    max_candidates <- min(k, 60)
  }
  if (is.null(n_iters)) {
    n_iters <- max(5, round(log2(ncol(data))))
  }
  tsmessage(
    thread_msg(
      "Running nearest neighbor descent for ",
      n_iters,
      " iterations",
      n_threads = n_threads
    )
  )

  nnd_args <- list(
    nn_idx = init$idx,
    nn_dist = init$dist,
    metric = actual_metric,
    n_iters = n_iters,
    max_candidates = max_candidates,
    delta = delta,
    low_memory = low_memory,
    n_threads = n_threads,
    verbose = verbose,
    progress_type = progress
  )
  if (is_sparse(data)) {
    nnd_fun <- rnn_sparse_descent
    nnd_args$data <- data@x
    nnd_args$ind <- data@i
    nnd_args$ptr <- data@p
    nnd_args$ndim <- nrow(data)
  } else if (is.logical(data)) {
    nnd_fun <- rnn_logical_descent
    nnd_args$data <- data
  } else {
    nnd_fun <- rnn_descent
    nnd_args$data <- data
  }
  res <- do.call(nnd_fun, nnd_args)

  if (use_alt_metric) {
    res$dist <-
      apply_alt_metric_correction(metric, res$dist, is_sparse(data))
  }
  if (any(res$idx == 0)) {
    tsmessage(
      "Warning: NN Descent failed to find ",
      k,
      " neighbors for all points"
    )
  }
  tsmessage("Finished")
  if (!is.null(forest)) {
    res$forest <- forest
  }
  res
}


# kNN Queries -------------------------------------------------------------

#' Query exact nearest neighbors by brute force
#'
#' Returns the exact nearest neighbors of query data to the reference data. A
#' brute force search is carried out: all possible pairs of reference and query
#' points are compared, and the nearest neighbors are returned.
#'
#' This is accurate but scales poorly with dataset size, so use with caution
#' with larger datasets. Having the exact neighbors as a ground truth to compare
#' with approximate results is useful for benchmarking and determining
#' parameter settings of the approximate methods.
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
#'   calculated from this data. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `query` data must be passed in the same format and
#'   orientation as `reference`. Possible formats are [base::data.frame()],
#'   [base::matrix()] or [Matrix::sparseMatrix()]. Sparse matrices should be in
#'   `dgCMatrix` format.
#' @param k Number of nearest neighbors to return.
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `query` and `reference`
#'   orientation stores each observation as a column (the orientation must be
#'   consistent). The default `"R"` means that observations are stored in each
#'   row. Storing the data by row is usually more convenient, but internally
#'   your data will be converted to column storage. Passing it already
#'   column-oriented will save some memory and (a small amount of) CPU usage.
#' @return the nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices in
#'   `reference`.
#'   * `dist` an n by k matrix containing the nearest neighbor distances to the
#'   items in `reference`.
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # For each item in iris_query find the 4 nearest neighbors in iris_ref
#' # If you pass a data frame, non-numeric columns are removed
#' # set verbose = TRUE to get details on the progress being made
#' iris_query_nn <- brute_force_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#'
#' # Manhattan (l1) distance
#' iris_query_nn <- brute_force_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "manhattan"
#' )
#' @export
brute_force_knn_query <- function(query,
                                  reference,
                                  k,
                                  metric = "euclidean",
                                  use_alt_metric = TRUE,
                                  n_threads = 0,
                                  verbose = FALSE,
                                  obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))

  actual_metric <- get_actual_metric(use_alt_metric, metric, reference, verbose)

  check_sparse(reference, query)
  reference <- x2m(reference)
  query <- x2m(query)
  if (obs == "R") {
    reference <- Matrix::t(reference)
    query <- Matrix::t(query)
  }
  check_k(k, ncol(reference))

  tsmessage(
    thread_msg(
      "Calculating brute force k-nearest neighbors from reference with k = ",
      k,
      n_threads = n_threads
    )
  )

  if (is_sparse(reference)) {
    res <- rnn_sparse_brute_force_query(
      ref_ind = reference@i,
      ref_ptr = reference@p,
      ref_data = reference@x,
      query_ind = query@i,
      query_ptr = query@p,
      query_data = query@x,
      ndim = nrow(query),
      nnbrs = k,
      metric = actual_metric,
      n_threads = n_threads,
      verbose = verbose
    )
  } else if (is.logical(reference)) {
    res <- rnn_logical_brute_force_query(reference,
      query,
      k,
      actual_metric,
      n_threads = n_threads,
      verbose = verbose
    )
  } else {
    res <- rnn_brute_force_query(reference,
      query,
      k,
      actual_metric,
      n_threads = n_threads,
      verbose = verbose
    )
  }
  res$idx <- res$idx + 1

  if (use_alt_metric) {
    res$dist <-
      apply_alt_metric_correction(metric, res$dist, is_sparse(reference))
  }
  tsmessage("Finished")

  res
}

#' Query nearest neighbors by random selection
#'
#' Run queries against reference data to return randomly selected neighbors.
#' This is not a useful query method on its own, but can be used with other
#' methods which require initialization.
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
#'   randomly selected from this data. Optionally, the data may be passed with
#'   the observations in the columns, by setting `obs = "C"`, which should be
#'   more efficient. The `query` data must be passed in the same orientation
#'   and format as `reference`. Possible formats are [base::data.frame()],
#'   [base::matrix()] or [Matrix::sparseMatrix()]. Sparse matrices should be in
#'   `dgCMatrix` format.
#' @param k Number of nearest neighbors to return.
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param order_by_distance If `TRUE` (the default), then results for each
#'   item are returned by increasing distance. If you don't need the results
#'   sorted, e.g. you are going to pass the results as initialization to another
#'   routine like [graph_knn_query()], set this to `FALSE` to save a
#'   small amount of computational time.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `query` and `reference`
#'   orientation stores each observation as a column (the orientation must be
#'   consistent). The default `"R"` means that observations are stored in each
#'   row. Storing the data by row is usually more convenient, but internally
#'   your data will be converted to column storage. Passing it already
#'   column-oriented will save some memory and (a small amount of) CPU usage.
#' @return an approximate nearest neighbor graph as a list containing:
#'   * `idx` an n by k matrix containing the nearest neighbor indices.
#'   * `dist` an n by k matrix containing the nearest neighbor distances.
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # For each item in iris_query find 4 random neighbors in iris_ref
#' # If you pass a data frame, non-numeric columns are removed
#' # set verbose = TRUE to get details on the progress being made
#' iris_query_random_nbrs <- random_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#'
#' # Manhattan (l1) distance
#' iris_query_random_nbrs <- random_knn_query(iris_query,
#'   reference = iris_ref,
#'   k = 4, metric = "manhattan"
#' )
#' @export
random_knn_query <-
  function(query,
           reference,
           k,
           metric = "euclidean",
           use_alt_metric = TRUE,
           order_by_distance = TRUE,
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

    actual_metric <-
      get_actual_metric(use_alt_metric, metric, reference, verbose)

    if (obs == "R") {
      reference <- Matrix::t(reference)
      query <- Matrix::t(query)
    }

    res <- random_knn_impl(
      reference = reference,
      query = query,
      k = k,
      actual_metric = actual_metric,
      order_by_distance = order_by_distance,
      n_threads = n_threads,
      verbose = verbose
    )
    res$idx <- res$idx + 1

    if (use_alt_metric) {
      res$dist <-
        apply_alt_metric_correction(metric, res$dist, is_sparse(reference))
    }

    tsmessage("Finished")
    res
  }

#' Query a search graph for nearest neighbors
#'
#' Run queries against a search graph, to return nearest neighbors taken from
#' the reference data used to build that graph.
#'
#' A greedy beam search is used to query the graph, combining two search pruning
#' strategies. The first, due to Iwasaki and Miyazaki (2018), only considers
#' new candidates within a relative distance of the current furthest neighbor
#' in the query's graph. The second, due to Harwood and Drummond (2016), puts a
#' limit on the absolute number of distance calculations to carry out. See the
#' `epsilon` and `max_search_fraction` parameters respectively.
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
#'   calculated from this data. Optionally, the data may be passed with the
#'   observations in the columns, by setting `obs = "C"`, which should be more
#'   efficient. The `query` data must be passed in the same format and
#'   orientation as `reference`. Possible formats are [base::data.frame()],
#'   [base::matrix()] or [Matrix::sparseMatrix()]. Sparse matrices should be in
#'   `dgCMatrix` format.
#' @param reference_graph Search graph of the `reference` data. A neighbor
#'   graph, such as that output from [nnd_knn()] can be used, but
#'   preferably a suitably prepared sparse search graph should be used, such as
#'   that output by [prepare_search_graph()].
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably the
#'   only reason to set this to `FALSE` is if you suspect that some sort of
#'   numeric issue is occurring with your data in the alternative code path. If
#'   a search forest is used for initialization via the `init` parameter, then
#'   the metric is fetched from there and this setting is ignored.
#' @param init Initial `query` neighbor graph to optimize. If not provided, `k`
#'   random neighbors are created. If provided, the input format must be one of:
#'   1. A list containing:
#'
#'       * `idx` an `n` by `k` matrix containing the nearest neighbor indices.
#'       * `dist` (optional) an `n` by `k` matrix containing the nearest neighbor
#'       distances.
#'
#'       If `k` and `init` are specified as arguments to this function, and the
#'       number of neighbors provided in `init` is not equal to `k` then:
#'
#'       * if `k` is smaller, only the `k` closest values in `init` are retained.
#'       * if `k` is larger, then random neighbors will be chosen to fill `init` to
#'       the size of `k`. Note that there is no checking if any of the random
#'       neighbors are duplicates of what is already in `init` so effectively fewer
#'       than `k` neighbors may be chosen for some observations under these
#'       circumstances.
#'
#'       If the input distances are omitted, they will be calculated for you.
#'  1. A random projection forest, such as that returned from [rpf_build()] or
#'     [rpf_knn()] with `ret_forest = TRUE`.
#' @param epsilon Controls trade-off between accuracy and search cost, as
#'   described by Iwasaki and Miyazaki (2018), by specifying a distance
#'   tolerance on whether to explore the neighbors of candidate points. The
#'   larger the value, the more neighbors will be searched. A value of 0.1
#'   allows query-candidate distances to be 10% larger than the current
#'   most-distant neighbor of the query point, 0.2 means 20%, and so on.
#'   Suggested values are between 0-0.5, although this value is highly dependent
#'   on the distribution of distances in the dataset (higher dimensional data
#'   should choose a smaller cutoff). Too large a value of `epsilon` will result
#'   in the query search approaching brute force comparison. Use this parameter
#'   in conjunction with `max_search_fraction` and [prepare_search_graph()] to
#'   prevent excessive run time. Default is 0.1. If you set `verbose = TRUE`,
#'   statistics of the number of distance calculations will be logged which
#'   can help you tune `epsilon`.
#' @param max_search_fraction Maximum fraction of the reference data to search.
#'  This is a value between 0 (search none of the reference data) and 1 (search
#'  all of the data if necessary). This works in conjunction with `epsilon` and
#'  will terminate the search early if the specified fraction of the reference
#'  data has been searched. Default is 1.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `query` and `reference`
#'   orientation stores each observation as a column (the orientation must be
#'   consistent). The default `"R"` means that observations are stored in each
#'   row. Storing the data by row is usually more convenient, but internally
#'   your data will be converted to column storage. Passing it already
#'   column-oriented will save some memory and (a small amount of) CPU usage.
#' @return the approximate nearest neighbor graph as a list containing:
#'   * `idx` a `n` by `k` matrix containing the nearest neighbor indices
#'     specifying the row of the neighbor in `reference`.
#'   * `dist` a `n` by `k` matrix containing the nearest neighbor distances.
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # First, find the approximate 4-nearest neighbor graph for the references:
#' iris_ref_graph <- nnd_knn(iris_ref, k = 4)
#'
#' # For each item in iris_query find the 4 nearest neighbors in iris_ref.
#' # You need to pass both the reference data and the reference graph.
#' # If you pass a data frame, non-numeric columns are removed.
#' # set verbose = TRUE to get details on the progress being made
#' iris_query_nn <- graph_knn_query(iris_query, iris_ref, iris_ref_graph,
#'   k = 4, metric = "euclidean", verbose = TRUE
#' )
#'
#' # A more complete example, converting the initial knn into a search graph
#' # and using a filtered random projection forest to initialize the search
#' # create initial knn and forest
#' iris_ref_graph <- nnd_knn(iris_ref, k = 4, init = "tree", ret_forest = TRUE)
#' # keep the best tree in the forest
#' forest <- rpf_filter(iris_ref_graph, n_trees = 1)
#' # expand the knn into a search graph
#' iris_ref_search_graph <- prepare_search_graph(iris_ref, iris_ref_graph)
#' # run the query with the improved graph and initialization
#' iris_query_nn <- graph_knn_query(iris_query, iris_ref, iris_ref_search_graph,
#'   init = forest, k = 4
#' )
#'
#' @references
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
#' (pp. 5713-5722).
#'
#' Iwasaki, M., & Miyazaki, D. (2018).
#' Optimization of indexing based on k-nearest neighbor graph for proximity
#' search in high-dimensional data.
#' *arXiv preprint arXiv:1810.07355*.
#' @export
graph_knn_query <- function(query,
                            reference,
                            reference_graph,
                            k = NULL,
                            metric = "euclidean",
                            init = NULL,
                            epsilon = 0.1,
                            max_search_fraction = 1.0,
                            use_alt_metric = TRUE,
                            n_threads = 0,
                            verbose = FALSE,
                            obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))
  check_sparse(reference, query)
  reference <- x2m(reference)
  query <- x2m(query)
  if (obs == "R") {
    reference <- Matrix::t(reference)
    query <- Matrix::t(query)
  }

  if (is.list(init) && is_rpforest(init)) {
    tsmessage("Reading metric data from forest")
    actual_metric <- init$actual_metric
    metric <- init$original_metric
    use_alt_metric <- init$use_alt_metric
    if (use_alt_metric && metric != actual_metric) {
      tsmessage("Using alt metric '", actual_metric, "' for '", metric, "'")
    } else {
      tsmessage("Using metric '", metric, "'")
    }
  } else {
    actual_metric <-
      get_actual_metric(use_alt_metric, metric, reference, verbose)
  }

  # reference and query must be column-oriented at this point
  if (is.null(init)) {
    if (is.null(k)) {
      if (is.list(reference_graph)) {
        k <- get_reference_graph_k(reference_graph)
        tsmessage("Using k = ", k, " from graph")
      } else {
        stop("Must provide k")
      }
    }
    check_k(k, ncol(reference))
    tsmessage("Initializing from random neighbors")
    init <- random_knn_impl(
      query = query,
      reference = reference,
      k = k,
      actual_metric = actual_metric,
      order_by_distance = FALSE,
      n_threads = n_threads,
      verbose = verbose
    )
    # FIXME: can we just do the unzeroing inside the init?
    init$idx <- init$idx + 1
  } else if (is.list(init) && is_rpforest(init)) {
    init <- rpf_knn_query(
      query = query,
      reference = reference,
      forest = init,
      k = k,
      cache = TRUE,
      n_threads = n_threads,
      verbose = verbose,
      obs = "C"
    )
    if (use_alt_metric) {
      init$dist <-
        apply_alt_metric_uncorrection(metric, init$dist, is_sparse(reference))
    }
  } else if ((is.list(init) && !is.null(init$idx)) || is.matrix(init)) {
    tsmessage("Initializing from user-supplied graph")
    if (is.matrix(init)) {
      init <- list(idx = init)
    }

    # user-supplied distances may need to be transformed to the actual metric
    if (use_alt_metric && !is.null(init$dist)) {
      init$dist <-
        apply_alt_metric_uncorrection(metric, init$dist, is_sparse(reference))
    }

    if (is.null(k)) {
      k <- ncol(init$idx)
      tsmessage("Using k = ", k, " from initial graph")
    }
  } else {
    stop("Unsupported type of 'init'")
  }

  init <-
    prepare_init_graph(
      nn = init,
      k = k,
      query = query,
      data = reference,
      metric = actual_metric,
      n_threads = n_threads,
      verbose = verbose
    )

  if (is.list(reference_graph) && any(reference_graph$idx == 0)) {
    tsmessage("Warning: reference knn graph contains missing data")
  }

  stopifnot(
    !is.null(query),
    (
      methods::is(query, "matrix") ||
        methods::is(query, "sparseMatrix")
    )
  )
  stopifnot(
    !is.null(init$idx),
    methods::is(init$idx, "matrix"),
    ncol(init$idx) == k,
    nrow(init$idx) == ncol(query)
  )
  stopifnot(
    !is.null(init$dist),
    methods::is(init$dist, "matrix"),
    ncol(init$dist) == k,
    nrow(init$dist) == ncol(query)
  )
  if (is.list(reference_graph)) {
    reference_dist <- reference_graph$dist
    reference_idx <- reference_graph$idx
    stopifnot(!is.null(reference), (
      methods::is(reference, "matrix") ||
        methods::is(reference, "sparseMatrix")
    ))
    stopifnot(
      !is.null(reference_idx),
      methods::is(reference_idx, "matrix"),
      nrow(reference_idx) == ncol(reference)
    )
    stopifnot(
      !is.null(reference_dist),
      methods::is(reference_dist, "matrix"),
      nrow(reference_dist) == ncol(reference)
    )
    reference_graph_list <- graph_to_list(reference_graph)
  } else {
    stopifnot(methods::is(reference_graph, "sparseMatrix"))
    reference_graph_list <- tcsparse_to_list(reference_graph)
  }

  tsmessage(
    thread_msg(
      "Searching nearest neighbor graph with epsilon = ",
      epsilon,
      " and max_search_fraction = ",
      max_search_fraction,
      n_threads = n_threads
    )
  )

  args <- list(
    reference_graph_list = reference_graph_list,
    nn_idx = init$idx,
    nn_dist = init$dist,
    metric = actual_metric,
    epsilon = epsilon,
    max_search_fraction = max_search_fraction,
    n_threads = n_threads,
    verbose = verbose
  )
  if (is_sparse(reference)) {
    res <- do.call(
      rnn_sparse_query,
      c(
        list(
          ref_ind = reference@i,
          ref_ptr = reference@p,
          ref_data = reference@x,
          query_ind = query@i,
          query_ptr = query@p,
          query_data = query@x,
          ndim = nrow(reference)
        ),
        args
      )
    )
  } else if (is.logical(reference)) {
    res <- do.call(
      rnn_logical_query,
      c(
        list(reference = reference, query = query),
        args
      )
    )
  } else {
    res <- do.call(
      rnn_query,
      c(
        list(reference = reference, query = query),
        args
      )
    )
  }
  if (use_alt_metric) {
    res$dist <-
      apply_alt_metric_correction(metric, res$dist, is_sparse(reference))
  }
  tsmessage("Finished")
  res
}

# Search Graph Preparation ------------------------------------------------

#' Convert a nearest neighbor graph into a search graph
#'
#' Create a graph using existing nearest neighbor data to balance search
#' speed and accuracy using the occlusion pruning and truncation strategies
#' of Harwood and Drummond (2016). The resulting search graph should be more
#' efficient for querying new data than the original nearest neighbor graph.
#'
#' An approximate nearest neighbor graph is not very useful for querying via
#' [graph_knn_query()], especially if the query data is initialized randomly:
#' some items in the data set may not be in the nearest neighbor list of any
#' other item and can therefore never be returned as a neighbor, no matter how
#' close they are to the query. Even those which do appear in at least one
#' neighbor list may not be reachable by expanding an arbitrary starting list if
#' the neighbor graph contains disconnected components.
#'
#' Converting the directed graph represented by the neighbor graph to an
#' undirected graph by adding an edge from item `j` to `i` if
#' an edge exists from `i` to `j` (i.e. creating the mutual neighbor
#' graph) solves the problems above, but can result in inefficient searches.
#' Although the out-degree of each item is restricted to the number of neighbors
#' the in-degree has no such restrictions: a given item could be very "popular"
#' and in a large number of neighbors lists. Therefore mutualizing the neighbor
#' graph can result in some items with a large number of neighbors to search.
#' These usually have very similar neighborhoods so there is nothing to be
#' gained from searching all of them.
#'
#' To balance accuracy and search time, the following procedure is carried out:
#'
#' 1. The graph is "diversified" by occlusion pruning.
#' 1. The reverse graph is formed by reversing the direction of all edges in
#' the pruned graph.
#' 1. The reverse graph is diversified by occlusion pruning.
#' 1. The pruned forward and pruned reverse graph are merged.
#' 1. The outdegree of each node in the merged graph is truncated.
#' 1. The truncated merged graph is returned as the prepared search graph.
#'
#' Explicit zero distances in the `graph` will be converted to a small positive
#' number to avoid being dropped in the sparse representation. The one exception
#' is the "self" distance, i.e. any edge in the `graph` which links a node to
#' itself (the diagonal of the sparse distance matrix). These trivial edges
#' aren't useful for search purposes and are always dropped.
#'
#' @param data Matrix of `n` items, with observations in the rows and features
#'   in the columns. Optionally, input can be passed with observations in the
#'   columns, by setting `obs = "C"`, which should be more efficient. Possible
#'   formats are [base::data.frame()], [base::matrix()] or
#'   [Matrix::sparseMatrix()]. Sparse matrices should be in `dgCMatrix` format.
#'   Dataframes will be converted to `numerical` matrix format internally, so if
#'   your data columns are `logical` and intended to be used with the
#'   specialized binary `metric`s, you should convert it to a logical matrix
#'   first (otherwise you will get the slower dense numerical version).
#' @param graph neighbor graph for `data`, a list containing:
#'   * `idx` an `n` by `k` matrix containing the nearest neighbor indices of
#'   the data in `data`.
#'   * `dist` an `n` by `k` matrix containing the nearest neighbor distances.
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
#' @param use_alt_metric If `TRUE`, use faster metrics that maintain the
#'   ordering of distances internally (e.g. squared Euclidean distances if using
#'   `metric = "euclidean"`), then apply a correction at the end. Probably
#'   the only reason to set this to `FALSE` is if you suspect that some
#'   sort of numeric issue is occurring with your data in the alternative code
#'   path.
#' @param diversify_prob the degree of diversification of the search graph
#'   by removing unnecessary edges through occlusion pruning. This should take a
#'   value between `0` (no diversification) and `1` (remove as many edges as
#'   possible) and is treated as the probability of a neighbor being removed if
#'   it is found to be an "occlusion". If item `p` and `q`, two members of the
#'   neighbor list of item `i`, are closer to each other than they are to `i`,
#'   then the nearer neighbor `p` is said to "occlude" `q`. It is likely that
#'   `q` will be in the neighbor list of `p` so there is no need to retain it in
#'   the neighbor list of `i`. You may also set this to `NULL` to skip any
#'   occlusion pruning. Note that occlusion pruning is carried out twice, once
#'   to the forward neighbors, and once to the reverse neighbors. Reducing this
#'   value will result in a more dense graph. This is similar to increasing the
#'   "alpha" parameter used by in the DiskAnn pruning method of Subramanya and
#'   co-workers (2014).
#' @param pruning_degree_multiplier How strongly to truncate the final neighbor
#'   list for each item. The neighbor list of each item will be truncated to
#'   retain only the closest `d` neighbors, where
#'   `d = k * pruning_degree_multiplier`, and `k` is the
#'   original number of neighbors per item in `graph`. Roughly, values
#'   larger than `1` will keep all the nearest neighbors of an item, plus
#'   the given fraction of reverse neighbors (if they exist). For example,
#'   setting this to `1.5` will keep all the forward neighbors and then
#'   half as many of the reverse neighbors, although exactly which neighbors are
#'   retained is also dependent on any occlusion pruning that occurs. Set this
#'   to `NULL` to skip this step.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @param obs set to `"C"` to indicate that the input `data` orientation stores
#'   each observation as a column. The default `"R"` means that observations are
#'   stored in each row. Storing the data by row is usually more convenient, but
#'   internally your data will be converted to column storage. Passing it
#'   already column-oriented will save some memory and (a small amount of) CPU
#'   usage.
#' @return a search graph for `data` based on `graph`, represented as a sparse
#'   matrix, suitable for use with [graph_knn_query()].
#' @examples
#' # 100 reference iris items
#' iris_ref <- iris[iris$Species %in% c("setosa", "versicolor"), ]
#'
#' # 50 query items
#' iris_query <- iris[iris$Species == "versicolor", ]
#'
#' # First, find the approximate 4-nearest neighbor graph for the references:
#' ref_ann_graph <- nnd_knn(iris_ref, k = 4)
#'
#' # Create a graph for querying with
#' ref_search_graph <- prepare_search_graph(iris_ref, ref_ann_graph)
#'
#' # Using the search graph rather than the ref_ann_graph directly may give
#' # more accurate or faster results
#' iris_query_nn <- graph_knn_query(
#'   query = iris_query, reference = iris_ref,
#'   reference_graph = ref_search_graph, k = 4, metric = "euclidean",
#'   verbose = TRUE
#' )
#' @references
#' Harwood, B., & Drummond, T. (2016).
#' Fanng: Fast approximate nearest neighbour graphs.
#' In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
#' (pp. 5713-5722).
#'
#' Jayaram Subramanya, S., Devvrit, F., Simhadri, H. V., Krishnawamy, R., & Kadekodi, R. (2019).
#' Diskann: Fast accurate billion-point nearest neighbor search on a single node.
#' *Advances in Neural Information Processing Systems*, *32*.
#' @seealso [graph_knn_query()]
#' @export
prepare_search_graph <- function(data,
                                 graph,
                                 metric = "euclidean",
                                 use_alt_metric = TRUE,
                                 diversify_prob = 1.0,
                                 pruning_degree_multiplier = 1.5,
                                 n_threads = 0,
                                 verbose = FALSE,
                                 obs = "R") {
  obs <- match.arg(toupper(obs), c("C", "R"))

  if (!is.null(pruning_degree_multiplier)) {
    stopifnot(pruning_degree_multiplier > 0)
  }
  if (!is.null(diversify_prob)) {
    stopifnot(
      diversify_prob <= 1,
      diversify_prob >= 0
    )
  }

  actual_metric <-
    get_actual_metric(use_alt_metric, metric, data, verbose)

  if (is_sparse(graph)) {
    sp <- Matrix::t(graph)
    n_nbrs <- mean(diff(sp@p))
  } else {
    n_nbrs <- check_graph(graph)$k
    tsmessage("Converting graph to sparse format")
    sp <- graph_to_csparse(graph)
  }

  sp <- preserve_zeros(sp)

  if (use_alt_metric) {
    sp@x <-
      apply_alt_metric_uncorrection(metric, sp@x, is_sparse(data))
  }

  data <- x2m(data)
  if (obs == "R") {
    data <- Matrix::t(data)
  }
  if (!is.null(diversify_prob) && diversify_prob > 0) {
    tsmessage("Diversifying forward graph")
    fdiv <- diversify(
      data,
      sp,
      metric = actual_metric,
      prune_probability = diversify_prob,
      verbose = verbose,
      n_threads = n_threads
    )
  } else {
    fdiv <- sp
    tsmessage(
      "Forward graph has # edges = ",
      Matrix::nnzero(fdiv),
      " (",
      formatC(100 * nn_sparsity_sp(fdiv)),
      "% sparse)"
    )
  }
  rsp <- reverse_knn_sp(fdiv)
  if (!is.null(diversify_prob) && diversify_prob > 0) {
    tsmessage("Diversifying reverse graph")
    rdiv <- diversify(
      data,
      rsp,
      metric = actual_metric,
      prune_probability = diversify_prob,
      verbose = verbose,
      n_threads = n_threads
    )
  } else {
    rdiv <- rsp
    tsmessage(
      "Reverse graph has # edges = ",
      Matrix::nnzero(rdiv),
      " (",
      formatC(100 * nn_sparsity_sp(rdiv)),
      "% sparse)"
    )
  }
  tsmessage("Merging diversified forward and reverse graph")
  merged <- merge_graphs_sp(fdiv, rdiv)

  if (!is.null(pruning_degree_multiplier) &&
    !is.infinite(pruning_degree_multiplier)) {
    max_degree <- max(round(n_nbrs * pruning_degree_multiplier), 1)
    tsmessage("Degree pruning merged graph to max degree: ", max_degree)
    res <-
      degree_prune(
        merged,
        max_degree = max_degree,
        verbose = verbose,
        n_threads = n_threads
      )
  } else {
    res <- merged
    tsmessage(
      "Merged graph has # edges = ",
      Matrix::nnzero(res),
      " (",
      formatC(100 * nn_sparsity_sp(res)),
      "% sparse)"
    )
  }

  if (use_alt_metric) {
    res@x <-
      apply_alt_metric_correction(metric, res@x, is_sparse(data))
  }

  tsmessage("Finished preparing search graph")
  # return this pre-transposed so we don't have to do it in graph_knn_query
  Matrix::t(res)
}

# Harwood, B., & Drummond, T. (2016).
# Fanng: Fast approximate nearest neighbour graphs.
# In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
# (pp. 5713-5722).
# "Occlusion pruning"
#
# prune_probability behaves a bit like (but in the opposite direction of) alpha
# Jayaram Subramanya, S., Devvrit, F., Simhadri, H. V., Krishnawamy, R., & Kadekodi, R. (2019).
# Diskann: Fast accurate billion-point nearest neighbor search on a single node.
# Advances in Neural Information Processing Systems, 32.
# the pynndescent implementation at cmuparlay/pbbsbench uses alpha instead of
# a prune_probability (see also https://arxiv.org/abs/2305.04359)
diversify <- function(data,
                      graph,
                      metric = "euclidean",
                      prune_probability = 1.0,
                      n_threads = 0,
                      verbose = FALSE) {
  nnz_before <- Matrix::nnzero(graph)
  sp_before <- nn_sparsity_sp(graph)
  stopifnot(methods::is(graph, "sparseMatrix"))
  gl <- csparse_to_list(graph)

  tsmessage("Occlusion pruning with probability: ", prune_probability)
  args <- list(
    graph_list = gl,
    metric = metric,
    prune_probability = prune_probability,
    n_threads = n_threads,
    verbose = verbose
  )
  if (is_sparse(data)) {
    gl_div <- do.call(rnn_sparse_diversify, c(args, list(
      ind = data@i,
      ptr = data@p,
      data = data@x,
      ndim = nrow(data)
    )))
  } else if (is.logical(data)) {
    gl_div <- do.call(rnn_logical_diversify, c(args, list(
      data = data
    )))
  } else {
    gl_div <- do.call(rnn_diversify, c(args, list(
      data = data
    )))
  }
  res <- list_to_sparse(gl_div)
  nnz_after <- Matrix::nnzero(res)
  tsmessage(
    "Diversifying reduced # edges from ",
    nnz_before,
    " to ",
    nnz_after,
    " (",
    formatC(100 * sp_before),
    "% to ",
    formatC(100 * nn_sparsity_sp(res)),
    "% sparse)"
  )
  res
}

# FANNG: Fast Approximate Nearest Neighbour Graphs
# "truncating"
# Harwood, B., & Drummond, T. (2016).
# Fanng: Fast approximate nearest neighbour graphs.
# In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
# (pp. 5713-5722).
degree_prune <-
  function(graph,
           max_degree = 20,
           n_threads = 0,
           verbose = FALSE) {
    stopifnot(methods::is(graph, "sparseMatrix"))
    nnz_before <- Matrix::nnzero(graph)
    sp_before <- nn_sparsity_sp(graph)
    gl <- csparse_to_list(graph)

    gl_div <-
      rnn_degree_prune(gl, max_degree, n_threads = n_threads)
    res <- list_to_sparse(gl_div)
    nnz_after <- Matrix::nnzero(res)
    tsmessage(
      "Degree pruning to max ",
      max_degree,
      " reduced # edges from ",
      nnz_before,
      " to ",
      nnz_after,
      " (",
      formatC(100 * sp_before),
      "% to ",
      formatC(100 * nn_sparsity_sp(res)),
      "% sparse)"
    )
    res
  }


merge_graphs_sp <- function(g1, g2) {
  gl1 <- csparse_to_list(g1)
  gl2 <- csparse_to_list(g2)

  gl_merge <- rnn_merge_graph_lists(gl1, gl2)
  list_to_sparse(gl_merge)
}

nn_sparsity <- function(graph) {
  sum(graph$idx == 0) / prod(dim(graph$idx))
}

nn_sparsity_sp <- function(graph) {
  Matrix::nnzero(graph) / prod(graph@Dim)
}

reverse_knn_sp <- function(graph) {
  stopifnot(methods::is(graph, "sparseMatrix"))
  Matrix::t(Matrix::drop0(graph))
}

# Merge -------------------------------------------------------------------

#' Merge multiple approximate nearest neighbors graphs
#'
#' `merge_knn` takes a list of nearest neighbor graphs and merges them into a
#' single graph, with the same number of neighbors as the first graph. This is
#' useful to combine the results of multiple different nearest neighbor
#' searches: the output will be at least as accurate as the most accurate of the
#' two input graphs, and ideally will be more accurate than either.
#'
#' @param graphs A list of nearest neighbor graphs to merge. Each item in the
#'   list should consist of a sub-list containing:
#'   * `idx` an n by k matrix containing the k nearest neighbor indices.
#'   * `dist` an n by k matrix containing k nearest neighbor distances.
#'   The number of neighbors can differ between graphs, but the merged result
#'   will have the same number of neighbors as the first graph in the list.
#' @param is_query If `TRUE` then the graphs are treated as the result of a knn
#'   query, not a knn building process. Or: is the graph bipartite? This should
#'   be set to `TRUE` if `nn_graphs` are the results of using e.g.
#'   [graph_knn_query()] or [random_knn_query()], and set to `FALSE` if these
#'   are the results of [nnd_knn()] or [random_knn()]. The difference is that if
#'   `is_query = FALSE`, if an index `p` is found in `nn_graph1[i, ]`, i.e. `p`
#'   is a neighbor of `i` with distance `d`, then it is assumed that `i` is a
#'   neighbor of `p` with the same distance. If `is_query = TRUE`, then `i` and
#'   `p` are indexes into two different datasets and the symmetry does not hold.
#'   If you aren't sure what case applies to you, it's safe (but potentially
#'   inefficient) to set `is_query = TRUE`.
#' @param n_threads Number of threads to use.
#' @param verbose If `TRUE`, log information to the console.
#' @return a list containing:
#'   * `idx` an n by k matrix containing the merged nearest neighbor indices.
#'   * `dist` an n by k matrix containing the merged nearest neighbor distances.
#'
#'   The size of `k` in the output graph is the same as that of the first
#'   item in `nn_graphs`.
#' @examples
#' set.seed(1337)
#' # Nearest neighbor descent with 15 neighbors for iris three times,
#' # starting from a different random initialization each time
#' iris_rnn1 <- nnd_knn(iris, k = 15, n_iters = 1)
#' iris_rnn2 <- nnd_knn(iris, k = 15, n_iters = 1)
#' iris_rnn3 <- nnd_knn(iris, k = 15, n_iters = 1)
#'
#' # Merged results should be an improvement over individual results
#' iris_mnn <- merge_knn(list(iris_rnn1, iris_rnn2, iris_rnn3))
#' sum(iris_mnn$dist) < sum(iris_rnn1$dist)
#' sum(iris_mnn$dist) < sum(iris_rnn2$dist)
#' sum(iris_mnn$dist) < sum(iris_rnn3$dist)
#' @export
merge_knn <- function(graphs,
                      is_query = FALSE,
                      n_threads = 0,
                      verbose = FALSE) {
  if (length(graphs) == 0) {
    return(list())
  }
  validate_are_mergeablel(graphs)

  rnn_merge_nn_all(graphs,
    is_query,
    n_threads = n_threads,
    verbose = verbose
  )
}

# Overlap -----------------------------------------------------------------

#' Overlap between the indices of two nearest neighbor graphs
#'
#' Calculates the mean average number of neighbors in common between the two
#' graphs. The per-item overlap can also be returned. This function can be
#' useful as a measure of accuracy of approximation algorithms, if the
#' exact nearest neighbors are known, or as a measure of diversity of two
#' different approximate graphs.
#'
#' The graph format is the same as that returned by e.g. [nnd_knn()] and should
#' be of dimensions n by k, where n is the number of points and k is the number
#' of neighbors. If you pass a neighbor graph directly, the index matrix will be
#' extracted if present. If the two graphs have different numbers of neighbors,
#' then the smaller number of neighbors is used.
#'
#' @param idx1 Indices of a nearest neighbor graph, i.e. a matrix of nearest
#'   neighbor indices. Can also be a list containing an `idx` element.
#' @param idx2 Indices of a nearest neighbor graph, i.e. a matrix of nearest
#'   neighbor indices. Can also be a list containing an `idx` element. This is
#'   considered to be the ground truth.
#' @param k Number of neighbors to consider. If `NULL`, then the minimum of the
#'   number of neighbors in `idx1` and `idx2` is used.
#' @param ret_vec If `TRUE`, also return a vector containing the per-item overlap.
#' @return The mean overlap between `idx1` and `idx2`. If `ret_vec = TRUE`,
#'  then a list containing the mean overlap and the overlap of each item in
#'  is returned with names `mean` and `overlaps`, respectively.
#' @examples
#' set.seed(1337)
#' # Generate two random neighbor graphs for iris
#' iris_rnn1 <- random_knn(iris, k = 15)
#' iris_rnn2 <- random_knn(iris, k = 15)
#'
#' # Overlap between the two graphs
#' mean_overlap <- neighbor_overlap(iris_rnn1, iris_rnn2)
#'
#' # Also get a vector of per-item overlap
#' overlap_res <- neighbor_overlap(iris_rnn1, iris_rnn2, ret_vec = TRUE)
#' summary(overlap_res$overlaps)
#' @export
neighbor_overlap <-
  function(idx1,
           idx2,
           k = NULL,
           ret_vec = FALSE) {
    vec <- neighbor_overlapv(idx1, idx2, k)
    mean_overlap <- mean(vec)
    if (ret_vec) {
      res <- list(
        mean = mean_overlap,
        overlaps = vec
      )
    } else {
      res <- mean_overlap
    }
    res
  }

neighbor_overlapv <-
  function(idx,
           ref_idx,
           k = NULL) {
    if (is.list(idx)) {
      idx <- idx$idx
    }
    if (is.list(ref_idx)) {
      ref_idx <- ref_idx$idx
    }

    if (is.null(k)) {
      k <- min(ncol(idx), ncol(ref_idx))
    }

    if (ncol(ref_idx) < k) {
      stop("Not enough columns in ref_idx for k = ", k)
    }

    n <- nrow(idx)
    if (nrow(ref_idx) != n) {
      stop("Not enough rows in ref_idx")
    }

    nbr_start <- 1
    nbr_end <- k

    ref_start <- nbr_start
    ref_end <- nbr_end

    nbr_range <- nbr_start:nbr_end
    ref_range <- ref_start:ref_end

    total_intersect <- rep(0, times = n)
    for (i in 1:n) {
      total_intersect[i] <-
        length(intersect(idx[i, nbr_range], ref_idx[i, ref_range]))
    }

    total_intersect / k
  }
