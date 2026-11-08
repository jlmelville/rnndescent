# Counting the private converter through public calls protects the performance
# contract; ordinary output tests cannot distinguish redundant finite scans.
for (representation in c("dense", "integer", "logical", "sparse")) {
  for (orientation in c("R", "C")) {
    test_that(
      paste("feature preparation is owned once", representation, orientation),
      {
        data <- matrix(as.double(seq_len(24)), nrow = 8)
        metric <- "euclidean"
        if (representation == "integer") {
          storage.mode(data) <- "integer"
        }
        if (representation == "logical") {
          data <- data %% 3 == 0
          metric <- "hamming"
        }
        if (representation == "sparse") {
          data <- Matrix::Matrix(data, sparse = TRUE)
        }
        if (orientation == "C") {
          data <- Matrix::t(data)
        }
        graph <- brute_force_knn(
          data,
          k = 2,
          metric = metric,
          obs = orientation
        )
        forest <- rpf_build(
          data,
          n_trees = 1,
          metric = metric,
          obs = orientation
        )
        index <- rnnd_build(
          data,
          k = 2,
          n_trees = 1,
          n_iters = 1,
          metric = metric,
          obs = orientation
        )
        builders <- list(
          brute = function() {
            brute_force_knn(data, k = 2, metric = metric, obs = orientation)
          },
          random = function() {
            random_knn(data, k = 2, metric = metric, obs = orientation)
          },
          nnd_random = function() {
            nnd_knn(
              data,
              k = 2,
              n_iters = 1,
              metric = metric,
              obs = orientation
            )
          },
          nnd_tree = function() {
            nnd_knn(
              data,
              k = 2,
              init = "tree",
              init_args = list(n_trees = 1),
              n_iters = 1,
              metric = metric,
              obs = orientation
            )
          },
          rnnd_knn = function() {
            rnnd_knn(
              data,
              k = 2,
              n_trees = 1,
              n_iters = 1,
              metric = metric,
              obs = orientation
            )
          },
          rnnd_build = function() {
            rnnd_build(
              data,
              k = 2,
              n_trees = 1,
              n_iters = 1,
              metric = metric,
              obs = orientation
            )
          },
          rp_knn = function() {
            rpf_knn(
              data,
              k = 2,
              n_trees = 1,
              metric = metric,
              obs = orientation
            )
          },
          rp_build = function() {
            rpf_build(data, n_trees = 1, metric = metric, obs = orientation)
          },
          search_graph = function() {
            prepare_search_graph(
              data,
              graph,
              metric = metric,
              obs = orientation
            )
          }
        )
        # Equal/aliased values still represent a reference role and a query role.
        queries <- list(
          brute = function() {
            brute_force_knn_query(
              data,
              data,
              k = 2,
              metric = metric,
              obs = orientation
            )
          },
          random = function() {
            random_knn_query(
              data,
              data,
              k = 2,
              metric = metric,
              obs = orientation
            )
          },
          rp = function() {
            rpf_knn_query(data, data, forest, k = 2, obs = orientation)
          },
          graph_random = function() {
            graph_knn_query(
              data,
              data,
              graph,
              k = 2,
              metric = metric,
              obs = orientation
            )
          },
          graph_forest = function() {
            graph_knn_query(
              data,
              data,
              graph,
              init = forest,
              k = 2,
              obs = orientation
            )
          },
          graph_supplied = function() {
            graph_knn_query(
              data,
              data,
              graph,
              init = graph,
              k = 2,
              metric = metric,
              obs = orientation
            )
          },
          index_forest = function() {
            rnnd_query(index, data, k = 2, obs = orientation)
          },
          index_supplied = function() {
            rnnd_query(index, data, k = 2, init = graph, obs = orientation)
          },
          index_indices = function() {
            rnnd_query(index, data, k = 2, init = graph$idx, obs = orientation)
          },
          index_random = function() {
            index$search_forest <- NULL
            rnnd_query(index, data, k = 2, obs = orientation)
          }
        )
        original_x2m <- getFromNamespace("x2m", "rnndescent")
        roles <- character()
        local_mocked_bindings(
          x2m = function(X, obs = "R", name = "data") {
            value <- original_x2m(X, obs, name)
            roles <<- c(roles, name)
            value
          },
          .package = "rnndescent"
        )
        for (name in names(builders)) {
          roles <- character()
          invisible(builders[[name]]())
          expect_identical(roles, "data", info = name)
        }
        for (name in names(queries)) {
          roles <- character()
          invisible(queries[[name]]())
          expected <- if (
            representation == "dense" &&
              name %in% c("index_forest", "index_supplied")
          ) {
            "query"
          } else {
            c("query", "reference")
          }
          expect_identical(sort(roles), expected, info = name)
        }
      }
    )
  }
}

test_that("direct and wrapper NND retain valid lazy argument forcing", {
  for (wrapper in c(FALSE, TRUE)) {
    events <- character()
    set.seed(1934)
    expected <- runif(13)
    set.seed(1934)
    dqrng::dqset.seed(1934)
    fun <- if (wrapper) rnnd_knn else nnd_knn
    invisible(fun(
      data = {
        events <- c(events, "data")
        realized <- matrix(runif(12), 6, 2)
        realized
      },
      k = 2,
      init = "rand",
      n_iters = 1,
      delta = {
        events <- c(events, "delta")
        invisible(runif(1))
        .001
      }
    ))
    expect_identical(
      events,
      if (wrapper) c("data", "delta") else c("delta", "data")
    )
    expect_identical(
      as.vector(realized),
      if (wrapper) expected[1:12] else expected[2:13]
    )
  }
})

test_that("index queries recheck the current stored reference on each call", {
  data <- matrix(as.double(seq_len(24)), nrow = 8)
  for (sparse in c(FALSE, TRUE)) {
    input <- if (sparse) Matrix::Matrix(data, sparse = TRUE) else data
    index <- rnnd_build(input, k = 2, n_trees = 1, n_iters = 1)
    supplied <- index$graph
    for (initializer in list(NULL, supplied)) {
      expect_no_error(rnnd_query(index, input, k = 2, init = initializer))
      for (bad in if (sparse) Inf else c(NA_real_, NaN, Inf, -Inf)) {
        bad_index <- index
        bad_index$data[1, 1] <- bad
        before <- bad_index
        set.seed(1934)
        dqrng::dqset.seed(1934)
        rng_before <- list(R = .Random.seed, dq = dqrng::dqrng_get_state())
        expect_error(
          rnnd_query(bad_index, input, k = 2, init = initializer),
          "^reference must contain only finite numeric feature values$"
        )
        expect_identical(
          list(R = .Random.seed, dq = dqrng::dqrng_get_state()),
          rng_before
        )
        expect_identical(bad_index, before)
      }
    }
  }
})

test_that("forest metric overrides remain lazy on direct graph queries", {
  data <- matrix(seq_len(24), nrow = 8)
  graph <- brute_force_knn(data, k = 2)
  forest <- rpf_build(data, n_trees = 1)
  expect_no_error(graph_knn_query(
    data,
    data,
    graph,
    k = 2,
    init = forest,
    metric = stop("overridden metric was forced"),
    use_alt_metric = stop("overridden flag was forced")
  ))
})
