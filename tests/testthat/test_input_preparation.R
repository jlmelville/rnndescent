test_that("building and querying an index prepare each feature input once", {
  data <- matrix(as.double(seq_len(24)), nrow = 8)
  original <- x2m
  calls <- 0L
  local_mocked_bindings(
    x2m = function(...) {
      calls <<- calls + 1L
      original(...)
    },
    .package = "rnndescent"
  )
  index <- rnnd_build(data, k = 2, n_trees = 1, n_iters = 1)
  expect_identical(calls, 1L)
  calls <- 0L
  rnnd_query(index, data, k = 2)
  expect_identical(calls, 1L)
})

test_that("index queries recheck the current stored reference on each call", {
  data <- matrix(as.double(seq_len(24)), nrow = 8)
  for (sparse in c(FALSE, TRUE)) {
    input <- if (sparse) Matrix::Matrix(data, sparse = TRUE) else data
    index <- rnnd_build(input, k = 2, n_trees = 1, n_iters = 1)
    supplied <- index$graph
    for (initializer in list(NULL, supplied)) {
      expect_no_error(rnnd_query(index, input, k = 2, init = initializer))
      bad_index <- index
      bad_index$data[1, 1] <- Inf
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
})

test_that("prepared forest query handles expire after success and error", {
  data <- matrix(as.double(seq_len(24)), nrow = 8)
  index <- rnnd_build(data, k = 2, n_trees = 1, n_iters = 1)
  native_query <- getFromNamespace("rnn_query", "rnndescent")
  traversal_args <- NULL
  fail <- FALSE
  local_mocked_bindings(
    rnn_query = function(...) {
      traversal_args <<- list(...)
      if (fail) {
        stop("traversal failed")
      }
      do.call(native_query, traversal_args)
    },
    .package = "rnndescent"
  )

  expect_no_error(rnnd_query(index, data, k = 2))
  expect_error(
    do.call(native_query, traversal_args),
    "external pointer is not valid"
  )

  fail <- TRUE
  expect_error(rnnd_query(index, data, k = 2), "traversal failed")
  expect_error(
    do.call(native_query, traversal_args),
    "external pointer is not valid"
  )
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
