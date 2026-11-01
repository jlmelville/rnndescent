test_that("supplied graph paths reject fractional neighbor identities", {
  data <- matrix(c(0, 1, 3, 6), ncol = 1)
  graph <- brute_force_knn(data, k = 2)
  graph$idx[1, 2] <- 2.5
  query <- data[1:2, , drop = FALSE]
  callers <- list(
    function(g) nnd_knn(data, k = 2, init = g, n_iters = 0),
    function(g) {
      nnd_knn(
        methods::as(Matrix::Matrix(data, sparse = TRUE), "dgCMatrix"),
        k = 2,
        init = g,
        n_iters = 0
      )
    },
    function(g) graph_knn_query(query, data, reference_graph = g, k = 2),
    function(g) {
      graph_knn_query(
        query,
        data,
        brute_force_knn(data, k = 2),
        k = 2,
        init = list(idx = g$idx[1:2, , drop = FALSE])
      )
    },
    function(g) prepare_search_graph(data, g),
    function(g) merge_knn(list(g)),
    function(g) merge_knn(list(g), is_query = TRUE)
  )
  for (call in callers) {
    expect_error(call(graph), "indices must be finite whole numbers")
  }
})

test_that("query merging rejects IDs that native integers cannot represent", {
  graph <- list(idx = matrix(c(12, 25), 2), dist = matrix(c(0.4, 0.7), 2))
  for (bad_id in c(2.5, Inf, .Machine$integer.max + 1)) {
    graph$idx[1, 1] <- bad_id
    expect_no_warning(
      expect_error(
        merge_knn(list(graph), is_query = TRUE),
        "indices must be finite whole numbers"
      )
    )
  }
})

test_that("whole-valued double IDs preserve supplied-graph results", {
  data <- matrix(c(0, 1, 3, 6), ncol = 1)
  graph <- brute_force_knn(data, k = 2)
  integer_graph <- graph
  storage.mode(integer_graph$idx) <- "integer"
  double_graph <- graph
  storage.mode(double_graph$idx) <- "double"
  callers <- list(
    function(g) nnd_knn(data, k = 2, init = g, n_iters = 0),
    function(g) graph_knn_query(data[1:2, , drop = FALSE], data, g, k = 2),
    function(g) prepare_search_graph(data, g),
    function(g) merge_knn(list(g)),
    function(g) merge_knn(list(g), is_query = TRUE)
  )
  for (call in callers) {
    set.seed(1934)
    integer_result <- call(integer_graph)
    integer_seed <- .Random.seed
    set.seed(1934)
    expect_identical(call(double_graph), integer_result)
    expect_identical(.Random.seed, integer_seed)
  }
})

test_that("partial query graphs retain missing rows and reference identities", {
  # Reference IDs can exceed the number of query rows.
  graph1 <- list(
    idx = rbind(c(12, 0, 0), c(0, 0, 0)),
    dist = rbind(c(0.4, NA, NA), c(NA, NA, NA))
  )
  graph2 <- graph1
  graph2$idx[1, 1] <- 25
  graph2$dist[1, 1] <- 0.7
  result <- merge_knn(list(graph1, graph2), is_query = TRUE)
  expect_equal(result$idx, rbind(c(12, 25, 0), c(0, 0, 0)))
  expect_equal(
    result$dist,
    rbind(c(0.4, 0.7, NA), c(NA, NA, NA)),
    tolerance = 1e-7
  )

  # The existing native interface can represent this reference identity.
  graph1$idx[1, 1] <- .Machine$integer.max
  expect_equal(
    merge_knn(list(graph1), is_query = TRUE)$idx,
    graph1$idx
  )
})
