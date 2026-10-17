library(rnndescent)

sparse_zero_fixture <- function(x) {
  Matrix::sparseMatrix(
    i = rep(seq_len(nrow(x)), ncol(x)),
    j = rep(seq_len(ncol(x)), each = nrow(x)),
    x = as.vector(x),
    dims = dim(x)
  )
}

test_that("stored feature zeros do not change defined metric distances", {
  query <- matrix(c(1, 1, 0, 0), nrow = 1)
  reference <- rbind(c(1, 0, 1, 0), c(1, 0, 1, 1))
  # Both pairs share one true coordinate; mismatches are two and three.
  expected <- list(
    dice = c(1 / 2, 3 / 5),
    hamming = c(1 / 2, 3 / 4),
    matching = c(1 / 2, 3 / 4),
    jaccard = c(2 / 3, 3 / 4),
    kulsinski = c(5 / 6, 6 / 7),
    rogerstanimoto = c(2 / 3, 6 / 7),
    russellrao = c(3 / 4, 3 / 4),
    sokalmichener = c(2 / 3, 6 / 7),
    sokalsneath = c(4 / 5, 6 / 7),
    yule = c(1, 2),
    correlation = 1 - as.vector(stats::cor(t(query), t(reference)))
  )
  for (metric in names(expected)) {
    for (stored in c(FALSE, TRUE)) {
      q <- sparse_zero_fixture(query)
      r <- Matrix::drop0(sparse_zero_fixture(reference))
      if (!stored) {
        q <- Matrix::drop0(q)
        r <- Matrix::drop0(r)
      }
      neighbors <- brute_force_knn_query(q, r, k = 2, metric = metric)
      expect_setequal(as.vector(neighbors$idx), 1:2)
      expect_equal(
        as.vector(neighbors$dist),
        expected[[metric]][neighbors$idx],
        tolerance = 1e-6,
        info = metric
      )
    }
  }
})

test_that("feature zero handling covers query and reference representations", {
  query <- matrix(c(1, 1, 0, 0), nrow = 1)
  reference <- rbind(c(1, 0, 1, 0), c(1, 0, 1, 1))
  cases <- list(
    list(
      q = sparse_zero_fixture(query),
      r = Matrix::drop0(reference),
      obs = "R"
    ),
    list(
      q = Matrix::t(Matrix::drop0(query)),
      r = methods::as(
        Matrix::t(sparse_zero_fixture(reference)),
        "RsparseMatrix"
      ),
      obs = "C"
    ),
    list(
      q = methods::as(sparse_zero_fixture(query), "TsparseMatrix"),
      r = methods::as(sparse_zero_fixture(reference), "TsparseMatrix"),
      obs = "R"
    )
  )
  for (case in cases) {
    for (n_threads in c(0L, 2L)) {
      neighbors <- brute_force_knn_query(
        case$q,
        case$r,
        k = 1,
        metric = "dice",
        obs = case$obs,
        n_threads = n_threads
      )
      expect_equal(neighbors$idx, matrix(1L))
      expect_equal(neighbors$dist, matrix(1 / 2), tolerance = 1e-6)
    }
  }
})

test_that("public sparse builders and queries use numerical feature zeros", {
  data <- rbind(c(1, 1, 0, 0), c(1, 0, 1, 0), c(1, 0, 1, 1))
  padded <- sparse_zero_fixture(data)
  expected <- matrix(c(0, .5, .6, .5, 0, .2, .6, .2, 0), nrow = 3)
  builders <- list(
    brute_force_knn = list(),
    random_knn = list(),
    nnd_knn = list(init = "rand", n_iters = 0),
    rpf_knn = list(n_trees = 1),
    rnnd_knn = list(init = "rand", n_iters = 0)
  )
  for (name in names(builders)) {
    set.seed(17)
    result <- do.call(
      name,
      c(list(data = padded, k = 3, metric = "dice"), builders[[name]])
    )
    expect_equal(
      as.vector(result$dist),
      expected[cbind(as.vector(row(result$idx)), as.vector(result$idx))],
      tolerance = 1e-6,
      info = name
    )
  }

  forest <- rpf_build(padded, metric = "dice", n_trees = 1)
  graph <- brute_force_knn(Matrix::drop0(padded), k = 3, metric = "dice")
  queries <- list(
    brute_force_knn_query = list(),
    random_knn_query = list(),
    rpf_knn_query = list(forest = forest),
    graph_knn_query = list(reference_graph = graph)
  )
  query <- sparse_zero_fixture(data[1, , drop = FALSE])
  for (name in names(queries)) {
    result <- do.call(
      name,
      c(
        list(query = query, reference = padded, k = 3),
        if (name != "rpf_knn_query") list(metric = "dice"),
        queries[[name]]
      )
    )
    expect_setequal(as.vector(result$idx), 1:3)
    expect_equal(
      as.vector(result$dist),
      expected[1, result$idx],
      tolerance = 1e-6,
      info = name
    )
  }

  make_index <- function(x) {
    set.seed(17)
    rnnd_build(
      x,
      k = 2,
      metric = "dice",
      init = "rand",
      n_iters = 0,
      diversify_prob = 0,
      pruning_degree_multiplier = NULL
    )
  }
  index <- make_index(padded)
  clean_index <- make_index(Matrix::drop0(padded))
  expect_equal(index$graph, clean_index$graph)
  expect_equal(index$search_graph, clean_index$search_graph)
  before <- serialize(list(index$graph, index$search_graph), NULL)
  result <- rnnd_query(index, query, k = 3, max_search_fraction = 1)
  expect_setequal(as.vector(result$idx), 1:3)
  expect_equal(
    as.vector(result$dist),
    expected[1, result$idx],
    tolerance = 1e-6
  )
  expect_identical(
    serialize(list(index$graph, index$search_graph), NULL),
    before
  )
})

test_that("feature cleanup keeps sparse storage and caller values", {
  data <- rbind(c(1, 1, 0, 0), c(1, 0, 1, 0))
  padded <- sparse_zero_fixture(data)
  for (class in c("CsparseMatrix", "RsparseMatrix", "TsparseMatrix")) {
    converted <- rnndescent:::x2m(methods::as(padded, class))
    # Numerical equality alone would not detect accidental densification.
    expect_s4_class(converted, "dgCMatrix")
    expect_identical(as.matrix(converted), data)
    expect_false(any(converted@x == 0))
  }
  expect_identical(padded@x, as.vector(data))
  expect_identical(
    rnndescent:::x2m(Matrix::drop0(padded)),
    Matrix::drop0(padded)
  )
  expect_identical(rnndescent:::x2m(data), data)
  expect_identical(rnndescent:::x2m(data != 0), data != 0)
  empty_values <- rnndescent:::x2m(sparse_zero_fixture(matrix(0, 2, 4)))
  expect_equal(dim(empty_values), c(2L, 4L))
  expect_length(empty_values@x, 0L)
  undefined <- matrix(c(NA_real_, NaN, Inf, -Inf, 1, 0), 2)
  expect_identical(
    as.matrix(rnndescent:::x2m(sparse_zero_fixture(undefined))),
    undefined
  )
})

test_that("feature cleanup preserves structural zero-distance graph edges", {
  data <- rbind(c(1, 1, 0, 0), c(1, 1, 0, 0), c(1, 0, 1, 0))
  graph <- Matrix::sparseMatrix(
    i = c(1, 2, 1, 3, 2, 3),
    j = c(2, 1, 3, 1, 3, 2),
    x = c(0, 0, rep(sqrt(2), 4)),
    dims = c(3, 3)
  )
  prepared <- prepare_search_graph(
    sparse_zero_fixture(data),
    graph,
    use_alt_metric = FALSE,
    diversify_prob = 0,
    pruning_degree_multiplier = NULL
  )
  expect_equal(diff(prepared@p), rep(2L, 3))
  expect_equal(prepared[1, 2], .Machine$double.eps)
  expect_equal(prepared[2, 1], .Machine$double.eps)
  expect_identical(graph@x, c(0, sqrt(2), 0, sqrt(2), sqrt(2), sqrt(2)))
})
