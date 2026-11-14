library(rnndescent)

test_that("cosine descent agrees across memory and thread settings", {
  # Cosine distance
  set.seed(1337)
  uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine")
  # expected sum from RcppHNSW
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)

  # Cosine distance
  set.seed(1337)
  uiris_rnn <- nnd_knn(uirism, 15, low_memory = FALSE, metric = "cosine")
  # expected sum from RcppHNSW
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)

  # multi-threading
  set.seed(1337)
  uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine", n_threads = 1)
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)

  # with caching
  set.seed(1337)
  uiris_rnn <- nnd_knn(
    uirism,
    15,
    metric = "cosine",
    low_memory = FALSE,
    n_threads = 1
  )
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)
})

test_that("dense cosine descent preserves scores for zero and scaled rows", {
  set.seed(3881L)
  data <- matrix(rnorm(29L * 64L), nrow = 29L)
  data[, 1:2] <- 0
  data[, 4:6] <- data[, 3L] + outer(rnorm(29L), c(1e-5, 1.001e-5, 5e-4))
  data[, 7:9] <- data[, 7:9] * 2^-48
  data[, 10:12] <- data[, 10:12] * 2^48
  data[, 13:14] <- data[, 13:14] * 2^-80
  init <- list(
    idx = matrix(0L, ncol(data), 8L),
    dist = matrix(NA_real_, ncol(data), 8L)
  )
  set.seed(710L)
  dqrng::dqset.seed(710L)
  graph <- rnnd_knn(
    data,
    k = 8L,
    metric = "cosine",
    init = init,
    n_iters = 1L,
    obs = "C"
  )
  expect_true(all(graph$idx > 0L))
  expected <- knn_distances(data, graph$idx, metric = "cosine", obs = "C")
  expect_equal(graph$dist, expected$dist, tolerance = 1e-6)
  expect_true(all(graph$dist[1:2, ] %in% c(0, 1)))
})

# Query

test_that("cosine queries return valid neighbors and distances", {
  set.seed(1337)
  ui6_nnd <- nnd_knn(ui6, k = 4, metric = "cosine")
  qnbrs4 <- graph_knn_query(
    reference = ui6,
    reference_graph = ui6_nnd,
    query = ui4,
    k = 4,
    metric = "cosine"
  )
  check_query_nbrs_idx(qnbrs4$idx, nref = nrow(ui6))
  expect_equal_abs(sum(qnbrs4$dist), ui4q_cdsum, tolerance = 1e-5)

  set.seed(1337)
  ui4_nnd <- nnd_knn(ui4, k = 4, metric = "cosine")
  qnbrs6 <- graph_knn_query(
    reference = ui4,
    reference_graph = ui4_nnd,
    query = ui6,
    k = 4,
    metric = "cosine"
  )
  check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
  expect_equal_abs(sum(qnbrs6$dist), ui6q_cdsum, tolerance = 1e-5)
})


# cosine-preprocess -------------------------------------------------------------

test_that("preprocessed cosine descent agrees across execution settings", {
  # Cosine distance
  set.seed(1337)
  uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine-preprocess")
  # expected sum from RcppHNSW
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)

  # Cosine distance
  set.seed(1337)
  uiris_rnn <- nnd_knn(
    uirism,
    15,
    low_memory = FALSE,
    metric = "cosine-preprocess"
  )
  # expected sum from RcppHNSW
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)

  # multi-threading
  set.seed(1337)
  uiris_rnn <- nnd_knn(uirism, 15, metric = "cosine-preprocess", n_threads = 1)
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)

  # with caching
  set.seed(1337)
  uiris_rnn <- nnd_knn(
    uirism,
    15,
    metric = "cosine-preprocess",
    low_memory = FALSE,
    n_threads = 1
  )
  expect_equal(sum(uiris_rnn$dist), 1.347357, tolerance = 1e-3)
})

# Query

test_that("preprocessed cosine queries return valid neighbors and distances", {
  set.seed(1337)
  ui6_nnd <- nnd_knn(ui6, k = 4, metric = "cosine-preprocess")
  qnbrs4 <- graph_knn_query(
    reference = ui6,
    reference_graph = ui6_nnd,
    query = ui4,
    k = 4,
    metric = "cosine-preprocess"
  )
  check_query_nbrs_idx(qnbrs4$idx, nref = nrow(ui6))
  expect_equal_abs(sum(qnbrs4$dist), ui4q_cdsum, tolerance = 1e-5)

  set.seed(1337)
  ui4_nnd <- nnd_knn(ui4, k = 4, metric = "cosine-preprocess")
  qnbrs6 <- graph_knn_query(
    reference = ui4,
    reference_graph = ui4_nnd,
    query = ui6,
    k = 4,
    metric = "cosine-preprocess"
  )
  check_query_nbrs_idx(qnbrs6$idx, nref = nrow(ui4))
  expect_equal_abs(sum(qnbrs6$dist), ui6q_cdsum, tolerance = 1e-5)
})

# sparse
test_that("sparse", {
  set.seed(1337)
  dznbrs <- nnd_knn(ui10z, k = 4, n_threads = 0, metric = "cosine")
  set.seed(1337)
  spnbrs <- nnd_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    metric = "cosine",
    use_alt_metric = TRUE
  )
  expect_equal(dznbrs, spnbrs, tolerance = 1e-5)
  set.seed(1337)
  spnbrs <- nnd_knn(
    ui10sp,
    k = 4,
    n_threads = 0,
    metric = "cosine",
    use_alt_metric = FALSE
  )
  expect_equal(dznbrs, spnbrs, tolerance = 1e-5)

  # make sure uncorrection is triggered: nnd with incorrect distances will not converge
  bfz <- brute_force_knn(ui10z, k = 4, metric = "cosine")
  set.seed(1337)
  cosrz <- random_knn(ui10z, k = 4, metric = "cosine")
  set.seed(1337)
  spannd <- nnd_knn(
    ui10sp,
    k = 4,
    metric = "cosine",
    init = cosrz,
    use_alt_metric = TRUE
  )
  set.seed(1337)
  spcnnd <- nnd_knn(
    ui10sp,
    k = 4,
    metric = "cosine",
    init = cosrz,
    use_alt_metric = FALSE
  )

  expect_equal(spannd, bfz, tolerance = 1e-3)
  expect_equal(spcnnd, bfz, tolerance = 1e-4)

  # make sure init graph can be prepared from sparse
  cosrz$dist <- NULL
  set.seed(1337)
  spannd <- nnd_knn(
    ui10sp,
    k = 4,
    metric = "cosine",
    init = cosrz,
    use_alt_metric = TRUE
  )
  set.seed(1337)
  spcnnd <- nnd_knn(
    ui10sp,
    k = 4,
    metric = "cosine",
    init = cosrz,
    use_alt_metric = FALSE
  )

  expect_equal(spannd, bfz, tolerance = 1e-5)
  expect_equal(spcnnd, bfz, tolerance = 1e-5)
})

test_that("dense dot-product queries respect candidate order and search budgets", {
  query <- matrix(c(1, -1, 0), nrow = 1L)
  reference <- seq_len(8L) %o%
    c(1, -1, 0) +
    matrix(rep(c(1, 1, -2), 8L), ncol = 3L, byrow = TRUE)
  graph <- Matrix::sparseMatrix(
    i = c(1:6, 7L),
    j = c(rep(1L, 6L), 6L),
    x = 1,
    dims = c(8L, 8L)
  )
  cases <- list(
    list("dot", TRUE),
    list("dot", FALSE),
    list("alternative-dot", FALSE),
    list("cosine-preprocess", FALSE),
    list("correlation-preprocess", FALSE)
  )
  for (case in cases) {
    metric <- case[[1L]]
    for (budget in 0:6) {
      expected <- matrix(budget + 1L)
      result <- graph_knn_query(
        query = query,
        reference = reference,
        reference_graph = graph,
        init = matrix(1L),
        k = 1L,
        metric = metric,
        use_alt_metric = case[[2L]],
        max_search_fraction = budget / nrow(reference)
      )
      expect_identical(result$idx, expected)
      expect_equal(
        result$dist,
        knn_distances(reference, expected, metric = metric, query = query)$dist,
        tolerance = 1e-6
      )
    }
  }
})
