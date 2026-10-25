cosine_sparse <- function(x) {
  methods::as(Matrix::Matrix(x, sparse = TRUE), "generalMatrix")
}

cosine_oracle <- function(reference, query = reference) {
  rn <- sqrt(rowSums(reference^2))
  qn <- sqrt(rowSums(query^2))
  result <- 1 - tcrossprod(query, reference) / outer(qn, rn)
  result[outer(qn == 0, rn == 0, `|`)] <- 1
  result[outer(qn == 0, rn == 0, `&`)] <- 0
  result
}

expect_cosine_pairs <- function(result, expected) {
  testthat::expect_true(all(result$idx > 0L))
  pairs <- cbind(
    rep(seq_len(nrow(result$idx)), ncol(result$idx)),
    c(result$idx)
  )
  testthat::expect_equal(
    result$dist,
    matrix(expected[pairs], nrow(result$idx)),
    tolerance = 1e-6
  )
}

test_that("default cosine retains negative similarities and exact neighbor order", {
  reference <- rbind(c(-1, 0), c(-0.5, sqrt(3) / 2))
  query <- matrix(c(1, 0), nrow = 1)
  cases <- list(
    list(reference, query, "R", 0L),
    list(t(reference), t(query), "C", 2L),
    list(cosine_sparse(reference), cosine_sparse(query), "R", 0L),
    list(
      methods::as(cosine_sparse(reference), "RsparseMatrix"),
      methods::as(cosine_sparse(query), "RsparseMatrix"),
      "R",
      2L
    ),
    list(
      methods::as(cosine_sparse(t(reference)), "TsparseMatrix"),
      methods::as(cosine_sparse(t(query)), "TsparseMatrix"),
      "C",
      0L
    )
  )
  for (case in cases) {
    result <- brute_force_knn_query(
      reference = case[[1]],
      query = case[[2]],
      k = 1,
      metric = "cosine",
      obs = case[[3]],
      n_threads = case[[4]]
    )
    expect_equal(result$idx, matrix(2L))
    expect_equal(result$dist, matrix(1.5), tolerance = 1e-6)
  }
})

test_that("cosine values preserve the full domain and existing zero conventions", {
  reference <- rbind(
    c(1, 0),
    c(0.5, sqrt(3) / 2),
    c(0, 1),
    c(-0.5, sqrt(3) / 2),
    c(-1, 0),
    c(0, 0)
  )
  query <- rbind(c(1, 0), c(0, 0))
  expected <- cosine_oracle(reference, query)
  for (sparse in c(FALSE, TRUE)) {
    ref <- if (sparse) cosine_sparse(reference) else reference
    qry <- if (sparse) cosine_sparse(query) else query
    for (alt in c(FALSE, TRUE)) {
      result <- brute_force_knn_query(
        reference = ref,
        query = qry,
        k = nrow(reference),
        metric = "cosine",
        use_alt_metric = alt
      )
      expect_cosine_pairs(result, expected)
    }
  }
})

test_that("sparse cosine keeps exact distances across alternating supports", {
  reference <- rbind(
    c(1, 1, 0, 1, 0, 0, 0, 1, 0),
    c(1, 0, 0, 1, 1, 0, 0, 1, 0),
    c(1, 0, 0, -1, 1, 0, 0, 1, 0),
    rep(0, 9)
  )
  query <- rbind(
    c(0, 1, 1, 1, 0, 0, 1, 0, 0),
    c(0, 1, 0, 1, 0, 1, 0, 0, 1)
  )
  expected <- matrix(c(0.5, 0.75, 1.25, 1), nrow = 2, ncol = 4, byrow = TRUE)
  for (threads in c(0L, 4L)) {
    result <- brute_force_knn_query(
      reference = cosine_sparse(reference),
      query = cosine_sparse(query),
      k = nrow(reference),
      metric = "cosine",
      use_alt_metric = FALSE,
      n_threads = threads
    )
    pairs <- cbind(
      rep(seq_len(nrow(result$idx)), ncol(result$idx)),
      c(result$idx)
    )
    expect_identical(result$dist, matrix(expected[pairs], nrow(query)))
  }
})

test_that("cosine distances and neighbors survive positive rescaling", {
  reference <- rbind(c(0, 1), c(1, 0), c(-1, 0))
  query <- matrix(c(1, 0), nrow = 1)
  for (scale in c(1, 1e10, 1e-15)) {
    for (sparse in c(FALSE, TRUE)) {
      ref <- reference * scale
      qry <- query * scale
      if (sparse) {
        ref <- Matrix::sparseMatrix(
          i = row(ref),
          j = col(ref),
          x = c(ref),
          dims = dim(ref)
        )
        qry <- Matrix::sparseMatrix(
          i = row(qry),
          j = col(qry),
          x = c(qry),
          dims = dim(qry)
        )
      }
      info <- paste("scale =", scale, "sparse =", sparse)
      result <- brute_force_knn_query(
        qry,
        ref,
        k = 3,
        metric = "cosine",
        n_threads = 0
      )
      expect_equal(result$idx, matrix(c(2L, 1L, 3L), nrow = 1), info = info)
      expect_equal(
        result$dist,
        matrix(c(0, 1, 2), nrow = 1),
        tolerance = 1e-6,
        info = info
      )
      nearest <- brute_force_knn_query(
        qry,
        ref,
        k = 1,
        metric = "cosine",
        n_threads = 0
      )
      expect_equal(nearest$idx, matrix(2L), info = info)
      expect_equal(nearest$dist, matrix(0), tolerance = 1e-6, info = info)
    }
  }
})

test_that("cosine initial graphs and search preparation use public distances", {
  data <- rbind(c(1, 0), c(-1, 0), c(-0.5, sqrt(3) / 2), c(0.5, sqrt(3) / 2))
  expected <- cosine_oracle(data)
  for (sparse in c(FALSE, TRUE)) {
    x <- if (sparse) cosine_sparse(data) else data
    initial <- brute_force_knn(
      x,
      k = nrow(data),
      metric = "cosine",
      use_alt_metric = FALSE
    )
    result <- nnd_knn(x, metric = "cosine", init = initial, n_iters = 0)
    expect_cosine_pairs(result, expected)
    graph <- prepare_search_graph(
      x,
      initial,
      metric = "cosine",
      diversify_prob = 0,
      pruning_degree_multiplier = NULL
    )
    expect_equal(as.matrix(graph)[1, 2], 2, tolerance = 1e-6)
    result <- graph_knn_query(
      x,
      x,
      graph,
      k = nrow(data),
      metric = "cosine",
      init = initial
    )
    expect_cosine_pairs(result, expected)
  }
})

test_that("cosine builders and stored indexes return consistent pair distances", {
  data <- rbind(c(1, 0), c(-1, 0), c(-0.5, sqrt(3) / 2), c(0.5, sqrt(3) / 2))
  expected <- cosine_oracle(data)
  for (sparse in c(FALSE, TRUE)) {
    x <- if (sparse) cosine_sparse(data) else data
    set.seed(627)
    expect_cosine_pairs(
      random_knn(x, k = nrow(data), metric = "cosine"),
      expected
    )
    set.seed(627)
    expect_cosine_pairs(
      rpf_knn(
        x,
        k = nrow(data),
        metric = "cosine",
        n_trees = 1,
        leaf_size = nrow(data)
      ),
      expected
    )
    set.seed(627)
    index <- rnnd_build(
      x,
      k = nrow(data),
      metric = "cosine",
      n_trees = 1,
      leaf_size = nrow(data),
      n_iters = 1
    )
    expect_cosine_pairs(index$graph, expected)
    saved <- index
    expect_cosine_pairs(rnnd_query(index, x, k = nrow(data)), expected)
    expect_identical(index, saved)
  }
})

test_that("legacy cosine forest tags calculate fresh cosine distances", {
  reference <- rbind(c(-1, 0), c(-0.5, sqrt(3) / 2))
  query <- matrix(c(1, 0), nrow = 1)
  for (sparse in c(FALSE, TRUE)) {
    ref <- if (sparse) cosine_sparse(reference) else reference
    qry <- if (sparse) cosine_sparse(query) else query
    # The raw builder reproduces the calculation and geometry of old forests.
    forest <- rpf_build(
      ref,
      metric = "alternative-cosine",
      n_trees = 1,
      leaf_size = nrow(reference),
      margin = if (sparse) "implicit" else "explicit"
    )
    forest$original_metric <- "cosine"
    forest$use_alt_metric <- TRUE
    saved <- forest
    result <- rpf_knn_query(qry, ref, forest, k = 1)
    expect_equal(result$idx, matrix(2L))
    expect_equal(result$dist, matrix(1.5), tolerance = 1e-6)
    graph <- brute_force_knn(
      ref,
      k = nrow(reference),
      metric = "cosine",
      use_alt_metric = FALSE
    )
    result <- graph_knn_query(
      qry,
      ref,
      graph,
      init = forest,
      k = 1,
      metric = "cosine"
    )
    expect_equal(result$idx, matrix(2L))
    expect_equal(result$dist, matrix(1.5), tolerance = 1e-6)
    expect_identical(forest, saved)
  }
})
