library(rnndescent)

test_that("sparse Canberra selects signed neighbors by mathematical distance", {
  query <- Matrix::sparseMatrix(i = 1, j = 1, x = 1)
  reference <- Matrix::sparseMatrix(i = 1:2, j = c(1, 1), x = c(-1, 2))
  cases <- list(
    list(storage = "CsparseMatrix", obs = "R", n_threads = 0),
    list(storage = "RsparseMatrix", obs = "C", n_threads = 2),
    list(storage = "TsparseMatrix", obs = "R", n_threads = 2)
  )

  for (case in cases) {
    query_sp <- query
    reference_sp <- reference
    if (case$obs == "C") {
      query_sp <- Matrix::t(query_sp)
      reference_sp <- Matrix::t(reference_sp)
    }
    query_sp <- methods::as(query_sp, case$storage)
    reference_sp <- methods::as(reference_sp, case$storage)

    for (k in c(1L, 2L)) {
      result <- brute_force_knn_query(
        query_sp,
        reference_sp,
        k = k,
        metric = "canberra",
        obs = case$obs,
        n_threads = case$n_threads
      )
      expect_equal(result$idx, matrix(c(2L, 1L)[seq_len(k)], nrow = 1))
      expect_equal(
        result$dist,
        matrix(c(1 / 3, 1)[seq_len(k)], nrow = 1),
        tolerance = 1e-6
      )
    }
  }
})

test_that("sparse Canberra build preserves signed distances", {
  data <- matrix(c(-1, 1, 2), ncol = 1)
  data_sp <- methods::as(Matrix::Matrix(data, sparse = TRUE), "dgCMatrix")
  # The negative point ties; compare its distances by reference index.
  for (n_threads in c(0, 2)) {
    result <- brute_force_knn(
      data_sp,
      k = 3,
      metric = "canberra",
      n_threads = n_threads
    )
    by_index <- t(vapply(
      seq_len(3),
      function(i) {
        result$dist[i, order(result$idx[i, ])]
      },
      numeric(3)
    ))
    expect_equal(
      by_index,
      matrix(c(0, 1, 1, 1, 0, 1 / 3, 1, 1 / 3, 0), nrow = 3),
      tolerance = 1e-6
    )
    expect_equal(
      result,
      brute_force_knn(data, k = 3, metric = "canberra", n_threads = n_threads)
    )
  }
})

test_that("sparse Canberra preserves zero handling and unequal support", {
  query <- Matrix::sparseMatrix(
    i = c(1, 1, 1),
    j = c(1, 2, 4),
    x = c(0, 1, 0),
    dims = c(1, 5)
  )
  reference <- Matrix::sparseMatrix(
    i = c(1, 2, 3, 3, 4, 4, 5),
    j = c(2, 2, 1, 5, 2, 4, 5),
    x = c(2, -1, 1, 3, 0, 0, 0),
    dims = c(5, 5)
  )
  expected <- c(1 / 3, 1, 3, 1, 1)

  for (query_sp in list(query, Matrix::drop0(query))) {
    for (reference_sp in list(reference, Matrix::drop0(reference))) {
      result <- brute_force_knn_query(
        query_sp,
        reference_sp,
        k = 5,
        metric = "canberra"
      )
      expect_equal(
        as.numeric(result$dist)[order(result$idx)],
        expected,
        tolerance = 1e-6
      )
      expect_equal(
        result$dist,
        brute_force_knn_query(
          as.matrix(query_sp),
          as.matrix(reference_sp),
          k = 5,
          metric = "canberra"
        )$dist,
        tolerance = 1e-6
      )
    }
  }

  zero <- Matrix::sparseMatrix(i = 1, j = 4, x = 0, dims = c(1, 5))
  for (query_sp in list(zero, Matrix::drop0(zero))) {
    result <- brute_force_knn_query(
      query_sp,
      reference,
      k = 5,
      metric = "canberra"
    )
    expect_equal(as.numeric(result$dist)[order(result$idx)], c(1, 1, 2, 0, 0))
  }
})
