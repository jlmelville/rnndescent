library(rnndescent)

test_that("TS-SS returns zero whenever either vector is zero", {
  query <- rbind(c(0, 0), c(0, 0), c(1, 0))
  reference <- rbind(c(0, 0), c(1, 0), c(0, 0))
  stored_zeros <- function(x) {
    Matrix::sparseMatrix(
      i = rep(seq_len(nrow(x)), ncol(x)),
      j = rep(seq_len(ncol(x)), each = nrow(x)),
      x = as.vector(x),
      dims = dim(x)
    )
  }
  for (convert in list(identity, Matrix::drop0, stored_zeros)) {
    for (i in seq_len(nrow(query))) {
      result <- brute_force_knn_query(
        convert(query[i, , drop = FALSE]),
        convert(reference[i, , drop = FALSE]),
        k = 1,
        metric = "tsss",
        n_threads = 0
      )
      expect_equal(result$idx, matrix(1L))
      expect_equal(result$dist, matrix(0))
    }
  }
})

test_that("TS-SS zero queries retain every reference neighbor", {
  reference <- rbind(c(0, 0), c(1, 0), c(0, 1))
  for (input in list(reference, Matrix::drop0(reference))) {
    result <- brute_force_knn_query(
      input[1, , drop = FALSE],
      input,
      k = 3,
      metric = "tsss",
      n_threads = 0
    )
    expect_equal(sort(as.vector(result$idx)), 1:3)
    expect_equal(result$dist, matrix(0, nrow = 1, ncol = 3))
  }
})
