test_that("graph dimension errors identify dist and idx", {
  idx <- matrix(rep(seq_len(4), 2), nrow = 4)
  cases <- list(
    list(nr = 3, nc = 2, message = "'dist' has 3 rows; 'idx' has 4"),
    list(nr = 5, nc = 2, message = "'dist' has 5 rows; 'idx' has 4"),
    list(nr = 4, nc = 1, message = "'dist' has 1 columns; 'idx' has 2"),
    list(nr = 4, nc = 3, message = "'dist' has 3 columns; 'idx' has 2")
  )

  for (case in cases) {
    nn <- list(idx = idx, dist = matrix(1, nrow = case$nr, ncol = case$nc))
    expect_error(merge_knn(list(nn)), case$message, fixed = TRUE)
    expect_error(
      rpf_filter(nn, forest = list(trees = list())),
      case$message,
      fixed = TRUE
    )
  }
})
