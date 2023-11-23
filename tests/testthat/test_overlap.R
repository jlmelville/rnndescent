library(rnndescent)
context("overlap")

ui10_bf <- brute_force_knn(ui10, k = 4)

test_that("overlap with self is always 1", {
  expect_equal(nn_overlap(ui10_bf, ui10_bf), 1)
  expect_equal(nn_overlap(ui10_bf$idx, ui10_bf$idx), 1)
  expect_equal(nn_overlap(ui10_bf, ui10_bf$idx), 1)
  expect_equal(nn_overlap(ui10_bf$idx, ui10_bf), 1)
  expect_equal(nn_overlap(ui10_bf, ui10_bf, k = 3), 1)
  expect_error(nn_overlap(ui10_bf, ui10_bf, k = 5))

  ov_bf_vec <- nn_overlap(ui10_bf, ui10_bf, k = 4, ret_vec = TRUE)
  expect_equal(ov_bf_vec$mean, nn_overlap(ui10_bf, ui10_bf))
  expect_equal(ov_bf_vec$overlaps, rep(1, 10))
})

test_that("overlap range", {
  set.seed(1337)
  rnnbrs <- random_knn(ui10, k = 4)
  rov <- nn_overlap(ui10_bf, rnnbrs$idx, ret_vec = TRUE)
  expect_equal(rov$mean, nn_overlap(rnnbrs$idx, ui10_bf), 1)
  expect_equal(rov$mean, mean(rov$overlaps))
  expect_gte(min(rov$overlaps), 0)
  expect_lte(min(rov$overlaps), 1)
})
