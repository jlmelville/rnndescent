library(rnndescent)
context("Reverse neighbors")

rnbrs <- brute_force_knn(ui10, k = 4, n_threads = 0)
hub10 <- k_occur(rnbrs$idx)
expect_equal(hub10, c(3, 5, 7, 3, 5, 3, 5, 4, 2, 3))
expect_equal(
  k_occur(rnbrs$idx, include_self = FALSE),
  c(2, 4, 6, 2, 4, 2, 4, 3, 1, 2)
)
expect_equal(
  k_occur(rnbrs$idx, k = 2),
  c(2, 1, 2, 2, 2, 3, 3, 2, 2, 1)
)

qnbrs4 <- brute_force_knn_query(reference = ui6, query = ui4, k = 4)
expect_equal(k_occur(qnbrs4$idx), c(1, 3, 4, 3, 4, 1))

qnbrs6 <- brute_force_knn_query(reference = ui4, query = ui6, k = 4)
expect_equal(k_occur(qnbrs6$idx, k = 2), c(5, 4, 1, 2))
