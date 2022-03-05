library(rnndescent)
context("Local scaling")

rnn <- brute_force_knn(ui10, k = 4, n_threads = 0)
expect_equal(get_local_scales(rnn$dist, k_begin = 2, k_end = 2), rnn$dist[, 2])
expect_equal(
  get_local_scales(rnn$dist, k_begin = 2, k_end = 3),
  c(
    0.5123216,
    0.3621322,
    0.2776346,
    0.3853371,
    0.2524338,
    0.4232051,
    0.2618034,
    0.2987346,
    0.4647526,
    0.6306641
  ),
  tol = 1e-7
)

expect_equal(local_scale_distances(rnn, k_begin = 2, k_end = 3),
  matrix(c(
    0, 0.5535, 1.424, 3.445,
    0, 0.9493, 1.79, 2.516,
    0, 0.6879, 1.57, 1.79,
    0, 0.6701, 1.564, 2.15,
    0, 0.3978, 1.57, 1.816,
    0, 0.5535, 2.128, 1.274,
    0, 0.6879, 0.9493, 1.816,
    0, 0.3978, 1.564, 2.685,
    0, 0.6701, 2.449, 2.258,
    0, 1.274, 1.424, 6.225
  ), byrow = TRUE, ncol = 4),
  tol = 1e-3
)


rnn6 <- brute_force_knn(ui10, k = 6, n_threads = 0)
srnn4 <- local_scale_nn(rnn6, k = 4, k_scale = c(2, 4), n_threads = 0)

scaled_idx <- matrix(c(1, 6, 10, 3,
         2, 7, 3, 8,
         3, 7, 5, 2,
         4, 9, 8, 2,
         5, 8, 3, 7,
         6, 1, 3, 10,
         7, 3, 2, 5,
         8, 5, 4, 2,
         9, 4, 8, 2,
         10, 6, 1, 3), byrow = TRUE, ncol = 4)

scaled_dist <- matrix(
  c(
    0, 0.3464, 0.6782, 0.7,
    0, 0.3, 0.4243, 0.4899,
    0, 0.2236, 0.3317, 0.4243,
    0, 0.3464, 0.4243, 0.5477,
    0, 0.1732, 0.3317, 0.3464,
    0, 0.3464, 0.5, 0.5831,
    0, 0.2236, 0.3, 0.3464,
    0, 0.1732, 0.4243, 0.4899,
    0, 0.3464, 0.5831, 0.6164,
    0, 0.5831, 0.6782, 1.044
  ),
  byrow = TRUE, ncol = 4)

expect_equal(srnn4$idx, scaled_idx)
expect_equal(srnn4$dist, scaled_dist, tol = 1e-4)

srnn4 <- local_scale_nn(rnn6, k = 4, k_scale = c(2, 4), n_threads = 2)
expect_equal(srnn4$idx, scaled_idx)
expect_equal(srnn4$dist, scaled_dist, tol = 1e-4)
