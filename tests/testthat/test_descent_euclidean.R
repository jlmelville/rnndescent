library(rnndescent)
context("NN descent Euclidean")


i10_rdist <- matrix(
  c(
    0.812403840463596, 0.346410161513775, 0.994987437106621, 1.45945195193264,
    0.479583152331272, 0.547722557505166, 0.489897948556636, 0.424264068711929,
    0.787400787401181, 0.424264068711929, 0.469041575982343, 0.223606797749979,
    0.346410161513776, 0.547722557505166, 0.424264068711928, 1.45945195193264,
    1.3114877048604, 0.728010988928052, 0.33166247903554, 0.479583152331272,
    1.36747943311773, 0.678232998312527, 0.346410161513775, 0.932737905308882,
    0.678232998312527, 0.818535277187245, 0.458257569495584, 1.2328828005938,
    0.173205080756888, 0.58309518948453, 0.458257569495584, 1.16189500386223,
    0.818535277187245, 0.616441400296897, 1.36747943311773, 0.346410161513776,
    1.80831413200251, 0.58309518948453, 1.04403065089105, 1.36014705087354
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

i10_ridx <- matrix(
  c(
    6, 5, 1, 3,
    4, 3, 7, 2,
    3, 1, 7, 6,
    8, 1, 7, 0,
    9, 8, 2, 1,
    8, 6, 0, 7,
    5, 8, 7, 9,
    4, 8, 6, 0,
    6, 1, 5, 3,
    8, 5, 2, 1
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

expected_heap_idx <- matrix(
  c(
    7, 6, 1, 5,
    3, 7, 2, 4,
    7, 1, 4, 6,
    2, 1, 7, 8,
    8, 1, 7, 2,
    7, 9, 6, 0,
    0, 7, 5, 2,
    2, 6, 3, 4,
    4, 1, 7, 3,
    4, 6, 5, 2
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

expected_heap_dist <- matrix(
  c(
    1.16189500386223, 0.812403840463596, 0.994987437106621, 0.346410161513775,
    0.547722557505166, 0.489897948556636, 0.424264068711929, 0.479583152331272,
    0.469041575982343, 0.424264068711929, 0.33166247903554, 0.223606797749979,
    0.787400787401181, 0.547722557505166, 0.424264068711928, 0.346410161513776,
    0.728010988928052, 0.479583152331272, 0.173205080756888, 0.33166247903554,
    0.932737905308882, 0.58309518948453, 0.678232998312527, 0.346410161513775,
    0.812403840463596, 0.458257569495584, 0.678232998312527, 0.223606797749979,
    0.469041575982343, 0.458257569495584, 0.424264068711928, 0.173205080756888,
    0.728010988928052, 0.616441400296897, 0.58309518948453, 0.346410161513776,
    1.3114877048604, 1.2328828005938, 0.58309518948453, 1.04403065089105
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

heap <- nn_to_heap(i10_ridx, i10_rdist)
expect_equal(heap[1, , ], expected_heap_idx, check.attributes = FALSE)
expect_equal(heap[2, , ], expected_heap_dist, check.attributes = FALSE)
expect_equal(heap[3, , ], matrix(rep(1, 40), nrow = 10), check.attributes = FALSE)

expected_heap_sort_idx <- matrix(
  c(
    5, 6, 1, 7,
    2, 4, 7, 3,
    6, 4, 1, 7,
    8, 7, 1, 2,
    7, 2, 1, 8,
    0, 9, 6, 7,
    2, 7, 5, 0,
    4, 3, 6, 2,
    3, 7, 1, 4,
    5, 2, 6, 4
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

expected_heap_sort_dist <- matrix(
  c(
    0.34641016, 0.81240384, 0.99498744, 1.161895,
    0.42426407, 0.47958315, 0.48989795, 0.54772256,
    0.2236068, 0.33166248, 0.42426407, 0.46904158,
    0.34641016, 0.42426407, 0.54772256, 0.78740079,
    0.17320508, 0.33166248, 0.47958315, 0.72801099,
    0.34641016, 0.58309519, 0.678233, 0.93273791,
    0.2236068, 0.45825757, 0.678233, 0.81240384,
    0.17320508, 0.42426407, 0.45825757, 0.46904158,
    0.34641016, 0.58309519, 0.6164414, 0.72801099,
    0.58309519, 1.04403065, 1.2328828, 1.3114877
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

heap_sorted <- deheap_sort(heap)
expect_equal(heap_sorted$idx, expected_heap_sort_idx, check.attributes = FALSE)
expect_equal(heap_sorted$dist, expected_heap_sort_dist, check.attributes = FALSE)


expected_idx <- matrix(
  c(
    1, 6, 10, 3,
    2, 7, 3, 5,
    3, 7, 5, 2,
    4, 9, 8, 2,
    5, 8, 3, 7,
    6, 1, 3, 10,
    7, 3, 2, 5,
    8, 5, 4, 7,
    9, 4, 8, 2,
    10, 6, 1, 3
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

# distances from FNN
expected_dist <- matrix(
  c(
    0, 0.3464102, 0.6782330, 0.7000000,
    0, 0.3000000, 0.4242641, 0.4795832,
    0, 0.2236068, 0.3316625, 0.4242641,
    0, 0.3464102, 0.4242641, 0.5477226,
    0, 0.1732051, 0.3316625, 0.3464102,
    0, 0.3464102, 0.5000000, 0.5830952,
    0, 0.2236068, 0.3000000, 0.3464102,
    0, 0.1732051, 0.4242641, 0.4582576,
    0, 0.3464102, 0.5830952, 0.6164414,
    0, 0.5830952, 0.6782330, 1.0440307
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

set.seed(1337)
rnn <- nnd_knn(ui10, 4, use_cpp = FALSE, verbose = FALSE)
expect_equal(rnn$idx, expected_idx, check.attributes = FALSE)
expect_equal(rnn$dist, expected_dist, check.attributes = FALSE, tol = 1e-7)


set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, use_cpp = FALSE)
# treat sum of distances an objective function
# expected sum from sum(FNN::get.knn(uirism, 14)$nn.dist)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

set.seed(1337)
rnnd <- nn_descent(ui10, i10_ridx, i10_rdist, verbose = FALSE)
expect_equal(rnnd$idx, expected_idx, check.attributes = FALSE)
expect_equal(rnnd$dist, expected_dist, check.attributes = FALSE, tol = 1e-6)


set.seed(1337)
iris_nbrs <- random_nbrs(uirism, 15)
iris_nnd <- nn_descent(uirism, iris_nbrs$idx - 1, iris_nbrs$dist, verbose = FALSE)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)

set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, use_cpp = TRUE)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)


# C++ with set backing
set.seed(1337)
rnnd <- nn_descent(ui10, i10_ridx, i10_rdist, verbose = FALSE, use_set = TRUE)
expect_equal(rnnd$idx, expected_idx, check.attributes = FALSE)
expect_equal(rnnd$dist, expected_dist, check.attributes = FALSE, tol = 1e-6)

set.seed(1337)
iris_nbrs <- random_nbrs(uirism, 15)
iris_nnd <- nn_descent(uirism, iris_nbrs$idx - 1, iris_nbrs$dist, verbose = FALSE, use_set = TRUE)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)

set.seed(1337)
uiris_rnn <- nnd_knn(uirism, 15, use_cpp = TRUE, use_set = TRUE)
expect_equal(sum(uiris_rnn$dist), 1016.834, tol = 1e-3)

# fast random
set.seed(1337)
rnnd <- nn_descent(ui10, i10_ridx, i10_rdist, verbose = FALSE, use_set = FALSE, fast_rand = TRUE)
expect_equal(rnnd$idx, expected_idx, check.attributes = FALSE)
expect_equal(rnnd$dist, expected_dist, check.attributes = FALSE, tol = 1e-6)

set.seed(1337)
iris_nbrs <- random_nbrs(uirism, 15)
iris_nnd <- nn_descent(uirism, iris_nbrs$idx - 1, iris_nbrs$dist, verbose = FALSE, use_set = FALSE, fast_rand = TRUE)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)

set.seed(1337)
rnnd <- nn_descent(ui10, i10_ridx, i10_rdist, verbose = FALSE, use_set = TRUE, fast_rand = TRUE)
expect_equal(rnnd$idx, expected_idx, check.attributes = FALSE)
expect_equal(rnnd$dist, expected_dist, check.attributes = FALSE, tol = 1e-6)

set.seed(1337)
iris_nbrs <- random_nbrs(uirism, 15)
iris_nnd <- nn_descent(uirism, iris_nbrs$idx - 1, iris_nbrs$dist, verbose = FALSE, use_set = TRUE, fast_rand = TRUE)
expect_equal(sum(iris_nnd$dist), 1016.834, tol = 1e-3)
