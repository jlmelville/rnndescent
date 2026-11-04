fill_query_fixture <- function() {
  set.seed(438)
  data <- matrix(runif(96), ncol = 3)
  query <- matrix(runif(36), ncol = 3)
  idx <- vapply(
    seq_len(5),
    function(j) {
      as.integer((seq_len(nrow(query)) + j - 2L) %% nrow(data) + 1L)
    },
    integer(nrow(query))
  )
  dist <- vapply(
    seq_len(5),
    function(j) {
      sqrt(rowSums((query - data[idx[, j], , drop = FALSE])^2))
    },
    numeric(nrow(query))
  )
  list(
    data = data,
    query = query,
    graph = brute_force_knn(data, k = 5, n_threads = 0),
    init = list(idx = idx, dist = dist)
  )
}

test_that("mixed query initializers preserve entries and complete uniquely", {
  fixture <- fill_query_fixture()
  for (sparse in c(FALSE, TRUE)) {
    data <- fixture$data
    query <- fixture$query
    if (sparse) {
      data <- Matrix::Matrix(data, sparse = TRUE)
      query <- Matrix::Matrix(query, sparse = TRUE)
    }
    init <- fixture$init
    init$idx[c(3, 12), 5] <- 0L
    init$idx[12, 3:4] <- 0L
    init$dist[c(3, 12), 5] <- NA_real_
    init$dist[12, 3:4] <- NA_real_
    for (threads in c(0, 2)) {
      run <- function(initial) {
        set.seed(731)
        graph_knn_query(
          query,
          data,
          fixture$graph,
          k = 5,
          init = initial,
          n_threads = threads,
          max_search_fraction = 0
        )
      }
      complete <- run(fixture$init)
      filled <- run(init)
      expect_identical(filled, run(init))
      untouched <- setdiff(seq_len(nrow(query)), c(3, 12))
      expect_identical(filled$idx[untouched, ], complete$idx[untouched, ])
      expect_identical(filled$dist[untouched, ], complete$dist[untouched, ])
      expect_true(all(filled$idx >= 1L & filled$idx <= nrow(data)))
      expect_true(all(is.finite(filled$dist)))
      expect_true(all(apply(filled$idx, 1, anyDuplicated) == 0L))
      retained <- vapply(
        seq_len(nrow(query)),
        function(i) {
          supplied <- init$idx[i, !is.na(init$idx[i, ]) & init$idx[i, ] > 0L]
          all(supplied %in% filled$idx[i, ])
        },
        logical(1)
      )
      expect_true(all(retained))
      expected <- vapply(
        seq_len(5),
        function(j) {
          sqrt(rowSums(
            (fixture$query -
              fixture$data[filled$idx[, j], , drop = FALSE])^2
          ))
        },
        numeric(nrow(query))
      )
      expect_equal(filled$dist, expected, tolerance = 1e-6)
    }
  }
})

test_that("filling can exhaust the reference population without duplicates", {
  data <- matrix(seq_len(12) / 12, ncol = 2)
  query <- data[1:2, , drop = FALSE] + 0.1
  graph <- brute_force_knn(data, k = 6, n_threads = 0)
  idx <- rbind(seq_len(6), c(1L, rep(0L, 5)))
  dist <- rbind(
    sqrt(rowSums((data - matrix(query[1, ], 6, 2, byrow = TRUE))^2)),
    c(sqrt(sum((query[2, ] - data[1, ])^2)), rep(NA_real_, 5))
  )
  set.seed(991)
  filled <- graph_knn_query(
    query,
    data,
    graph,
    k = 6,
    init = list(idx = idx, dist = dist),
    max_search_fraction = 0,
    n_threads = 0
  )
  expect_equal(t(apply(filled$idx, 1, sort)), rbind(seq_len(6), seq_len(6)))
  expect_true(all(is.finite(filled$dist)))
})
