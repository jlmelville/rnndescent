build_candidates <- function(current_graph, n_vertices, n_neighbors,
                             max_candidates = 50) {
  candidate_neighbors <- make_heap(n_vertices, max_candidates)
  for (i in 1:n_vertices) {
    for (j in 1:n_neighbors) {
      if (current_graph[1, i, j] < 0) {
        next
      }
      idx <- current_graph[1, i, j]
      isn <- current_graph[3, i, j]
      d <- stats::runif(1)

      candidate_neighbors <- heap_push(candidate_neighbors, i - 1, d, idx, isn)$heap
      candidate_neighbors <- heap_push(candidate_neighbors, idx, d, i - 1, isn)$heap

      current_graph[3, i, j] <- 0
    }
  }

  list(
    candidate_neighbors = candidate_neighbors,
    current_graph = current_graph
  )
}

nn_descent_opt <- function(data, metric, indices, dist, n_iters = 10,
                           max_candidates = 20, delta = 0.001,
                           verbose = FALSE) {
  dist_fn <- create_dist_fn(metric)
  n_vertices <- nrow(indices)
  n_neighbors <- ncol(indices)
  current_graph <- nn_to_heap(indices, dist)
  for (iter in 1:n_iters) {
    tsmessage(iter, " / ", n_iters, " ", formatC(sum(current_graph[2, , ])))

    res <- build_candidates(
      current_graph, n_vertices, n_neighbors,
      max_candidates
    )
    candidate_neighbors <- res$candidate_neighbors
    current_graph <- res$current_graph

    c <- 0
    for (i in 1:n_vertices) {
      for (j in 1:max_candidates) {
        p <- candidate_neighbors[1, i, j]

        if (p < 0) {
          next
        }

        for (k in 1:max_candidates) {
          q <- candidate_neighbors[1, i, k]
          if (q < 0 || candidate_neighbors[3, i, j] == 0 &&
            candidate_neighbors[3, i, k] == 0) {
            next
          }
          d <- dist_fn(data, p + 1, q + 1)
          res <- heap_push(current_graph, p, d, q, 1)
          current_graph <- res$heap
          c <- c + res$success

          res <- heap_push(current_graph, q, d, p, 1)
          current_graph <- res$heap
          c <- c + res$success
        }
      }
    }

    if (c <= delta * n_neighbors * n_vertices) {
      tsmessage("c = ", c, " crit = ", delta * n_neighbors * n_vertices)
      break
    }
  }

  deheap_sort(current_graph)
}

nn_descent_optl <- function(data, l, metric = "euclidean", n_iters = 10,
                            max_candidates = 20,
                            delta = 0.001, verbose = FALSE) {
  nn_descent_opt(data, metric, l$idx, l$dist,
    n_iters = n_iters, max_candidates = max_candidates,
    delta = delta, verbose = verbose
  )
}
