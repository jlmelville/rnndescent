
### R implementation of distances

R_knn_hamming <- function(q, r = q, k = nrow(q)) {
  N <- nrow(q)
  D <- t(sapply(1:N, function(i) colSums(q[i, ] != t(r))))
  idx <- t(sapply(1:N, function(i) order(D[i, ])))
  dist <- matrix(D[cbind(1:N, c(idx))], N)
  list(
    idx = idx[, 1:k],
    dist = dist[, 1:k]
  )
}
