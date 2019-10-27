l2d <- function(a, b) {
  diff <- a - b
  sum(diff * diff)
}

l2_dist <- function(X, i, j) {
  l2d(X[i, ], X[j, ])
}

eucd <- function(a, b) {
  sqrt(l2d(a, b))
}

euc_dist <- function(X, i, j) {
  eucd(X[i, ], X[j, ])
}

dot <- function(a, b = a) {
  sum(a * b)
}

normv <- function(a) {
  a / sqrt(dot(a))
}

norm2 <- function(a) {
  sqrt(dot(a))
}

coss <- function(a, b) {
  dot(a, b) / sqrt(dot(a) * dot(b))
}

cosd <- function(a, b) {
  1.0 - coss(a, b)
}

cos_dist <- function(X, i, j) {
  cosd(X[i, ], X[j, ])
}

manhattand <- function(a, b) {
  sum(abs(a - b))
}

manhattan_dist <- function(X, i, j) {
  manhattand(X[i, ], X[j, ])
}

hammingd <- function(a, b) {
  sum(bitwXor(a, b))
}

hamming_dist <- function(X, i, j) {
  hammingd(X[i, ], X[j, ])
}
