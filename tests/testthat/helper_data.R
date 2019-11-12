# ten iris entries where the 4 nearest neighbors are distinct
uiris <- unique(iris)
uirism <- as.matrix(uiris[, -5])
ui10 <- uirism[6:15, ]


ui6 <- ui10[1:6, ]
ui4 <- ui10[7:10, ]
ui10_eucd <- as.matrix(dist(ui10))

# sum(RcppHNSW::hnsw_search(ui4, RcppHNSW::hnsw_build(ui6), k = 4)$dist)
ui4q_edsum <- 9.310494
# sum(RcppHNSW::hnsw_search(ui6, RcppHNSW::hnsw_build(ui4), k = 4)$dist)
ui6q_edsum <-  18.98666

# NB Annoy and HNSW don't agree to more than this # of decimal places
# sum(RcppHNSW::hnsw_search(ui4, RcppHNSW::hnsw_build(ui6, distance = "cosine"), k = 4)$dist)
ui4q_cdsum <- 0.02072
# sum(RcppHNSW::hnsw_search(ui6, RcppHNSW::hnsw_build(ui4, distance = "cosine"), k = 4)$dist)
ui6q_cdsum <-  0.04220

# Manhattan: Taken from RcppAnnoy
ui4q_mdsum <- 15.4
ui6q_mdsum <-  31.6

# Hamming
bitm <- function(nrow, ncol, prob = 0.5) {
  matrix(rbinom(n = nrow * ncol, size = 1, prob = prob), ncol = ncol)
}

set.seed(1337)
bitdata <- bitm(nrow = 10, ncol = 160)

bit6 <- bitdata[1:6, ]
bit4 <- bitdata[7:10, ]

# Taken from RcppAnnoy
bit4q_hdsum <- 1275
bit6q_hdsum <-  1986
