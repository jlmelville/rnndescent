# ten iris entries where the 4 nearest neighbors are distinct
uiris <- unique(iris)
uirism <- as.matrix(uiris[, -5])
ui10 <- uirism[6:15, ]

ui10sp_full <- Matrix::drop0(ui10)

set.seed(1337)
ui10z <- ui10
ui10z[sample(prod(dim(ui10z)), 10)] <- 0
ui10sp <- Matrix::drop0(ui10z)

ui10z6 <- head(ui10z, 6)
ui10z4 <- tail(ui10z, 4)

ui10sp6 <- head(ui10sp, 6)
ui10sp4 <- tail(ui10sp, 4)

# treat sum of distances an objective function
# expected sum from sum(FNN::get.knn(uirism, 14)$nn.dist)
ui_edsum <- 1016.834
# sum(FNN::get.knn(ui10, 3)$nn.dist)
ui10_edsum <- 13.28425

ui6 <- ui10[1:6, ]
ui4 <- ui10[7:10, ]
ui10_eucd <- as.matrix(dist(ui10))

# sum(RcppHNSW::hnsw_search(ui4, RcppHNSW::hnsw_build(ui6), k = 4)$dist)
ui4q_edsum <- 9.310494
# sum(RcppHNSW::hnsw_search(ui6, RcppHNSW::hnsw_build(ui4), k = 4)$dist)
ui6q_edsum <- 18.98666

# NB Annoy and HNSW don't agree to more than this # of decimal places
# sum(RcppHNSW::hnsw_search(ui4, RcppHNSW::hnsw_build(ui6, distance = "cosine"), k = 4)$dist)
ui4q_cdsum <- 0.02072
# sum(RcppHNSW::hnsw_search(ui6, RcppHNSW::hnsw_build(ui4, distance = "cosine"), k = 4)$dist)
ui6q_cdsum <- 0.04220

# Manhattan: Taken from RcppAnnoy
ui4q_mdsum <- 15.4
ui6q_mdsum <- 31.6

# Hamming
bitm <- function(nrow, ncol, prob = 0.5) {
  matrix(rbinom(n = nrow * ncol, size = 1, prob = prob), ncol = ncol)
}

set.seed(1337)
bitdata <- bitm(nrow = 10, ncol = 160)
intdata <- matrix(sample.int(5, 40, replace = TRUE), 10)

bitdatasp <- Matrix::drop0(bitdata)
lbitdata <- matrix(as.logical(bitdata), nrow = nrow(bitdatasp))

bit6 <- bitdata[1:6, ]
bit4 <- bitdata[7:10, ]

lbit6 <- matrix(as.logical(bit6), nrow = nrow(bit6))
lbit4 <- matrix(as.logical(bit4), nrow = nrow(bit4))

# Hamming
# from Annoy
expected_hamm_idx <- matrix(
  c(
    1, 7, 4, 5,
    2, 10, 3, 9,
    3, 4, 2, 7,
    4, 3, 1, 7,
    5, 6, 7, 1,
    6, 5, 10, 3,
    7, 1, 10, 5,
    8, 9, 10, 7,
    9, 8, 10, 4,
    10, 2, 9, 7
  ),
  byrow = TRUE, nrow = 10, ncol = 4
)

# distances normalized wrt ndim for consistency with PyNNDescent
expected_hamm_dist <- matrix(
  c(
    0, 72, 74, 77,
    0, 69, 78, 79,
    0, 65, 78, 79,
    0, 65, 74, 76,
    0, 67, 75, 77,
    0, 67, 80, 81,
    0, 72, 74, 75,
    0, 69, 77, 81,
    0, 69, 72, 78,
    0, 69, 72, 74
  ),
  byrow = TRUE, nrow = 10, ncol = 4
) / ncol(bitdata)

int6 <- intdata[1:6, ]
int4 <- intdata[7:10, ]

int6hd <- matrix(c(
  0,   2,   3,   3,   3,   2,
  2,   0,   2,   2,   4,   4,
  3,   2,   0,   4,   4,   4,
  3,   2,   4,   0,   3,   3,
  3,   4,   4,   3,   0,   2,
  2,   4,   4,   3,   2,   0
), nrow = 6) / ncol(int6)

# Taken from RcppAnnoy (and then normalize wrt num features for consistency with PyNNDescent)
bit4q_hdsum <- 1275 / ncol(bitdata)
bit6q_hdsum <- 1986 / ncol(bitdata)

# Distance matrices generated with Annoy

ui10_cosd <- matrix(c(
  0.0000000, 0.0001315, 0.0007371, 1.131e-03, 2.628e-03, 0.0009933, 0.0004830, 2.966e-03, 1.830e-03, 0.0039657,
  0.0001315, 0.0000000, 0.0009095, 1.648e-03, 3.132e-03, 0.0009375, 0.0006808, 3.429e-03, 1.426e-03, 0.0033142,
  0.0007371, 0.0009095, 0.0000000, 2.441e-04, 6.824e-04, 0.0001688, 0.0004221, 8.124e-04, 8.191e-04, 0.0027981,
  0.0011313, 0.0016477, 0.0002441, 1.372e-09, 3.832e-04, 0.0007671, 0.0005494, 6.114e-04, 1.888e-03, 0.0043782,
  0.0026276, 0.0031324, 0.0006824, 3.832e-04, 0.000e+00, 0.0011068, 0.0014791, 7.131e-05, 2.038e-03, 0.0043755,
  0.0009933, 0.0009375, 0.0001688, 7.671e-04, 1.107e-03, 0.0000000, 0.0009307, 1.066e-03, 2.660e-04, 0.0016037,
  0.0004830, 0.0006808, 0.0004221, 5.494e-04, 1.479e-03, 0.0009307, 0.0000000, 1.947e-03, 1.850e-03, 0.0047715,
  0.0029658, 0.0034286, 0.0008124, 6.114e-04, 7.131e-05, 0.0010661, 0.0019470, 0.000e+00, 1.851e-03, 0.0037512,
  0.0018304, 0.0014256, 0.0008191, 1.888e-03, 2.038e-03, 0.0002660, 0.0018498, 1.851e-03, 1.945e-09, 0.0007827,
  0.0039657, 0.0033142, 0.0027981, 4.378e-03, 4.376e-03, 0.0016037, 0.0047715, 3.751e-03, 7.827e-04, 0.0000000
), nrow = 10)

ui10_mand <- matrix(c(
  0.0,  1.7,  1.3, 2.5, 1.8, 0.6, 1.4, 2.1, 2.9,  1.2,
  1.7,  0.0,  0.6, 0.8, 0.9, 1.3, 0.5, 0.8, 1.2,  2.1,
  1.3,  0.6,  0.0, 1.2, 0.5, 0.7, 0.3, 0.8, 1.6,  1.7,
  2.5,  0.8,  1.2, 0.0, 0.9, 1.9, 1.1, 0.6, 0.6,  2.7,
  1.8,  0.9,  0.5, 0.9, 0.0, 1.2, 0.6, 0.3, 1.1,  2.2,
  0.6,  1.3,  0.7, 1.9, 1.2, 0.0, 1.0, 1.5, 2.3,  1.0,
  1.4,  0.5,  0.3, 1.1, 0.6, 1.0, 0.0, 0.7, 1.5,  2.0,
  2.1,  0.8,  0.8, 0.6, 0.3, 1.5, 0.7, 0.0, 0.8,  2.3,
  2.9,  1.2,  1.6, 0.6, 1.1, 2.3, 1.5, 0.8, 0.0,  2.7,
  1.2,  2.1,  1.7, 2.7, 2.2, 1.0, 2.0, 2.3, 2.7,  0.0
), nrow = 10)

bit10_hamd <- matrix(c(
  0, 85, 81, 74, 77, 90, 72, 93, 90, 90,
  85, 0, 78, 83, 82, 83, 89, 84, 79, 69,
  81, 78, 0, 65, 86, 81, 79, 82, 81, 89,
  74, 83, 65, 0, 83, 90, 76, 85, 78, 92,
  77, 82, 86, 83, 0, 67, 75, 82, 83, 81,
  90, 83, 81, 90, 67, 0, 82, 87, 88, 80,
  72, 89, 79, 76, 75, 82, 0, 81, 92, 74,
  93, 84, 82, 85, 82, 87, 81, 0, 69, 77,
  90, 79, 81, 78, 83, 88, 92, 69, 0, 72,
  90, 69, 89, 92, 81, 80, 74, 77, 72, 0
), nrow = 10) / ncol(bitdata)

# for uirism[1:10, ]
uirism10_cord <- matrix(
  c(
    0, 0.00400133876, 2.60889537e-05, 0.00183154822, 0.0006526685,
    0.000413946853, 0.00118880691, 0.000461852891, 0.00192335771,
    0.00344803882, 0.00400133876, 0, 0.00339291433, 0.00260336976,
    0.00776732119, 0.00640810993, 0.00927944638, 0.00288194472, 0.00145365862,
    0.00096714455, 2.60889537e-05, 0.00339291433, 0, 0.00166652079,
    0.000938868047, 0.000622703492, 0.00156232107, 0.000395491414,
    0.00164390757, 0.00301440442, 0.00183154822, 0.00260336976, 0.00166652079,
    0, 0.00328117923, 0.00216742062, 0.00386063303, 0.000454440488,
    0.000166690505, 0.000693152364, 0.0006526685, 0.00776732119,
    0.000938868047, 0.00328117923, 0, 0.000116680316, 8.60443273e-05,
    0.00149684289, 0.00396913822, 0.00623882524, 0.000413946853,
    0.00640810993, 0.000622703492, 0.00216742062, 0.000116680316,
    0, 0.000277394147, 0.000821174669, 0.00278434617, 0.00473942264,
    0.00118880691, 0.00927944638, 0.00156232107, 0.00386063303, 8.60443273e-05,
    0.000277394147, 1.11022302e-16, 0.00204787836, 0.00478602797,
    0.0072727783, 0.000461852891, 0.00288194472, 0.000395491414,
    0.000454440488, 0.00149684289, 0.000821174669, 0.00204787836,
    0, 0.000593789371, 0.00162634825, 0.00192335771, 0.00145365862,
    0.00164390757, 0.000166690505, 0.00396913822, 0.00278434617,
    0.00478602797, 0.000593789371, 0, 0.000260225275, 0.00344803882,
    0.00096714455, 0.00301440442, 0.000693152364, 0.00623882524,
    0.00473942264, 0.0072727783, 0.00162634825, 0.000260225275, 0
  ),
  nrow = 10
)

ui10_nn4 <- list(
  idx = matrix(
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
    byrow = TRUE, ncol = 4
  ),
  dist = matrix(
    c(
      0, 0.3464, 0.6782, 0.7,
      0, 0.3, 0.4243, 0.4796,
      0, 0.2236, 0.3317, 0.4243,
      0, 0.3464, 0.4243, 0.5477,
      0, 0.1732, 0.3317, 0.3464,
      0, 0.3464, 0.5, 0.5831,
      0, 0.2236, 0.3, 0.3464,
      0, 0.1732, 0.4243, 0.4583,
      0, 0.3464, 0.5831, 0.6164,
      0, 0.5831, 0.6782, 1.044
    ),
    byrow = TRUE, ncol = 4
  )
)

# set.seed(1337)
# rpf_build(ui10, metric = "euclidean", leaf_size = 4, margin = "explicit")
rpf_index_ls4e <-
  list(
    trees = list(
      list(hyperplanes = structure(c(
        -0.5, 0.300000190734863,
        0, 0, 0, -0.800000190734863, -0.300000190734863, 0, 0, 0, -0.200000047683716,
        0.100000023841858, 0, 0, 0, -0.300000011920929, -0.200000017881393,
        0, 0, 0
      ), dim = 5:4), offsets = c(
        5.77000093460083, -0.555000305175781,
        NaN, NaN, NaN
      ), children = structure(c(
        1L, 2L, 0L, 3L, 7L, 4L,
        3L, 3L, 7L, 10L
      ), dim = c(5L, 2L)), indices = c(
        2L, 4L, 7L, 1L,
        3L, 6L, 8L, 0L, 5L, 9L
      ), leaf_size = 4), list(
        hyperplanes = structure(c(
          0.599999904632568,
          0.0999999046325684, 0.400000095367432, -0.400000095367432, 0,
          0, 0, 0, 0, 0.0999999046325684, 0.300000190734863, 0, -0.299999952316284,
          0, 0, 0, 0, 0, 0.399999976158142, 0, 0.100000023841858, 0, 0,
          0, 0, 0, 0, 0, 0.100000001490116, -0.100000008940697, 0, 0, 0,
          0, 0, 0
        ), dim = c(9L, 4L)), offsets = c(
          -3.58499956130981, -1.4850001335144,
          -2.04000043869019, 3.14500045776367, NaN, NaN, NaN, NaN, NaN
        ),
        children = structure(c(
          1L, 2L, 3L, 4L, 0L, 2L, 5L, 6L, 8L,
          8L, 7L, 6L, 5L, 2L, 5L, 6L, 8L, 10L
        ), dim = c(9L, 2L)), indices = c(
          2L,
          6L, 0L, 5L, 9L, 1L, 4L, 7L, 3L, 8L
        ), leaf_size = 3
      ), list(
        hyperplanes = structure(c(
          1.09999990463257, -0.599999904632568,
          0, 0, 0.5, 0, 0, 0.700000047683716, -0.5, 0, 0, 0.199999809265137,
          0, 0, 0.399999976158142, -0.100000023841858, 0, 0, 0.100000023841858,
          0, 0, 0.100000001490116, -0.200000002980232, 0, 0, -0.100000001490116,
          0, 0
        ), dim = c(7L, 4L)), offsets = c(
          -8.21500015258789, 5.10999965667725,
          NaN, NaN, -3.05499935150146, NaN, NaN
        ), children = structure(c(
          1L,
          2L, 0L, 2L, 5L, 5L, 8L, 4L, 3L, 2L, 5L, 6L, 8L, 10L
        ), dim = c(
          7L,
          2L
        )), indices = c(2L, 6L, 0L, 5L, 9L, 1L, 4L, 7L, 3L, 8L),
        leaf_size = 3
      ), list(hyperplanes = structure(c(
        -0.0999999046325684,
        0, 0.800000190734863, 0, 0, -0.0999999046325684, 0, 0.299999952316284,
        0, 0, -0.100000023841858, 0, 0.100000023841858, 0, 0, 0, 0, -0.100000008940697,
        0, 0
      ), dim = 5:4), offsets = c(
        0.934999287128448, NaN, -5.18500089645386,
        NaN, NaN
      ), children = structure(c(
        1L, 0L, 3L, 3L, 6L, 2L, 3L,
        4L, 6L, 10L
      ), dim = c(5L, 2L)), indices = c(
        3L, 7L, 8L, 0L, 5L,
        9L, 1L, 2L, 4L, 6L
      ), leaf_size = 4), list(hyperplanes = structure(c(
        0.300000190734863,
        0.599999904632568, 0, 0, 0.400000095367432, 0, 0, -0.300000190734863,
        0.700000047683716, 0, 0, 0.5, 0, 0, 0.100000023841858, 0.100000023841858,
        0, 0, 0.200000047683716, 0, 0, -0.200000017881393, 0.100000001490116,
        0, 0, 0, 0, 0
      ), dim = c(7L, 4L)), offsets = c(
        -0.555000305175781,
        -5.5649995803833, NaN, NaN, -3.71500062942505, NaN, NaN
      ), children = structure(c(
        1L,
        2L, 0L, 2L, 5L, 5L, 8L, 4L, 3L, 2L, 5L, 6L, 8L, 10L
      ), dim = c(
        7L,
        2L
      )), indices = c(5L, 9L, 2L, 4L, 7L, 0L, 1L, 6L, 3L, 8L), leaf_size = 3),
      list(hyperplanes = structure(c(
        -0.599999904632568, -0.0999999046325684,
        0, 0, 0, -0.5, -0.0999999046325684, 0, 0, 0, -0.100000023841858,
        -0.100000023841858, 0, 0, 0, -0.200000002980232, 0, 0, 0,
        0
      ), dim = 5:4), offsets = c(
        5.10999965667725, 0.934999287128448,
        NaN, NaN, NaN
      ), children = structure(c(
        1L, 2L, 0L, 3L, 7L,
        4L, 3L, 3L, 7L, 10L
      ), dim = c(5L, 2L)), indices = c(
        3L, 7L,
        8L, 1L, 2L, 4L, 6L, 0L, 5L, 9L
      ), leaf_size = 4), list(hyperplanes = structure(c(
        0.900000095367432,
        0, -0.599999904632568, 0, 0, 0.900000095367432, 0, -0.5,
        0, 0, -0.299999952316284, 0, -0.100000023841858, 0, 0, 0.100000001490116,
        0, 0, 0, 0
      ), dim = 5:4), offsets = c(
        -7.62000131607056, NaN,
        4.53999948501587, NaN, NaN
      ), children = structure(c(
        1L, 0L,
        3L, 3L, 6L, 2L, 3L, 4L, 6L, 10L
      ), dim = c(5L, 2L)), indices = c(
        0L,
        5L, 9L, 3L, 7L, 8L, 1L, 2L, 4L, 6L
      ), leaf_size = 4)
    ), margin = "explicit",
    actual_metric = "sqeuclidean", version = "0.0.12", use_alt_metric = TRUE,
    original_metric = "euclidean", sparse = FALSE, type = "rnndescent:rpforest"
  )

# set.seed(1337)
# rpf_build(ui10, metric = "euclidean", leaf_size = 4, margin = "implicit")
rpf_index_ls4i <-
  list(
    trees = list(list(normal_indices = structure(c(
      4L, 4L, -1L,
      -1L, -1L, 0L, 1L, -1L, -1L, -1L
    ), dim = c(5L, 2L)), children = structure(c(
      1L,
      2L, 0L, 3L, 7L, 4L, 3L, 3L, 7L, 10L
    ), dim = c(5L, 2L)), indices = c(
      2L,
      4L, 7L, 1L, 3L, 6L, 8L, 0L, 5L, 9L
    ), leaf_size = 4), list(normal_indices = structure(c(
      4L,
      2L, 2L, 2L, -1L, -1L, -1L, -1L, -1L, 8L, 4L, 1L, 5L, -1L, -1L,
      -1L, -1L, -1L
    ), dim = c(9L, 2L)), children = structure(c(
      1L,
      2L, 3L, 4L, 0L, 2L, 5L, 6L, 8L, 8L, 7L, 6L, 5L, 2L, 5L, 6L, 8L,
      10L
    ), dim = c(9L, 2L)), indices = c(
      2L, 6L, 0L, 5L, 9L, 1L, 4L,
      7L, 3L, 8L
    ), leaf_size = 3), list(normal_indices = structure(c(
      5L,
      6L, -1L, -1L, 4L, -1L, -1L, 8L, 0L, -1L, -1L, 3L, -1L, -1L
    ), dim = c(
      7L,
      2L
    )), children = structure(c(
      1L, 2L, 0L, 2L, 5L, 5L, 8L, 4L,
      3L, 2L, 5L, 6L, 8L, 10L
    ), dim = c(7L, 2L)), indices = c(
      2L, 6L,
      0L, 5L, 9L, 1L, 4L, 7L, 3L, 8L
    ), leaf_size = 3), list(
      normal_indices = structure(c(
        7L,
        -1L, 5L, -1L, -1L, 4L, -1L, 1L, -1L, -1L
      ), dim = c(5L, 2L)),
      children = structure(c(
        1L, 0L, 3L, 3L, 6L, 2L, 3L, 4L, 6L,
        10L
      ), dim = c(5L, 2L)), indices = c(
        3L, 7L, 8L, 0L, 5L, 9L,
        1L, 2L, 4L, 6L
      ), leaf_size = 4
    ), list(normal_indices = structure(c(
      4L,
      5L, -1L, -1L, 6L, -1L, -1L, 1L, 7L, -1L, -1L, 3L, -1L, -1L
    ), dim = c(
      7L,
      2L
    )), children = structure(c(
      1L, 2L, 0L, 2L, 5L, 5L, 8L, 4L,
      3L, 2L, 5L, 6L, 8L, 10L
    ), dim = c(7L, 2L)), indices = c(
      5L, 9L,
      2L, 4L, 7L, 0L, 1L, 6L, 3L, 8L
    ), leaf_size = 3), list(
      normal_indices = structure(c(
        6L,
        7L, -1L, -1L, -1L, 0L, 4L, -1L, -1L, -1L
      ), dim = c(5L, 2L)),
      children = structure(c(
        1L, 2L, 0L, 3L, 7L, 4L, 3L, 3L, 7L,
        10L
      ), dim = c(5L, 2L)), indices = c(
        3L, 7L, 8L, 1L, 2L, 4L,
        6L, 0L, 5L, 9L
      ), leaf_size = 4
    ), list(
      normal_indices = structure(c(
        9L,
        -1L, 3L, -1L, -1L, 4L, -1L, 2L, -1L, -1L
      ), dim = c(5L, 2L)),
      children = structure(c(
        1L, 0L, 3L, 3L, 6L, 2L, 3L, 4L, 6L,
        10L
      ), dim = c(5L, 2L)), indices = c(
        0L, 5L, 9L, 3L, 7L, 8L,
        1L, 2L, 4L, 6L
      ), leaf_size = 4
    )), margin = "implicit", actual_metric = "sqeuclidean",
    version = "0.0.12", use_alt_metric = TRUE, original_metric = "euclidean",
    sparse = FALSE, type = "rnndescent:rpforest"
  )
