# ten iris entries where the 4 nearest neighbors are distinct
uiris <- unique(iris)
uirism <- as.matrix(uiris[, -5])
ui10 <- uirism[6:15, ]


ui6 <- ui10[1:6, ]
ui4 <- ui10[7:10, ]
ui10_eucd <- as.matrix(dist(ui10))
