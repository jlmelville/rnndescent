# ten iris entries where the 4 nearest neighbors are distinct
uiris <- unique(iris)
uirism <- as.matrix(uiris[, -5])
ui10 <- uirism[6:15, ]

ui10_eucd <- as.matrix(dist(ui10))
