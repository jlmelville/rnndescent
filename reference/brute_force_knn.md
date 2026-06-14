# Find exact nearest neighbors by brute force

Returns the exact nearest neighbors of a dataset. A brute force search
is carried out: all possible pairs of points are compared, and the
nearest neighbors are returned.

## Usage

``` r
brute_force_knn(
  data,
  k,
  metric = "euclidean",
  use_alt_metric = TRUE,
  n_threads = 0,
  verbose = FALSE,
  obs = "R"
)
```

## Arguments

- data:

  Matrix of `n` items to generate neighbors for, with observations in
  the rows and features in the columns. Optionally, input can be passed
  with observations in the columns, by setting `obs = "C"`, which should
  be more efficient. Possible formats are
  [`base::data.frame()`](https://rdrr.io/r/base/data.frame.html),
  [`base::matrix()`](https://rdrr.io/r/base/matrix.html) or
  [`Matrix::sparseMatrix()`](https://rdrr.io/pkg/Matrix/man/sparseMatrix.html).
  Sparse matrices should be in `dgCMatrix` format. Dataframes will be
  converted to `numerical` matrix format internally, so if your data
  columns are `logical` and intended to be used with the specialized
  binary `metric`s, you should convert it to a logical matrix first
  (otherwise you will get the slower dense numerical version).

- k:

  Number of nearest neighbors to return.

- metric:

  Type of distance calculation to use. One of:

  - `"braycurtis"`

  - `"canberra"`

  - `"chebyshev"`

  - `"correlation"` (1 minus the Pearson correlation)

  - `"cosine"`

  - `"dice"`

  - `"euclidean"`

  - `"hamming"`

  - `"haversine"` (great-circle distance for 2D latitude/longitude in
    *radians*; an error will be raised if values that appear to be
    supplied in degrees are encountered)

  - `"hellinger"`

  - `"jaccard"`

  - `"jensenshannon"`

  - `"kulsinski"`

  - `"sqeuclidean"` (squared Euclidean)

  - `"manhattan"`

  - `"rogerstanimoto"`

  - `"russellrao"`

  - `"sokalmichener"`

  - `"sokalsneath"`

  - `"spearmanr"` (1 minus the Spearman rank correlation)

  - `"symmetrickl"` (symmetric Kullback-Leibler divergence)

  - `"tsss"` (Triangle Area Similarity-Sector Area Similarity or TS-SS
    metric)

  - `"yule"`

  For non-sparse data, the following variants are available with
  preprocessing: this trades memory for a potential speed up during the
  distance calculation. Some minor numerical differences should be
  expected compared to the non-preprocessed versions:

  - `"cosine-preprocess"`: `cosine` with preprocessing.

  - `"correlation-preprocess"`: `correlation` with preprocessing.

  For non-sparse binary data passed as a `logical` matrix, the following
  metrics have specialized variants which should be substantially faster
  than the non-binary variants (in other cases the logical data will be
  treated as a dense numeric vector of 0s and 1s):

  - `"dice"`

  - `"hamming"`

  - `"jaccard"`

  - `"kulsinski"`

  - `"matching"`

  - `"rogerstanimoto"`

  - `"russellrao"`

  - `"sokalmichener"`

  - `"sokalsneath"`

  - `"yule"`

- use_alt_metric:

  If `TRUE`, use faster metrics that maintain the ordering of distances
  internally (e.g. squared Euclidean distances if using
  `metric = "euclidean"`), then apply a correction at the end. Probably
  the only reason to set this to `FALSE` is if you suspect that some
  sort of numeric issue is occurring with your data in the alternative
  code path.

- n_threads:

  Number of threads to use.

- verbose:

  If `TRUE`, log information to the console.

- obs:

  set to `"C"` to indicate that the input `data` orientation stores each
  observation as a column. The default `"R"` means that observations are
  stored in each row. Storing the data by row is usually more
  convenient, but internally your data will be converted to column
  storage. Passing it already column-oriented will save some memory and
  (a small amount of) CPU usage.

## Value

the nearest neighbor graph as a list containing:

- `idx` an n by k matrix containing the nearest neighbor indices.

- `dist` an n by k matrix containing the nearest neighbor distances.

## Details

This method is accurate but scales poorly with dataset size, so use with
caution with larger datasets. Having the exact neighbors as a ground
truth to compare with approximate results is useful for benchmarking and
determining parameter settings of the approximate methods.

## Examples

``` r
# Find the 4 nearest neighbors using Euclidean distance
# If you pass a data frame, non-numeric columns are removed
iris_nn <- brute_force_knn(iris, k = 4, metric = "euclidean")

# Manhattan (l1) distance
iris_nn <- brute_force_knn(iris, k = 4, metric = "manhattan")

# Multi-threading: you can choose the number of threads to use: in real
# usage, you will want to set n_threads to at least 2
iris_nn <- brute_force_knn(iris, k = 4, metric = "manhattan", n_threads = 1)

# Use verbose flag to see information about progress
iris_nn <- brute_force_knn(iris, k = 4, metric = "euclidean", verbose = TRUE)
#> 00:38:30 Using alt metric 'sqeuclidean' for 'euclidean'
#> 00:38:30 Calculating brute force k-nearest neighbors with k = 4
#> 00:38:30 Finished
```
