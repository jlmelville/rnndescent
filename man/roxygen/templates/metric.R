#' @param metric Type of distance calculation to use. One of:
#'   - `"braycurtis"`
#'   - `"canberra"`
#'   - `"chebyshev"`
#'   - `"correlation"` (1 minus the Pearson correlation)
#'   - `"cosine"`
#'   - `"dice"`
#'   - `"dot"` (1 minus the normalized dot product, capped at 1;
#'   1 if either vector is zero)
#'   - `"euclidean"`
#'   - `"hamming"`
#'   - `"haversine"` (great-circle distance for 2D latitude/longitude in
#' *radians*; an error will be raised if values that appear to be supplied in
#' degrees are encountered)
#'   - `"hellinger"`
#'   - `"jaccard"`
#'   - `"jensenshannon"`
#'   - `"kulsinski"`
#'   - `"sqeuclidean"` (squared Euclidean)
#'   - `"manhattan"`
#'   - `"matching"` (fraction of features whose zero/nonzero indicators
#'   differ)
#'   - `"rogerstanimoto"`
#'   - `"russellrao"`
#'   - `"sokalmichener"`
#'   - `"sokalsneath"`
#'   - `"spearmanr"` (1 minus the Spearman rank correlation)
#'   - `"symmetrickl"` (symmetric Kullback-Leibler divergence)
#'   - `"trueangular"` (angle between vectors divided by pi; 0 for two zero
#'   vectors and 1/2 when exactly one vector is zero)
#'   - `"tsss"` (Triangle Area Similarity-Sector Area Similarity or TS-SS
#'   metric)
#'   - `"yule"`
#'
#'   For non-sparse data, the following variants are available with
#'   preprocessing: this trades memory for a potential speed up during the
#'   distance calculation. Some minor numerical differences should be expected
#'   compared to the non-preprocessed versions:
#'   - `"cosine-preprocess"`: `cosine` with preprocessing.
#'   - `"correlation-preprocess"`: `correlation` with preprocessing.
#'
#'   For non-sparse binary data passed as a `logical` matrix, the following
#'   metrics have specialized variants which should be substantially faster than
#'   the non-binary variants (in other cases the logical data will be treated as
#'   a dense numeric vector of 0s and 1s):
#'   - `"dice"`
#'   - `"hamming"`
#'   - `"jaccard"`
#'   - `"kulsinski"`
#'   - `"matching"`
#'   - `"rogerstanimoto"`
#'   - `"russellrao"`
#'   - `"sokalmichener"`
#'   - `"sokalsneath"`
#'   - `"yule"`
#' <% if (isTRUE(get0("explicit_margin", inherits = FALSE) == "TRUE")) { %>
#'
#'   Note that if `margin = "explicit"`, the metric is only used to determine
#'   whether an "angular" or "Euclidean" distance is used to measure the
#'   distance between split points in the tree.
#' <% } %>
