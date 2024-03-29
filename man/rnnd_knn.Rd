% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/rnndescent.R
\name{rnnd_knn}
\alias{rnnd_knn}
\title{Find approximate nearest neighbors}
\usage{
rnnd_knn(
  data,
  k = 30,
  metric = "euclidean",
  use_alt_metric = TRUE,
  init = "tree",
  n_trees = NULL,
  leaf_size = NULL,
  max_tree_depth = 200,
  margin = "auto",
  n_iters = NULL,
  delta = 0.001,
  max_candidates = NULL,
  weight_by_degree = FALSE,
  low_memory = TRUE,
  n_threads = 0,
  verbose = FALSE,
  progress = "bar",
  obs = "R"
)
}
\arguments{
\item{data}{Matrix of \code{n} items to generate neighbors for, with observations
in the rows and features in the columns. Optionally, input can be passed
with observations in the columns, by setting \code{obs = "C"}, which should be
more efficient. Possible formats are \code{\link[base:data.frame]{base::data.frame()}}, \code{\link[base:matrix]{base::matrix()}}
or \code{\link[Matrix:sparseMatrix]{Matrix::sparseMatrix()}}. Sparse matrices should be in \code{dgCMatrix}
format. Dataframes will be converted to \code{numerical} matrix format
internally, so if your data columns are \code{logical} and intended to be used
with the specialized binary \code{metric}s, you should convert it to a logical
matrix first (otherwise you will get the slower dense numerical version).}

\item{k}{Number of nearest neighbors to return. Optional if \code{init} is
specified.}

\item{metric}{Type of distance calculation to use. One of:
\itemize{
\item \code{"braycurtis"}
\item \code{"canberra"}
\item \code{"chebyshev"}
\item \code{"correlation"} (1 minus the Pearson correlation)
\item \code{"cosine"}
\item \code{"dice"}
\item \code{"euclidean"}
\item \code{"hamming"}
\item \code{"hellinger"}
\item \code{"jaccard"}
\item \code{"jensenshannon"}
\item \code{"kulsinski"}
\item \code{"sqeuclidean"} (squared Euclidean)
\item \code{"manhattan"}
\item \code{"rogerstanimoto"}
\item \code{"russellrao"}
\item \code{"sokalmichener"}
\item \code{"sokalsneath"}
\item \code{"spearmanr"} (1 minus the Spearman rank correlation)
\item \code{"symmetrickl"} (symmetric Kullback-Leibler divergence)
\item \code{"tsss"} (Triangle Area Similarity-Sector Area Similarity or TS-SS
metric)
\item \code{"yule"}
}

For non-sparse data, the following variants are available with
preprocessing: this trades memory for a potential speed up during the
distance calculation. Some minor numerical differences should be expected
compared to the non-preprocessed versions:
\itemize{
\item \code{"cosine-preprocess"}: \code{cosine} with preprocessing.
\item \code{"correlation-preprocess"}: \code{correlation} with preprocessing.
}

For non-sparse binary data passed as a \code{logical} matrix, the following
metrics have specialized variants which should be substantially faster than
the non-binary variants (in other cases the logical data will be treated as
a dense numeric vector of 0s and 1s):
\itemize{
\item \code{"dice"}
\item \code{"hamming"}
\item \code{"jaccard"}
\item \code{"kulsinski"}
\item \code{"matching"}
\item \code{"rogerstanimoto"}
\item \code{"russellrao"}
\item \code{"sokalmichener"}
\item \code{"sokalsneath"}
\item \code{"yule"}
}}

\item{use_alt_metric}{If \code{TRUE}, use faster metrics that maintain the
ordering of distances internally (e.g. squared Euclidean distances if using
\code{metric = "euclidean"}), then apply a correction at the end. Probably
the only reason to set this to \code{FALSE} is if you suspect that some
sort of numeric issue is occurring with your data in the alternative code
path.}

\item{init}{Name of the initialization strategy or initial \code{data} neighbor
graph to optimize. One of:
\itemize{
\item \code{"rand"} random initialization (the default).
\item \code{"tree"} use the random projection tree method of Dasgupta and Freund
(2008).
\item a pre-calculated neighbor graph. A list containing:
\itemize{
\item \code{idx} an \code{n} by \code{k} matrix containing the nearest neighbor indices.
\item \code{dist} (optional) an \code{n} by \code{k} matrix containing the nearest
neighbor distances. If the input distances are omitted, they will be
calculated for you.'
}
}

If \code{k} and \code{init} are specified as arguments to this function, and the
number of neighbors provided in \code{init} is not equal to \code{k} then:
\itemize{
\item if \code{k} is smaller, only the \code{k} closest values in \code{init} are retained.
\item if \code{k} is larger, then random neighbors will be chosen to fill \code{init} to
the size of \code{k}. Note that there is no checking if any of the random
neighbors are duplicates of what is already in \code{init} so effectively fewer
than \code{k} neighbors may be chosen for some observations under these
circumstances.
}}

\item{n_trees}{The number of trees to use in the RP forest. A larger number
will give more accurate results at the cost of a longer computation time.
The default of \code{NULL} means that the number is chosen based on the number
of observations in \code{data}. Only used if \code{init = "tree"}.}

\item{leaf_size}{The maximum number of items that can appear in a leaf. This
value should be chosen to match the expected number of neighbors you will
want to retrieve when running queries (e.g. if you want find 50 nearest
neighbors set \code{leaf_size = 50}) and should not be set to a value smaller
than \code{10}. Only used if \code{init = "tree"}.}

\item{max_tree_depth}{The maximum depth of the tree to build (default = 200).
If the maximum tree depth is exceeded then the leaf size of a tree may
exceed \code{leaf_size} which can result in a large number of neighbor distances
being calculated. If \code{verbose = TRUE} a message will be logged to indicate
that the leaf size is large. However, increasing the \code{max_tree_depth} may
not help: it may be that there is something unusual about the distribution
of your data set under your chose \code{metric} that makes a tree-based
initialization inappropriate. Only used if \code{init = "tree"}.}

\item{margin}{A character string specifying the method used to  assign points
to one side of the hyperplane or the other. Possible values are:
\itemize{
\item \code{"explicit"} categorizes all distance metrics as either Euclidean or
Angular (Euclidean after normalization), explicitly calculates a hyperplane
and offset, and then calculates the margin based on the dot product with
the hyperplane.
\item \code{"implicit"} calculates the distance from a point to each of the
points defining the normal vector. The margin is calculated by comparing the
two distances: the point is assigned to the side of the hyperplane that
the normal vector point with the closest distance belongs to.
\item \code{"auto"} (the default) picks the margin method depending on whether a
binary-specific \code{metric} such as \code{"bhammming"} is chosen, in which case
\code{"implicit"} is used, and \code{"explicit"} otherwise: binary-specific metrics
involve storing the data in a way that isn't very efficient for the
\code{"explicit"} method and the binary-specific metric is usually a lot faster
than the generic equivalent such that the cost of two distance calculations
for the margin method is still faster.
}

Only used if \code{init = "tree"}.}

\item{n_iters}{Number of iterations of nearest neighbor descent to carry out.
By default, this will be chosen based on the number of observations in
\code{data}.}

\item{delta}{The minimum relative change in the neighbor graph allowed before
early stopping. Should be a value between 0 and 1. The smaller the value,
the smaller the amount of progress between iterations is allowed. Default
value of \code{0.001} means that at least 0.1\% of the neighbor graph must
be updated at each iteration.}

\item{max_candidates}{Maximum number of candidate neighbors to try for each
item in each iteration. Use relative to \code{k} to emulate the "rho"
sampling parameter in the nearest neighbor descent paper. By default, this
is set to \code{k} or \code{60}, whichever is smaller.}

\item{weight_by_degree}{If \code{TRUE}, then candidates for the local join are
weighted according to their in-degree, so that if there are more than
\code{max_candidates} in a candidate list, candidates with a smaller degree are
favored for retention. This prevents items with large numbers of edges
crowding out other items and for high-dimensional data is likely to provide
a small improvement in accuracy. Because this incurs a small extra cost of
counting the degree of each node, and because it tends to delay early
convergence, by default this is \code{FALSE}.}

\item{low_memory}{If \code{TRUE}, use a lower memory, but more
computationally expensive approach to index construction. If set to
\code{FALSE}, you should see a noticeable speed improvement, especially when
using a smaller number of threads, so this is worth trying if you have the
memory to spare.}

\item{n_threads}{Number of threads to use.}

\item{verbose}{If \code{TRUE}, log information to the console.}

\item{progress}{Determines the type of progress information logged during the
nearest neighbor descent stage when \code{verbose = TRUE}. Options are:
\itemize{
\item \code{"bar"}: a simple text progress bar.
\item \code{"dist"}: the sum of the distances in the approximate knn graph at the
end of each iteration.
}}

\item{obs}{set to \code{"C"} to indicate that the input \code{data} orientation stores
each observation as a column. The default \code{"R"} means that observations are
stored in each row. Storing the data by row is usually more convenient, but
internally your data will be converted to column storage. Passing it
already column-oriented will save some memory and (a small amount of) CPU
usage.}
}
\value{
the approximate nearest neighbor index, a list containing:
\itemize{
\item \code{idx} an n by k matrix containing the nearest neighbor indices.
\item \code{dist} an n by k matrix containing the nearest neighbor distances.
}
}
\description{
This function builds an approximate nearest neighbors graph of the provided
data using convenient defaults. It does not return an index for later
querying, to speed the graph construction and reduce the size and complexity
of the return value.
}
\details{
The process of k-nearest neighbor graph construction using Random Projection
Forests (Dasgupta and Freund, 2008) for initialization and Nearest Neighbor
Descent (Dong and co-workers, 2011) for refinement. If you are sure you will
not want to query new data then compared to \code{\link[=rnnd_build]{rnnd_build()}} this function has
the advantage of not storing the index, which can be very large.
}
\examples{

# Find 4 (approximate) nearest neighbors using Euclidean distance
iris_knn <- rnnd_knn(iris, k = 4)

}
\references{
Dasgupta, S., & Freund, Y. (2008, May).
Random projection trees and low dimensional manifolds.
In \emph{Proceedings of the fortieth annual ACM symposium on Theory of computing}
(pp. 537-546).
\doi{10.1145/1374376.1374452}.

Dong, W., Moses, C., & Li, K. (2011, March).
Efficient k-nearest neighbor graph construction for generic similarity measures.
In \emph{Proceedings of the 20th international conference on World Wide Web}
(pp. 577-586).
ACM.
\doi{10.1145/1963405.1963487}.
}
\seealso{
\code{\link[=rnnd_build]{rnnd_build()}}, \code{\link[=rnnd_query]{rnnd_query()}}
}
