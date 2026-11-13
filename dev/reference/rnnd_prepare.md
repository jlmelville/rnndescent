# Prepare an index for repeated queries

Copy an index into native memory once and pass the returned handle to
[`rnnd_query()`](https://jlmelville.github.io/rnndescent/dev/reference/rnnd_query.md)
for repeated queries. Dense numeric indexes support the dense metrics
with an explicit search forest. Dense logical indexes support packed
binary metrics with an implicit search forest. Sparse indexes support
the sparse metrics with either explicit or implicit search forests.

## Usage

``` r
rnnd_prepare(index)
```

## Arguments

- index:

  An index from
  [`rnnd_build()`](https://jlmelville.github.io/rnndescent/dev/reference/rnnd_build.md)
  with a compatible search forest.

## Value

A prepared index handle for use in the current R session.

## Details

The handle owns a snapshot of the reference, search graph and forest.
Changes to the original index require a new call to `rnnd_prepare()`.
Garbage collection releases the native memory when the handle is no
longer in use. Unloading the package also releases the snapshot and
invalidates the handle. To use the index in another session, save the
original index and prepare it again after loading it.

Preparation applies any metric-specific reference preprocessing once.
For dense `"spearmanr"`, the native snapshot stores reference ranks and
queries rank their features once for distance scoring. Dense logical
preparation packs the reference once and packs only the incoming queries
on subsequent calls. Graph traversal uses the same distance kernels as
ordinary queries. Floating-point rounding can differ for dense numeric
indexes and may affect the ordering of near ties.

## See also

[`rnnd_query()`](https://jlmelville.github.io/rnndescent/dev/reference/rnnd_query.md),
[`rnnd_build()`](https://jlmelville.github.io/rnndescent/dev/reference/rnnd_build.md)

## Examples

``` r
index <- rnnd_build(iris[1:75, ], k = 5)
prepared <- rnnd_prepare(index)
rnnd_query(prepared, iris[76:80, ], k = 5)
#> $idx
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]   66   59   75   52   55
#> [2,]   59   53   55   51   66
#> [3,]   53   55   51   59   57
#> [4,]   64   62   56   74   67
#> [5,]   70   65   68   54   63
#> 
#> $dist
#>           [,1]      [,2]      [,3]      [,4]      [,5]
#> [1,] 0.1414212 0.2449488 0.2645749 0.3162277 0.3162277
#> [2,] 0.3162282 0.3464101 0.3741661 0.4582576 0.5099021
#> [3,] 0.3162279 0.5291503 0.5567767 0.5830953 0.5916078
#> [4,] 0.2449488 0.3316626 0.3741660 0.3872982 0.4123106
#> [5,] 0.4358899 0.4472136 0.6164414 0.6855654 0.7071068
#> 
```
