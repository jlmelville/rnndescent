## Release summary

This release adds `rnnd_prepare()` for repeated queries and new nearest-neighbor
graph utilities. It corrects several distance calculations, improves input
validation, and improves index construction and query performance. Users should
rebuild graphs or indexes affected by the distance corrections described in NEWS.

TODO before submission — set the final release version and reconcile these
results with the submitted source tarball. R-hub and GitHub Actions checked
development version 0.2.0.9000 at b7b9c1c0. The local macOS check also includes
the cosine comparison fix.

## Completed checks

### R-hub

All seven R-hub jobs completed successfully on 2026-11-17 UTC. Each returned
`Status: OK` with 0 errors, 0 warnings, and 0 notes. Each job reported 6,075
passing testthat assertions with no failures, warnings, or skips.

* Ubuntu 26.04.1, R-devel r90890, Clang 23.1.6, AddressSanitizer.
* Ubuntu 26.04.1, R-devel r90890, Clang 23.1.6, UndefinedBehaviorSanitizer.
* Fedora 44, R-devel r90899, GCC 16.2.1, AddressSanitizer.
* Fedora 44, R-devel r90899, GCC 16.2.1, Valgrind. Valgrind reported
  0 memory errors and 0 bytes definitely lost.
* Ubuntu 26.04.1, R 4.6.2, GCC 15.2.0, link-time optimization.
* Ubuntu 22.04.5, R-devel r90685, Clang 22.1.8.
* Fedora 44, R-devel r90899, GCC 16.2.1.

All jobs used `--no-manual`. The GCC ASan, GCC 16, Clang 22, and LTO jobs used
`--as-cran`. The Clang sanitizer and Valgrind jobs used their platform-specific
check settings.

[R-hub run](https://github.com/jlmelville/rnndescent/actions/runs/44028574259)

### Local macOS

* macOS arm64, Homebrew R 4.6.2, Apple Clang 21.0.0.
  `R CMD check --as-cran --no-manual` returned `Status: OK` with 0 errors,
  0 warnings, and 0 notes. All 6,077 testthat assertions passed with no
  failures, warnings, or skips.

### GitHub Actions

The following jobs completed with 0 errors, 0 warnings, and 0 notes, using
`--no-manual --as-cran`.

* macOS Tahoe 26.7.2, R 4.6.2, Apple Clang 21.0.0.
* Windows x64, R 4.6.2, GCC 14.3.0.
* Windows x64, R 4.5.3, GCC 14.3.0.

[GitHub Actions run](https://github.com/jlmelville/rnndescent/actions/runs/43823267204)

## Remaining checks

* TODO — final source tarball, `R CMD check --as-cran` on current R-devel,
  including the PDF manual. Record the version, environment, and results.
* TODO — rerun the Ubuntu release, oldrel-1, and devel jobs with the cosine
  comparison fix. The previous b7b9c1c0 run has 1 check error per Ubuntu job
  from two dense-versus-sparse cosine assertions.
* TODO — local WSL Linux on the Intel laptop. Record the distribution, R and
  compiler versions, check options, and results.
* TODO — local native Windows on the Intel laptop. Record the Windows, R, and
  Rtools versions, check options, and results.
* TODO — win-builder R-devel. Record the R revision and results.
* TODO — mac-builder R-devel. Record the environment and results.

## Reverse dependencies

TODO — complete comparisons of CRAN rnndescent 0.2.0 with development
0.2.0.9000 at b7b9c1c0 for the following cohort.

* All direct CRAN reverse dependencies, including reverse Suggests —
  bbknnR 2.0.2, blocking 1.0.3, and uwot 0.2.5.
* One randomly sampled indirect reverse dependency — XYomics 0.1.5, through
  uwot, selected with seed 20261117.

TODO — record completed comparisons, new problems, and packages that could
not be checked. Describe any coverage limits from unavailable optional
dependencies.
