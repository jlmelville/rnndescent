## Minor release

This release adds prepared indexes for repeated queries and fixes several distance calculations.

## Test environments

* local Ubuntu 26.04 LTS (WSL2), R 4.5.2, GCC 15.2.0
* Ubuntu 26.04 (on GitHub Actions), R 4.5.3, R 4.6.2 and R-devel, GCC 15.2.0
* local Windows 11, R 4.6.2, GCC 14.2.0
* Windows Server 2025 (on GitHub Actions), R 4.5.3 and R 4.6.2, GCC 14.3.0
* Windows Server 2022 (on Win-Builder), R-devel r90926, GCC 14.3.0
* local macOS arm64, R 4.6.2, Apple Clang 21.0.0
* macOS Tahoe 26 arm64 (on GitHub Actions), R 4.6.2, Apple Clang 21.0.0
* Ubuntu 22.04 (on R-hub), R-devel, Clang 22.1.8
* Ubuntu 26.04 (on R-hub), R-devel, Clang 23.1.6 (ASan and UBSan)
* Fedora 44 (on R-hub), R-devel, GCC 16.2.1 (standard, ASan and Valgrind)
* Ubuntu 26.04 (on R-hub), R 4.6.2, GCC 15.2.0 (LTO)

## R CMD check results

0 errors | 0 warnings

Win-Builder reported one NOTE for six Wikipedia links (HTTP 429) and a broken vignette link.
The Wikipedia links resolved on recheck. The corrected vignette link works in the built package.

## CRAN checks

All CRAN checks for rnndescent 0.2.0 report OK.

## revdepcheck results

We checked all 3 direct CRAN reverse dependencies and 1 sampled indirect dependency,
comparing R CMD check results for CRAN and development versions of rnndescent.

* We saw 0 new problems.
* We failed to check 0 packages.
