## New Minor Release

This release adds prepared indexes for repeated queries and fixes several distance calculations.

## Test environments

* R-hub, Ubuntu and Fedora, R-devel (ASan, UBSan and Valgrind) and R 4.6.2 (LTO)
* GitHub Actions, Ubuntu (R 4.5.3, 4.6.2 and R-devel), Windows (R 4.5.3 and 4.6.2)
  and macOS (R 4.6.2)
* Local macOS arm64 and Windows 11 (R 4.6.2), Ubuntu 26.04 on WSL2 (R 4.5.2)
* win-builder (R-devel)

## CRAN Checks

All CRAN checks for the current release (0.2.0) are OK.
The local macOS, Windows and WSL2 checks and win-builder all finished with 0 errors and 0 warnings.

Win-builder reported one NOTE for six Wikipedia links (HTTP 429) and a broken link in a vignette.
The Wikipedia links resolved on recheck. The vignette link has been corrected.
The corrected tarball passed incoming checks locally. Ubuntu's only NOTE was for a system
compiler flag.

## Downstream dependencies

I checked all 3 direct CRAN reverse dependencies and 1 sampled indirect dependency,
comparing R CMD check results against the CRAN and development versions.

* I saw 0 new problems.
* I failed to check 0 packages.
