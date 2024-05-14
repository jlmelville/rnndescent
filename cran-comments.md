## New Patch Release

This is a new patch release to fix an error that resulted from the recent CRAN
release of the 'dqrng' package, which is a dependency of this package.

## Recent Resubmission

The previous release of 'rnndescent' was submitted to CRAN in April. I am aware
of the policy against frequent submission to CRAN. The previous submission was
intended to support the changes to 'dqrng'. Unfortunately, some new problems
were found only after the submission of 'dqrng' to CRAN. I regret any
inconvenience caused to the CRAN maintainers.

## Test environments

* ubuntu 22.04 (on rhub) devel clang-ASAN
* Fedora 38 (on rhub) devel gcc13
* Fedora 38 (on rhub) devel valgrind
* local ubuntu 23.04 R 4.3.1
* ubuntu 22.04 (on github actions), R 4.3.3, R 4.4.0, devel
* Windows Server 2012 (on appveyor) R 4.4.0 Patched
* Windows Server 2022 (on github actions), R 4.3.3, R 4.4.0
* local Windows 11 build, R 4.4.0
* win-builder (devel)
* mac OS X Monterey (on github actions) R 4.4.0
* local mac OS X Sonoma R 4.4.0

## CRAN Checks

There are no WARNINGs or ERRORs.

There is 1 NOTE remaining from the previous release on r-release-macos-arm64, 
r-release-macos-x86_64, r-oldrel-macos-arm64, r-oldrel-macos-x86_64

Check: installed package size
Result: NOTE 
    installed size is 13.5Mb
    sub-directories of 1Mb or more:
      libs  12.4Mb

This is due to the majority of the package being written in C++ and is not
affected by this patch release.

## Downstream dependencies

I checked 1 reverse dependency from CRAN, comparing R CMD check results across
CRAN and dev versions of this package.

 * I saw 0 new problems
 * I failed to check 0 packages
