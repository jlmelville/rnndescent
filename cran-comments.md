## New Patch Release

This is a new patch release to fix a CRAN M1 Mac check error.

## Test environments

* ubuntu 22.04 (on rhub) devel clang-ASAN
* Fedora 38 (on rhub) devel gcc15
* Fedora 38 (on rhub) devel valgrind
* local ubuntu 25.04 R 4.4.3
* ubuntu 24.04 (on github actions), R 4.4.3, R 4.5.1 devel
* Windows Server 2012 (on appveyor) R 4.5.1 Patched
* Windows Server 2022 (on github actions), R 4.4.3, R 4.5.1
* local Windows 11 build, R 4.5.1
* win-builder (devel)
* mac OS X Sonoma (on github actions) R 4.5.1
* local mac OS X Sequoia R 4.5.1
* mac-builder (devel)

## CRAN Checks

There are no WARNINGs or ERRORs.

There is 1 NOTE for r-oldrel-macos-arm64, r-oldrel-macos-x86_64:

Check: installed package size
Result: NOTE 
    installed size is 13.5Mb
    sub-directories of 1Mb or more:
      libs  12.4Mb

This is due to the majority of the package being written in C++ and is not
affected by this patch release.

There is an "M1mac" problem flagged as part of "Additional Issues". This release is intended to
fix this.

## Downstream dependencies

I checked 3 reverse dependencies from CRAN, comparing R CMD check results across
CRAN and dev versions of this package.

 * I saw 0 new problems
 * I failed to check 0 packages
