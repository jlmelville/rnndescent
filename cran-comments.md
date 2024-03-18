## New Patch Release

This is a new patch release to fix a bug.

## Test environments

* ubuntu 22.04 (on github actions), R 4.2.3, R 4.3.3, devel
* local ubuntu 23.04 R 4.3.1
* Debian Linux, R-devel, GCC ASAN/UBSAN (via rhub)
* Debian Linux, R-release, GCC (via rhub)
* Ubuntu Linux 20.04.1 LTS, R-release, GCC (via rhub)
* Fedora Linux, R-devel, clang, gfortran (via rhub)
* Windows Server 2012 (on appveyor) R 4.3.3
* Windows Server 2022 (on github actions), R 4.2.3, R 4.3.3
* Windows Server 2022, R-devel, 64 bit (via rhub)
* local Windows 11 build, R 4.3.3
* win-builder (devel)
* mac OS X Monterey (on github actions) R 4.3.3
* local mac OS X Sonoma R 4.3.3

## CRAN Checks

There are no WARNINGs or ERRORs.

There is 1 NOTE remaining from the previous release on r-release-macos-arm64, 
r-release-macos-x86_64, r-oldrel-macos-arm64:

Check: installed package size
Result: NOTE 
    installed size is 13.9Mb
    sub-directories of 1Mb or more:
      libs  12.4Mb

This is due to the majority of the package being written in C++ and is not
affected by this patch release.

## Downstream dependencies

There are no downstream dependencies.
