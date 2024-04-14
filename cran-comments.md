## New Patch Release

This is a new patch release to support an upcoming release of 'dqrng' which is
a dependency of this package. This is required to prevent 'rnndescent' breaking
when the new release of 'dqrng' is submitted to CRAN.

## Recent Resubmission

The previous release of 'rnndescent' was submitted to CRAN in March. I am aware
of the policy against frequent submission to CRAN. This current submission is
purely to support the release of 'dqrng'.

## Test environments

* Fedora Linux 38, R-devel, GCC, gfortran, valgrind (via rhub)
* local ubuntu 23.04 R 4.3.1
* ubuntu 22.04 (on github actions), R 4.2.3, R 4.3.3, devel
* Windows Server 2012 (on appveyor) R 4.3.3
* Windows Server 2022 (on github actions), R 4.2.3, R 4.3.3
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
    installed size is 13.5Mb
    sub-directories of 1Mb or more:
      libs  12.4Mb

This is due to the majority of the package being written in C++ and is not
affected by this patch release.

## Downstream dependencies

There are no downstream dependencies.
