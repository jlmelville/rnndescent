## Resubmission

This is a resubmission to fix ASAN/UBSAN issues that were not fixed in the
previous resubmission.

## Test environments

* ubuntu 22.04 (on github actions), R 4.2.3, R 4.3.2, devel
* local ubuntu 23.04 R 4.2.2
* Debian Linux, R-devel, GCC ASAN/UBSAN (via rhub)
* Debian Linux, R-release, GCC (via rhub)
* Ubuntu Linux 20.04.1 LTS, R-release, GCC (via rhub)
* Fedora Linux, R-devel, clang, gfortran (via rhub)
* Windows Server 2012 (on appveyor) R 4.3.2
* Windows Server 2022 (on github actions), R 4.2.3, R 4.3.2
* Windows Server 2022, R-devel, 64 bit (via rhub)
* local Windows 11 build, R 4.3.2
* win-builder (devel)
* mac OS X Monterey (on github actions) R 4.3.2
* local mac OS X Sonoma R 4.3.2

## CRAN Checks

There are no WARNINGs or ERRORs.

There is 1 NOTE remaining from the previous release. On Linux, there is a 
message about package size:

* checking installed package size ... NOTE
  installed size is 21.6Mb
  sub-directories of 1Mb or more:
    libs 20.2Mb

This is due to the majority of the package being written in C++ and is not
affected by this patch release.

### Additional Issues

There are 3 Additional Issues on the CRAN checks page:

clang-ASAN clang-UBSAN gcc-ASAN

This release is intended to fix these. The previous resubmission fixed the ASAN
errors so only the UBSAN error remains to be fixed by this resubmission.
