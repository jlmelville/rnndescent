## New Patch Release

This is a new minor release to fix a CRAN warning, plus some new features.

## Test environments

* ubuntu 22.04 (on rhub) devel clang-ASAN
* Fedora 42 (on rhub) devel valgrind
* local ubuntu 26.04 R 4.5.2
* ubuntu 24.04 (on github actions), R 4.5.3, R 4.6.0, devel
* Windows Server 2012 (on appveyor) R 4.6.0 Patched
* Windows Server 2022 (on github actions), R 4.5.3, R 4.6.0
* local Windows 11 build, R 4.6.0
* win-builder (devel)
* mac OS X Sequoia (on github actions) R 4.6.0
* local mac OS X Tahoe R 4.6.0
* mac-builder (devel)

## CRAN Checks

There are no ERRORs or NOTEs.

There is 1 WARNING for r-devel-linux-x86_64-debian-gcc, r-release-linux-x86_64:

```
Version: 0.1.8
Check: whether package can be installed
Result: WARN 
  Found the following significant warnings:
    ../inst/include/tdoann/distancebase.h:143:11: warning: array subscript ‘tdoann::QueryDistanceCalculator<float, float, unsigned int>[0]’ is partly outside array bounds of ‘unsigned char [56]’ [-Warray-bounds=]
  See ‘/home/hornik/tmp/R.check/r-devel-gcc/Work/PKGS/rnndescent.Rcheck/00install.out’ for details.
  * used C++ compiler: ‘g++-16 (Debian 16-20260425-1) 16.0.1 20260425 (prerelease) [gcc-16 r16-8812-gd9c07462a22]’
```

This release is intended to fix this.

## Downstream dependencies

I checked 3 reverse dependencies from CRAN, comparing R CMD check results across
CRAN and dev versions of this package.

 * I saw 0 new problems
 * I failed to check 0 packages
