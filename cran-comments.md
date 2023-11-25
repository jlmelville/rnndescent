## R CMD check results

0 errors | 0 warnings | 2 notes

There are 2 NOTEs:

* This is a new submission:
checking CRAN incoming feasibility ... NOTE
Maintainer: 'James Melville <jlmelville@gmail.com>'

New submission

* On Linux, there is a message about package size:
N  checking installed package size ...
     installed size is 21.7Mb
     sub-directories of 1Mb or more:
       libs  20.2Mb
This is due to the majority of the package being written in C++.

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
