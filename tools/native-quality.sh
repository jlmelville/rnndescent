#!/usr/bin/env bash

set -euo pipefail

usage() {
  printf 'Usage: %s\n' "${0##*/}"
}

case "${1-}" in
  "")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    printf '%s: unexpected argument: %s\n' "${0##*/}" "$1" >&2
    usage >&2
    exit 2
    ;;
esac

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/.." && pwd)
cd "$repo_root"

required_commands=(Rscript g++ clang++ mktemp)
for required_command in "${required_commands[@]}"; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    printf '%s: required command not found: %s\n' \
      "${0##*/}" "$required_command" >&2
    exit 1
  fi
done

r_include=$(Rscript --vanilla -e 'cat(R.home("include"))')
if [[ ! -d $r_include ]]; then
  printf '%s: R include directory not found: %s\n' \
    "${0##*/}" "$r_include" >&2
  exit 1
fi

linking_to_packages=()
while IFS= read -r linking_to_package; do
  linking_to_packages+=("$linking_to_package")
done < <(
  Rscript --vanilla -e '
    description <- read.dcf("DESCRIPTION")
    packages <- trimws(strsplit(description[1L, "LinkingTo"], ",")[[1L]])
    cat(packages, sep = "\n")
  '
)
if ((${#linking_to_packages[@]} == 0)); then
  printf '%s: DESCRIPTION does not declare any LinkingTo packages\n' \
    "${0##*/}" >&2
  exit 1
fi

dependency_include_flags=()
for linking_to_package in "${linking_to_packages[@]}"; do
  package_include=$(
    Rscript --vanilla -e '
      package <- commandArgs(trailingOnly = TRUE)[[1L]]
      cat(system.file("include", package = package))
    ' "$linking_to_package"
  )
  if [[ ! -d $package_include ]]; then
    printf '%s: include directory unavailable for LinkingTo package %s\n' \
      "${0##*/}" "$linking_to_package" >&2
    exit 1
  fi
  dependency_include_flags+=(-isystem "$package_include")
done

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/rnndescent-native-quality.XXXXXX")
cleanup() {
  rm -rf -- "$scratch_dir"
}
trap cleanup EXIT HUP INT TERM

common_flags=(
  -pthread
  -Iinst/include
  -isystem "$r_include"
  "${dependency_include_flags[@]}"
  -Wall
  -Wextra
  -Wpedantic
  -Wformat=2
  -Wnull-dereference
  -Werror
  -fsyntax-only
)
compilers=(g++ clang++)
cxx11_headers=(pforr.h)
cxx17_headers=(rnndescent/random.h tdoann/distance.h)

probe_headers() {
  local language_standard=$1
  shift
  local header_file
  local compiler
  local probe_source

  for header_file in "$@"; do
    probe_source="$scratch_dir/${header_file//\//-}.cpp"
    cat >"$probe_source" <<EOF
#include <$header_file>

int main() { return 0; }
EOF
    for compiler in "${compilers[@]}"; do
      if [[ $compiler == clang++ ]]; then
        "$compiler" -stdlib=libc++ "-std=$language_standard" \
          "${common_flags[@]}" "$probe_source"
      else
        "$compiler" "-std=$language_standard" "${common_flags[@]}" \
          "$probe_source"
      fi
    done
  done
}

probe_headers c++11 "${cxx11_headers[@]}"
probe_headers gnu++17 "${cxx17_headers[@]}"
tools/test-pforr-exception-transport.sh

printf 'native-quality: PASS (supported headers with GCC and Clang/libc++)\n'
