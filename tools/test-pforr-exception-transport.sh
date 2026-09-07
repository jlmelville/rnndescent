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

required_commands=(clang++ env g++ mktemp uname)
for required_command in "${required_commands[@]}"; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    printf '%s: required command not found: %s\n' \
      "${0##*/}" "$required_command" >&2
    exit 1
  fi
done

if [[ $(uname -s) != Linux ]]; then
  printf '%s: mixed-runtime exception test requires Linux\n' "${0##*/}" >&2
  exit 1
fi

libstdcpp=$(g++ -print-file-name=libstdc++.so.6)
if [[ $libstdcpp == libstdc++.so.6 || ! -f $libstdcpp ]]; then
  printf '%s: unable to locate libstdc++.so.6 with g++\n' "${0##*/}" >&2
  exit 1
fi

scratch_dir=$(mktemp -d "${TMPDIR:-/tmp}/rnndescent-pforr-test.XXXXXX")
cleanup() {
  rm -rf -- "$scratch_dir"
}
trap cleanup EXIT HUP INT TERM

compiler_flags=(
  -std=c++11
  -pthread
  -Iinst/include
  -Wall
  -Wextra
  -Wpedantic
  -Wformat=2
  -Wnull-dereference
  -Werror
)
test_source=tools/test-pforr-exception-transport.cpp

gcc_executable="$scratch_dir/pforr-exceptions-gcc"
g++ "${compiler_flags[@]}" "$test_source" -o "$gcc_executable"
"$gcc_executable"

libcxx_executable="$scratch_dir/pforr-exceptions-libcxx"
clang++ -stdlib=libc++ "${compiler_flags[@]}" \
  "$test_source" -o "$libcxx_executable"
"$libcxx_executable"
env LD_PRELOAD="$libstdcpp" "$libcxx_executable"

printf 'pforr exception transport: PASS (GCC, libc++, mixed runtime)\n'
