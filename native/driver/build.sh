#!/usr/bin/env bash
# Build asr_decode.c for the host (x86_64 Linux check) or Android arm64
# (phone, NPU). Works from any CWD.
#
#   build.sh host     -> asr_decode_host   (LiteRT from $HOST_LITERT_DIR: headers + libLiteRt.so)
#   build.sh android  -> asr_decode        (NDK r27d aarch64, android28; LiteRT headers from
#                                          $LITERT_SRC, libs from $LITERT_LIB)
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

NDK_CLANG="$HOME/toolchains/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"

case "${1:-}" in
  host)
    if [ -z "${HOST_LITERT_DIR:-}" ]; then
      echo "build.sh: HOST_LITERT_DIR is not set (a directory with the LiteRT headers and libLiteRt.so)" >&2
      exit 1
    fi
    cc -std=c11 -O2 -Wall \
      -I"$HOST_LITERT_DIR" \
      asr_decode.c -o asr_decode_host \
      -L"$HOST_LITERT_DIR" -lLiteRt \
      -Wl,-rpath,"$HOST_LITERT_DIR" \
      -lm
    ;;
  android)
    if [ ! -x "$NDK_CLANG" ]; then
      echo "build.sh: NDK clang not found at $NDK_CLANG" >&2
      exit 1
    fi
    "$NDK_CLANG" -std=c11 -O2 -Wall \
      -I"${LITERT_SRC:-../../third_party/LiteRT}" \
      asr_decode.c -o asr_decode \
      -L"${LITERT_LIB:-../../prebuilt}" -lLiteRt \
      -lm
    ;;
  *)
    echo "usage: $0 host|android" >&2
    exit 2
    ;;
esac
