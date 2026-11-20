#!/usr/bin/env bash
# Copies the prebuilt native libraries and tokenizer into prebuilt/ (next to this script's
# parent directory) and checks them against prebuilt.sha256. Works from any CWD.
#
#   scripts/fetch-prebuilt.sh <dir>
# where <dir> holds libqwen3encsrv.so, libasrdecode.so, libonnxruntime.so,
# libsherpa-onnx-jni.so, libLiteRt.so, libLiteRtDispatch_MediaTek.so and tokenizer/.
set -euo pipefail

ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
if [ $# -ne 1 ]; then
  echo "usage: $0 <dir with the .so files and tokenizer/>" >&2
  exit 2
fi
SRC=$1

mkdir -p "$ROOT/prebuilt/tokenizer"
for name in libqwen3encsrv.so libasrdecode.so libonnxruntime.so libsherpa-onnx-jni.so \
            libLiteRt.so libLiteRtDispatch_MediaTek.so; do
  cp "$SRC/$name" "$ROOT/prebuilt/$name"
done
cp "$SRC/tokenizer/vocab.json" "$SRC/tokenizer/merges.txt" "$ROOT/prebuilt/tokenizer/"
(cd "$ROOT" && sha256sum -c prebuilt.sha256)
