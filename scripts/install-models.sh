#!/usr/bin/env bash
# Installs the ASR model files onto the phone adb picks (set ANDROID_SERIAL for a specific one).
# Checks the files exist under <models dir>, pushes them with adb and copies them into the
# app's private directory. Works from any CWD.
#
#   scripts/install-models.sh <models dir>
# where <models dir> holds asr/ (conv_frontend.onnx, encoder.int8.onnx, tokenizer/) and npu/
# (the tflite sections, including the prefill section).
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <models dir with asr/ and npu/>" >&2
  exit 2
fi
M=$1

need="asr/conv_frontend.onnx
      asr/encoder.int8.onnx
      asr/decoder.placeholder.onnx
      asr/tokenizer/vocab.json
      npu/Section0_tf_lite_prefill_decode_decode.tflite
      npu/Section1_TFLiteModel_tf_lite_embedder.tflite
      npu/Section2_TFLiteModel_tf_lite_aux.tflite"
for f in $need; do
  if [ ! -f "$M/$f" ]; then
    echo "missing $M/$f" >&2
    exit 1
  fi
done
if ! ls "$M"/npu/Section0_tf_lite_prefill_decode_prefill_*.tflite >/dev/null 2>&1; then
  echo "missing $M/npu/Section0_tf_lite_prefill_decode_prefill_<N>.tflite" >&2
  exit 1
fi

adb push "$M/asr" /data/local/tmp/npu-dictate/asr
adb push "$M/npu" /data/local/tmp/npu-dictate/npu
adb shell run-as com.bfu.qwenasr sh -c 'mkdir -p files/models && cp -r /data/local/tmp/npu-dictate/asr /data/local/tmp/npu-dictate/npu files/models/'
adb shell rm -rf /data/local/tmp/npu-dictate
