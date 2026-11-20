#!/usr/bin/env bash
# Builds app/out/qwen-asr.apk without Gradle: aapt2 + javac + d8 + zipalign + apksigner.
# Native executables are packaged as lib*.so so that Android extracts them to nativeLibraryDir
# (the only app location that may be executed).
set -euo pipefail
cd "$(dirname "$0")"
SDK=~/toolchains/android-sdk
BT=$SDK/build-tools/35.0.0
JAR=$SDK/platforms/android-35/android.jar
export JAVA_HOME=~/toolchains/jdk17 PATH=~/toolchains/jdk17/bin:$PATH
PREBUILT=${PREBUILT:-$(cd .. && pwd)/prebuilt}
OUT=out
rm -rf $OUT && mkdir -p $OUT/classes $OUT/apk/lib/arm64-v8a

# native payload (from prebuilt/, fetched by scripts/fetch-prebuilt.sh)
L=$OUT/apk/lib/arm64-v8a
for name in libqwen3encsrv.so libasrdecode.so libonnxruntime.so libsherpa-onnx-jni.so \
            libLiteRt.so libLiteRtDispatch_MediaTek.so; do
  if [ ! -f "$PREBUILT/$name" ]; then
    echo "missing $PREBUILT/$name: run scripts/fetch-prebuilt.sh" >&2
    exit 1
  fi
  cp "$PREBUILT/$name" "$L/$name"
done

# assets: the shared ones, plus the personal word lists and polish config
PERSONAL=${PERSONAL:-../personal}
if [ ! -d "$PERSONAL" ]; then
  echo "using personal.example (generic word lists, no polish server)" >&2
  PERSONAL=../personal.example
fi
mkdir -p $OUT/assets
cp assets/* $OUT/assets/
cp "$PERSONAL/terms.txt" "$PERSONAL/polish.conf" $OUT/assets/
# PYTHON: an interpreter with `tokenizers`; defaults to the repo's .venv when it exists
if [ -z "${PYTHON:-}" ]; then
  if [ -x ../.venv/bin/python ]; then PYTHON=../.venv/bin/python; else PYTHON=python3; fi
fi
$PYTHON ../tools/make_context_ids.py "$PREBUILT/tokenizer" "$PERSONAL/context.txt" $OUT/assets/context_ids.i32

$BT/aapt2 compile --dir res -o $OUT/res.zip
$BT/aapt2 link -o $OUT/base.apk --manifest AndroidManifest.xml -I $JAR -A $OUT/assets $OUT/res.zip
javac -source 11 -target 11 -Xlint:-options -cp $JAR -d $OUT/classes $(find src -name '*.java')
$BT/d8 --release --min-api 29 --lib $JAR --output $OUT $(find $OUT/classes -name '*.class')

cp $OUT/base.apk $OUT/unsigned.apk
(cd $OUT && zip -q -j unsigned.apk classes.dex)
(cd $OUT/apk && zip -q -0 -r ../unsigned.apk lib)   # stored: needed for page-aligned extraction
$BT/zipalign -f -P 16 4 $OUT/unsigned.apk $OUT/aligned.apk

KS=$HOME/.android/qwenasr-debug.keystore
if [ ! -f "$KS" ]; then
  mkdir -p ~/.android
  keytool -genkeypair -keystore "$KS" -storepass android -keypass android -alias key \
    -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=qwen-asr debug" >/dev/null
fi
$BT/apksigner sign --ks "$KS" --ks-pass pass:android --key-pass pass:android \
  --out $OUT/qwen-asr.apk $OUT/aligned.apk
ls -l $OUT/qwen-asr.apk
