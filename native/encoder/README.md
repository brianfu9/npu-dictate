# Qwen3-ASR encoder server (Android arm64)

The encoder server is one added file on top of k2-fsa/sherpa-onnx, tag v2.0.1:

1. Copy `sherpa-onnx-qwen3-encoder-server.cc` into `sherpa-onnx/csrc/`.
2. Apply the CMake change from the sherpa-onnx root: `git apply cmakelists.patch`.
3. Build with upstream's script, with the binaries enabled:

       SHERPA_ONNX_ENABLE_BINARY=ON ./build-android-arm64-v8a.sh

The binary is `install/bin/sherpa-onnx-qwen3-encoder-server`; the app ships it as
`libqwen3encsrv.so` (Android only executes files from nativeLibraryDir).
`libonnxruntime.so` and `libsherpa-onnx-jni.so` come from the same build's `install/lib`.
