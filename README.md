# npu-dictate

An Android voice keyboard built around Qwen3-ASR-1.7B speech recognition. The audio encoder
runs on the CPU via sherpa-onnx; the text decoder runs on a MediaTek Dimensity 9500 NPU via
LiteRT. While you speak, the recognized words appear as composing text in the focused field.
Each finished segment is optionally cleaned up by a language model over any OpenAI-compatible
chat endpoint; every line of the model output is checked against the raw text (missing lines
and large length changes are rejected), so the raw transcript is kept whenever the model
output looks wrong.

## Requirements

- A phone with a MediaTek Dimensity 9500 (MT6993) NPU.
- Android SDK 35 with build-tools 35.0.0 and platform android-35.
- JDK 17.
- NDK r27d (only for building the native driver).
- Python with the `tokenizers` package (the build tokenizes the context word list; set
  `PYTHON` in `app/build.sh` if your interpreter is not on the path as `python3`).

## Repository layout

    app/                  the Android app (Java, no Gradle): sources, resources, assets
      build.sh            builds app/out/qwen-asr.apk with aapt2 + javac + d8
      vocab/mine_terms.py mines domain terms from your own typed messages
    native/driver/        asr_decode.c: the NPU decoder server (LiteRT), host and Android builds
    native/encoder/       the sherpa-onnx encoder-server addition (one .cc file + a CMake patch)
    prebuilt.sha256       checksums of the files in prebuilt/
    scripts/              fetch-prebuilt.sh, install-models.sh
    tools/                make_context_ids.py (word list -> token ids)
    personal.example/     example personal config (generic word lists, no polish server)

## Build

1. Get the prebuilt native libraries and tokenizer into `prebuilt/`:

       scripts/fetch-prebuilt.sh <dir with the .so files and tokenizer/>

   The script copies the six `.so` files and `tokenizer/` and checks them against
   `prebuilt.sha256`.
2. Create your personal config as a directory next to `app/`, e.g. by copying
   `personal.example/` to `personal/`:

       terms.txt    one line of comma-separated terms for the clean-up step
       context.txt  one line of comma-separated terms for the recognizer
       polish.conf  url= and model= of the polish server (any OpenAI-compatible chat
                    endpoint; leave url empty to skip the clean-up)

   `app/build.sh` uses `personal/` when it exists, otherwise `personal.example/`; the
   directory can be overridden with the `PERSONAL` environment variable.
3. Build:

       app/build.sh

   The Android SDK and JDK are expected under `~/toolchains`, as in the script. The output is
   `app/out/qwen-asr.apk`.
4. Install:

       adb install app/out/qwen-asr.apk

## Installing the models

The NPU decoder model files (the tflite sections) are produced by a separate conversion
pipeline that is not in this repo. Once you have them, install both model sets with:

    scripts/install-models.sh <models dir>

where `<models dir>` contains `asr/` (`conv_frontend.onnx`, `encoder.int8.onnx`,
`decoder.placeholder.onnx`, `tokenizer/`) and `npu/` (the tflite sections, including the
prefill section).

## Enabling the keyboard

Settings > System > Languages & input > On-screen keyboard > Qwen voice, or:

    adb shell ime enable com.bfu.qwenasr/.VoiceIme
    adb shell ime set com.bfu.qwenasr/.VoiceIme

## Testing with a wav file

The test activity plays a wav through the streaming pipeline at real-time speed (see the
comment in `app/src/com/bfu/qwenasr/MainActivity.java`):

    am start -n com.bfu.qwenasr/.MainActivity --es wav <file>

Results go to the screen and to logcat (tag QwenASR).

## Known limits

- The NPU prefill window is about 256 tokens, so audio segments are cut at roughly
  12-20 s.
- Live text is not word-by-word streaming: the open segment is decoded again about every 2 s
  while you speak, and a pause (0.6 s) ends the segment and gives its final text.
- The app keeps the models loaded while its process lives; they load again on demand after
  the process dies.

## License and credits

Apache-2.0. Built on:

- sherpa-onnx (Apache-2.0)
- Silero VAD (MIT)
- Qwen3-ASR (Apache-2.0)
- LiteRT (Apache-2.0)
