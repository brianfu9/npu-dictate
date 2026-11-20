// sherpa-onnx/csrc/sherpa-onnx-qwen3-encoder-server.cc
//
// Persistent Qwen3-ASR audio-encoder server for an external (NPU) decoder.
// Loads the conv frontend + audio encoder once (the ONNX decoder is skipped via
// QWEN3_DUMP_ONLY), prints READY, then serves one request per stdin line:
//
//   <wav_path> <out_dir>
//
// writing <out_dir>/audio_features.f32 and <out_dir>/prompt_ids.i32 (see the
// QWEN3_DUMP_DIR hook in offline-recognizer-qwen3-asr-impl.cc) and answering
//
//   FEATURES encoder_ms=<x> duration_s=<y>   then   DONE
//
// or ERROR <message> then DONE. A line QUIT (or EOF) stops the server.

#include <stdio.h>
#include <stdlib.h>

#include <chrono>  // NOLINT
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Persistent Qwen3-ASR encoder server (features for an external decoder).

  ./bin/sherpa-onnx-qwen3-encoder-server \
    --qwen3-asr-conv-frontend=conv_frontend.onnx \
    --qwen3-asr-encoder=encoder.int8.onnx \
    --qwen3-asr-decoder=decoder.int8.onnx (path is only validated, not loaded) \
    --qwen3-asr-tokenizer=tokenizer_dir

Then write "<wav_path> <out_dir>" lines to stdin.
)usage";

  // Must be set before the recognizer is created: it makes the model skip the
  // ONNX decoder and makes the recognizer stop right after dumping features.
  setenv("QWEN3_DUMP_ONLY", "1", 1);

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineRecognizerConfig config;
  config.Register(&po);
  po.Read(argc, argv);
  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  const auto t0 = std::chrono::steady_clock::now();
  sherpa_onnx::OfflineRecognizer recognizer(config);
  const auto t1 = std::chrono::steady_clock::now();
  printf("READY setup_ms=%.1f\n",
         std::chrono::duration<double, std::milli>(t1 - t0).count());
  fflush(stdout);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "QUIT") break;
    std::istringstream is(line);
    std::string wav, out_dir;
    if (!(is >> wav >> out_dir)) {
      printf("ERROR expected: <wav_path> <out_dir>\nDONE\n");
      fflush(stdout);
      continue;
    }
    int32_t sampling_rate = -1;
    bool is_ok = false;
    std::vector<float> samples =
        sherpa_onnx::ReadWave(wav, &sampling_rate, &is_ok);
    if (!is_ok) {
      printf("ERROR cannot read wave '%s'\nDONE\n", wav.c_str());
      fflush(stdout);
      continue;
    }
    setenv("QWEN3_DUMP_DIR", out_dir.c_str(), 1);
    const auto b = std::chrono::steady_clock::now();
    auto s = recognizer.CreateStream();
    s->AcceptWaveform(sampling_rate, samples.data(), samples.size());
    recognizer.DecodeStream(s.get());
    const auto e = std::chrono::steady_clock::now();
    printf("FEATURES encoder_ms=%.1f duration_s=%.3f\nDONE\n",
           std::chrono::duration<double, std::milli>(e - b).count(),
           samples.size() / static_cast<float>(sampling_rate));
    fflush(stdout);
  }
  return 0;
}
