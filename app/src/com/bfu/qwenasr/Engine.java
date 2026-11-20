package com.bfu.qwenasr;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Holds the two persistent model servers:
 *  - encoder: sherpa-onnx Qwen3-ASR audio encoder (CPU). wav -> audio_features.f32 + prompt_ids.i32
 *  - decoder: asr_decode --server (Qwen3-1.7B on the MediaTek NPU). features + ids -> token ids
 * Both read one request per stdin line and answer with lines ending in DONE.
 * If the app process dies, their stdin closes (EOF) and they exit.
 */
final class Engine {
    static final class Result {
        String language = "";
        String text = "";
        double audioSec, encoderMs, prefillMs, decodeMsPerTok, totalMs;
        int tokens;
        boolean retried;
        boolean hitLimit;
        int contextTokens;
        List<Integer> textIds = new ArrayList<>();
        List<Integer> rawIds = new ArrayList<>();
    }

    static final class Server {
        final String name;
        Process proc;
        BufferedWriter in;
        BufferedReader out;

        Server(String name) { this.name = name; }

        String start(List<String> cmd, File nativeDir, File workDir, File log) throws IOException {
            return start(cmd, nativeDir, workDir, log, null);
        }

        String start(List<String> cmd, File nativeDir, File workDir, File log, Map<String, String> extraEnv)
                throws IOException {
            ProcessBuilder pb = new ProcessBuilder(cmd);
            Map<String, String> env = pb.environment();
            if (extraEnv != null) env.putAll(extraEnv);
            env.put("LD_LIBRARY_PATH", nativeDir.getAbsolutePath());
            env.put("DISPATCH_DIR", nativeDir.getAbsolutePath());
            env.put("TMPDIR", workDir.getAbsolutePath());
            pb.directory(workDir);
            pb.redirectError(log);
            proc = pb.start();
            in = new BufferedWriter(new OutputStreamWriter(proc.getOutputStream(), StandardCharsets.UTF_8));
            out = new BufferedReader(new InputStreamReader(proc.getInputStream(), StandardCharsets.UTF_8));
            String line;
            while ((line = out.readLine()) != null) {
                if (line.startsWith("READY")) return line;
            }
            throw new IOException(name + " exited during startup (see " + log.getName() + ")");
        }

        List<String> request(String line) throws IOException {
            in.write(line);
            in.write('\n');
            in.flush();
            List<String> lines = new ArrayList<>();
            String l;
            while ((l = out.readLine()) != null) {
                if (l.equals("DONE")) return lines;
                lines.add(l);
            }
            throw new IOException(name + " stopped");
        }

        boolean running() { return proc != null; }

        void stop() {
            if (proc == null) return;
            try { in.write("QUIT\n"); in.flush(); } catch (IOException ignored) { }
            // Wait for the exit, so its NPU/DMA-BUF memory is really free before another model loads.
            try {
                if (!proc.waitFor(3, java.util.concurrent.TimeUnit.SECONDS)) {
                    proc.destroy();
                    if (!proc.waitFor(2, java.util.concurrent.TimeUnit.SECONDS)) proc.destroyForcibly();
                }
            } catch (InterruptedException e) { proc.destroyForcibly(); Thread.currentThread().interrupt(); }
            proc = null;
        }
    }

    private final Server enc = new Server("encoder");
    private final Server dec = new Server("decoder");
    private Tokenizer tokenizer;
    final File modelDir, workDir;
    String encReady = "", decReady = "";
    private File nativeDir;

    private void startDecoder() throws IOException {
        List<String> d = new ArrayList<>();
        d.add(new File(nativeDir, "libasrdecode.so").getAbsolutePath());
        d.add("--model-dir");
        d.add(new File(modelDir, "npu").getAbsolutePath());
        d.add("--max-new");
        d.add("256");
        d.add("--accel");
        d.add("cpu");
        d.add("--server");
        decReady = dec.start(d, nativeDir, workDir, new File(workDir, "decoder.log"));
    }

    synchronized boolean decoderRunning() { return dec.running(); }

    /** Lock-free (for the UI status panel): whether the speech decoder / encoder processes exist. */
    boolean decoderLoadedNow() { return dec.running(); }

    boolean encoderLoadedNow() { return enc.running(); }

    Engine(File filesDir, File cacheDir) {
        modelDir = new File(filesDir, "models");
        workDir = cacheDir;
    }

    /** Returns null when all model files exist, else a description of what is missing. */
    String missingModels() {
        String[] need = {
            "asr/conv_frontend.onnx", "asr/encoder.int8.onnx", "asr/tokenizer/vocab.json",
            "npu/Section0_tf_lite_prefill_decode_decode.tflite",
            "npu/Section1_TFLiteModel_tf_lite_embedder.tflite",
            "npu/Section2_TFLiteModel_tf_lite_aux.tflite",
        };
        StringBuilder sb = new StringBuilder();
        for (String n : need) if (!new File(modelDir, n).isFile()) sb.append(n).append('\n');
        if (maxPrompt() == 0) sb.append("npu/Section0_tf_lite_prefill_decode_prefill_<N>.tflite\n");
        return sb.length() == 0 ? null : sb.toString();
    }

    /** Loads both models. Blocking; call from a worker thread. */
    void start(File nativeDir) throws Exception {
        File asr = new File(modelDir, "asr");
        // The ONNX decoder is never loaded in encoder-server mode; the path only has to exist.
        File placeholder = new File(asr, "decoder.placeholder.onnx");
        if (!placeholder.exists()) placeholder.createNewFile();

        final Exception[] err = new Exception[1];
        Thread t = new Thread(() -> {
            try {
                List<String> c = new ArrayList<>();
                c.add(new File(nativeDir, "libqwen3encsrv.so").getAbsolutePath());
                c.add("--provider=cpu");
                c.add("--num-threads=4");
                c.add("--qwen3-asr-conv-frontend=" + new File(asr, "conv_frontend.onnx"));
                c.add("--qwen3-asr-encoder=" + new File(asr, "encoder.int8.onnx"));
                c.add("--qwen3-asr-decoder=" + placeholder);
                c.add("--qwen3-asr-tokenizer=" + new File(asr, "tokenizer"));
                encReady = enc.start(c, nativeDir, workDir, new File(workDir, "encoder.log"));
            } catch (Exception e) { err[0] = e; }
        });
        t.start();

        this.nativeDir = nativeDir;
        synchronized (this) { startDecoder(); }

        tokenizer = new Tokenizer(new File(asr, "tokenizer/vocab.json"));
        t.join();
        if (err[0] != null) throw err[0];
    }

    static final int CACHE_LEN = 1024;   // KV cache length the NPU model was compiled with

    /** "language English<asr_text>": forces English output (Qwen3-ASR language forcing). */
    static final List<Integer> ENGLISH = java.util.Arrays.asList(11528, 6364, Tokenizer.ASR_TEXT);

    /** Prefix that forces the output language, used when a request has no prefix; empty = auto-detect. */
    volatile List<Integer> forcedLanguage = new ArrayList<>(ENGLISH);

    /** Word list put into every prompt's context (Qwen3-ASR context biasing); empty = off. */
    volatile List<Integer> wordList = new ArrayList<>();

    /**
     * Largest prompt the NPU takes: the compiled prefill length. A second prefill chunk gives garbage
     * on the NPU, so every prompt must fit one chunk.
     */
    int maxPrompt() {
        int best = 0;
        File[] fs = new File(modelDir, "npu").listFiles();
        if (fs != null) for (File f : fs) {
            java.util.regex.Matcher m = java.util.regex.Pattern.compile("prefill_decode_prefill_(\\d+)\\.tflite").matcher(f.getName());
            if (m.find()) best = Math.max(best, Integer.parseInt(m.group(1)));
        }
        return best;
    }

    /** wav (16 kHz mono s16) -> text, no prefix. Blocking; call from a worker thread. */
    Result transcribe(File wav) throws IOException {
        return transcribe(wav, null, 256);
    }

    /**
     * wav -> text. prefix: token ids (from an earlier result) appended after the prompt, so that the
     * model continues that text (Qwen3-ASR streaming); null for none. The raw output ids
     * (prefix + continuation, without EOS) are in Result.rawIds.
     */
    synchronized Result transcribe(File wav, List<Integer> prefix, int maxNew) throws IOException {
        return transcribe(wav, prefix, maxNew, null);
    }

    /**
     * context: token ids put into the (empty) system message of the prompt, as Qwen3-ASR's
     * "context" (for example the text of the previous window); null for none.
     */
    synchronized Result transcribe(File wav, List<Integer> prefix, int maxNew, List<Integer> context)
            throws IOException {
        return transcribe(wav, prefix, maxNew, context, false);
    }

    /** Output tokens kept free in the cache for a long (full-pass) prompt. */
    static final int LONG_OUTPUT_RESERVE = 200;

    /**
     * allowLong: the prompt may exceed one NPU prefill chunk; the driver then feeds the tokens after
     * the first chunk through the decode step (correct, but ~55 ms per extra token). Used by the
     * full pass only; streaming keeps every prompt inside one chunk.
     */
    synchronized Result transcribe(File wav, List<Integer> prefix, int maxNew, List<Integer> context,
                                   boolean allowLong) throws IOException {
        long t0 = System.nanoTime();
        if ((prefix == null || prefix.isEmpty()) && !forcedLanguage.isEmpty()) prefix = forcedLanguage;
        File req = new File(workDir, "req");
        req.mkdirs();
        Result r = new Result();
        for (String l : enc.request(wav.getAbsolutePath() + " " + req.getAbsolutePath())) {
            if (l.startsWith("ERROR")) throw new IOException("encoder: " + l);
            if (l.startsWith("FEATURES")) {
                r.encoderMs = num(l, "encoder_ms=");
                r.audioSec = num(l, "duration_s=");
            }
        }
        File idsFile = new File(req, "prompt_ids.i32");
        ByteBuffer in = ByteBuffer.wrap(Files.readAllBytes(idsFile.toPath())).order(ByteOrder.LITTLE_ENDIAN);
        List<Integer> prompt = new ArrayList<>();
        while (in.remaining() >= 4) prompt.add(in.getInt());
        // Prompt = <|im_start|>system\n [context] <|im_end|>\n<|im_start|>user\n<|audio_start|>pads...
        // Context = word list + caller context. If the prompt is too long for one prefill chunk, drop
        // the word list first, then the caller context.
        int limit = allowLong ? CACHE_LEN - LONG_OUTPUT_RESERVE : maxPrompt();
        List<Integer> words = wordList;
        int base = prompt.size() + (prefix == null ? 0 : prefix.size());
        int ctxLen = words.size() + (context == null ? 0 : context.size());
        if (base + ctxLen > limit) { words = new ArrayList<>(); ctxLen = context == null ? 0 : context.size(); }
        if (base + ctxLen > limit) { context = null; ctxLen = 0; }
        if (base > limit) throw new IOException("audio too long for one NPU prefill (" + base + " > " + limit + " tokens)");
        List<Integer> ctx = new ArrayList<>(words);
        if (context != null) ctx.addAll(context);
        if (!ctx.isEmpty()) {
            if (prompt.size() < 3 || prompt.get(0) != 151644 || prompt.get(1) != 8948 || prompt.get(2) != 198)
                throw new IOException("unexpected prompt layout, cannot add context");
            prompt.addAll(3, ctx);
        }
        r.contextTokens = ctx.size();
        if (prefix != null) prompt.addAll(prefix);
        int promptLen = prompt.size();
        if (!ctx.isEmpty() || (prefix != null && !prefix.isEmpty())) {
            ByteBuffer bb = ByteBuffer.allocate(promptLen * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int id : prompt) bb.putInt(id);
            try (FileOutputStream o = new FileOutputStream(idsFile)) { o.write(bb.array()); }
        }
        int room = CACHE_LEN - promptLen - 1;
        if (room < 4) throw new IOException("segment too long for the " + CACHE_LEN + "-token cache");
        maxNew = Math.min(maxNew, room);
        List<Integer> ids = new ArrayList<>();
        if (prefix != null) ids.addAll(prefix);
        if (!dec.running()) startDecoder();   // load it again if it is not running
        for (String l : dec.request(new File(req, "audio_features.f32") + " " + idsFile + " " + maxNew)) {
            if (l.startsWith("ERROR")) throw new IOException("decoder: " + l);
            if (l.startsWith("TOKENS")) {
                for (String s : l.substring(6).trim().split("\\s+")) if (!s.isEmpty()) ids.add(Integer.parseInt(s));
            } else if (l.startsWith("prefill_ms=")) {
                r.prefillMs = num(l, "prefill_ms=");
                r.decodeMsPerTok = num(l, "decode_ms_per_tok=");
            }
        }
        r.tokens = ids.size() - (prefix == null ? 0 : prefix.size());
        int end = ids.size();
        for (int i = 0; i < ids.size(); i++) {
            int id = ids.get(i);
            if (id == 151645 || id == 151643) { end = i; break; }
        }
        // A repetition loop never ends: the output runs into the token limit without an EOS. Only
        // then collapse repeats; a finished output keeps real repeats ("five five five five").
        r.hitLimit = end == ids.size();
        r.rawIds = new ArrayList<>(r.hitLimit ? Tokenizer.collapseRepeats(ids.subList(0, end)) : ids.subList(0, end));
        int mark = r.rawIds.indexOf(Tokenizer.ASR_TEXT);
        r.textIds = new ArrayList<>(r.rawIds.subList(mark + 1, r.rawIds.size()));
        tokenizer.parse(ids, r);
        r.totalMs = (System.nanoTime() - t0) / 1e6;
        // The quantized model sometimes writes "language X" and then no <asr_text> marker (it repeats
        // the language or stops). Run once more with "language X<asr_text>" forced as the prompt end.
        boolean hasMark = r.rawIds.contains(Tokenizer.ASR_TEXT);
        if (!hasMark && (prefix == null || prefix.isEmpty()) && r.rawIds.size() >= 2
                && r.rawIds.get(0) == Tokenizer.LANGUAGE) {
            List<Integer> forced = new ArrayList<>(r.rawIds.subList(0, 2));
            forced.add(Tokenizer.ASR_TEXT);
            Result r2 = transcribe(wav, forced, maxNew, context, allowLong);
            r2.encoderMs += r.encoderMs;
            r2.prefillMs += r.prefillMs;
            r2.totalMs += r.totalMs;
            r2.retried = true;
            return r2;
        }
        return r;
    }

    void stop() {
        enc.stop();
        dec.stop();
    }

    private static double num(String line, String key) {
        int i = line.indexOf(key);
        if (i < 0) return 0;
        int j = i + key.length(), k = j;
        while (k < line.length() && " \t".indexOf(line.charAt(k)) < 0) k++;
        try { return Double.parseDouble(line.substring(j, k)); } catch (NumberFormatException e) { return 0; }
    }
}
