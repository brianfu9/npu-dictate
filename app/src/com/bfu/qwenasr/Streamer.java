package com.bfu.qwenasr;

import android.util.Log;

import com.k2fsa.sherpa.onnx.SileroVadModelConfig;
import com.k2fsa.sherpa.onnx.Vad;
import com.k2fsa.sherpa.onnx.VadModelConfig;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Streaming transcription on top of the offline Qwen3-ASR servers.
 *
 * Audio is cut into segments with Silero VAD (a pause ends a segment; a segment is force-cut at
 * MAX_SEG_S because the NPU KV cache holds 512 tokens). While a segment grows, a "partial" pass
 * runs on all audio of the segment so far, about every PARTIAL_EVERY_S, following Qwen3-ASR's own
 * streaming recipe: after the first UNFIXED_CHUNKS passes, the previous output minus its last
 * ROLLBACK tokens is given as a prefix, and the model continues it. When the segment ends, one
 * full pass without prefix gives the final text (same accuracy as a one-shot run).
 *
 * Partials are "latest wins": when a pass takes longer than PARTIAL_EVERY_S, updates are skipped,
 * never queued. Finals are always queued and run before any partial.
 *
 * When a server backend is configured as primary, each segment is pinned to the backend its first
 * pass used (serverFor): a mid-sentence server recovery does not switch an in-progress sentence,
 * and a segment that started on the NPU finishes on it.
 */
final class Streamer {
    interface Listener {
        /** committed: final text of a finished segment (append). partial: live text of the open segment. */
        void onFinal(String text, String info);
        void onPartial(String text, String info);
        void onError(String msg);
        /** All input processed (after finish()). */
        void onDone();
        /** The full pass over the whole session starts / produced the session's new text. */
        default void onFullPassStart(int windows) { }
        default void onFullPass(String text, String info) { }
        /** The backend that produced the final ("local" or "server"); called just before onFinal. */
        default void onFinalSource(String source) { }
        /** Active backend that produced a partial or final ("server" or "local"); fallback=true when primary failed/skipped. */
        default void onBackend(String source, boolean fallback) { }
    }

    static final int RATE = 16000;
    static final float MAX_SEG_S = 20f;          // upper bound; lowered to fit the NPU prefill
    static final float PADS_PER_S = 13.1f;        // <|audio_pad|> tokens per second of audio
    static final float PARTIAL_EVERY_S = 2f;
    static final float FIRST_PARTIAL_S = 1.5f;
    static final float PREROLL_S = 0.6f;
    static final int UNFIXED_CHUNKS = 2;
    static final int ROLLBACK = 5;
    static final int PARTIAL_MAX_NEW = 64;
    static final int CTX_TAIL = 48;               // tokens of the previous window's text given as context

    private final Engine engine;
    private final Vad vad;
    private final File workDir;
    private final Listener listener;
    private final boolean live;
    private ServerAsr serverAsr;          // null = the on-device NPU only
    private boolean serverPrimary;        // true = the server first, the NPU as fallback
    private static int segCounter;
    private final float maxSegS;
    private boolean fullPass = true;
    private float windowS = 30f;
    private volatile boolean cancelled;
    private final List<int[]> done = new ArrayList<>();          // finished segments {start, end}
    private final List<String> doneText = new ArrayList<>();

    // Audio of this session (all samples since start()).
    private short[] audio = new short[RATE * 60];
    private int n;                       // samples written
    private int segStart = -1;           // start sample of the open segment, -1 = none
    private int segId;                   // increments for each new segment
    private final ArrayDeque<int[]> finals = new ArrayDeque<>();  // {start, end, segId}
    private boolean running;
    private Thread loop;
    private final float[] vadBuf = new float[512];
    private int vadFill;

    // Partial state of the open segment (only touched on the loop thread).
    private int partialSeg = -1, partialCount;
    private int lastPartialEnd;
    private List<Integer> partialIds = new ArrayList<>();

    // Backend pinning per segment (only touched on the loop thread): once a segment's first pass
    // picks a backend, its later passes and its final stay on it.
    private int stickySeg = -1;
    private boolean stickyServer;

    Streamer(Engine engine, File vadModel, File workDir, boolean live, Listener l) {
        this.engine = engine;
        this.workDir = workDir;
        this.listener = l;
        this.live = live;
        // prompt = ~15 fixed + pads + partial prefix (<= ~80); the word list is dropped when it does not fit
        // maxSegS stays NPU-prefill-bounded even in server mode, so a mid-segment NPU fallback can never exceed engine.maxPrompt().
        this.maxSegS = Math.min(MAX_SEG_S, (engine.maxPrompt() - 15 - 80) / PADS_PER_S - 0.5f);
        SileroVadModelConfig silero = SileroVadModelConfig.builder()
                .setModel(vadModel.getAbsolutePath())
                .setThreshold(0.5f)
                .setMinSilenceDuration(0.6f)
                .setMinSpeechDuration(0.25f)
                .setMaxSpeechDuration(60f)   // the force-cut is done here, at MAX_SEG_S
                .setWindowSize(512)
                .build();
        vad = new Vad(VadModelConfig.builder().setSileroVadModelConfig(silero)
                .setSampleRate(RATE).setNumThreads(1).setProvider("cpu").build());
    }

    /** After the session: one pass over the whole recording, in windows of at most windowS seconds. */
    Streamer setFullPass(boolean on, float windowS) {
        this.fullPass = on;
        this.windowS = windowS;
        return this;
    }

    /** The optional server ASR backend (asr.conf); primary when on. Neither set: the local NPU only. */
    Streamer useServerAsr(ServerAsr s, boolean primary) {
        this.serverAsr = s;
        this.serverPrimary = primary;
        return this;
    }

    /** Skip any remaining work of this session (a new session starts). */
    void cancel() {
        cancelled = true;
        finish();
    }

    synchronized void start() {
        n = 0;
        segStart = -1;
        finals.clear();
        vad.reset();
        vadFill = 0;
        running = true;
        loop = new Thread(this::run, "asr-stream");
        loop.start();
    }

    /** Feed 16 kHz mono samples (any block size). Called from the audio thread. */
    void feed(short[] pcm, int len) {
        synchronized (this) {
            if (n + len > audio.length) {
                short[] a = new short[Math.max(audio.length * 2, n + len)];
                System.arraycopy(audio, 0, a, 0, n);
                audio = a;
            }
            System.arraycopy(pcm, 0, audio, n, len);
            n += len;
        }
        for (int i = 0; i < len; i++) {
            vadBuf[vadFill++] = pcm[i] / 32768f;
            if (vadFill == vadBuf.length) {
                vad.acceptWaveform(vadBuf.clone());
                vadFill = 0;
                onVad();
            }
        }
    }

    private void onVad() {
        boolean ended = false;
        while (!vad.empty()) { vad.pop(); ended = true; }
        boolean speaking = vad.isSpeechDetected();
        synchronized (this) {
            if (segStart >= 0 && ended) {
                closeSegment(n);
            }
            if (segStart < 0 && speaking) {
                int lastEnd = finals.isEmpty() ? 0 : finals.peekLast()[1];
                segStart = Math.max(Math.max(0, lastEnd), n - (int) ((PREROLL_S + 0.25f) * RATE));
                segId++;
            }
            if (segStart >= 0 && n - segStart >= maxSegS * RATE) {
                // Cut at the quietest 100 ms of the last second, not mid-word.
                int cut = quietestCut(Math.max(segStart + RATE, n - RATE), n);
                closeSegment(cut);
                segStart = cut;   // speech continues: open the next segment right away
                segId++;
            }
            notifyAll();
        }
    }

    /** Start sample of the 100 ms window with the lowest energy in [from, to) (caller holds the lock). */
    private int quietestCut(int from, int to) {
        int win = RATE / 10, best = to;
        double bestE = Double.MAX_VALUE;
        for (int p = from; p + win <= to; p += win / 2) {
            double e = 0;
            for (int i = p; i < p + win; i++) e += (double) audio[i] * audio[i];
            if (e < bestE) { bestE = e; best = p + win / 2; }
        }
        return best;
    }

    private void closeSegment(int end) {
        finals.add(new int[]{segStart, end, segId});
        segStart = -1;
    }

    /** Stop input; the open segment is finalized, queued finals still run. */
    void finish() {
        synchronized (this) {
            if (segStart >= 0) closeSegment(n);
            running = false;
            notifyAll();
        }
        // finish() can run twice (cancel() and the mic thread); write the WAV only once.
        if (saveTo != null && n > 0 && !wavSaved) {
            wavSaved = true;
            try {
                saveTo.getParentFile().mkdirs();
                writeWavTo(0, n, saveTo);
                Log.i("QwenASR", "saved session " + saveTo.getName() + String.format(Locale.US, " %.1f s", n / (float) RATE));
                if (keepSessions > 0) {
                    File[] old = saveTo.getParentFile().listFiles((d, f) -> f.endsWith(".wav"));
                    if (old != null && old.length > keepSessions) {
                        java.util.Arrays.sort(old, (a, b) -> a.getName().compareTo(b.getName()));
                        for (int i = 0; i < old.length - keepSessions; i++) old[i].delete();
                    }
                }
            } catch (IOException e) {
                Log.e("QwenASR", "saving the session failed", e);
            }
        }
    }

    static final int KEEP_SESSIONS = 50;
    private File saveTo;
    private int keepSessions = KEEP_SESSIONS;   // <= 0: never delete older files
    private boolean wavSaved;

    /** Save the whole session audio to this file when it finishes (microphone sessions); null = no. */
    Streamer saveSessionTo(File f) {
        return saveSessionTo(f, KEEP_SESSIONS);
    }

    /** Same, but prunes older .wav files in the same directory down to `keep`; keep <= 0 never deletes. */
    Streamer saveSessionTo(File f, int keep) {
        saveTo = f;
        keepSessions = keep;
        return this;
    }

    /**
     * The backend for this segment's passes: pinned when the segment's first pass runs, decided
     * from the live server state otherwise. A mid-sentence server recovery (cooldown expiry or
     * toggle) only affects segments that have not started yet. Loop thread only.
     */
    private boolean serverFor(int seg) {
        if (stickySeg == seg) return stickyServer;
        stickySeg = seg;
        stickyServer = serverPrimary && serverAsr != null && !serverAsr.isInCooldown();
        return stickyServer;
    }

    private void run() {
        while (true) {
            int[] fin = null;
            int pStart = -1, pEnd = -1, pSeg = -1;
            synchronized (this) {
                while (true) {
                    if (!finals.isEmpty()) { fin = finals.poll(); break; }
                    if (!running) break;
                    if (live && segStart >= 0) {
                        int len = n - segStart;
                        boolean newSeg = partialSeg != segId;
                        int since = newSeg ? len : n - lastPartialEnd;
                        if (len >= FIRST_PARTIAL_S * RATE && (newSeg ? true : since >= PARTIAL_EVERY_S * RATE)) {
                            pStart = segStart; pEnd = n; pSeg = segId;
                            break;
                        }
                    }
                    try { wait(100); } catch (InterruptedException e) { return; }
                }
            }
            if (fin == null && pStart < 0) {   // input finished and all finals done
                try { vad.release(); } catch (Throwable ignored) { }
                if (fullPass && !cancelled && done.size() >= 2) runFullPass();
                listener.onDone();
                return;
            }
            if (cancelled) continue;
            try {
                if (fin != null) {
                    if (fin[1] - fin[0] < RATE / 4) continue;
                    // keep the last 20 segment recordings for debugging (cache/segs/)
                    new File(workDir, "segs").mkdirs();
                    writeWav(fin[0], fin[1], String.format(Locale.US, "segs/seg_%02d.wav", segCounter++ % 20));
                    Engine.Result r;
                    boolean server = false, fallback = false;
                    if (serverPrimary && serverAsr != null) {
                        if (serverFor(fin[2])) {   // pinned at this segment's first pass
                            try {
                                r = serverTranscribe(fin[0], fin[1]);
                                server = true;
                            } catch (IOException e) {
                                Log.w("QwenASR", "server ASR failed, using the local model: " + e.getMessage());
                                r = engine.transcribe(writeWav(fin[0], fin[1], "final.wav"), null, maxNewFor(fin[1] - fin[0]));
                                fallback = true;
                            }
                        } else {
                            if (serverAsr.isInCooldown()) Log.i("QwenASR", "server ASR in cooldown, using the local model");
                            else Log.i("QwenASR", "segment pinned to the local model, staying on it");
                            r = engine.transcribe(writeWav(fin[0], fin[1], "final.wav"), null, maxNewFor(fin[1] - fin[0]));
                            fallback = true;
                        }
                    } else {
                        try {
                            r = engine.transcribe(writeWav(fin[0], fin[1], "final.wav"), null, maxNewFor(fin[1] - fin[0]));
                        } catch (IOException e) {
                            if (serverAsr == null) throw e;   // no server: the existing error path
                            Log.w("QwenASR", "local ASR failed, using the server: " + e.getMessage());
                            r = serverTranscribe(fin[0], fin[1]);   // last resort: bypasses the cooldown
                            server = true;
                            fallback = true;
                        }
                    }
                    if (partialSeg == fin[2]) { partialIds = new ArrayList<>(); }
                    if (cancelled) continue;   // a new session started while this ran
                    String src = server ? "server" : "local";
                    String info = server
                            ? String.format(Locale.US, "final %.1f s audio: %.2f s (server %s%s)", r.audioSec, r.totalMs / 1000, serverAsr.model(), fallback ? ", local failed" : "")
                            : String.format(Locale.US, "final %s%.1f s audio: %.2f s (enc %.2f, prefill %.2f, %d tok%s)",
                                    fallback ? "NPU fallback; " : "", r.audioSec, r.totalMs / 1000, r.encoderMs / 1000, r.prefillMs / 1000, r.tokens, r.retried ? ", retried" : "");
                    Log.i("QwenASR", info + " text=" + r.text);
                    done.add(new int[]{fin[0], fin[1]});
                    doneText.add(r.text);
                    listener.onBackend(src, fallback);
                    listener.onFinalSource(src);
                    listener.onFinal(r.text, info);
                    // The partial of the next (open) segment, if any, is redrawn by its next pass.
                } else {
                    if (partialSeg != pSeg) { partialSeg = pSeg; partialCount = 0; partialIds = new ArrayList<>(); }
                    lastPartialEnd = pEnd;   // before either backend: a fast failure cannot spin the loop
                    boolean useServer = serverFor(pSeg);   // pinned at this segment's first pass
                    boolean fallback = serverPrimary && serverAsr != null && !useServer;
                    if (useServer) {
                        // Server partials: re-upload the whole segment on the same cadence; no prefix/rollback.
                        try {
                            long t0 = System.nanoTime();
                            String text = serverAsr.transcribe(wavBytes(pStart, pEnd), ServerAsr.PARTIAL_READ_TIMEOUT_MS);
                            partialCount++;
                            synchronized (this) {
                                if (cancelled || pSeg != segId || segStart < 0) continue;   // segment already closed: drop
                            }
                            String info = String.format(Locale.US, "live %.1f s audio: %.2f s (server %s)",
                                    (pEnd - pStart) / (float) RATE, (System.nanoTime() - t0) / 1e9, serverAsr.model());
                            Log.i("QwenASR", info + " text=" + text);
                            listener.onBackend("server", false);
                            listener.onPartial(text, info);
                            continue;
                        } catch (IOException e) {
                            // The server just failed: fall through to the local NPU for this partial.
                            Log.w("QwenASR", "server partial failed: " + e.getMessage());
                            fallback = true;
                            synchronized (this) {
                                if (cancelled || pSeg != segId || segStart < 0) continue;
                                pEnd = n;            // refresh to latest audio (guaranteed < maxSegS while segment is open)
                                lastPartialEnd = pEnd;
                            }
                        }
                    }
                    List<Integer> prefix = null;
                    if (partialCount >= UNFIXED_CHUNKS && !partialIds.isEmpty()) {
                        prefix = new ArrayList<>(partialIds.subList(0, Math.max(0, partialIds.size() - ROLLBACK)));
                    }
                    int maxNew = prefix == null ? maxNewFor(pEnd - pStart) : PARTIAL_MAX_NEW;
                    Engine.Result r = engine.transcribe(writeWav(pStart, pEnd, "partial.wav"), prefix, maxNew);
                    partialCount++;
                    synchronized (this) {
                        if (cancelled || pSeg != segId || segStart < 0) continue;   // segment already closed: drop
                    }
                    partialIds = r.rawIds;
                    String info = String.format(Locale.US, "live %s%.1f s audio: %.2f s (enc %.2f, prefill %.2f, %d new tok%s)",
                            fallback ? "NPU fallback; " : "", r.audioSec, r.totalMs / 1000, r.encoderMs / 1000, r.prefillMs / 1000, r.tokens,
                            prefix == null ? "" : ", prefix " + prefix.size());
                    Log.i("QwenASR", info + " text=" + r.text);
                    listener.onBackend("local", fallback);
                    listener.onPartial(r.text, info);
                }
            } catch (Exception e) {
                Log.e("QwenASR", "stream pass failed", e);
                listener.onError(e.getMessage());
            }
        }
    }

    private void runFullPass() {
        // Group the finished segments into windows (cut only at segment boundaries, i.e. at pauses).
        // A window must fit one NPU prefill with the word list and the previous window's tail as context.
        // The full pass may exceed one prefill chunk (the driver feeds the rest through decode steps).
        float budgetS = (Engine.CACHE_LEN - Engine.LONG_OUTPUT_RESERVE - 15 - engine.wordList.size() - CTX_TAIL)
                / PADS_PER_S - 0.5f;
        float windowS = Math.max(maxSegS, Math.min(this.windowS, budgetS));
        List<int[]> wins = new ArrayList<>();
        List<String> winSegText = new ArrayList<>();
        StringBuilder segText = new StringBuilder(doneText.get(0));
        int ws = done.get(0)[0], we = done.get(0)[1];
        for (int i = 1; i < done.size(); i++) {
            int[] d = done.get(i);
            if (d[1] - ws <= windowS * RATE) { we = d[1]; segText.append(join(segText, doneText.get(i))); }
            else {
                wins.add(new int[]{ws, we}); winSegText.add(segText.toString());
                ws = d[0]; we = d[1]; segText = new StringBuilder(doneText.get(i));
            }
        }
        wins.add(new int[]{ws, we});
        winSegText.add(segText.toString());
        listener.onFullPassStart(wins.size());
        long t0 = System.nanoTime();
        StringBuilder text = new StringBuilder();
        List<Integer> context = null;
        int kept = 0;
        double audio = 0;
        int tokens = 0;
        try {
            for (int wi = 0; wi < wins.size(); wi++) {
                int[] w = wins.get(wi);
                if (cancelled) return;
                // ~3 tokens/s of speech is typical; 5/s + 16 leaves room and stops a repetition loop early.
                int maxNew = 16 + (int) (5 * (w[1] - w[0]) / (float) RATE);
                Engine.Result r = engine.transcribe(writeWav(w[0], w[1], "full.wav"), null, maxNew, context, true);
                // Never lose or garble text: keep the sentence text when the window result is much shorter
                // (empty output) or much longer (the int4 model can fall into a repetition loop on long input).
                String seg = winSegText.get(wi);
                if (r.text.length() < 0.6 * seg.length() || r.text.length() > 1.5 * seg.length() + 10) {
                    Log.w("QwenASR", "full window kept sentence text: window gave '" + r.text + "'");
                    r.text = seg;
                    r.textIds = new ArrayList<>();
                    kept++;
                }
                Log.i("QwenASR", String.format(Locale.US, "full window %.1f s: %.2f s (enc %.2f, prefill %.2f, %d tok%s, ctx %d) text=%s",
                        r.audioSec, r.totalMs / 1000, r.encoderMs / 1000, r.prefillMs / 1000, r.tokens,
                        r.retried ? ", retried" : "", context == null ? 0 : context.size(), r.text));
                text.append(join(text, r.text));
                audio += r.audioSec;
                tokens += r.tokens;
                List<Integer> t = r.textIds;
                context = t.isEmpty() ? null : new ArrayList<>(t.subList(Math.max(0, t.size() - CTX_TAIL), t.size()));
            }
        } catch (Exception e) {
            Log.e("QwenASR", "full pass failed", e);
            listener.onError("full pass failed, keeping the sentence text: " + e.getMessage());
            return;
        }
        if (cancelled) return;
        String info = String.format(Locale.US, "full pass: %.1f s audio in %d window(s): %.1f s, %d tok%s",
                audio, wins.size(), (System.nanoTime() - t0) / 1e9, tokens,
                kept > 0 ? ", " + kept + " window(s) kept the sentence text" : "");
        Log.i("QwenASR", info + " text=" + text);
        listener.onFullPass(text.toString(), info);
    }

    /** Separator between two pieces of text: none next to CJK characters, else a space. */
    static String join(CharSequence a, String b) {
        if (a.length() == 0 || b.isEmpty()) return b;
        char x = a.charAt(a.length() - 1), y = b.charAt(0);
        return (isCjk(x) || isCjk(y)) ? b : " " + b;
    }

    private static boolean isCjk(char c) {
        Character.UnicodeScript u = Character.UnicodeScript.of(c);
        return u == Character.UnicodeScript.HAN || u == Character.UnicodeScript.HIRAGANA
                || u == Character.UnicodeScript.KATAKANA || u == Character.UnicodeScript.HANGUL
                || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFF00 && c <= 0xFFEF);
    }

    /**
     * Output token limit for a segment: speech is ~3 tokens/s, so 16 + 6/s never cuts real text but stops
     * a repetition loop of the int4 model after ~1-2 s instead of running to 256 tokens (~15 s).
     */
    static int maxNewFor(int samples) {
        return Math.min(256, 16 + (int) (6f * samples / RATE));
    }

    private File writeWav(int from, int to, String name) throws IOException {
        return writeWavTo(from, to, new File(workDir, name));
    }

    private File writeWavTo(int from, int to, File f) throws IOException {
        try (FileOutputStream o = new FileOutputStream(f)) {
            o.write(wavBytes(from, to));
        }
        return f;
    }

    /** The segment's audio as an in-memory 16 kHz mono WAV (for the server backend). */
    private byte[] wavBytes(int from, int to) {
        short[] s;
        synchronized (this) {
            s = new short[to - from];
            System.arraycopy(audio, from, s, 0, to - from);
        }
        byte[] pcm = new byte[s.length * 2];
        for (int i = 0; i < s.length; i++) { pcm[2 * i] = (byte) s[i]; pcm[2 * i + 1] = (byte) (s[i] >> 8); }
        return Wav.toBytes(pcm, RATE);
    }

    /** The final segment on the server backend, timed like a local pass. */
    private Engine.Result serverTranscribe(int from, int to) throws IOException {
        long t0 = System.nanoTime();
        String text = serverAsr.transcribe(wavBytes(from, to));
        Engine.Result r = new Engine.Result();
        r.text = text;
        r.audioSec = (to - from) / (float) RATE;
        r.totalMs = (System.nanoTime() - t0) / 1e6;
        return r;
    }
}
