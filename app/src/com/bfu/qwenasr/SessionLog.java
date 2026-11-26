package com.bfu.qwenasr;

import android.content.Context;
import android.os.SystemClock;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * One voice-keyboard session kept for training data: the whole audio as a WAV plus a JSON record
 * of the recognized segments, every polish request and response (including the model's raw output),
 * and the committed text. segment/polish/commit are called on the main thread; save() runs on the
 * net executor and never throws.
 */
final class SessionLog {
    private final File dir;
    private final String id;
    private final String fieldPackage;
    private final String model;
    private final Context ctx;          // nullable, for the app version in the JSON
    private final long startWallMs;
    private final long t0;              // SystemClock.elapsedRealtime() at session start
    private long commitTMs;             // elapsed since t0 when the text was committed, 0 = not yet
    private long endWallMs;
    private boolean saved;

    private final JSONArray segments = new JSONArray();
    private final JSONArray polish = new JSONArray();
    private String committedText;
    private String commitReason;

    SessionLog(File dir, String fieldPackage, String model) { this(dir, fieldPackage, model, null); }

    SessionLog(File dir, String fieldPackage, String model, Context ctx) {
        Date now = new Date();
        this.dir = dir;
        this.id = new SimpleDateFormat("yyyyMMdd-HHmmss-SSS", Locale.US).format(now);
        this.fieldPackage = fieldPackage;
        this.model = model;
        this.ctx = ctx == null ? null : ctx.getApplicationContext();
        this.startWallMs = now.getTime();
        this.t0 = SystemClock.elapsedRealtime();
    }

    File wavFile() { return new File(dir, id + ".wav"); }

    File jsonFile() { return new File(dir, id + ".json"); }

    /** A recognized segment (main thread); index is its position in the session's raw list. */
    synchronized void segment(int index, String text) { segment(index, text, "local"); }

    /** Same, with the backend that produced the text ("local" or "server"). */
    synchronized void segment(int index, String text, String backend) {
        if (saved) return;
        try {
            segments.put(new JSONObject()
                    .put("i", index)
                    .put("text", text)
                    .put("backend", backend)
                    .put("t_ms", SystemClock.elapsedRealtime() - t0));
        } catch (Exception e) {
            Log.w("QwenASR", "session log segment failed: " + e.getMessage());
        }
    }

    /** A polish request and response (main thread); startMs is an elapsedRealtime value. */
    synchronized void polish(List<String> input, String context, Polisher.Result r, String error, long startMs, long durationMs) {
        if (saved) return;
        JSONObject o = new JSONObject();
        try {
            o.put("t_ms", startMs - t0);
            o.put("ms", durationMs);
            JSONArray in = new JSONArray();
            for (String s : input) in.put(s);
            o.put("input", in);
            if (context != null && !context.isEmpty()) o.put("context", context);
            if (r == null) {
                o.put("output", JSONObject.NULL);
                o.put("kept", JSONObject.NULL);
                o.put("raw", JSONObject.NULL);
            } else {
                JSONArray out = new JSONArray();
                for (String s : r.lines) out.put(s);
                o.put("output", out);
                o.put("kept", r.kept);
                o.put("raw", r.raw);
            }
            o.put("error", error == null ? JSONObject.NULL : error);   // put(k, null) would drop the key
        } catch (Exception e) {
            Log.w("QwenASR", "session log polish entry failed: " + e.getMessage());
            return;
        }
        polish.put(o);
    }

    /** The committed text (main thread); ends the session. */
    synchronized void commit(String text, String reason) {
        if (saved) return;
        committedText = text;
        commitReason = reason;
        commitTMs = SystemClock.elapsedRealtime() - t0;
        endWallMs = System.currentTimeMillis();
    }

    /** Writes the JSON record next to the WAV (net executor); never throws. */
    synchronized void save() {
        if (saved) return;
        saved = true;
        JSONObject o = new JSONObject();
        try {
            o.put("version", 1);
            o.put("id", id);
            o.put("start", Instant.ofEpochMilli(startWallMs).toString());
            if (endWallMs > 0) {
                o.put("end", Instant.ofEpochMilli(endWallMs).toString());
                o.put("duration_ms", commitTMs);
            } else {
                o.put("duration_ms", 0);
            }
            if (ctx != null) {
                try {
                    o.put("app_version", ctx.getPackageManager().getPackageInfo(ctx.getPackageName(), 0).versionName);
                } catch (Exception e) {
                    Log.w("QwenASR", "session log app version failed: " + e.getMessage());
                }
            }
            o.put("field_package", fieldPackage);
            o.put("polish_model", model);
            o.put("audio", wavFile().getName());
            o.put("segments", segments);
            o.put("polish", polish);
            if (committedText != null) {
                o.put("committed_text", committedText);
                o.put("commit_reason", commitReason);
            }
        } catch (Exception e) {
            Log.w("QwenASR", "session log build failed: " + e.getMessage());
            return;
        }
        File tmp = new File(dir, id + ".json.tmp");
        try (FileOutputStream out = new FileOutputStream(tmp)) {
            out.write(o.toString().getBytes(StandardCharsets.UTF_8));
        } catch (Exception e) {
            Log.w("QwenASR", "saving session log " + id + " failed: " + e.getMessage());
            tmp.delete();
            return;
        }
        if (!tmp.renameTo(jsonFile())) {
            Log.w("QwenASR", "renaming session log " + id + ".json.tmp failed");
            tmp.delete();
        }
    }

    /** Delete the oldest sessions (by file name) while .wav + .json together exceed maxBytes. */
    static void prune(File dir, long maxBytes) {
        File[] files = dir.listFiles((d, f) -> f.endsWith(".wav") || f.endsWith(".json"));
        if (files == null || files.length == 0) return;
        java.util.Arrays.sort(files, (a, b) -> a.getName().compareTo(b.getName()));
        long total = 0;
        for (File f : files) total += f.length();
        int deleted = 0;
        String lastId = null;
        for (int i = 0; i < files.length && total > maxBytes; i++) {
            String name = files[i].getName();
            String id = name.endsWith(".json") ? name.substring(0, name.length() - 5)
                    : name.substring(0, name.length() - 4);
            if (id.equals(lastId)) continue;   // the other half of a session already deleted
            lastId = id;
            File wav = new File(dir, id + ".wav");
            File json = new File(dir, id + ".json");
            total -= wav.length() + json.length();
            wav.delete();
            json.delete();
            deleted++;
        }
        if (deleted > 0) Log.w("QwenASR", "pruned " + deleted + " old session(s) from " + dir);
    }
}
