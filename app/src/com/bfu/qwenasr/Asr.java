package com.bfu.qwenasr;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Process-wide ASR state: the model servers stay loaded while the app process lives, and any
 * component (MainActivity today, the voice keyboard later) loads and uses them through this class.
 */
public final class Asr {
    /** Load result; both callbacks are posted to the main looper. */
    public interface Callback {
        void onReady(Engine e);
        void onError(String message);
    }

    private static Engine engine;
    private static volatile boolean ready;
    private static final ExecutorService worker = Executors.newSingleThreadExecutor();
    private static Polisher polisher;
    private static String polishModel = "";
    private static List<Integer> words = new ArrayList<>();
    private static long loadMs;
    private static boolean loading;
    private static final List<Callback> pending = new ArrayList<>();
    private static final Handler ui = new Handler(Looper.getMainLooper());

    private Asr() { }

    /** The process-wide ASR engine (null until the models are loaded). */
    public static Engine engine() { return ready ? engine : null; }

    /** The process-wide text polisher (null until the models are loaded). */
    public static Polisher polisher() { return polisher; }

    /** The configured polish model name ("" when unset). */
    public static String polishModel() { return polishModel; }

    /** The word list token ids, once loaded. */
    public static List<Integer> words() { return words; }

    /** Milliseconds the last model load took. */
    public static long loadMs() { return loadMs; }

    /** Whether voice-keyboard sessions are saved for training data (off by default). */
    public static boolean keepRecordings(Context c) {
        return c.getApplicationContext().getSharedPreferences("settings", Context.MODE_PRIVATE)
                .getBoolean("keep_recordings", false);
    }

    public static void setKeepRecordings(Context c, boolean on) {
        c.getApplicationContext().getSharedPreferences("settings", Context.MODE_PRIVATE)
                .edit().putBoolean("keep_recordings", on).apply();
    }

    /** Where the session recordings (WAV + JSON) are kept; created if needed. */
    public static File sessionsDir(Context c) {
        File d = c.getApplicationContext().getExternalFilesDir("sessions");
        if (!d.isDirectory()) d.mkdirs();
        return d;
    }

    /** The shared single-thread worker for model loading and other background work. */
    public static ExecutorService worker() { return worker; }

    /** The Silero VAD model ships as an APK asset; copy it to a file once. */
    public static File vadModel(Context c) throws Exception {
        Context app = c.getApplicationContext();
        File f = new File(app.getFilesDir(), "silero_vad.onnx");
        if (!f.isFile()) {
            try (InputStream in = app.getAssets().open("silero_vad.onnx");
                 FileOutputStream o = new FileOutputStream(f)) {
                byte[] buf = new byte[65536];
                int k;
                while ((k = in.read(buf)) > 0) o.write(buf, 0, k);
            }
        }
        return f;
    }

    /** Word list token ids (assets/context_ids.i32, made from personal/context.txt at build time). */
    public static List<Integer> loadWordList(Context c) throws Exception {
        Context app = c.getApplicationContext();
        List<Integer> ids = new ArrayList<>();
        try (InputStream in = app.getAssets().open("context_ids.i32")) {
            byte[] b = new byte[4];
            while (in.read(b) == 4) ids.add((b[0] & 255) | (b[1] & 255) << 8 | (b[2] & 255) << 16 | (b[3] & 255) << 24);
        }
        return ids;
    }

    /** Reads a UTF-8 text asset. */
    public static String asset(Context c, String name) throws Exception {
        Context app = c.getApplicationContext();
        try (InputStream in = app.getAssets().open(name)) {
            java.io.ByteArrayOutputStream b = new java.io.ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int k;
            while ((k = in.read(buf)) > 0) b.write(buf, 0, k);
            return b.toString("UTF-8");
        }
    }

    /**
     * The polisher from assets: polish.conf (url= and model= lines, # comments) plus terms.txt
     * and polish_prompt.txt. Null when the conf is missing or has no url; sets polishModel().
     */
    private static Polisher newPolisher(Context app) throws Exception {
        String url = "", model = "";
        try {
            for (String line : asset(app, "polish.conf").split("\n")) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#") || !line.contains("=")) continue;
                int i = line.indexOf('=');
                String k = line.substring(0, i).trim(), v = line.substring(i + 1).trim();
                if (k.equals("url")) url = v;
                else if (k.equals("model")) model = v;
            }
        } catch (Exception e) {
            return null;   // no polish.conf: the app runs without clean-up
        }
        polishModel = model;
        if (url.isEmpty()) return null;
        return new Polisher(asset(app, "polish_prompt.txt"), asset(app, "terms.txt"), url, model);
    }

    /** Applies the language and word-list settings to the loaded engine (no-op until it exists). */
    public static void setLanguageAndWords(boolean englishOnly, boolean useWords) {
        Engine e = engine;
        if (e == null) return;
        e.forcedLanguage = englishOnly ? new ArrayList<>(Engine.ENGLISH) : new ArrayList<>();
        e.wordList = useWords ? words : new ArrayList<>();
    }

    /**
     * Loads the speech model once per process. If it is already loaded, cb.onReady is posted at
     * once; if a load is already running, cb joins the pending list and is called when it finishes.
     */
    public static synchronized void load(Context c, boolean englishOnly, boolean useWords, Callback cb) {
        if (ready) { ui.post(() -> cb.onReady(engine)); return; }
        if (loading) { pending.add(cb); return; }
        final Context app = c.getApplicationContext();
        loading = true;
        pending.add(cb);
        engine = new Engine(app.getFilesDir(), app.getCacheDir());
        String missing = engine.missingModels();
        if (missing != null) {
            String msg = "Model files are missing in " + engine.modelDir + ":\n" + missing;
            engine = null;
            loading = false;
            pending.clear();
            ui.post(() -> cb.onError(msg));
            return;
        }
        AsrService.begin(app, false);   // keep the app from being frozen while it loads
        final File nativeDir = new File(app.getApplicationInfo().nativeLibraryDir);
        worker.execute(() -> {
            long t0 = SystemClock.elapsedRealtime();
            try {
                vadModel(app);
                words = loadWordList(app);
                polisher = newPolisher(app);   // null when polish.conf is missing or has no url
                engine.forcedLanguage = englishOnly ? new ArrayList<>(Engine.ENGLISH) : new ArrayList<>();
                engine.wordList = useWords ? words : new ArrayList<>();
                engine.start(nativeDir);
                ready = true;
                loadMs = SystemClock.elapsedRealtime() - t0;
                AsrService.end(app);
                Log.i("QwenASR", "ready load_ms=" + loadMs + " enc=" + engine.encReady + " dec=" + engine.decReady);
                List<Callback> cbs;
                synchronized (Asr.class) { cbs = new ArrayList<>(pending); pending.clear(); loading = false; }
                for (Callback p : cbs) ui.post(() -> p.onReady(engine));
            } catch (Exception e) {
                AsrService.end(app);
                engine.stop();
                engine = null;
                List<Callback> cbs;
                synchronized (Asr.class) { cbs = new ArrayList<>(pending); pending.clear(); loading = false; }
                for (Callback p : cbs) ui.post(() -> p.onError(e.getMessage()));
            }
        });
    }
}
