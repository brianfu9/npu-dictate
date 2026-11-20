package com.bfu.qwenasr;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.ForegroundColorSpan;
import android.util.Log;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.util.Locale;

public class MainActivity extends Activity implements Streamer.Listener {
    private static final int RATE = Streamer.RATE;

    // UI state (the process-wide model state lives in Asr).
    private static final StringBuilder history = new StringBuilder();
    private static String partial = "";
    private static String lastFinal = "";
    private static boolean liveText = true;
    private static boolean useWords = true;
    // Off by default: the NPU prefill is only correct up to 256 tokens (~12 s windows), too short to help.
    private static boolean fullPass = false;
    private static boolean polishOn = true;
    private static boolean englishOnly = true;
    // Last polished session: its start offset in history and both versions (for the Original button).
    private static int lastOffset = -1;
    private static String lastRaw, lastPolished;
    private static final java.util.List<String> sessionSegments = new java.util.ArrayList<>();
    private static int sessionOffset;          // history length when the current session started
    private static Streamer current;           // the latest session (its full pass may still run)

    private final Handler ui = new Handler(Looper.getMainLooper());
    private TextView status, transcript, models;
    private Button origBtn, settingsBtn;
    private ScrollView settingsScroll;
    private static boolean settingsOpen = false;
    private Button recordBtn;
    private ScrollView scroll;
    private CheckBox keepCheck;

    private volatile boolean recording;
    private Streamer streamer;
    private long recStart;
    private String lastInfo = "";

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        applyKeepRecordings(getIntent());
        buildUi();
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO, Manifest.permission.POST_NOTIFICATIONS}, 1);
        }
        redraw();
        if (Asr.engine() == null) setStatus("Loading the speech model (about 5 s)...");
        else setStatus("Ready.");
        Asr.load(this, englishOnly, useWords, new Asr.Callback() {
            @Override public void onReady(Engine e) {
                setStatus(String.format(Locale.US, "Ready. Speech model loaded in %.1f s. Tap Record.", Asr.loadMs() / 1000.0));
                recordBtn.setEnabled(true);
            }
            @Override public void onError(String message) {
                setStatus("Model load failed: " + message + "\nLogs: " + getCacheDir() + "/encoder.log, decoder.log");
            }
        });
        recordBtn.setEnabled(Asr.engine() != null);
        if (b == null) handleTestIntent(getIntent());   // a restarted activity gets the old intent again
    }

    @Override
    protected void onNewIntent(Intent i) {
        super.onNewIntent(i);
        applyKeepRecordings(i);
        handleTestIntent(i);
    }

    /** The --ez keep_recordings extra turns the "Keep recordings" setting on or off. */
    private void applyKeepRecordings(Intent i) {
        if (i == null || !i.hasExtra("keep_recordings")) return;
        boolean on = i.getBooleanExtra("keep_recordings", false);
        Asr.setKeepRecordings(this, on);
        Log.i("QwenASR", "keep_recordings=" + on);
        if (keepCheck != null) keepCheck.setChecked(on);
    }

    /**
     * Test hook: plays a wav file through the streaming pipeline at real-time speed, as if spoken:
     *   am start -n com.bfu.qwenasr/.MainActivity --es wav /data/user/0/com.bfu.qwenasr/files/x.wav [--ez live false]
     * Results go to the screen and to logcat (tag QwenASR).
     */
    private void handleTestIntent(Intent i) {
        final String path = i == null ? null : i.getStringExtra("wav");
        if (path == null) return;
        final boolean live = i.getBooleanExtra("live", true);
        final boolean full = i.getBooleanExtra("full", false);
        final float window = i.getFloatExtra("window", 30f);
        final boolean useW = i.getBooleanExtra("words", true);
        i.removeExtra("wav");
        new Thread(() -> {
            try {
                while (Asr.engine() == null) Thread.sleep(200);
                Engine e = Asr.engine();
                short[] s = Wav.read(new File(path));
                ui.post(() -> setStatus("Test: playing " + new File(path).getName()));
                e.wordList = useW ? Asr.words() : new java.util.ArrayList<>();
                Log.i("QwenASR", "test words=" + useW + " (" + e.wordList.size() + " tokens) maxPrompt=" + e.maxPrompt());
                AsrService.begin(this, false);
                Streamer st = newSession(live).setFullPass(full, window);
                Log.i("QwenASR", String.format(Locale.US, "test start %s %.2f s live=%b", path, s.length / (float) RATE, live));
                st.start();
                short[] blk = new short[RATE / 10];
                long t0 = SystemClock.elapsedRealtime();
                for (int p = 0; p < s.length; p += blk.length) {
                    int len = Math.min(blk.length, s.length - p);
                    System.arraycopy(s, p, blk, 0, len);
                    st.feed(blk, len);
                    long due = t0 + (p + len) * 1000L / RATE;
                    long wait = due - SystemClock.elapsedRealtime();
                    if (wait > 0) Thread.sleep(wait);
                }
                Log.i("QwenASR", "test input done");
                st.finish();
            } catch (Exception e) {
                Log.e("QwenASR", "test failed", e);
            }
        }).start();
    }

    /** Starts a new session: cancels the previous one (and its full pass), marks where its text begins. */
    private Streamer newSession(boolean live) throws Exception {
        if (current != null) current.cancel();
        Streamer st = new Streamer(Asr.engine(), Asr.vadModel(this), getCacheDir(), live, this);
        current = st;
        final Object lock = new Object();
        synchronized (lock) {
            ui.post(() -> {
                synchronized (history) { sessionOffset = history.length(); sessionSegments.clear(); }
                synchronized (lock) { lock.notifyAll(); }
            });
            if (Looper.myLooper() == Looper.getMainLooper()) {
                synchronized (history) { sessionOffset = history.length(); sessionSegments.clear(); }
            } else {
                lock.wait(2000);
            }
        }
        return st;
    }

    private void buildUi() {
        float dp = getResources().getDisplayMetrics().density;
        int pad = (int) (16 * dp);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad * 2, pad, pad);

        // Always visible: status, what runs where, Record, text buttons. The settings open in a panel.
        LinearLayout ctl = new LinearLayout(this);
        ctl.setOrientation(LinearLayout.VERTICAL);
        root.addView(ctl);

        status = new TextView(this);
        status.setTextSize(14);
        ctl.addView(status);

        models = new TextView(this);
        models.setTextSize(12);
        models.setTextColor(0xFF9E9E9E);
        models.setPadding(0, pad / 3, 0, 0);
        ctl.addView(models);

        recordBtn = new Button(this);
        recordBtn.setText("Record");
        recordBtn.setTextSize(22);
        recordBtn.setEnabled(false);
        recordBtn.setOnClickListener(v -> toggleRecord());
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, (int) (80 * dp));
        lp.topMargin = pad / 2;
        ctl.addView(recordBtn, lp);
        ctl.addView(help("Speech to text runs on this phone (no network needed). Recordings are saved in the app."));

        settingsBtn = small("");
        settingsBtn.setGravity(Gravity.START | Gravity.CENTER_VERTICAL);
        settingsBtn.setOnClickListener(v -> {
            settingsOpen = !settingsOpen;
            settingsScroll.setVisibility(settingsOpen ? android.view.View.VISIBLE : android.view.View.GONE);
            scroll.setVisibility(settingsOpen ? android.view.View.GONE : android.view.View.VISIBLE);   // settings take the space
            refreshSettingsSummary();
        });
        ctl.addView(settingsBtn);

        LinearLayout set = new LinearLayout(this);
        set.setOrientation(LinearLayout.VERTICAL);
        settingsScroll = new ScrollView(this);
        settingsScroll.addView(set);
        settingsScroll.setVisibility(settingsOpen ? android.view.View.VISIBLE : android.view.View.GONE);
        LinearLayout ctl2 = ctl;   // the text actions stay in the always-visible part
        ctl = set;

        // --- while you speak ---
        ctl.addView(header("While you speak"));
        CheckBox live = check("Live text", liveText, on -> { liveText = on; refreshSettingsSummary(); });
        ctl.addView(live);
        ctl.addView(help("Show the words as you speak. Off: the text appears after each pause."));
        CheckBox wl = check("Word list", useWords, on -> {
            useWords = on;
            Asr.setLanguageAndWords(englishOnly, useWords);
            refreshSettingsSummary();
        });
        ctl.addView(wl);
        ctl.addView(help("Help the recognizer spell your terms (rad, Immich, Tailscale, ...)."));
        CheckBox en = check("English only", englishOnly, on -> {
            englishOnly = on;
            Asr.setLanguageAndWords(englishOnly, useWords);
            refreshSettingsSummary();
        });
        ctl.addView(en);
        ctl.addView(help("Always write English. Off: detect the spoken language."));

        // --- after you stop ---
        ctl.addView(header("After you stop: clean-up"));
        CheckBox fp = check("Clean up the text", polishOn, on -> { polishOn = on; refreshSettingsSummary(); });
        ctl.addView(fp);
        ctl.addView(help("A language model fixes punctuation, sentence breaks and the spelling of your terms. "
                + "It does not hear the audio. \"Show original\" undoes it."));

        // --- training data ---
        ctl.addView(header("Training data"));
        keepCheck = check("Keep recordings (audio + text, for training data)", Asr.keepRecordings(this), on -> {
            Asr.setKeepRecordings(this, on);
        });
        ctl.addView(keepCheck);
        ctl.addView(help("Every voice-keyboard session saves its audio and text to the app's files, for later review."));

        Button close = small("Close settings");
        close.setOnClickListener(v -> settingsBtn.performClick());
        ctl.addView(close);

        ctl = ctl2;
        // --- text actions (always visible) ---
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        Button copy = small("Copy all");
        copy.setOnClickListener(v -> {
            ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            String all;
            synchronized (history) { all = history.toString(); }
            cm.setPrimaryClip(ClipData.newPlainText("transcript", all));
            Toast.makeText(this, "Copied all text", Toast.LENGTH_SHORT).show();
        });
        Button clear = small("Clear all");
        clear.setOnClickListener(v -> {
            synchronized (history) { history.setLength(0); }
            partial = "";
            lastOffset = -1;
            refreshOriginalButton();
            redraw();
        });
        origBtn = small("Show original");
        origBtn.setOnClickListener(v -> toggleOriginal());
        row.addView(copy, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        row.addView(clear, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1));
        row.addView(origBtn, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1.3f));
        ctl.addView(row);
        refreshOriginalButton();

        root.addView(settingsScroll, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));
        refreshSettingsSummary();

        scroll = new ScrollView(this);
        transcript = new TextView(this);
        transcript.setTextSize(18);
        transcript.setTextIsSelectable(true);
        transcript.setGravity(Gravity.TOP);
        transcript.setPadding(0, pad / 2, 0, pad);
        transcript.setHint("Your text appears here. \"Show original\" switches the last recording between the raw "
                + "and the cleaned-up text.");
        scroll.addView(transcript);
        root.addView(scroll, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1));
        scroll.setVisibility(settingsOpen ? android.view.View.GONE : android.view.View.VISIBLE);
        setContentView(root);
    }

    private TextView header(String t) {
        TextView v = new TextView(this);
        v.setText(t);
        v.setTextSize(15);
        v.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        v.setPadding(0, (int) (14 * getResources().getDisplayMetrics().density), 0, 0);
        return v;
    }

    private TextView help(String t) {
        TextView v = new TextView(this);
        v.setText(t);
        v.setTextSize(12);
        v.setTextColor(0xFF9E9E9E);
        v.setPadding((int) (32 * getResources().getDisplayMetrics().density), 0, 0, 0);
        return v;
    }

    private CheckBox check(String t, boolean on, java.util.function.Consumer<Boolean> f) {
        CheckBox c = new CheckBox(this);
        c.setText(t);
        c.setTextSize(14);
        c.setChecked(on);
        c.setOnCheckedChangeListener((v, x) -> f.accept(x));
        return c;
    }

    private Button small(String t) {
        Button b = new Button(this);
        b.setText(t);
        b.setTextSize(13);
        b.setAllCaps(false);
        return b;
    }

    private void refreshSettingsSummary() {
        if (settingsBtn == null) return;
        String m = Asr.polishModel();
        String clean = !polishOn ? "off" : (m.isEmpty() ? "the polish server" : m + " on the polish server");
        settingsBtn.setText((settingsOpen ? "\u25BE Settings  " : "\u25B8 Settings  ")
                + String.format("Live text: %s \u00B7 Word list: %s \u00B7 %s \u00B7 Clean-up: %s",
                        liveText ? "on" : "off", useWords ? "on" : "off", englishOnly ? "English only" : "any language", clean));
    }

    private void refreshOriginalButton() {
        if (origBtn == null) return;
        boolean showingRaw = false;
        if (lastOffset >= 0 && lastRaw != null) {
            synchronized (history) {
                if (lastOffset <= history.length()) showingRaw = history.substring(lastOffset).trim().equals(lastRaw);
            }
        }
        origBtn.setText(showingRaw ? "Show cleaned" : "Show original");
        origBtn.setEnabled(lastOffset >= 0 && lastRaw != null);
    }

    /** "What runs where", refreshed every second while the screen is visible. */
    private final Runnable modelTicker = new Runnable() {
        @Override public void run() {
            refreshModels();
            ui.postDelayed(this, 1000);
        }
    };

    private void refreshModels() {
        if (models == null) return;
        Engine e = Asr.engine();
        StringBuilder sb = new StringBuilder();
        if (e == null) {
            sb.append("Speech model (Qwen3-ASR 1.7B): loading...");
        } else {
            sb.append("Speech: Qwen3-ASR 1.7B. Audio encoder on the CPU ")
              .append(e.encoderLoadedNow() ? "(loaded)" : "(not loaded)")
              .append(", text decoder on the NPU ")
              .append(e.decoderLoadedNow() ? "(loaded)" : "(unloaded; loads again when you record)");
        }
        sb.append("\nan OpenAI-compatible chat endpoint from polish.conf; used for text clean-up");
        models.setText(sb.toString());
    }

    @Override
    protected void onResume() {
        super.onResume();
        ui.post(modelTicker);
        refreshOriginalButton();
    }

    @Override
    protected void onPause() {
        ui.removeCallbacks(modelTicker);
        super.onPause();
    }

    private void setStatus(String s) { status.setText(s); }

    /** Final text in the normal colour, live (partial) text in grey after it. */
    private void redraw() {
        SpannableStringBuilder sb = new SpannableStringBuilder();
        synchronized (history) { sb.append(history); }
        if (!partial.isEmpty()) {
            if (sb.length() > 0) sb.append(' ');
            int s = sb.length();
            sb.append(partial);
            sb.setSpan(new ForegroundColorSpan(Color.GRAY), s, sb.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        transcript.setText(sb);
        scroll.post(() -> scroll.fullScroll(ScrollView.FOCUS_DOWN));
    }

    // Streamer.Listener (called on the streaming thread)
    @Override
    public void onFinal(String text, String info) {
        ui.post(() -> {
            if (!text.isEmpty()) {
                synchronized (history) {
                    if (history.length() > 0) history.append(' ');
                    history.append(text);
                    sessionSegments.add(text);
                }
                lastFinal = text;
            }
            partial = "";
            lastInfo = info;
            if (!recording) { setStatus(info); recordBtn.setEnabled(Asr.engine() != null); }
            redraw();
        });
    }

    @Override
    public void onPartial(String text, String info) {
        ui.post(() -> {
            partial = text;
            lastInfo = info;
            redraw();
        });
    }

    @Override
    public void onFullPassStart(int windows) {
        ui.post(() -> setStatus("Checking the full recording" + (windows > 1 ? " (" + windows + " parts)" : "") + "..."));
    }

    @Override
    public void onFullPass(String text, String info) {
        ui.post(() -> {
            synchronized (history) {
                if (sessionOffset <= history.length()) {
                    history.setLength(sessionOffset);
                    if (sessionOffset > 0 && !text.isEmpty()) history.append(' ');
                    history.append(text);
                }
            }
            partial = "";
            setStatus(info);
            redraw();
        });
    }

    @Override
    public void onDone() {
        Log.i("QwenASR", "session done");
        AsrService.end(this);
        ui.post(() -> {
            if (!recording) recordBtn.setEnabled(Asr.engine() != null);
            Log.i("QwenASR", "onDone polishOn=" + polishOn + " polisher=" + (Asr.polisher() != null));
            if (polishOn && Asr.polisher() != null) startPolish();
        });
    }

    /** UI thread: sends the last session's text to the polish server; replaces it only if nothing changed meanwhile. */
    private void startPolish() {
        final int off;
        final String raw;
        final java.util.List<String> segs;
        synchronized (history) {
            off = Math.min(sessionOffset, history.length());
            raw = history.substring(off).trim();
            segs = new java.util.ArrayList<>(sessionSegments);
        }
        if (raw.isEmpty() || segs.isEmpty() || !raw.equals(String.join(" ", segs))) {
            Log.w("QwenASR", "polish skipped: raw=" + raw.length() + " chars, segments=" + segs.size()
                    + ", joined=" + String.join(" ", segs).length() + " chars, off=" + off
                    + "\nRAW=" + raw + "\nJOINED=" + String.join(" ", segs));
            return;
        }
        final String m = Asr.polishModel();
        final String who = m.isEmpty() ? "the polish server" : m + " on the polish server";
        setStatus("Cleaning up with " + who + "...");
        final Polisher p = Asr.polisher();
        new Thread(() -> {
            long t0 = SystemClock.elapsedRealtime();
            String out = null, err = null;
            int keptN = 0;
            try {
                Polisher.Result pr = p.polish(segs);
                java.util.List<String> nonEmpty = new java.util.ArrayList<>();
                for (String l : pr.lines) if (!l.isEmpty()) nonEmpty.add(l);
                out = String.join(" ", nonEmpty);
                keptN = pr.kept;
            } catch (Exception e) { err = e.getMessage(); }
            final String o = out, er = err;
            final int kept = keptN, total = segs.size();
            final double s = (SystemClock.elapsedRealtime() - t0) / 1000.0;
            Log.i("QwenASR", "polish " + (o != null ? String.format(Locale.US, "ok %.1f s text=%s", s, o) : "failed: " + er));
            ui.post(() -> {
                if (o == null) { setStatus("Clean-up failed (" + who + ": " + er + "). The raw text is kept."); return; }
                synchronized (history) {
                    String cur = history.substring(Math.min(off, history.length())).trim();
                    if (!cur.equals(raw)) { setStatus("The text changed during the clean-up, so it was not applied."); return; }
                    history.setLength(off);
                    if (off > 0) history.append(' ');
                    history.append(o);
                }
                lastOffset = off; lastRaw = raw; lastPolished = o;
                setStatus(String.format(Locale.US, "Cleaned up by %s in %.1f s%s. \"Show original\" undoes it.", who, s,
                        kept > 0 ? String.format(Locale.US, " (%d of %d lines kept as they were)", kept, total) : ""));
                refreshOriginalButton();
                redraw();
            });
        }, "polish").start();
    }

    /** Switches the last polished session between the polish server's text and the raw text. */
    private void toggleOriginal() {
        if (lastOffset < 0 || lastRaw == null) { Toast.makeText(this, "Nothing has been cleaned up yet", Toast.LENGTH_SHORT).show(); return; }
        synchronized (history) {
            if (lastOffset > history.length()) return;
            String cur = history.substring(lastOffset).trim();
            String next = cur.equals(lastPolished) ? lastRaw : cur.equals(lastRaw) ? lastPolished : null;
            if (next == null) { Toast.makeText(this, "Text changed since the polish", Toast.LENGTH_SHORT).show(); return; }
            history.setLength(lastOffset);
            if (lastOffset > 0) history.append(' ');
            history.append(next);
            setStatus(next.equals(lastRaw) ? "Showing the original (raw) text of the last recording."
                                           : "Showing the cleaned-up text of the last recording.");
        }
        refreshOriginalButton();
        redraw();
    }

    @Override
    public void onError(String msg) {
        ui.post(() -> setStatus("Error: " + msg));
    }

    private void toggleRecord() {
        if (!recording) startRecording(); else stopRecording();
    }

    private void startRecording() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 1);
            return;
        }
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        final AudioRecord rec;
        try {
            rec = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, RATE,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, Math.max(min, RATE * 2));
        } catch (SecurityException e) {
            setStatus("No microphone permission.");
            return;
        }
        if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
            setStatus("Cannot open the microphone.");
            rec.release();
            return;
        }
        try {
            streamer = newSession(liveText).setFullPass(false, 30f)   // the polish server cleans up instead
                    .saveSessionTo(new File(getFilesDir(), "recordings/"
                            + new java.text.SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new java.util.Date()) + ".wav"));
        } catch (Exception e) {
            setStatus("Cannot start: " + e.getMessage());
            rec.release();
            return;
        }
        AsrService.begin(this, true);
        recording = true;
        recStart = SystemClock.elapsedRealtime();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        recordBtn.setText("Stop");
        Engine e = Asr.engine();
        if (e != null && !e.decoderLoadedNow())
            lastInfo = "(loading the speech decoder on the NPU again, first words in ~5 s)";
        final Streamer st = streamer;
        st.start();
        rec.startRecording();
        new Thread(() -> {
            short[] buf = new short[RATE / 10];
            while (recording) {
                int k = rec.read(buf, 0, buf.length);
                if (k > 0) st.feed(buf, k);
            }
            rec.stop();
            rec.release();
            st.finish();
        }, "mic").start();
        tickTimer();
    }

    private void tickTimer() {
        if (!recording) return;
        long s = (SystemClock.elapsedRealtime() - recStart) / 1000;
        setStatus(String.format(Locale.US, "Recording %d:%02d   %s", s / 60, s % 60, lastInfo));
        ui.postDelayed(this::tickTimer, 250);
    }

    private void stopRecording() {
        recording = false;
        recordBtn.setText("Record");
        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setStatus("Stopped. Finishing the last sentence...");
    }

    @Override
    protected void onDestroy() {
        recording = false;
        super.onDestroy();
    }
}
