package com.bfu.qwenasr;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.inputmethodservice.InputMethodService;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Voice-only input method: records from the microphone and shows the live transcript as composing
 * text in the focused field. Each finished segment is replaced by its qwen27b-polished version
 * while recording continues; the text is committed when recording stops or the keyboard goes away.
 */
public class VoiceIme extends InputMethodService {
    private static final int RATE = Streamer.RATE;

    // Session state is touched on the main thread only (recording is also read by the "mic" thread).
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService net = Executors.newSingleThreadExecutor();   // qwen27b polish calls

    private LinearLayout root;
    private TextView status;
    private Button micBtn;

    private volatile boolean recording;
    private int sessionId;                 // incremented per session; late callbacks are ignored
    private int committedSid = -1;         // commitNow() is idempotent per session
    private Streamer st;
    private long recStart;
    private final List<String> raw = new ArrayList<>();
    private final List<String> polished = new ArrayList<>();   // same length as raw; null = not yet, "" = merged into the previous line
    private String partial = "";
    private String lead = "";              // a leading space when the field already has text before the cursor
    private boolean polishBusy, polishAgain, waitingToCommit, wantStart;
    private SessionLog log;                // session record for training data (null when off)
    private String lastRendered = "";      // what render() last gave to setComposingText

    @Override
    public View onCreateInputView() {
        float dp = getResources().getDisplayMetrics().density;
        int pad = (int) (16 * dp);
        LinearLayout r = new LinearLayout(this);
        r.setOrientation(LinearLayout.VERTICAL);
        r.setBackgroundColor(0xFF212121);
        r.setPadding(pad, pad / 2, pad, 0);

        status = new TextView(this);
        status.setTextSize(12);
        status.setTextColor(0xFFBDBDBD);
        status.setSingleLine(true);
        status.setText("Ready");
        r.addView(status);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        Button kb = smallBtn("\u2328");
        kb.setOnClickListener(v -> switchKeyboard());
        micBtn = new Button(this);
        micBtn.setText(recording ? "\u25A0 Stop" : "\uD83C\uDFA4 Start");
        micBtn.setTextSize(13);
        micBtn.setAllCaps(false);
        micBtn.setOnClickListener(v -> toggleMic());
        Button del = smallBtn("\u232B");
        del.setOnClickListener(v -> backspace());
        Button send = smallBtn("Send \u21B5");
        send.setOnClickListener(v -> sendEnter());
        int h = (int) (56 * dp);   // one height for every button
        row.addView(kb, new LinearLayout.LayoutParams(0, h, 1));
        row.addView(micBtn, new LinearLayout.LayoutParams(0, h, 2));
        row.addView(del, new LinearLayout.LayoutParams(0, h, 1));
        row.addView(send, new LinearLayout.LayoutParams(0, h, 1));
        r.addView(row);

        root = r;
        return r;
    }

    private Button smallBtn(String t) {
        Button b = new Button(this);
        b.setText(t);
        b.setTextSize(13);
        b.setAllCaps(false);
        return b;
    }

    private void setStatus(String s) { if (status != null) status.setText(s); }

    /** Recording, or session text still composing in the field. */
    private boolean dictationActive() { return recording || hasComposing(); }

    private boolean hasComposing() { return !raw.isEmpty() || !partial.isEmpty(); }

    // --- buttons ---

    private void toggleMic() {
        if (recording) { stopRecording(); return; }
        if (hasComposing()) commitNow("new_session");   // the old text goes into the field before a new session starts
        startDictation();
    }

    private void switchKeyboard() {
        if (dictationActive()) commitNow("switch");
        // switchToPreviousInputMethod() only hid the keyboard on the OPPO phone (2026-11-20), so switch by
        // id: Gboard if enabled, else the first other enabled IME with a typing ("keyboard") subtype
        // (skips voice-only IMEs such as Google's voice input); fall back to the picker.
        InputMethodManager imm = (InputMethodManager) getSystemService(INPUT_METHOD_SERVICE);
        String target = null;
        for (android.view.inputmethod.InputMethodInfo info : imm.getEnabledInputMethodList()) {
            if (info.getPackageName().equals(getPackageName())) continue;
            if (info.getPackageName().equals("com.google.android.inputmethod.latin")) { target = info.getId(); break; }
            if (target == null && hasTypingSubtype(info)) target = info.getId();
        }
        if (target != null) {
            try {
                switchInputMethod(target);
                return;
            } catch (RuntimeException e) {
                Log.w("QwenASR", "switch to " + target + " failed: " + e);
            }
        }
        imm.showInputMethodPicker();
    }

    private static boolean hasTypingSubtype(android.view.inputmethod.InputMethodInfo info) {
        int n = info.getSubtypeCount();
        if (n == 0) return true;
        for (int i = 0; i < n; i++) if ("keyboard".equals(info.getSubtypeAt(i).getMode())) return true;
        return false;
    }

    private void backspace() {
        if (!recording && !hasComposing()) sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL);
    }

    private void sendEnter() {
        commitNow("send");
        sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER);
    }

    // --- recording ---

    private void startDictation() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            setStatus("Open the Qwen ASR app once to allow the microphone.");
            startActivity(new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
            return;
        }
        if (Asr.engine() == null) {
            setStatus("Loading the speech model (about 5 s)...");
            wantStart = true;
            Asr.load(this, true, true, new Asr.Callback() {
                @Override public void onReady(Engine e) {
                    if (!wantStart) return;   // the user did something else meanwhile
                    wantStart = false;
                    if (isInputViewShown()) startRecording();
                }
                @Override public void onError(String message) {
                    wantStart = false;
                    setStatus(message);
                }
            });
            return;
        }
        startRecording();
    }

    private void startRecording() {
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
        sessionId++;
        final int sid = sessionId;
        final Streamer streamer;
        try {
            streamer = new Streamer(Asr.engine(), Asr.vadModel(this), getCacheDir(), true, listenerFor(sid))
                    .setFullPass(false, 30f);   // qwen27b polishes instead of the full pass
        } catch (Exception e) {
            setStatus("Cannot start: " + e.getMessage());
            rec.release();
            return;
        }
        if (Asr.keepRecordings(this)) {
            EditorInfo ei = getCurrentInputEditorInfo();
            String fieldPkg = ei == null || ei.packageName == null ? "" : ei.packageName;
            log = new SessionLog(Asr.sessionsDir(this), fieldPkg, Asr.polishModel(), this);
            streamer.saveSessionTo(log.wavFile(), 0);   // the Streamer keeps no older files; SessionLog.prune() does
        } else {
            log = null;
        }
        st = streamer;
        raw.clear(); polished.clear(); partial = "";
        InputConnection ic = getCurrentInputConnection();
        CharSequence before = ic == null ? null : ic.getTextBeforeCursor(1, 0);
        lead = (before != null && before.length() > 0 && !Character.isWhitespace(before.charAt(0))) ? " " : "";
        AsrService.begin(this, true);
        recording = true;
        recStart = SystemClock.elapsedRealtime();
        root.setKeepScreenOn(true);
        micBtn.setText("\u25A0 Stop");
        streamer.start();
        rec.startRecording();
        new Thread(() -> {
            short[] buf = new short[RATE / 10];   // 100 ms blocks
            while (recording) {
                int k = rec.read(buf, 0, buf.length);
                if (k > 0) streamer.feed(buf, k);
            }
            rec.stop();
            rec.release();
            streamer.finish();
        }, "mic").start();
        tickTimer();
    }

    private void stopRecording() {
        recording = false;   // the mic thread then calls st.finish()
        micBtn.setText("\uD83C\uDFA4 Start");
        setStatus("Finishing...");
    }

    private void tickTimer() {
        if (!recording) return;
        long s = (SystemClock.elapsedRealtime() - recStart) / 1000;
        setStatus(String.format(Locale.US, "Listening %d:%02d%s", s / 60, s % 60, polishBusy ? " \u00B7 polishing" : ""));
        ui.postDelayed(this::tickTimer, 250);
    }

    // --- session text ---

    /** Composing text: the segments (polished where available) joined with spaces, then the live partial. */
    private void render() {
        StringBuilder body = new StringBuilder();
        for (int i = 0; i < raw.size(); i++) {
            String s = polished.get(i) != null ? polished.get(i) : raw.get(i);
            if (s.isEmpty()) continue;
            if (body.length() > 0) body.append(' ');
            body.append(s);
        }
        if (!partial.isEmpty()) {
            if (body.length() > 0) body.append(' ');
            body.append(partial);
        }
        String text = body.length() == 0 ? "" : lead + body;
        lastRendered = text;
        InputConnection ic = getCurrentInputConnection();
        if (ic != null) ic.setComposingText(text, 1);
    }

    /** Sends the not-yet-polished segments to qwen27b; replaces them only if nothing changed meanwhile. */
    private void schedulePolish() {
        if (Asr.polisher() == null) return;
        if (!hasUnpolished()) return;
        if (polishBusy) { polishAgain = true; return; }
        polishBusy = true;
        final int sid = sessionId;
        final List<String> snapshot = new ArrayList<>(raw);
        final SessionLog sl = log;   // the record belongs to the session that sent this request
        final long started = SystemClock.elapsedRealtime();
        net.execute(() -> {
            Polisher.Result r = null;
            String err = null;
            try {
                r = Asr.polisher().polish(snapshot);
            } catch (IOException e) {
                Log.w("QwenASR", "polish failed: " + e.getMessage());
                err = e.getMessage();
            }
            final Polisher.Result res = r;
            final String error = err;
            ui.post(() -> {
                if (sl != null) sl.polish(snapshot, res, error, started, SystemClock.elapsedRealtime() - started);
                if (sid == sessionId) {
                    if (res != null) {
                        if (raw.size() >= snapshot.size() && raw.subList(0, snapshot.size()).equals(snapshot)) {
                            for (int i = 0; i < snapshot.size(); i++) polished.set(i, res.lines.get(i));
                            render();
                        }
                    } else {
                        setStatus("qwen27b unavailable, raw text kept");
                        // Keep the raw text for these lines so the session does not retry forever.
                        for (int i = 0; i < snapshot.size() && i < polished.size(); i++)
                            if (polished.get(i) == null) polished.set(i, raw.get(i));
                    }
                }
                polishBusy = false;
                if (polishAgain) { polishAgain = false; schedulePolish(); }
                else if (waitingToCommit) commitNow("stop");
            });
        });
    }

    private boolean hasUnpolished() {
        for (String p : polished) if (p == null) return true;
        return false;
    }

    /** Commits whatever text exists into the field. Idempotent per session. */
    private void commitNow(String reason) {
        if (committedSid == sessionId) return;
        committedSid = sessionId;
        if (recording) { recording = false; if (st != null) st.cancel(); }
        if (micBtn != null) micBtn.setText("🎤 Start");
        if (root != null) root.setKeepScreenOn(false);
        render();   // partial kept
        InputConnection ic = getCurrentInputConnection();
        if (ic != null) ic.finishComposingText();
        if (log != null) {
            log.commit(lastRendered, reason);
            final SessionLog sl = log;
            log = null;
            net.execute(() -> {
                sl.save();
                SessionLog.prune(Asr.sessionsDir(VoiceIme.this), 2L << 30);   // keep at most 2 GB
            });
        }
        raw.clear(); polished.clear(); partial = "";
        waitingToCommit = false;
        wantStart = false;
        sessionId++;
        setStatus("Ready");
    }

    // --- Streamer.Listener: callbacks arrive on the streaming thread, are posted to the main looper,
    // and a session's listener ignores every callback of an older session. ---
    private Streamer.Listener listenerFor(final int sid) {
        return new Streamer.Listener() {
            @Override public void onFinal(String text, String info) {
                ui.post(() -> {
                    if (sid != sessionId) return;
                    if (!text.isEmpty()) {
                        raw.add(text);
                        polished.add(null);
                        if (log != null) log.segment(raw.size() - 1, text);
                    }
                    partial = "";
                    render();
                    schedulePolish();
                });
            }
            @Override public void onPartial(String text, String info) {
                ui.post(() -> {
                    if (sid != sessionId) return;
                    partial = text;
                    render();
                });
            }
            @Override public void onError(String msg) {
                ui.post(() -> {
                    if (sid != sessionId) return;
                    setStatus(msg);
                });
            }
            @Override public void onDone() {
                AsrService.end(VoiceIme.this);   // release this session's hold even when it is stale
                ui.post(() -> {
                    if (sid != sessionId) return;
                    root.setKeepScreenOn(false);
                    if (polishBusy || (hasUnpolished() && Asr.polisher() != null)) {
                        waitingToCommit = true;
                        setStatus("Polishing...");
                        ui.postDelayed(() -> { if (sid == sessionId) commitNow("stop"); }, 8000);
                    } else {
                        commitNow("stop");
                    }
                });
            }
        };
    }

    // --- InputMethodService lifecycle ---

    /**
     * Preload: when the keyboard opens and the model is not loaded (first use, or Android killed the
     * process), start loading it now so the ~5 s load runs while the user gets ready to speak. A
     * Start tap during the load joins it (Asr.load keeps one load and a list of waiters).
     */
    @Override
    public void onStartInputView(EditorInfo info, boolean restarting) {
        super.onStartInputView(info, restarting);
        if (Asr.engine() != null || recording) return;
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return;
        setStatus("Loading the speech model...");
        Asr.load(this, true, true, new Asr.Callback() {
            @Override public void onReady(Engine e) {
                Log.i("QwenASR", "preload done load_ms=" + Asr.loadMs());
                if (!recording && !wantStart) setStatus("Ready");
            }
            @Override public void onError(String message) {
                if (!recording) setStatus(message);
            }
        });
    }

    /** Called while the old field is still connected: its dictation goes into it, not the next field. */
    @Override
    public void onFinishInput() {
        if (dictationActive()) commitNow("field_change");
        super.onFinishInput();
    }

    @Override
    public void onFinishInputView(boolean finishingInput) {
        super.onFinishInputView(finishingInput);
        if (dictationActive()) commitNow("hidden");
    }

    @Override
    public void onWindowHidden() {
        super.onWindowHidden();
        if (dictationActive()) commitNow("hidden");
    }
}
