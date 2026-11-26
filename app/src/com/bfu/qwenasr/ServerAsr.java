package com.bfu.qwenasr;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * Server-side ASR: sends a segment's WAV to an OpenAI-compatible /v1/audio/transcriptions
 * endpoint (url and model from asr.conf) and returns the text. Any failure throws IOException,
 * so the caller falls back to the on-phone NPU model. A failure also sets a 30 s cooldown so the
 * caller can skip the server for that window instead of stalling every partial and final.
 */
final class ServerAsr {
    private static final String BOUNDARY = "qwenasr-form";
    static final int CONNECT_TIMEOUT_MS = 2500;
    static final int PARTIAL_READ_TIMEOUT_MS = 10000;
    static final int FINAL_READ_TIMEOUT_MS = 15000;
    static final long COOLDOWN_NS = 30_000_000_000L;

    private final String url;
    private final String model;
    private final String name;
    private final int connectTimeoutMs;
    private final long cooldownNs;
    private volatile long downUntilNanos;

    ServerAsr(String url, String model) {
        this(url, model, "", CONNECT_TIMEOUT_MS, COOLDOWN_NS);
    }

    ServerAsr(String url, String model, String name) {
        this(url, model, name, CONNECT_TIMEOUT_MS, COOLDOWN_NS);
    }

    ServerAsr(String url, String model, int connectTimeoutMs, long cooldownNs) {
        this(url, model, "", connectTimeoutMs, cooldownNs);
    }

    ServerAsr(String url, String model, String name, int connectTimeoutMs, long cooldownNs) {
        this.url = url;
        this.model = model;
        this.name = name;
        this.connectTimeoutMs = connectTimeoutMs;
        this.cooldownNs = cooldownNs;
    }

    /** The configured model name (for status lines). */
    String model() { return model; }

    /** The endpoint's host[:port] for status lines ("server" when unparseable). */
    String route() {
        int s = url.indexOf("://");
        if (s < 0) return "server";
        int e = url.indexOf('/', s + 3);
        String host = e < 0 ? url.substring(s + 3) : url.substring(s + 3, e);
        return host.isEmpty() ? "server" : host;
    }

    /** The status-line name for this endpoint (name= from asr.conf, else host[:port]). */
    String label() { return name.isEmpty() ? route() : name; }

    /** Whether a failure's cooldown is still running (wrap-safe). */
    boolean isInCooldown() {
        long until = downUntilNanos;
        return until != 0 && System.nanoTime() - until < 0;
    }

    void clearCooldown() { downUntilNanos = 0; }

    /** Blocking; call from a worker thread. An empty text is valid (silence). */
    String transcribe(byte[] wav) throws IOException {
        return transcribe(wav, FINAL_READ_TIMEOUT_MS);
    }

    String transcribe(byte[] wav, int readTimeoutMs) throws IOException {
        HttpURLConnection c = null;
        try {
            c = (HttpURLConnection) new URL(url).openConnection();
            c.setConnectTimeout(connectTimeoutMs);
            c.setReadTimeout(readTimeoutMs);
            c.setRequestMethod("POST");
            c.setDoOutput(true);
            c.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + BOUNDARY);
            try (OutputStream o = c.getOutputStream()) {
                o.write(part("model", null, "text/plain", model.getBytes(StandardCharsets.UTF_8)));
                o.write(part("file", "segment.wav", "audio/wav", wav));
                o.write(("--" + BOUNDARY + "--\r\n").getBytes(StandardCharsets.UTF_8));
            }
            int code = c.getResponseCode();
            InputStream in = code < 400 ? c.getInputStream() : c.getErrorStream();
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            if (in != null) {
                try (InputStream s = in) {
                    byte[] b = new byte[8192];
                    int k;
                    while ((k = s.read(b)) > 0) buf.write(b, 0, k);
                }
            }
            String resp = buf.toString("UTF-8");
            if (code < 200 || code >= 300)
                throw new IOException(model + " HTTP " + code + ": " + resp.substring(0, Math.min(200, resp.length())));
            String text;
            try {
                text = new JSONObject(resp).getString("text").trim();
            } catch (Exception e) {
                throw new IOException("bad response from the ASR server: " + e.getMessage());
            }
            downUntilNanos = 0;
            return text;
        } catch (IOException e) {
            downUntilNanos = System.nanoTime() + cooldownNs;
            if (c != null) c.disconnect();
            throw e;
        }
    }

    /** One multipart part: headers + data + CRLF. */
    private static byte[] part(String name, String filename, String type, byte[] data) throws IOException {
        StringBuilder h = new StringBuilder();
        h.append("--").append(BOUNDARY).append("\r\n");
        h.append("Content-Disposition: form-data; name=\"").append(name).append('"');
        if (filename != null) h.append("; filename=\"").append(filename).append('"');
        h.append("\r\nContent-Type: ").append(type).append("\r\n\r\n");
        ByteArrayOutputStream b = new ByteArrayOutputStream();
        byte[] hb = h.toString().getBytes(StandardCharsets.UTF_8);
        b.write(hb, 0, hb.length);
        b.write(data, 0, data.length);
        b.write('\r');
        b.write('\n');
        return b.toByteArray();
    }
}
