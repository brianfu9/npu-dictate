package com.bfu.qwenasr;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Text clean-up of a finished transcript by the polish server (any OpenAI-compatible chat
 * endpoint; url and model come from polish.conf). The model cannot hear the audio: it only
 * fixes term spelling, punctuation, split sentences and repetition loops. Any failure (server
 * unreachable, model not loaded, odd result) keeps the on-phone text.
 */
final class Polisher {
    private final String systemPrompt;
    private final String url;
    private final String model;
    private final List<String> terms = new ArrayList<>();
    private final List<Pattern> termPatterns = new ArrayList<>();

    /** promptTemplate contains {TERMS}; termList is the comma-separated word list. */
    Polisher(String promptTemplate, String termList, String url, String model) {
        this.url = url;
        this.model = model;
        for (String t : termList.split(",")) {
            t = t.trim();
            if (t.isEmpty()) continue;
            terms.add(t);
            // Letters/digits groups of the term, joined by an optional space or hyphen, case-insensitive.
            String[] parts = t.split("[\\s\\-.]+");
            StringBuilder re = new StringBuilder("(?<![\\p{L}\\p{N}])");
            for (int i = 0; i < parts.length; i++) {
                if (i > 0) re.append("[\\s\\-.]?");
                re.append(Pattern.quote(parts[i]));
            }
            re.append("(?![\\p{L}\\p{N}])");
            termPatterns.add(Pattern.compile(re.toString(), Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE));
        }
        systemPrompt = promptTemplate.replace("{TERMS}", String.join(", ", terms));
    }

    /** Writes every case/space/hyphen variant of a listed term exactly as listed. */
    String normalizeTerms(String s) {
        for (int i = 0; i < terms.size(); i++) {
            s = termPatterns.get(i).matcher(s).replaceAll(Matcher.quoteReplacement(terms.get(i)));
        }
        return s;
    }

    private static String letters(String s) {
        return s.toLowerCase(java.util.Locale.ROOT).replaceAll("[^\\p{L}\\p{N}]", "");
    }

    /** True when a short line (<= 3 words) now ends the previous polished line. */
    static boolean absorbed(String prevPolished, String raw) {
        String w = letters(raw);
        return !w.isEmpty() && raw.trim().split("\\s+").length <= 3 && letters(prevPolished).endsWith(w);
    }

    /** Result of a polish: one line per input segment, and how many lines kept the original text. */
    static final class Result {
        List<String> lines = new ArrayList<>();
        int kept;
    }

    private static final Pattern LINE = Pattern.compile("^\\[(\\d+)\\]\\s?(.*)$", Pattern.MULTILINE);

    /**
     * Blocking; call from a worker thread. Sends the segments as numbered lines "[n] text" and
     * expects the same numbered lines back. Each line is checked on its own: a missing line, or one
     * whose length changed too much, keeps the phone text. Throws only when the polish server
     * is unreachable.
     */
    /** A model that answers one system + user message (the polish server over HTTP). */
    interface Backend {
        String complete(String system, String user, int maxTokens) throws IOException;
    }

    static String userText(List<String> segments) {
        StringBuilder user = new StringBuilder();
        for (int i = 0; i < segments.size(); i++) {
            if (i > 0) user.append('\n');
            user.append('[').append(i + 1).append("] ").append(segments.get(i).replace('\n', ' '));
        }
        return user.toString();
    }

    Result polish(List<String> segments) throws IOException {
        return polish(segments, this::chatComplete, model);
    }

    /** Same numbered-line protocol and per-line guards for any backend. */
    Result polish(List<String> segments, Backend backend, String label) throws IOException {
        long t0 = System.nanoTime();
        String user = userText(segments);
        String out = backend.complete(systemPrompt, user, 200 + user.length() / 3);
        Result r = parse(segments, out);
        Log.i("QwenASR", String.format("polish (%s) %d lines, %d kept original, %.2f s", label, segments.size(), r.kept,
                (System.nanoTime() - t0) / 1e9));
        return r;
    }

    private String chatComplete(String system, String userText, int maxTokens) throws IOException {
        StringBuilder user = new StringBuilder(userText);
        JSONObject body = new JSONObject();
        try {
            body.put("model", model);
            body.put("temperature", 0);
            body.put("max_tokens", maxTokens);
            body.put("chat_template_kwargs", new JSONObject().put("enable_thinking", false));
            JSONArray msgs = new JSONArray();
            msgs.put(new JSONObject().put("role", "system").put("content", system));
            msgs.put(new JSONObject().put("role", "user").put("content", user.toString()));
            body.put("messages", msgs);
        } catch (Exception e) {
            throw new IOException(e);
        }
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(3000);
        c.setReadTimeout(20000);
        c.setRequestMethod("POST");
        c.setDoOutput(true);
        c.setRequestProperty("Content-Type", "application/json");
        try (OutputStream o = c.getOutputStream()) {
            o.write(body.toString().getBytes(StandardCharsets.UTF_8));
        }
        int code = c.getResponseCode();
        InputStream in = code < 400 ? c.getInputStream() : c.getErrorStream();
        ByteArrayOutputStream buf = new ByteArrayOutputStream();
        if (in != null) {
            byte[] b = new byte[8192];
            int k;
            while ((k = in.read(b)) > 0) buf.write(b, 0, k);
            in.close();
        }
        String resp = buf.toString("UTF-8");
        if (code != 200) throw new IOException(model + " HTTP " + code + ": " + resp.substring(0, Math.min(200, resp.length())));
        try {
            return new JSONObject(resp).getJSONArray("choices").getJSONObject(0)
                    .getJSONObject("message").getString("content");
        } catch (Exception e) {
            throw new IOException("bad response from the polish server: " + e.getMessage());
        }
    }

    /** Model output -> one line per segment, with the per-line guards (missing / length ratio) and term pass. */
    Result parse(List<String> segments, String out) {
        String[] got = new String[segments.size()];
        Matcher m = LINE.matcher(out);
        while (m.find()) {
            int n = Integer.parseInt(m.group(1)) - 1;
            if (n >= 0 && n < got.length && got[n] == null) got[n] = m.group(2).trim();
        }
        Result r = new Result();
        for (int i = 0; i < got.length; i++) {
            String raw = segments.get(i), g = got[i];
            double ratio = g == null ? 0 : g.length() / (double) Math.max(1, raw.length());
            if (g != null && g.isEmpty() && i > 0 && absorbed(r.lines.get(i - 1), raw)) {
                r.lines.add("");   // the model moved this short piece (e.g. "back" of "fallback") into the previous line
                continue;
            }
            if (g == null || g.isEmpty() || ratio < 0.6 || ratio > 1.4) {
                r.lines.add(raw);
                r.kept++;
            } else {
                r.lines.add(normalizeTerms(g));
            }
        }
        return r;
    }
}
