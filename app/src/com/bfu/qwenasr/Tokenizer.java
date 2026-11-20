package com.bfu.qwenasr;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Iterator;
import java.util.List;

import org.json.JSONException;
import org.json.JSONObject;

/** Byte-level BPE detokenizer for the Qwen3 vocabulary (vocab.json). Decoding only. */
final class Tokenizer {
    static final int ASR_TEXT = 151704;     // <asr_text>
    static final int LANGUAGE = 11528;      // "language"
    static final int FIRST_SPECIAL = 151643; // <|endoftext|> and above are special tokens

    private final String[] idToTok = new String[152064];
    private final int[] unicodeToByte = new int[512];

    Tokenizer(File vocabJson) throws IOException {
        // GPT-2 bytes_to_unicode, inverted.
        int n = 0;
        for (int b = 0; b < 256; b++) {
            boolean direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            int cp = direct ? b : 256 + n++;
            unicodeToByte[cp] = b;
        }
        try {
            JSONObject o = new JSONObject(new String(Files.readAllBytes(vocabJson.toPath()), StandardCharsets.UTF_8));
            for (Iterator<String> it = o.keys(); it.hasNext(); ) {
                String k = it.next();
                int id = o.getInt(k);
                if (id >= 0 && id < idToTok.length) idToTok[id] = k;
            }
        } catch (JSONException e) {
            throw new IOException("bad vocab.json: " + e.getMessage());
        }
    }

    String decode(List<Integer> ids, int from, int to) {
        if (to < 0) to = ids.size();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        for (int i = from; i < to; i++) {
            int id = ids.get(i);
            if (id >= FIRST_SPECIAL || id < 0 || idToTok[id] == null) continue;
            String t = idToTok[id];
            for (int j = 0; j < t.length(); j++) {
                char c = t.charAt(j);
                bytes.write(c < unicodeToByte.length ? unicodeToByte[c] : '?');
            }
        }
        return new String(bytes.toByteArray(), StandardCharsets.UTF_8);
    }

    /** Max copies of a repeated 1-4 token unit kept in a row (the int4 model can loop on one word). */
    static final int MAX_REPEATS = 2;

    /**
     * Removes repetition loops: wherever a unit of 1..4 tokens occurs more than 3 times in a row,
     * keeps MAX_REPEATS copies. Normal speech ("yeah, yeah") is untouched.
     */
    static java.util.List<Integer> collapseRepeats(List<Integer> in) {
        java.util.List<Integer> out = new java.util.ArrayList<>(in);
        boolean changed = true;
        while (changed) {
            changed = false;
            outer:
            for (int unit = 1; unit <= 4; unit++) {
                for (int start = 0; start + unit * 4 <= out.size(); start++) {
                    int reps = 1;
                    while (start + (reps + 1) * unit <= out.size()
                            && out.subList(start + reps * unit, start + (reps + 1) * unit)
                                  .equals(out.subList(start, start + unit))) reps++;
                    if (reps > 3) {
                        out.subList(start + MAX_REPEATS * unit, start + reps * unit).clear();
                        changed = true;
                        break outer;
                    }
                }
            }
        }
        return out;
    }

    /** Output is "language X<asr_text>text<|im_end|>". */
    void parse(List<Integer> ids, Engine.Result r) {
        int end = ids.size();
        for (int i = 0; i < ids.size(); i++) {
            int id = ids.get(i);
            if (id == 151645 || id == 151643) { end = i; break; }
        }
        int mark = ids.indexOf(ASR_TEXT);
        if (mark >= 0 && mark < end) {
            r.language = decode(ids, 0, mark).replaceFirst("^language\\s*", "").trim();
            List<Integer> t = ids.subList(mark + 1, end);
            r.text = decode(r.hitLimit ? collapseRepeats(t) : t, 0, -1).trim();
        } else {
            List<Integer> t = ids.subList(0, end);
            r.text = decode(r.hitLimit ? collapseRepeats(t) : t, 0, -1).trim();
        }
    }
}
