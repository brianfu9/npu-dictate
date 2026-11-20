package com.bfu.qwenasr;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Spoken-phrase to fixed-text rules from personal/commands.txt ("slash clear => /clear"), applied
 * on the phone to every live and final transcript segment. Pure JVM (no android imports) so it can
 * be unit-tested on the host: selfTest() returns null when all examples hold.
 */
public final class Commands {
    private static final class Rule {
        final Pattern pattern;
        final String replacement;
        Rule(Pattern p, String r) { pattern = p; replacement = r; }
    }

    /** The whole text is now only commands/mentions: "/clear", "@claude /new-session", "/clear." ... */
    private static final Pattern COMMANDS_ONLY =
            Pattern.compile("^\\s*[/@][\\w\\-]+(\\s+[/@][\\w\\-]+)*\\s*[.!?]?\\s*$");

    private final List<Rule> rules = new ArrayList<>();

    private Commands() { }

    /** The rule-less instance (no commands.txt, or one with no rules). */
    public static Commands empty() { return new Commands(); }

    /**
     * One rule per line: `spoken phrase => replacement`. Blank lines and lines starting with # are
     * ignored; a phrase starting with ^ matches only at the start of a segment (after optional
     * spaces). Rules apply in file order, so list longer phrases first. Null/empty text = no rules.
     */
    public static Commands parse(String text) {
        Commands c = new Commands();
        if (text == null || text.isEmpty()) return c;
        for (String line : text.split("\n")) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            int i = line.indexOf("=>");
            if (i < 0) continue;
            String phrase = line.substring(0, i).trim();
            String replacement = line.substring(i + 2).trim();
            if (phrase.isEmpty()) continue;
            boolean anchored = phrase.startsWith("^");
            if (anchored) phrase = phrase.substring(1).trim();
            String[] words = phrase.split("\\s+");
            StringBuilder re = new StringBuilder(anchored ? "^\\s*" : "(?<![\\p{L}\\p{N}/@])");
            for (int k = 0; k < words.length; k++) {
                if (k > 0) re.append("[\\s,.\\-]*");   // the recognizer may put commas, full stops or hyphens between spoken words
                re.append(Pattern.quote(words[k]));
            }
            re.append("(?![\\p{L}\\p{N}])");
            c.rules.add(new Rule(Pattern.compile(re.toString(), Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE),
                    Matcher.quoteReplacement(replacement)));
        }
        return c;
    }

    /** Whether no rules were parsed. */
    public boolean isEmpty() { return rules.isEmpty(); }

    /**
     * Replaces every rule's phrase, in file order. When the result is only commands/mentions, a
     * trailing . or ! (not ?) is removed and the text trimmed; for ^ rules the matched leading
     * spaces are not kept.
     */
    public String apply(String s) {
        if (s == null || rules.isEmpty()) return s;
        for (Rule r : rules) s = r.pattern.matcher(s).replaceAll(r.replacement);
        if (COMMANDS_ONLY.matcher(s).matches()) {
            s = s.trim();
            int n = s.length();
            if (n > 0 && (s.charAt(n - 1) == '.' || s.charAt(n - 1) == '!')) s = s.substring(0, n - 1);
        }
        return s;
    }

    /** Returns null when all examples hold, else a description of the first failure. Not called by the app. */
    public static String selfTest() {
        Commands c = parse("slash clear => /clear\nslash new session => /new-session\n^at claude => @claude");
        String[][] cases = {
                {"Slash clear.", "/clear"},
                {"slash, new session", "/new-session"},
                {"Slash new-session.", "/new-session"},
                {"At Claude, review this diff.", "@claude, review this diff."},
                {"I looked at Claude's answer.", "I looked at Claude's answer."},
                {"run slash clear now", "run /clear now"},
                {"/clear", "/clear"},
        };
        for (String[] tc : cases) {
            String got = c.apply(tc[0]);
            if (!got.equals(tc[1])) return "apply(\"" + tc[0] + "\") = \"" + got + "\", want \"" + tc[1] + "\"";
        }
        return null;
    }
}
