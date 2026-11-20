#!/usr/bin/env python3
"""Mine domain terms from the user's own typed messages (Claude Code transcripts + opencode DB).
Counts words that are not plain dictionary words: mixed case (LiteRT), digits (Qwen3), or not in
/usr/share/dict/words (opencode, llama.cpp). Writes a ranked TSV: term, count, sessions."""
import collections, glob, json, os, re, sqlite3, sys

DICT = set(w.strip().lower() for w in open('/usr/share/dict/words', encoding='utf-8', errors='ignore'))
WORD = re.compile(r"(?<![\w/.\-])([A-Za-z][A-Za-z0-9]*(?:[.\-+][A-Za-z0-9]+)*)(?![\w/])")
SKIP = re.compile(r"<system-reminder>.*?</system-reminder>|<command-[a-z]+>.*?</command-[a-z]+>|"
                  r"<local-command-[a-z]+>.*?</local-command-[a-z]+>|```.*?```|https?://\S+|\S+/\S+|\S+@\S+", re.S)

def texts_claude():
    for f in glob.glob(os.path.expanduser('~/.claude/projects/*/*.jsonl')):
        sid = os.path.basename(f)
        for line in open(f, encoding='utf-8', errors='ignore'):
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get('type') != 'user' or o.get('isMeta') or o.get('isCompactSummary'):
                continue
            c = (o.get('message') or {}).get('content')
            if isinstance(c, str):
                yield sid, c
            elif isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get('type') == 'text':
                        yield sid, p.get('text', '')

def texts_opencode():
    db = os.path.expanduser('~/.local/share/opencode/opencode.db')
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    q = """select m.session_id, p.data from part p join message m on m.id = p.message_id
           where json_extract(m.data,'$.role')='user' and json_extract(p.data,'$.type')='text'"""
    for sid, d in con.execute(q):
        t = json.loads(d).get('text', '')
        if t.startswith('TASK:') or 'VERIFY:' in t or json.loads(d).get('synthetic'):
            continue   # delegate.sh briefs, not typed by the user
        yield sid, t

cnt, sess = collections.Counter(), collections.defaultdict(set)
for src in (texts_claude(), texts_opencode()):
    for sid, t in src:
        if 'This session is being continued' in t or 'Caveat:' in t[:40]:
            continue
        t = SKIP.sub(' ', t)
        for w in WORD.findall(t):
            if len(w) < 3 or len(w) > 24:
                continue
            lw = w.lower()
            special = (w != w.lower() and w != w.capitalize() and w != w.upper()) or any(ch.isdigit() for ch in w) \
                      or lw.strip('.-+') not in DICT
            if w.isupper() and len(w) <= 4 and lw in DICT:
                special = False
            if not special or not re.search('[aeiouy]', lw):
                continue
            cnt[w] += 1
            sess[w].add(sid)
for w, n in cnt.most_common():
    if n < 3 or len(sess[w]) < 2:
        continue
    print(f"{w}\t{n}\t{len(sess[w])}")
