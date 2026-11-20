#!/usr/bin/env bash
# Pulls the voice keyboard's session recordings (WAV + JSON) from the phone adb picks
# (set ANDROID_SERIAL for a specific one) into a local directory.
#
#   scripts/pull-sessions.sh <dest dir> [--delete]
#
# A session is pulled when its .json or .wav is missing locally or has a different size.
# With --delete, sessions that are complete and the same byte size on both sides after the
# pull are deleted from the phone; anything else stays for the next run.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 <dest dir> [--delete]" >&2
  exit 2
fi
DEST=$1
DELETE=0
if [ "${2:-}" = "--delete" ]; then DELETE=1; fi

REMOTE=/sdcard/Android/data/com.bfu.qwenasr/files/sessions
mkdir -p "$DEST"

lsize() {   # local file size in bytes ("" when missing)
  if [ -f "$1" ]; then wc -c < "$1" | tr -d '[:space:]'; else echo ""; fi
}
rsize() {   # remote file size in bytes ("" when missing)
  adb shell "stat -c %s $REMOTE/$1 2>/dev/null || true" | tr -d '[:space:]\r'
}

remote_list=$(adb shell "ls $REMOTE 2>/dev/null || true")

pulled=0
present=0
deleted=0
for j in $remote_list; do
  case "$j" in *.json) ;; *) continue ;; esac
  id=${j%.json}
  rs_j=$(rsize "$j")
  rs_w=$(rsize "$id.wav")
  if [ -n "$rs_j" ] && [ "$(lsize "$DEST/$id.json")" = "$rs_j" ] && [ "$(lsize "$DEST/$id.wav")" = "$rs_w" ]; then
    present=$((present + 1))
    continue
  fi
  adb pull "$REMOTE/$j" "$DEST/"
  if [ -n "$rs_w" ]; then
    adb pull "$REMOTE/$id.wav" "$DEST/"
  fi
  pulled=$((pulled + 1))
done

if [ $DELETE -eq 1 ]; then
  for j in $remote_list; do
    case "$j" in *.json) ;; *) continue ;; esac
    id=${j%.json}
    rs_j=$(rsize "$j")
    rs_w=$(rsize "$id.wav")
    if [ -n "$rs_j" ] && [ -n "$rs_w" ] \
       && [ "$(lsize "$DEST/$id.json")" = "$rs_j" ] && [ "$(lsize "$DEST/$id.wav")" = "$rs_w" ]; then
      adb shell "rm $REMOTE/$j $REMOTE/$id.wav"
      deleted=$((deleted + 1))
    fi
  done
fi

echo "pulled $pulled, already present $present, deleted $deleted"
