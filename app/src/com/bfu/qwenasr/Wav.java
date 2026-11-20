package com.bfu.qwenasr;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;

/** 16-bit mono PCM WAV read/write. */
final class Wav {
    static void write(File f, byte[] pcm, int rate) throws IOException {
        ByteBuffer h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN);
        h.put(new byte[]{'R', 'I', 'F', 'F'}).putInt(36 + pcm.length)
         .put(new byte[]{'W', 'A', 'V', 'E', 'f', 'm', 't', ' '}).putInt(16)
         .putShort((short) 1).putShort((short) 1).putInt(rate).putInt(rate * 2)
         .putShort((short) 2).putShort((short) 16)
         .put(new byte[]{'d', 'a', 't', 'a'}).putInt(pcm.length);
        try (FileOutputStream o = new FileOutputStream(f)) {
            o.write(h.array());
            o.write(pcm);
        }
    }

    /** Reads the samples of a 16-bit mono WAV (finds the "data" chunk). */
    static short[] read(File f) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(f.toPath())).order(ByteOrder.LITTLE_ENDIAN);
        int p = 12;
        while (p + 8 <= b.limit()) {
            int id = b.getInt(p), size = b.getInt(p + 4);
            if (id == 0x61746164) {   // "data"
                int len = Math.min(size, b.limit() - p - 8) / 2;
                short[] s = new short[len];
                b.position(p + 8);
                b.asShortBuffer().get(s);
                return s;
            }
            p += 8 + size + (size & 1);
        }
        throw new IOException("no data chunk in " + f);
    }
}
