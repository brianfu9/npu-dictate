package com.k2fsa.sherpa.onnx;

/** Android: the JNI library ships in the APK's nativeLibraryDir. */
public class LibraryLoader {
    private static boolean loaded;

    static synchronized void maybeLoad() {
        if (!loaded) {
            System.loadLibrary("sherpa-onnx-jni");
            loaded = true;
        }
    }
}
