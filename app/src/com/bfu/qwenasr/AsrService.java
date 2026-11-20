package com.bfu.qwenasr;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.IBinder;
import android.util.Log;

/**
 * Foreground service held while a transcription session runs. Without it, Android (and OPPO's
 * freezer) freezes the app's whole UID as soon as the screen locks or the app leaves the
 * foreground: recording, VAD and both model servers stop.
 */
public class AsrService extends Service {
    private static final String CH = "asr";

    private static int holds;   // model loading and sessions each hold the service

    static synchronized void begin(Context c, boolean mic) {
        holds++;
        Intent i = new Intent(c.getApplicationContext(), AsrService.class).putExtra("mic", mic);
        try {
            c.getApplicationContext().startForegroundService(i);
        } catch (RuntimeException e) {
            Log.w("QwenASR", "foreground service refused: " + e);
            holds--;
        }
    }

    static synchronized void end(Context c) {
        if (holds > 0) holds--;
        if (holds == 0) c.getApplicationContext().stopService(new Intent(c.getApplicationContext(), AsrService.class));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CH, "Transcription", NotificationManager.IMPORTANCE_LOW));
        boolean mic = intent == null || intent.getBooleanExtra("mic", true);
        Notification n = new Notification.Builder(this, CH)
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setContentTitle(mic ? "Recording and transcribing" : "Transcribing")
                .setOngoing(true)
                .build();
        startForeground(1, n, mic ? ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
                                  : ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC);
        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }
}
