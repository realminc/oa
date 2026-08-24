package com.oa.mobilelab;

import android.app.Service;
import android.content.Intent;
import android.os.Bundle;
import android.os.IBinder;
import android.os.ResultReceiver;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

abstract class ProbeService extends Service {
    static final String EXTRA_RECEIVER = "com.oa.mobilelab.RECEIVER";
    static final String EXTRA_REPORT = "com.oa.mobilelab.REPORT";
    static final String EXTRA_SOURCE = "com.oa.mobilelab.SOURCE";
    static final int RESULT_OK = 1;
    static final int RESULT_ERROR = 2;
    private static final String TAG = "OA";

    static {
        System.loadLibrary("oa_mobile_lab");
    }

    protected abstract String driverSource();

    private static native String nativeRunProbe(
            String driverSource,
            String driverDirectory,
            String nativeLibraryDirectory,
            String cacheDirectory);

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        ResultReceiver receiver = intent == null
                ? null
                : intent.getParcelableExtra(EXTRA_RECEIVER, ResultReceiver.class);
        String source = driverSource();

        Thread worker = new Thread(() -> {
            Bundle result = new Bundle();
            int resultCode = RESULT_OK;
            String report;
            try {
                File driverDirectory = source.equals("turnip")
                        ? DriverAssets.prepareTurnip(this)
                        : new File(getFilesDir(), "drivers/turnip-26.1.4");
                report = nativeRunProbe(
                        source,
                        driverDirectory.getAbsolutePath(),
                        getApplicationInfo().nativeLibraryDir,
                        getCacheDir().getAbsolutePath());
                saveReport(source, report);
            } catch (Throwable error) {
                resultCode = RESULT_ERROR;
                report = "OA MOBILE LAB\n\nProbe failed before Vulkan initialization.\n\n"
                        + error.getClass().getSimpleName() + ": " + error.getMessage();
                Log.e(TAG, "Android Vulkan probe failed", error);
            }

            result.putString(EXTRA_SOURCE, source);
            result.putString(EXTRA_REPORT, report);
            if (receiver != null) {
                receiver.send(resultCode, result);
            }
            stopSelf(startId);
        }, "oa-vulkan-probe-" + source);
        worker.start();
        return START_NOT_STICKY;
    }

    private void saveReport(String source, String report) throws IOException {
        File reports = new File(getFilesDir(), "reports");
        if (!reports.exists() && !reports.mkdirs()) {
            throw new IOException("Could not create " + reports);
        }
        try (FileOutputStream output = new FileOutputStream(
                new File(reports, "vulkan-" + source + ".txt"))) {
            output.write(report.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }
    }
}
