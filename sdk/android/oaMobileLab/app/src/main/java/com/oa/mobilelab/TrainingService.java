package com.oa.mobilelab;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.os.IBinder;
import android.os.PowerManager;
import android.os.ResultReceiver;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public final class TrainingService extends Service {
	// Checkpoints encode token IDs, architecture topology, and optimizer state.
	// Bump this contract whenever any of those or the canonical corpus changes;
	// silently resuming an older tutorial checkpoint can show a low loss while
	// decoding with different semantics.
	private static final String NLP_CHECKPOINT_CONTRACT = "v2";
	static final String ACTION_CANCEL = "com.oa.mobilelab.action.CANCEL_TRAINING";
	static final String EXTRA_RECEIVER = "com.oa.mobilelab.TRAINING_RECEIVER";
	static final String EXTRA_REPORT = "com.oa.mobilelab.TRAINING_REPORT";
	static final String EXTRA_ARCHITECTURE = "com.oa.mobilelab.TRAINING_ARCHITECTURE";
	static final String EXTRA_TOKENIZER = "com.oa.mobilelab.TRAINING_TOKENIZER";
	static final String EXTRA_STEPS = "com.oa.mobilelab.TRAINING_STEPS";
	static final String EXTRA_BATCH = "com.oa.mobilelab.TRAINING_BATCH";
	static final String EXTRA_RESUME = "com.oa.mobilelab.TRAINING_RESUME";
	static final String EXTRA_STEP = "com.oa.mobilelab.TRAINING_STEP";
	static final String EXTRA_TOTAL = "com.oa.mobilelab.TRAINING_TOTAL";
	static final String EXTRA_LOSS = "com.oa.mobilelab.TRAINING_LOSS";
	static final String EXTRA_GPU_MS = "com.oa.mobilelab.TRAINING_GPU_MS";
	static final String EXTRA_WALL_MS = "com.oa.mobilelab.TRAINING_WALL_MS";

	static final int RESULT_PROGRESS = 10;
	static final int RESULT_COMPLETE = 11;
	static final int RESULT_ERROR = 12;

	private static final String TAG = "OA";
	private static final String REPORT_TAG = "OaMobileReport";
	private static final String CHANNEL_ID = "oa_mobile_training";
	private static final int NOTIFICATION_ID = 0x0A17;
	private volatile ResultReceiver receiver;
	private volatile Thread worker;
	private PowerManager.WakeLock trainingWakeLock;
	private volatile String activeArchitecture = "model";
	private volatile String activeTokenizer = "OA";
	private volatile long lastNotificationMs;

	static {
		System.loadLibrary("oa_mobile_lab");
	}

	private static native String nativeRunTraining(
		String driverDirectory,
		String nativeLibraryDirectory,
		String cacheDirectory,
		String checkpointPath,
		String architecture,
		String tokenizer,
		int steps,
		int batchSize,
		boolean resume,
		TrainingService callback);

	private static native void nativeRequestCancel();

	@Override
	public IBinder onBind(Intent intent) {
		return null;
	}

	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {
		if (intent != null && ACTION_CANCEL.equals(intent.getAction())) {
			nativeRequestCancel();
			return START_NOT_STICKY;
		}
		if (intent == null) {
			stopSelf(startId);
			return START_NOT_STICKY;
		}

		receiver = intent.getParcelableExtra(EXTRA_RECEIVER, ResultReceiver.class);
		if (worker != null && worker.isAlive()) {
			return START_NOT_STICKY;
		}

		int steps = Math.max(1, intent.getIntExtra(EXTRA_STEPS, 50));
		int batch = Math.max(1, intent.getIntExtra(EXTRA_BATCH, 8));
		boolean resume = intent.getBooleanExtra(EXTRA_RESUME, false);
		String architecture = sanitizeId(
			intent.getStringExtra(EXTRA_ARCHITECTURE), "gru");
		String tokenizer = sanitizeId(intent.getStringExtra(EXTRA_TOKENIZER), "byte");
		activeArchitecture = architecture;
		activeTokenizer = tokenizer;
		ensureNotificationChannel();
		startForeground(
			NOTIFICATION_ID,
			buildNotification("Initializing Vulkan training", 0, steps));
		acquireTrainingWakeLock();
		worker = new Thread(
			() -> runTraining(
				startId, architecture, tokenizer, steps, batch, resume),
			"oa-nlp-training");
		worker.start();
		return START_NOT_STICKY;
	}

	public void onNativeProgress(
		int step,
		int total,
		float loss,
		double gpuMs,
		double wallMs) {
		ResultReceiver target = receiver;
		if (target == null) {
			updateNotification(step, total, loss);
			return;
		}
		Bundle progress = new Bundle();
		progress.putInt(EXTRA_STEP, step);
		progress.putInt(EXTRA_TOTAL, total);
		progress.putFloat(EXTRA_LOSS, loss);
		progress.putDouble(EXTRA_GPU_MS, gpuMs);
		progress.putDouble(EXTRA_WALL_MS, wallMs);
		target.send(RESULT_PROGRESS, progress);
		updateNotification(step, total, loss);
		if (step == 1 || step == total || step % 25 == 0) {
			Log.i(TAG, String.format(
				java.util.Locale.US,
				"%s/%s progress %d/%d loss %.4f gpu %.2f ms wall %.2f ms",
				activeTokenizer, activeArchitecture,
				step, total, loss, gpuMs, wallMs));
		}
	}

	private void runTraining(
		int startId,
		String architecture,
		String tokenizer,
		int steps,
		int batch,
		boolean resume) {
		Bundle result = new Bundle();
		int code = RESULT_COMPLETE;
		String report;
		try {
			File driverDirectory = DriverAssets.prepareTurnip(this);
			File checkpointDirectory = new File(getFilesDir(), "checkpoints");
			if (!checkpointDirectory.exists() && !checkpointDirectory.mkdirs()) {
				throw new IllegalStateException(
					"Could not create " + checkpointDirectory);
			}
			String runId = tokenizer + "-" + architecture;
			File checkpoint = new File(
				checkpointDirectory, runId + "-" + NLP_CHECKPOINT_CONTRACT + ".oam");
			report = nativeRunTraining(
				driverDirectory.getAbsolutePath(),
				getApplicationInfo().nativeLibraryDir,
				getCacheDir().getAbsolutePath(),
				checkpoint.getAbsolutePath(),
				architecture,
				tokenizer,
				steps,
				batch,
				resume && checkpoint.isFile(),
				this);
			if (report.contains("Fatal:")
				|| report.contains("GenerationQuality: FAIL")) {
				code = RESULT_ERROR;
			}
		} catch (Throwable error) {
			code = RESULT_ERROR;
			report = "OaMobileLab(\n  Tokenizer: " + tokenizer
				+ "\n  Architecture: " + architecture
				+ "\n  Fatal: " + error.getClass().getSimpleName()
				+ ": " + error.getMessage() + "\n)\n";
			Log.e(TAG, "Android ML training failed", error);
		}
		String runId = tokenizer + "-" + architecture;
		saveReport(runId, report);
		logReportForAdb(runId, report);

		result.putString(EXTRA_REPORT, report);
		ResultReceiver target = receiver;
		if (target != null) {
			target.send(code, result);
		}
		worker = null;
		releaseTrainingWakeLock();
		stopForeground(STOP_FOREGROUND_REMOVE);
		stopSelf(startId);
	}

	private void ensureNotificationChannel() {
		NotificationManager manager = getSystemService(NotificationManager.class);
		NotificationChannel channel = new NotificationChannel(
			CHANNEL_ID, "OA on-device training", NotificationManager.IMPORTANCE_LOW);
		channel.setDescription("Progress for Vulkan-native model training");
		channel.enableLights(false);
		channel.setLightColor(Color.WHITE);
		channel.setSound(null, null);
		manager.createNotificationChannel(channel);
	}

	private Notification buildNotification(String status, int step, int total) {
		Intent openIntent = new Intent(this, MainActivity.class);
		PendingIntent open = PendingIntent.getActivity(
			this, 0, openIntent,
			PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
		Intent cancelIntent = new Intent(this, TrainingService.class)
			.setAction(ACTION_CANCEL);
		PendingIntent cancel = PendingIntent.getService(
			this, 1, cancelIntent,
			PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
		Notification.Builder builder = new Notification.Builder(this, CHANNEL_ID)
			.setSmallIcon(android.R.drawable.stat_sys_download)
			.setContentTitle(activeTokenizer.toUpperCase() + " / "
				+ activeArchitecture.toUpperCase())
			.setContentText(status)
			.setContentIntent(open)
			.setOngoing(true)
			.setOnlyAlertOnce(true)
			.addAction(new Notification.Action.Builder(
				0, "Stop and checkpoint", cancel).build());
		if (total > 0) {
			builder.setProgress(total, Math.min(step, total), false);
		}
		return builder.build();
	}

	private void updateNotification(int step, int total, float loss) {
		long now = android.os.SystemClock.elapsedRealtime();
		if (step < total && now - lastNotificationMs < 1000L) {
			return;
		}
		lastNotificationMs = now;
		String status = String.format(
			java.util.Locale.US, "%d/%d · loss %.4f", step, total, loss);
		getSystemService(NotificationManager.class).notify(
			NOTIFICATION_ID, buildNotification(status, step, total));
	}

	private void saveReport(String runId, String report) {
		try {
			File reports = new File(getFilesDir(), "reports");
			if (!reports.exists() && !reports.mkdirs()) {
				return;
			}
			try (FileOutputStream output = new FileOutputStream(
				new File(reports, runId + "-training.txt"))) {
				output.write(report.getBytes(StandardCharsets.UTF_8));
			}
		} catch (Throwable error) {
			Log.w(TAG, "Could not persist training report", error);
		}
	}

	private void logReportForAdb(String runId, String report) {
		Log.i(REPORT_TAG, "OA_REPORT_BEGIN " + runId);
		for (String line : report.split("\\n", -1)) {
			Log.i(REPORT_TAG, "OA_REPORT " + runId + " " + line);
		}
		Log.i(REPORT_TAG, "OA_REPORT_END " + runId);
	}

	private synchronized void acquireTrainingWakeLock() {
		if (trainingWakeLock == null) {
			PowerManager power = getSystemService(PowerManager.class);
			trainingWakeLock = power.newWakeLock(
				PowerManager.PARTIAL_WAKE_LOCK,
				getPackageName() + ":NlpTraining");
			trainingWakeLock.setReferenceCounted(false);
		}
		if (!trainingWakeLock.isHeld()) {
			trainingWakeLock.acquire();
		}
	}

	private synchronized void releaseTrainingWakeLock() {
		if (trainingWakeLock != null && trainingWakeLock.isHeld()) {
			trainingWakeLock.release();
		}
	}

	private static String sanitizeId(String value, String fallback) {
		if (value == null || !value.matches("[a-z0-9]+")) {
			return fallback;
		}
		return value;
	}

	@Override
	public void onDestroy() {
		if (worker != null && worker.isAlive()) {
			nativeRequestCancel();
		}
		if (worker == null || !worker.isAlive()) {
			releaseTrainingWakeLock();
		}
		super.onDestroy();
	}
}
