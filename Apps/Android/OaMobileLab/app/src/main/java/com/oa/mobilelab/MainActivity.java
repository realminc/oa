package com.oa.mobilelab;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ComponentName;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.util.Locale;

public final class MainActivity extends Activity {
	private static final int BG = Color.rgb(10, 10, 10);
	private static final int PANEL = Color.rgb(26, 26, 26);
	private static final int PANEL_ACTIVE = Color.rgb(34, 34, 34);
	private static final int TEXT = Color.rgb(248, 248, 248);
	private static final int MUTED = Color.rgb(155, 155, 155);
	private static final int OUTLINE = Color.rgb(55, 55, 55);
	private static final int WHITE = Color.WHITE;
	private static final long WELCOME_DURATION_MS = 7000L;

	private static final String[] ARCHITECTURE_IDS = {
		"rnn", "gru", "transformer", "moe", "mamba3"
	};
	private static final String[] ARCHITECTURE_NAMES = {
		"RNN", "GRU", "TRANSFORMER", "MOE", "MAMBA-3"
	};
	private static final String[] ARCHITECTURE_CAPABILITIES = {
		"Verified on Adreno 610 · decomposed mobile scan",
		"Verified on Adreno 610 · decomposed mobile scan",
		"Verified on Adreno 610 · core OA attention kernels",
		"Verified on Adreno 610 · shared-memory atomics",
		"Mobile-bounded backward · global scratch · validation required"
	};
	private static final String[] TOKENIZER_IDS = {"byte", "bpe", "char"};
	private static final String[] TOKENIZER_NAMES = {"BYTE", "BPE", "CHAR"};

	private final Handler handler = new Handler(Looper.getMainLooper());
	private final Button[] architectureButtons = new Button[ARCHITECTURE_IDS.length];
	private final Button[] tokenizerButtons = new Button[TOKENIZER_IDS.length];

	private Typeface sans;
	private Typeface sansSemibold;
	private Typeface mono;
	private Typeface monoMedium;
	private TextView status;
	private TextView report;
	private Button systemButton;
	private Button turnipButton;
	private Button copyButton;
	private Button quickTrainButton;
	private Button suiteTrainButton;
	private Button resumeButton;
	private Button cancelButton;
	private TextView modelTitle;
	private TextView modelDetail;
	private TextView capability;
	private TextView trainingStatus;
	private TextView trainingMetrics;
	private ProgressBar trainingProgress;
	private HorizontalScrollView architectureScroll;
	private FrameLayout welcomeOverlay;
	private boolean trainingRunning;
	private String activeSource = "turnip";
	private int selectedArchitecture = 1;
	private int selectedTokenizer = 0;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		loadTypefaces();
		getWindow().setStatusBarColor(BG);
		getWindow().setNavigationBarColor(BG);
		getWindow().getDecorView().setSystemUiVisibility(0);
		setContentView(buildUi());

		String requestedArchitecture = getIntent().getStringExtra("architecture");
		String requestedTokenizer = getIntent().getStringExtra("tokenizer");
		selectedArchitecture = findId(ARCHITECTURE_IDS, requestedArchitecture, 1);
		selectedTokenizer = findId(TOKENIZER_IDS, requestedTokenizer, 0);
		updateModelSelection();

		if (getIntent().getBooleanExtra("train", false)) {
			dismissWelcome();
			startTraining(
				getIntent().getBooleanExtra("resume", false),
				Math.max(1, getIntent().getIntExtra("steps", 10)),
				Math.max(1, getIntent().getIntExtra("batch", 8)));
		} else {
			String requestedDriver = getIntent().getStringExtra("driver");
			runProbe("system".equals(requestedDriver) ? "system" : "turnip");
			handler.postDelayed(this::dismissWelcome, WELCOME_DURATION_MS);
		}
	}

	private void loadTypefaces() {
		sans = Typeface.createFromAsset(
			getAssets(), "fonts/IBMPlexSans-Regular.ttf");
		sansSemibold = Typeface.createFromAsset(
			getAssets(), "fonts/IBMPlexSans-SemiBold.ttf");
		mono = Typeface.createFromAsset(
			getAssets(), "fonts/IntelOneMono-Regular.ttf");
		monoMedium = Typeface.createFromAsset(
			getAssets(), "fonts/IntelOneMono-Medium.ttf");
	}

	private View buildUi() {
		FrameLayout frame = new FrameLayout(this);
		frame.setBackgroundColor(BG);
		frame.addView(buildContent(), new FrameLayout.LayoutParams(-1, -1));
		welcomeOverlay = buildWelcome();
		frame.addView(welcomeOverlay, new FrameLayout.LayoutParams(-1, -1));
		return frame;
	}

	private View buildContent() {
		ScrollView scroll = new ScrollView(this);
		scroll.setFillViewport(true);
		scroll.setBackgroundColor(BG);

		LinearLayout root = new LinearLayout(this);
		root.setOrientation(LinearLayout.VERTICAL);
		root.setPadding(dp(20), dp(24), dp(20), dp(40));
		scroll.addView(root, new ScrollView.LayoutParams(-1, -2));

		LinearLayout brand = new LinearLayout(this);
		brand.setOrientation(LinearLayout.HORIZONTAL);
		brand.setGravity(Gravity.CENTER_VERTICAL);
		brand.addView(realmMark(), new LinearLayout.LayoutParams(dp(20), dp(20)));
		TextView eyebrow = text("OA / MOBILE LAB", 11, MUTED, true);
		eyebrow.setLetterSpacing(0.18f);
		LinearLayout.LayoutParams eyebrowParams = new LinearLayout.LayoutParams(-2, -2);
		eyebrowParams.setMarginStart(dp(8));
		brand.addView(eyebrow, eyebrowParams);
		root.addView(brand);
		root.addView(text("Train where the data lives.", 31, TEXT, true),
			margin(-1, -2, 0, 8, 0, 0));
		TextView subtitle = text(
			"The OA NLP Suite on Vulkan. Same modules, kernels, autograd, optimizer and metrics—running on your phone.",
			15, MUTED, false);
		subtitle.setLineSpacing(0f, 1.18f);
		root.addView(subtitle, margin(-1, -2, 0, 8, 0, 24));

		LinearLayout statusCard = card();
		root.addView(statusCard, margin(-1, -2, 0, 0, 0, 16));
		statusCard.addView(label("VULKAN / MODERNCOMPUTE"));
		status = text("INITIALIZING", 20, TEXT, true);
		statusCard.addView(status, margin(-1, -2, 0, 8, 0, 12));
		LinearLayout driverRow = new LinearLayout(this);
		driverRow.setOrientation(LinearLayout.HORIZONTAL);
		statusCard.addView(driverRow, new LinearLayout.LayoutParams(-1, dp(44)));
		turnipButton = actionButton("TURNIP", true);
		turnipButton.setOnClickListener(view -> runProbe("turnip"));
		driverRow.addView(turnipButton, new LinearLayout.LayoutParams(0, -1, 1f));
		systemButton = actionButton("SYSTEM", false);
		systemButton.setOnClickListener(view -> runProbe("system"));
		LinearLayout.LayoutParams systemParams = new LinearLayout.LayoutParams(0, -1, 1f);
		systemParams.setMarginStart(dp(8));
		driverRow.addView(systemButton, systemParams);

		root.addView(sectionTitle("MODEL"), margin(-1, -2, 0, 8, 0, 10));
		architectureScroll = new HorizontalScrollView(this);
		architectureScroll.setHorizontalScrollBarEnabled(false);
		architectureScroll.setClipToPadding(false);
		architectureScroll.setPadding(0, 0, dp(20), 0);
		LinearLayout architectureRow = new LinearLayout(this);
		architectureRow.setOrientation(LinearLayout.HORIZONTAL);
		architectureScroll.addView(architectureRow, new HorizontalScrollView.LayoutParams(-2, -2));
		for (int index = 0; index < ARCHITECTURE_IDS.length; ++index) {
			final int selection = index;
			Button button = modelButton(ARCHITECTURE_NAMES[index]);
			button.setOnClickListener(view -> {
				if (!trainingRunning) {
					selectedArchitecture = selection;
					updateModelSelection();
				}
			});
			architectureButtons[index] = button;
			LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(dp(152), dp(72));
			if (index > 0) {
				params.setMarginStart(dp(8));
			}
			architectureRow.addView(button, params);
		}
		root.addView(architectureScroll, margin(-1, -2, 0, 0, 0, 16));

		root.addView(sectionTitle("TOKENIZER"), margin(-1, -2, 0, 0, 0, 10));
		LinearLayout tokenizerRow = new LinearLayout(this);
		tokenizerRow.setOrientation(LinearLayout.HORIZONTAL);
		root.addView(tokenizerRow, margin(-1, dp(44), 0, 0, 0, 16));
		for (int index = 0; index < TOKENIZER_IDS.length; ++index) {
			final int selection = index;
			Button button = actionButton(TOKENIZER_NAMES[index], index == 0);
			button.setOnClickListener(view -> {
				if (!trainingRunning) {
					selectedTokenizer = selection;
					updateModelSelection();
				}
			});
			tokenizerButtons[index] = button;
			LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, -1, 1f);
			if (index > 0) {
				params.setMarginStart(dp(8));
			}
			tokenizerRow.addView(button, params);
		}

		LinearLayout modelCard = card();
		root.addView(modelCard, margin(-1, -2, 0, 0, 0, 16));
		modelCard.addView(label("SELECTED RECIPE"));
		modelTitle = text("BYTE / GRU", 20, TEXT, true);
		modelCard.addView(modelTitle, margin(-1, -2, 0, 8, 0, 0));
		modelDetail = text("", 13, TEXT, false);
		modelDetail.setTypeface(mono);
		modelDetail.setLineSpacing(0f, 1.14f);
		modelCard.addView(modelDetail, margin(-1, -2, 0, 8, 0, 0));
		capability = text("", 12, MUTED, false);
		capability.setLineSpacing(0f, 1.14f);
		modelCard.addView(capability, margin(-1, -2, 0, 9, 0, 16));

		trainingStatus = text("READY", 18, TEXT, true);
		modelCard.addView(trainingStatus);
		trainingMetrics = text("Loss —   GPU —   Wall —", 12, MUTED, false);
		trainingMetrics.setTypeface(mono);
		modelCard.addView(trainingMetrics, margin(-1, -2, 0, 5, 0, 12));
		trainingProgress = new ProgressBar(
			this, null, android.R.attr.progressBarStyleHorizontal);
		trainingProgress.setMax(1000);
		trainingProgress.setProgress(0);
		trainingProgress.setProgressTintList(ColorStateList.valueOf(WHITE));
		trainingProgress.setProgressBackgroundTintList(
			ColorStateList.valueOf(OUTLINE));
		modelCard.addView(trainingProgress, margin(-1, dp(4), 0, 0, 0, 14));

		LinearLayout trainRow = new LinearLayout(this);
		trainRow.setOrientation(LinearLayout.HORIZONTAL);
		modelCard.addView(trainRow, margin(-1, dp(48), 0, 0, 0, 8));
		quickTrainButton = actionButton("SMOKE 50 × 8", true);
		quickTrainButton.setOnClickListener(view -> startTraining(false, 50, 8));
		trainRow.addView(quickTrainButton, new LinearLayout.LayoutParams(0, -1, 1f));
		suiteTrainButton = actionButton("SUITE 300 × 64", false);
		suiteTrainButton.setOnClickListener(view -> startTraining(false, 300, 64));
		LinearLayout.LayoutParams suiteParams = new LinearLayout.LayoutParams(0, -1, 1f);
		suiteParams.setMarginStart(dp(8));
		trainRow.addView(suiteTrainButton, suiteParams);

		LinearLayout checkpointRow = new LinearLayout(this);
		checkpointRow.setOrientation(LinearLayout.HORIZONTAL);
		modelCard.addView(checkpointRow, new LinearLayout.LayoutParams(-1, dp(44)));
		resumeButton = actionButton("RESUME 50", false);
		resumeButton.setOnClickListener(view -> startTraining(true, 50, 8));
		checkpointRow.addView(resumeButton, new LinearLayout.LayoutParams(0, -1, 1f));
		cancelButton = actionButton("STOP", false);
		cancelButton.setEnabled(false);
		cancelButton.setOnClickListener(view -> cancelTraining());
		LinearLayout.LayoutParams cancelParams = new LinearLayout.LayoutParams(0, -1, 1f);
		cancelParams.setMarginStart(dp(8));
		checkpointRow.addView(cancelButton, cancelParams);

		LinearLayout reportCard = card();
		root.addView(reportCard, margin(-1, -2, 0, 0, 0, 12));
		reportCard.addView(label("NATIVE REPORT"));
		report = text("Waiting for the Vulkan probe…", 12, TEXT, false);
		report.setTypeface(mono);
		report.setTextIsSelectable(true);
		report.setLineSpacing(0f, 1.16f);
		reportCard.addView(report, margin(-1, -2, 0, 12, 0, 0));
		copyButton = actionButton("COPY REPORT", false);
		copyButton.setEnabled(false);
		copyButton.setOnClickListener(view -> copyReport());
		root.addView(copyButton, margin(-1, dp(46), 0, 0, 0, 0));
		return scroll;
	}

	private FrameLayout buildWelcome() {
		FrameLayout overlay = new FrameLayout(this);
		overlay.setBackgroundColor(BG);
		ImageView image = new ImageView(this);
		image.setImageResource(R.drawable.office_library);
		image.setScaleType(ImageView.ScaleType.CENTER_CROP);
		overlay.addView(image, new FrameLayout.LayoutParams(-1, -1));

		View scrim = new View(this);
		GradientDrawable scrimGradient = new GradientDrawable(
			GradientDrawable.Orientation.TOP_BOTTOM,
			new int[] {
				Color.TRANSPARENT,
				Color.argb(30, 0, 0, 0),
				Color.argb(210, 0, 0, 0)
			});
		scrim.setBackground(scrimGradient);
		FrameLayout.LayoutParams scrimParams = new FrameLayout.LayoutParams(
			-1, dp(620));
		scrimParams.gravity = Gravity.BOTTOM;
		overlay.addView(scrim, scrimParams);

		LinearLayout copy = new LinearLayout(this);
		copy.setOrientation(LinearLayout.VERTICAL);
		copy.setPadding(dp(24), dp(24), dp(24), dp(42));
		FrameLayout.LayoutParams copyParams = new FrameLayout.LayoutParams(-1, -2);
		copyParams.gravity = Gravity.BOTTOM;
		overlay.addView(copy, copyParams);
		copy.addView(realmMark(), margin(48, 48, 0, 0, 0, 12));
		TextView mark = text("OA / MOBILE LAB", 11, WHITE, true);
		mark.setLetterSpacing(0.2f);
		copy.addView(mark);
		copy.addView(text("Vulkan intelligence.\nIn your hand.", 38, WHITE, true),
			margin(-1, -2, 0, 12, 0, 0));
		copy.addView(text("Preparing Turnip and the OA runtime", 13, WHITE, false));
		TextView gestureHint = text("TAP OR SWIPE TO CONTINUE  ·  7 S", 10, WHITE, true);
		gestureHint.setAlpha(0.72f);
		copy.addView(gestureHint, margin(-1, -2, 0, 10, 0, 0));
		View line = new View(this);
		line.setBackgroundColor(WHITE);
		copy.addView(line, margin(dp(88), dp(2), 0, 16, 0, 0));
		overlay.setOnClickListener(view -> dismissWelcome());
		final float[] gestureStart = new float[2];
		overlay.setOnTouchListener((view, event) -> {
			if (event.getActionMasked() == MotionEvent.ACTION_DOWN) {
				gestureStart[0] = event.getX();
				gestureStart[1] = event.getY();
				return false;
			}
			if (event.getActionMasked() == MotionEvent.ACTION_UP) {
				float distance = Math.abs(event.getX() - gestureStart[0])
					+ Math.abs(event.getY() - gestureStart[1]);
				if (distance >= dp(18)) {
					dismissWelcome();
					return true;
				}
			}
			return false;
		});
		return overlay;
	}

	private void dismissWelcome() {
		if (welcomeOverlay == null || welcomeOverlay.getVisibility() != View.VISIBLE) {
			return;
		}
		welcomeOverlay.animate()
			.alpha(0f)
			.setDuration(420)
			.withEndAction(() -> welcomeOverlay.setVisibility(View.GONE))
			.start();
	}

	private void updateModelSelection() {
		String architecture = ARCHITECTURE_NAMES[selectedArchitecture];
		String tokenizer = TOKENIZER_NAMES[selectedTokenizer];
		int vocab = selectedTokenizer == 0 ? 256 : selectedTokenizer == 1 ? 320 : 27;
		modelTitle.setText(tokenizer + " / " + architecture);
		modelDetail.setText(String.format(
			Locale.US,
			"vocab %d · width 32 · hidden 64\ncontext 16 · AdamW %.3f · FP32 stable",
			vocab,
			selectedArchitecture == 4 ? 0.003 : 0.01));
		capability.setText(ARCHITECTURE_CAPABILITIES[selectedArchitecture]);
		for (int index = 0; index < architectureButtons.length; ++index) {
			styleModelButton(architectureButtons[index], index == selectedArchitecture);
		}
		for (int index = 0; index < tokenizerButtons.length; ++index) {
			styleButton(tokenizerButtons[index], index == selectedTokenizer);
		}
		architectureScroll.post(() -> architectureScroll.smoothScrollTo(
			Math.max(0, selectedArchitecture * dp(160) - dp(20)), 0));
		if (!trainingRunning) {
			quickTrainButton.setEnabled(true);
			suiteTrainButton.setEnabled(true);
			resumeButton.setEnabled(true);
		}
	}

	private void startTraining(boolean resume, int steps, int batch) {
		if (trainingRunning) {
			return;
		}
		trainingRunning = true;
		String runName = TOKENIZER_NAMES[selectedTokenizer] + " / "
			+ ARCHITECTURE_NAMES[selectedArchitecture];
		trainingStatus.setText(resume ? "RESUMING " + runName : "INITIALIZING " + runName);
		trainingMetrics.setText("Loading Turnip and compiling first-use pipelines…");
		trainingProgress.setProgress(0);
		report.setText("The native OA training process is starting.");
		setTrainingBusy(true);

		ProbeResultReceiver receiver = new ProbeResultReceiver(
			new Handler(Looper.getMainLooper()),
			(resultCode, data) -> {
				if (resultCode == TrainingService.RESULT_PROGRESS) {
					int step = data.getInt(TrainingService.EXTRA_STEP);
					int total = Math.max(1, data.getInt(TrainingService.EXTRA_TOTAL));
					float loss = data.getFloat(TrainingService.EXTRA_LOSS);
					double gpuMs = data.getDouble(TrainingService.EXTRA_GPU_MS);
					double wallMs = data.getDouble(TrainingService.EXTRA_WALL_MS);
					trainingStatus.setText("TRAINING  " + step + " / " + total);
					trainingMetrics.setText(String.format(
						Locale.US,
						"Loss %.4f   GPU %.2f ms   Wall %.2f ms",
						loss, gpuMs, wallMs));
					trainingProgress.setProgress(Math.round(1000f * step / total));
					return;
				}

				String trainingReport = data.getString(
					TrainingService.EXTRA_REPORT,
					"Training returned no report.");
				report.setText(trainingReport);
				boolean failed = resultCode == TrainingService.RESULT_ERROR
					|| trainingReport.contains("Fatal:");
				boolean cancelled = trainingReport.contains("Cancelled: yes");
				trainingStatus.setText(failed
					? "TRAINING FAILED"
					: cancelled ? "CHECKPOINTED / STOPPED" : "TRAINING COMPLETE");
				if (!failed && !cancelled) {
					trainingProgress.setProgress(1000);
				}
				trainingRunning = false;
				setTrainingBusy(false);
				copyButton.setEnabled(true);
			});

		Intent intent = new Intent(this, TrainingService.class);
		intent.putExtra(TrainingService.EXTRA_RECEIVER, receiver);
		intent.putExtra(
			TrainingService.EXTRA_ARCHITECTURE,
			ARCHITECTURE_IDS[selectedArchitecture]);
		intent.putExtra(TrainingService.EXTRA_TOKENIZER, TOKENIZER_IDS[selectedTokenizer]);
		intent.putExtra(TrainingService.EXTRA_STEPS, steps);
		intent.putExtra(TrainingService.EXTRA_BATCH, batch);
		intent.putExtra(TrainingService.EXTRA_RESUME, resume);
		ComponentName started = startForegroundService(intent);
		if (started == null) {
			trainingRunning = false;
			trainingStatus.setText("SERVICE ERROR");
			setTrainingBusy(false);
		}
	}

	private void cancelTraining() {
		if (!trainingRunning) {
			return;
		}
		trainingStatus.setText("STOP REQUESTED");
		cancelButton.setEnabled(false);
		Intent intent = new Intent(this, TrainingService.class);
		intent.setAction(TrainingService.ACTION_CANCEL);
		startService(intent);
	}

	private void setTrainingBusy(boolean busy) {
		quickTrainButton.setEnabled(!busy);
		suiteTrainButton.setEnabled(!busy);
		resumeButton.setEnabled(!busy);
		cancelButton.setEnabled(busy);
		for (Button button : architectureButtons) {
			button.setEnabled(!busy);
		}
		for (Button button : tokenizerButtons) {
			button.setEnabled(!busy);
		}
		styleButton(quickTrainButton, !busy);
		styleButton(suiteTrainButton, false);
		styleButton(resumeButton, false);
		styleButton(cancelButton, false);
		updateModelSelection();
	}

	private void runProbe(String source) {
		activeSource = source;
		setProbeBusy(true);
		status.setText("PROBING " + source.toUpperCase(Locale.ROOT));
		report.setText("Starting isolated " + source + " driver process…");
		styleDriverSelector();

		ProbeResultReceiver receiver = new ProbeResultReceiver(
			new Handler(Looper.getMainLooper()),
			(resultCode, data) -> {
				String resultSource = data.getString(ProbeService.EXTRA_SOURCE, source);
				String resultReport = data.getString(
					ProbeService.EXTRA_REPORT,
					"Probe returned no report.");
				activeSource = resultSource;
				report.setText(resultReport);
				boolean pass = resultReport.contains("ModernCompute: PASS")
					&& resultReport.contains("Dispatch: PASS");
				status.setText(pass ? "READY / PASS" : "NOT READY / INSPECT");
				setProbeBusy(false);
				copyButton.setEnabled(true);
				styleDriverSelector();
			});

		Class<?> service = source.equals("turnip")
			? TurnipProbeService.class
			: SystemProbeService.class;
		Intent intent = new Intent(this, service);
		intent.putExtra(ProbeService.EXTRA_RECEIVER, receiver);
		ComponentName started = startService(intent);
		if (started == null) {
			report.setText("Android refused to start the probe service.");
			status.setText("SERVICE ERROR");
			setProbeBusy(false);
		}
	}

	private void setProbeBusy(boolean busy) {
		turnipButton.setEnabled(!busy);
		systemButton.setEnabled(!busy);
		copyButton.setEnabled(!busy && report != null && report.length() > 0);
	}

	private void copyReport() {
		ClipboardManager clipboard = getSystemService(ClipboardManager.class);
		clipboard.setPrimaryClip(ClipData.newPlainText(
			"OA Mobile Lab report", report.getText()));
		Toast.makeText(this, "Report copied", Toast.LENGTH_SHORT).show();
	}

	private void styleDriverSelector() {
		styleButton(turnipButton, activeSource.equals("turnip"));
		styleButton(systemButton, activeSource.equals("system"));
	}

	private Button modelButton(String value) {
		Button button = new Button(this);
		button.setText(value);
		button.setTextSize(12);
		button.setTypeface(sansSemibold);
		button.setLetterSpacing(0.08f);
		button.setAllCaps(false);
		button.setGravity(Gravity.START | Gravity.CENTER_VERTICAL);
		button.setPadding(dp(14), 0, dp(14), 0);
		button.setMinHeight(0);
		button.setMinWidth(0);
		button.setFocusable(false);
		return button;
	}

	private Button actionButton(String value, boolean selected) {
		Button button = new Button(this);
		button.setText(value);
		button.setTextSize(11);
		button.setTypeface(sansSemibold);
		button.setLetterSpacing(0.08f);
		button.setAllCaps(false);
		button.setPadding(dp(10), 0, dp(10), 0);
		button.setMinHeight(0);
		button.setMinWidth(0);
		styleButton(button, selected);
		return button;
	}

	private void styleModelButton(Button button, boolean selected) {
		if (button == null) {
			return;
		}
		GradientDrawable background = solidBackground(
			selected ? WHITE : PANEL,
			selected ? WHITE : OUTLINE,
			6);
		button.setBackground(background);
		button.setTextColor(selected ? BG : TEXT);
		button.setAlpha(button.isEnabled() ? 1f : 0.42f);
	}

	private void styleButton(Button button, boolean selected) {
		if (button == null) {
			return;
		}
		button.setBackground(solidBackground(
			selected ? WHITE : PANEL_ACTIVE,
			selected ? WHITE : OUTLINE,
			6));
		button.setTextColor(selected ? BG : TEXT);
		button.setAlpha(button.isEnabled() ? 1f : 0.42f);
	}

	private LinearLayout card() {
		LinearLayout card = new LinearLayout(this);
		card.setOrientation(LinearLayout.VERTICAL);
		card.setPadding(dp(16), dp(16), dp(16), dp(16));
		card.setBackground(solidBackground(PANEL, OUTLINE, 6));
		return card;
	}

	private GradientDrawable solidBackground(int color, int stroke, int radius) {
		GradientDrawable background = new GradientDrawable();
		background.setColor(color);
		background.setCornerRadius(dp(radius));
		background.setStroke(dp(1), stroke);
		return background;
	}

	private TextView sectionTitle(String value) {
		TextView view = text(value, 12, TEXT, true);
		view.setLetterSpacing(0.12f);
		return view;
	}

	private TextView label(String value) {
		TextView view = text(value, 10, MUTED, true);
		view.setLetterSpacing(0.14f);
		return view;
	}

	private TextView text(String value, int sizeSp, int color, boolean semibold) {
		TextView view = new TextView(this);
		view.setText(value);
		view.setTextSize(sizeSp);
		view.setTextColor(color);
		view.setTypeface(semibold ? sansSemibold : sans);
		view.setGravity(Gravity.START);
		return view;
	}

	private ImageView realmMark() {
		ImageView view = new ImageView(this);
		view.setImageResource(R.drawable.realm_mark);
		view.setScaleType(ImageView.ScaleType.FIT_CENTER);
		view.setContentDescription("Realm");
		return view;
	}

	private LinearLayout.LayoutParams margin(
		int width, int height, int left, int top, int right, int bottom) {
		LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(width, height);
		params.setMargins(dp(left), dp(top), dp(right), dp(bottom));
		return params;
	}

	private int findId(String[] ids, String requested, int fallback) {
		if (requested != null) {
			for (int index = 0; index < ids.length; ++index) {
				if (ids[index].equals(requested)) {
					return index;
				}
			}
		}
		return fallback;
	}

	private int dp(int value) {
		return Math.round(value * getResources().getDisplayMetrics().density);
	}
}
