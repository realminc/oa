//! OA Tutorial — speech commands · mel features + CNN (autograd)
//!
//! Mirrors the TensorFlow "simple audio" tutorial end to end on OA's
//! GPU-resident pipeline: decode one-second clips, extract mel features for a
//! whole batch in one mel_spectrogram dispatch, and train a small CNN
//! classifier with ItTraining capture/replay.
//!
//! The bundled subset at sdk/asset/audio/commands holds four clips per
//! command (32 total, ~1 MB) so the tutorial runs quickly out of the box.
//! Point --dataset at an extracted mini_speech_commands archive for the full
//! ~8,000-clip workload and meaningful generalization numbers.

use std::path::PathBuf;

use oa::ml::{
	GradientTape, ItTraining, ItTrainingConfig, LossMetric, Module, ProgressBar, TrainingSummary,
};
use oa::sdk::ml::audio;

fn arg_value(flag: &str) -> Option<String> {
	std::env::args().skip_while(|arg| arg != flag).nth(1)
}

fn arg_usize(flag: &str, default: usize) -> usize {
	arg_value(flag)
		.and_then(|value| value.parse().ok())
		.unwrap_or(default)
}

/// Chunked held-out evaluation pass: mean accuracy and clip-mean loss over
/// standardized host features, run eagerly between training steps.
fn evaluate(
	engine: &oa::Engine,
	model: &impl Module,
	features: &[f32],
	labels: &[u32],
	mel_bands: usize,
	mel_frames: usize,
	chunk_clips: usize,
) -> oa::Result<(f64, f64)> {
	let clip_feature_len = mel_bands * mel_frames;
	let mut correct = 0_u64;
	let mut loss_sum = 0.0_f64;
	for begin in (0..labels.len()).step_by(chunk_clips) {
		let end = (begin + chunk_clips).min(labels.len());
		let input = oa::Matrix::new_from_slice(
			engine,
			[end - begin, 1, mel_bands, mel_frames],
			&features[begin * clip_feature_len..end * clip_feature_len],
		)?;
		let target = oa::Matrix::new_from_slice(engine, [end - begin], &labels[begin..end])?;
		let logits = model.forward(&input)?;
		correct += oa::matrix::categorical_accuracy_count(&logits, &target)?.read::<u32>()?[0] as u64;
		loss_sum +=
			oa::ml::loss::cross_entropy(&logits, &target)?.read_f32()?[0] as f64 * (end - begin) as f64;
	}
	let total = labels.len().max(1) as f64;
	Ok((correct as f64 / total, loss_sum / total))
}

fn main() -> oa::Result<()> {
	let engine = oa::Engine::new()?;
	let dataset_dir =
		arg_value("--dataset").map_or_else(|| PathBuf::from("sdk/asset/audio/commands"), PathBuf::from);
	let epochs = arg_usize("--epochs", audio::TRAINING_EPOCHS);
	let batch = arg_usize("--batch", audio::BATCH_SIZE);
	let dataset = audio::SpeechCommandsDataset::load(&engine, &dataset_dir)?;
	let clip_count = dataset.clips.len();

	println!("\n╔══════════════════════════════════════════════════════════════════╗");
	println!("║  OA Tutorial — speech commands · mel features + CNN (autograd)   ║");
	println!("╚══════════════════════════════════════════════════════════════════╝\n");
	println!(
		"dataset: {} · {} clips · {} commands ({} s16kHz samples each)",
		dataset_dir.display(),
		clip_count,
		audio::COMMAND_LABELS.len(),
		audio::SAMPLES_PER_CLIP / audio::CLIP_SAMPLE_RATE as usize,
	);
	println!(
		"features: {} mel bands · fft {} · hop {} — one dispatch per batch",
		audio::MEL_BANDS,
		audio::MEL_FFT_SIZE,
		audio::MEL_HOP_SIZE,
	);

	// Deterministic split: every 4th clip per sorted label goes to test.
	// The loader emits clips in label order, so the training indices get a
	// seeded shuffle — unshuffled windows would be single-label blocks and
	// the optimizer would collapse at every label boundary.
	let (train_idx, test_idx): (Vec<usize>, Vec<usize>) =
		(0..clip_count).partition(|&index| index % 4 != 3);
	let mut shuffled: Vec<i64> = train_idx.iter().map(|&index| index as i64).collect();
	oa::data::shuffle(&mut shuffled, audio::RNG_SEED)?;
	let train_idx: Vec<usize> = shuffled.iter().map(|&index| index as usize).collect();
	println!(
		"split: {} train · {} test\n",
		train_idx.len(),
		test_idx.len(),
	);

	// Chunked extraction keeps each mel dispatch bounded — one call over a
	// multi-thousand-clip planar upload can exceed device memory or driver
	// watchdog limits. ItTraining::step requires an empty execution session,
	// and the prepare closure may only upload fresh inputs, so features
	// materialize to host once and windows slice host-side per step.
	let mut mel_bands = 0_usize;
	let mut mel_frames = 0_usize;
	let mut train_features_host = Vec::new();
	for chunk in train_idx.chunks(audio::MEL_CHUNK_CLIPS) {
		let features = dataset.mel_features(&engine, chunk)?;
		if mel_bands == 0 {
			// mel_features guarantees the NCHW rank-four layout.
			let shape = features.shape();
			(mel_bands, mel_frames) = (shape[2], shape[3]);
		}
		train_features_host.extend_from_slice(&features.read_f32()?);
	}
	let clip_feature_len = mel_bands * mel_frames;
	let train_labels_host = dataset.labels(&engine, &train_idx)?.read::<u32>()?;
	// Standardize with training-set statistics — the host-side equivalent of
	// the TF tutorial's Normalization layer.
	let (mean, inv_std) = audio::standardization(&train_features_host);
	audio::standardize(&mut train_features_host, mean, inv_std);
	// Held-out split materializes upfront — the per-epoch validation pass runs
	// eagerly between training steps.
	let mut test_features_host = Vec::new();
	for chunk in test_idx.chunks(audio::MEL_CHUNK_CLIPS) {
		let features = dataset.mel_features(&engine, chunk)?;
		test_features_host.extend_from_slice(&features.read_f32()?);
	}
	audio::standardize(&mut test_features_host, mean, inv_std);
	let test_labels_host: Vec<u32> = test_idx
		.iter()
		.map(|&index| dataset.clips[index].label)
		.collect();
	let model = audio::commands_cnn(&engine, mel_bands, mel_frames, audio::COMMAND_LABELS.len())?;
	model.train(true);
	// Complete asynchronous parameter initialization before isolated capture.
	engine.checkpoint()?.wait()?;
	// TF's Adam default — 1e-2 diverges on the full dataset's early batches.
	let learning_rate = 0.001;
	let mut optimizer = oa::ml::AdamW::new(model.all_parameters()?, learning_rate)?;

	println!(
		"Model: Conv2d(1→8,3×3)→ReLU→MaxPool→Conv2d(8→16,3×3)→ReLU→MaxPool→Linear(→64)→Dropout→Linear(→8)"
	);
	println!(
		"params: {}    Optimizer: AdamW(lr={learning_rate})\n",
		model.num_parameters()?
	);

	// Full-width windows only — a fixed batch shape keeps the captured
	// training program stable; the tail remainder is dropped.
	let windows = (train_idx.len() / batch).max(1);
	// The canonical recipe counts data passes like TensorFlow: epochs ×
	// windows-per-pass gives the total step count. --steps overrides it.
	let steps = arg_usize("--steps", epochs * windows);
	let mut loss_metric = LossMetric::default();
	let mut progress = ProgressBar::default();
	let mut summary_callback = TrainingSummary::default();
	let mut training = ItTraining::new(
		&engine,
		&mut optimizer,
		ItTrainingConfig {
			total_steps: steps as u64,
			steps_per_epoch: windows as u64,
			batch_size: batch as u64,
			source_unit: "clip".into(),
			timer_name: "commands_training_step".into(),
			enable_gpu_timing: true,
			..ItTrainingConfig::default()
		},
	)?;
	training.add_metric(&mut loss_metric);
	training.add_callback(&mut progress);
	training.add_callback(&mut summary_callback);

	let mut initial_loss = 0.0_f32;
	// The per-step accuracy count is recorded as an extra program output and
	// read back after each step — the same readback the loss metric uses.
	let step_correct = std::cell::RefCell::new(Option::<oa::Matrix>::None);
	let mut epoch_correct = 0_u64;
	let mut epoch_clips = 0_u64;
	for step in 0..steps {
		training.record_source_units(batch as u64);
		// Reshuffle the window order each data pass, as TF's
		// dataset.shuffle does per epoch.
		let mut window_order: Vec<i64> = (0..windows as i64).collect();
		oa::data::shuffle(
			&mut window_order,
			audio::RNG_SEED.wrapping_add((step / windows) as u64),
		)?;
		assert!(training.step(
			|| {
				let begin = (window_order[step % windows] as usize * batch)
					.min(train_idx.len().saturating_sub(batch));
				let end = begin + batch.min(train_idx.len());
				let input = oa::Matrix::new_from_slice(
					&engine,
					[end - begin, 1, mel_bands, mel_frames],
					&train_features_host[begin * clip_feature_len..end * clip_feature_len],
				)?;
				let target =
					oa::Matrix::new_from_slice(&engine, [end - begin], &train_labels_host[begin..end])?;
				Ok((input, target))
			},
			|(input, target)| {
				let tape = GradientTape::new();
				let logits = model.forward(&input)?;
				let loss = oa::ml::loss::cross_entropy(&logits, &target)?;
				tape.backward(&loss)?;
				*step_correct.borrow_mut() =
					Some(oa::matrix::categorical_accuracy_count(&logits, &target)?);
				Ok(loss)
			},
		)?);
		if let Some(count) = &*step_correct.borrow() {
			epoch_correct += count.read::<u32>()?[0] as u64;
		}
		epoch_clips += batch.min(train_idx.len()) as u64;
		let snapshot = training.snapshot();
		if snapshot.step_count() == 1 {
			initial_loss = snapshot.last_loss().unwrap_or_default();
		}
		if snapshot.is_epoch_boundary() {
			model.eval();
			let (val_accuracy, val_loss) = evaluate(
				&engine,
				&model,
				&test_features_host,
				&test_labels_host,
				mel_bands,
				mel_frames,
				audio::MEL_CHUNK_CLIPS,
			)?;
			model.train(true);
			let seconds = snapshot.epoch_elapsed().as_secs_f64();
			println!(
				"  {}/{} — {seconds:.1}s {:.0} ms/step · accuracy: {:.4} · loss: {:.4} · val_accuracy: {val_accuracy:.4} · val_loss: {val_loss:.4}",
				snapshot.step_in_epoch(),
				snapshot.steps_in_epoch(),
				seconds * 1e3 / snapshot.steps_in_epoch().max(1) as f64,
				epoch_correct as f64 / epoch_clips.max(1) as f64,
				snapshot.epoch_mean_loss(),
			);
			epoch_correct = 0;
			epoch_clips = 0;
		}
	}
	let program_diagnostics = training
		.training_program()
		.map(|program| program.diagnostics());
	let result = training.finish()?;
	let final_training_loss = result.last_loss().unwrap_or_default();

	// Final held-out evaluation through the same chunked pass the per-epoch
	// validation used.
	model.eval();
	let (accuracy, evaluation_loss) = evaluate(
		&engine,
		&model,
		&test_features_host,
		&test_labels_host,
		mel_bands,
		mel_frames,
		audio::MEL_CHUNK_CLIPS,
	)?;

	println!("\nEvaluation:");
	println!(
		"  Random-loss baseline ln({}) = {:.4}",
		audio::COMMAND_LABELS.len(),
		(audio::COMMAND_LABELS.len() as f64).ln()
	);
	println!("  test accuracy: {:.1}%", accuracy * 100.0);
	println!("  post-update evaluation loss: {evaluation_loss:.6}");
	if let Some(diagnostics) = program_diagnostics {
		println!(
			"program: {} nodes · {} recording · {} cache hits · {} uploads",
			diagnostics.node_count(),
			diagnostics.command_recording_count(),
			diagnostics.command_cache_hit_count(),
			diagnostics.input_upload_count(),
		);
	}

	assert!(initial_loss > 1.0);
	assert!(final_training_loss < initial_loss);
	// The recorded oracles describe the canonical bundled-subset recipe only.
	let canonical = arg_value("--dataset").is_none()
		&& arg_value("--steps").is_none()
		&& epochs == audio::TRAINING_EPOCHS
		&& batch == audio::BATCH_SIZE;
	if canonical {
		assert!((final_training_loss - audio::COMMANDS_FINAL_LOSS).abs() < 0.01);
		assert!(accuracy >= f64::from(audio::COMMANDS_TEST_ACCURACY));
	}

	let checkpoint = oa::Path::temp().join("oars_commands").with_extension("oam");
	oa::ml::save_checkpoint(&checkpoint, &model, &optimizer)?;
	let reloaded = audio::commands_cnn(&engine, mel_bands, mel_frames, audio::COMMAND_LABELS.len())?;
	reloaded.eval();
	let mut reloaded_optimizer = oa::ml::AdamW::new(reloaded.all_parameters()?, learning_rate)?;
	oa::ml::load_checkpoint(&engine, &checkpoint, &reloaded, &mut reloaded_optimizer)?;
	assert_eq!(reloaded_optimizer.step_count(), optimizer.step_count());
	assert_eq!(
		evaluate(
			&engine,
			&reloaded,
			&test_features_host,
			&test_labels_host,
			mel_bands,
			mel_frames,
			audio::MEL_CHUNK_CLIPS,
		)?
		.0,
		accuracy
	);
	oa::Filesystem::remove_file(&checkpoint).expect("remove completed commands checkpoint");
	Ok(())
}
