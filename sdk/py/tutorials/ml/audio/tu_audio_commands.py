#!/usr/bin/env python3
"""OA Tutorial — speech commands · mel features + CNN (autograd).

Mirrors sdk/rs/tutorials/ml/audio/tu_audio_commands.rs.

Decodes one-second clips, extracts mel features for a whole batch in one
mel_spectrogram dispatch, and trains a small CNN classifier with ItTraining
capture/replay. The bundled subset at sdk/asset/audio/commands holds four
clips per command (32 total, ~1 MB) so the tutorial runs quickly out of the
box; point --dataset at an extracted mini_speech_commands archive for the
full ~8,000-clip workload and meaningful generalization numbers.
"""

import argparse
import math
import os
import sys

import oa
import oa.ml
import oa.ml.audio as audio


def _num_parameters(model) -> int:
	return sum(p.data().num_elements for p in model.all_parameters())


def _evaluate(engine, model, features, labels, mel_bands, mel_frames, chunk):
	"""Chunked held-out pass: mean accuracy and clip-mean loss, run eagerly
	between training steps."""
	clip_feature_len = mel_bands * mel_frames
	correct = 0
	loss_sum = 0.0
	for begin in range(0, len(labels), chunk):
		end = min(begin + chunk, len(labels))
		input_matrix = oa.Matrix.new_from_f32(
			engine,
			[end - begin, 1, mel_bands, mel_frames],
			features[begin * clip_feature_len : end * clip_feature_len],
		)
		target = oa.Matrix.new_from_u32(engine, [end - begin], labels[begin:end])
		logits = model.forward(input_matrix)
		correct += oa.matrix.categorical_accuracy_count(logits, target).read_u32()[0]
		loss_sum += oa.ml.loss.cross_entropy(logits, target).read_f32()[0] * (end - begin)
	total = max(1, len(labels))
	return correct / total, loss_sum / total


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="tu_audio_commands",
		description="OA Tutorial — speech commands · mel features + CNN",
	)
	parser.add_argument(
		"--dataset",
		default="sdk/asset/audio/commands",
		help="speech-commands root containing <label>/*.wav directories",
	)
	parser.add_argument("--epochs", type=int, default=0, help="data passes")
	parser.add_argument(
		"--steps", type=int, default=0, help="training steps (overrides --epochs)"
	)
	parser.add_argument("--batch", type=int, default=0, help="clips per step")
	args = parser.parse_args()
	epochs = args.epochs if args.epochs > 0 else audio.AUDIO_TRAINING_EPOCHS
	batch = args.batch if args.batch > 0 else audio.AUDIO_BATCH_SIZE

	engine = oa.Engine()
	dataset = audio.SpeechCommandsDataset(engine, args.dataset)
	clip_count = len(dataset)

	print()
	print("╔══════════════════════════════════════════════════════════════════╗")
	print("║  OA Tutorial — speech commands · mel features + CNN (autograd)   ║")
	print("╚══════════════════════════════════════════════════════════════════╝")
	print()
	print(
		f"dataset: {args.dataset} · {clip_count} clips · "
		f"{len(audio.AUDIO_COMMAND_LABELS)} commands "
		f"({audio.AUDIO_SAMPLES_PER_CLIP // audio.AUDIO_CLIP_SAMPLE_RATE} s"
		f"{audio.AUDIO_CLIP_SAMPLE_RATE // 1000}kHz samples each)"
	)
	print(
		f"features: {audio.AUDIO_MEL_BANDS} mel bands · fft {audio.AUDIO_MEL_FFT_SIZE} · "
		f"hop {audio.AUDIO_MEL_HOP_SIZE} — one dispatch per batch"
	)

	# Deterministic split: every 4th clip per sorted label goes to test.
	train_idx = [index for index in range(clip_count) if index % 4 != 3]
	test_idx = [index for index in range(clip_count) if index % 4 == 3]
	# The loader emits clips in label order — unshuffled windows would be
	# single-label blocks and the optimizer collapses at label boundaries.
	train_idx = oa.data.shuffle(train_idx, audio.AUDIO_RNG_SEED)
	print(f"split: {len(train_idx)} train · {len(test_idx)} test\n")

	# Chunked extraction keeps each mel dispatch bounded — one call over a
	# multi-thousand-clip planar upload can exceed device memory or driver
	# watchdog limits. ItTraining.step requires an empty execution session,
	# and the prepare closure may only upload fresh inputs, so features
	# materialize to host once and windows slice host-side per step.
	mel_bands = mel_frames = 0
	train_features_host = []
	chunk = audio.AUDIO_MEL_CHUNK_CLIPS
	for begin in range(0, len(train_idx), chunk):
		features = dataset.mel_features(engine, train_idx[begin : begin + chunk])
		if mel_bands == 0:
			# mel_features guarantees the NCHW rank-four layout.
			_, _, mel_bands, mel_frames = features.shape
		train_features_host.extend(features.read_f32())
	clip_feature_len = mel_bands * mel_frames
	train_labels_host = dataset.labels(engine, train_idx).read_u32()
	# Standardize with training-set statistics — the host-side equivalent of
	# the TF tutorial's Normalization layer.
	count = len(train_features_host)
	mean = sum(train_features_host) / count
	variance = sum((v - mean) ** 2 for v in train_features_host) / count
	inv_std = 1.0 / (variance**0.5 + 1e-6)
	train_features_host = [(v - mean) * inv_std for v in train_features_host]
	# Held-out split materializes upfront — the per-epoch validation pass
	# runs eagerly between training steps.
	test_features_host = []
	for begin in range(0, len(test_idx), chunk):
		features = dataset.mel_features(engine, test_idx[begin : begin + chunk])
		test_features_host.extend(features.read_f32())
	test_features_host = [(v - mean) * inv_std for v in test_features_host]
	test_labels_host = dataset.labels(engine, test_idx).read_u32()
	num_classes = len(audio.AUDIO_COMMAND_LABELS)
	model = audio.commands_cnn(engine, mel_bands, mel_frames, num_classes)
	model.train()
	# Complete asynchronous parameter initialization before isolated capture.
	engine.checkpoint().wait()

	# TF's Adam default — 1e-2 diverges on the full dataset's early batches.
	learning_rate = 0.001
	optimizer = oa.ml.AdamW(model.all_parameters(), learning_rate=learning_rate)

	print(
		"Model: Conv2d(1→8,3×3)→ReLU→MaxPool→Conv2d(8→16,3×3)→ReLU→MaxPool→"
		"Linear(→64)→Dropout→Linear(→8)"
	)
	print(f"params: {_num_parameters(model)}    Optimizer: AdamW(lr={learning_rate})\n")

	# Full-width windows only — a fixed batch shape keeps the captured
	# training program stable; the tail remainder is dropped.
	windows = max(1, len(train_idx) // batch)
	# The canonical recipe counts data passes like TensorFlow: epochs ×
	# windows-per-pass gives the total step count. --steps overrides it.
	steps = args.steps if args.steps > 0 else epochs * windows
	loss_metric = oa.ml.LossMetric()
	progress = oa.ml.ProgressBar()
	summary_callback = oa.ml.TrainingSummary()
	training = oa.ml.ItTraining(
		engine,
		optimizer,
		oa.ml.ItTrainingConfig(
			total_steps=steps,
			steps_per_epoch=windows,
			batch_size=batch,
			source_unit="clip",
			timer_name="commands_training_step",
			enable_gpu_timing=True,
		),
	)
	training.add_metric(loss_metric)
	training.add_callback(progress)
	training.add_callback(summary_callback)

	initial_loss = 0.0
	# The per-step accuracy count is recorded as an extra program output and
	# read back after each step — the same readback the loss metric uses.
	step_correct = []
	epoch_correct = 0
	epoch_clips = 0
	for step in range(steps):
		training.record_source_units(batch)
		# Reshuffle the window order each data pass, as TF's
		# dataset.shuffle does per epoch.
		window_order = oa.data.shuffle(
			list(range(windows)), audio.AUDIO_RNG_SEED + step // windows
		)

		def prepare():
			begin = min(
				window_order[step % windows] * batch,
				max(0, len(train_idx) - batch),
			)
			end = begin + min(batch, len(train_idx))
			return (
				oa.Matrix.new_from_f32(
					engine,
					[end - begin, 1, mel_bands, mel_frames],
					train_features_host[begin * clip_feature_len : end * clip_feature_len],
				),
				oa.Matrix.new_from_u32(engine, [end - begin], train_labels_host[begin:end]),
			)

		def record(prepared):
			input_matrix, target = prepared
			with oa.ml.GradientTape() as tape:
				logits = model.forward(input_matrix)
				loss = oa.ml.loss.cross_entropy(logits, target)
				tape.backward(loss)
			step_correct.clear()
			step_correct.append(oa.matrix.categorical_accuracy_count(logits, target))
			return loss

		assert training.step(prepare, record)
		if step_correct:
			epoch_correct += step_correct[0].read_u32()[0]
		epoch_clips += min(batch, len(train_idx))
		snap = training.snapshot()
		if snap.step_count == 1:
			initial_loss = snap.last_loss or 0.0
		if snap.is_epoch_boundary:
			model.eval()
			val_accuracy, val_loss = _evaluate(
				engine,
				model,
				test_features_host,
				test_labels_host,
				mel_bands,
				mel_frames,
				chunk,
			)
			model.train()
			elapsed = snap.epoch_elapsed_secs
			print(
				f"  {snap.step_in_epoch}/{snap.steps_in_epoch} — {elapsed:.1f}s "
				f"{elapsed * 1e3 / max(1, snap.steps_in_epoch):.0f} ms/step · "
				f"accuracy: {epoch_correct / max(1, epoch_clips):.4f} · "
				f"loss: {snap.epoch_mean_loss:.4f} · "
				f"val_accuracy: {val_accuracy:.4f} · val_loss: {val_loss:.4f}"
			)
			epoch_correct = 0
			epoch_clips = 0

	program_diagnostics = training.program_diagnostics()
	result = training.finish()
	final_training_loss = result.last_loss or 0.0

	# Final held-out evaluation through the same chunked pass the per-epoch
	# validation used.
	model.eval()
	accuracy, evaluation_loss = _evaluate(
		engine,
		model,
		test_features_host,
		test_labels_host,
		mel_bands,
		mel_frames,
		chunk,
	)

	print("\nEvaluation:")
	print(
		f"  Random-loss baseline ln({num_classes}) = {math.log(num_classes):.4f}"
	)
	print(f"  test accuracy: {accuracy * 100.0:.1f}%")
	print(f"  post-update evaluation loss: {evaluation_loss:.6f}")
	if program_diagnostics is not None:
		print(
			f"program: {program_diagnostics.node_count} nodes · "
			f"{program_diagnostics.command_recording_count} recording · "
			f"{program_diagnostics.command_cache_hit_count} cache hits · "
			f"{program_diagnostics.input_upload_count} uploads"
		)

	assert initial_loss > 1.0
	assert final_training_loss < initial_loss
	# The recorded oracles describe the canonical bundled-subset recipe only.
	canonical = (
		args.dataset == "sdk/asset/audio/commands"
		and args.steps <= 0
		and epochs == audio.AUDIO_TRAINING_EPOCHS
		and batch == audio.AUDIO_BATCH_SIZE
	)
	if canonical:
		assert abs(final_training_loss - audio.COMMANDS_FINAL_LOSS) < 0.01
		assert accuracy >= audio.COMMANDS_TEST_ACCURACY

	checkpoint = oa.Path.temp().join("oars_commands").with_extension("oam")
	oa.ml.save_checkpoint(str(checkpoint), model.all_parameters(), optimizer)
	reloaded = audio.commands_cnn(engine, mel_bands, mel_frames, num_classes)
	reloaded.eval()
	reloaded_optimizer = oa.ml.AdamW(
		reloaded.all_parameters(), learning_rate=learning_rate
	)
	oa.ml.load_checkpoint(
		engine, str(checkpoint), reloaded.all_parameters(), reloaded_optimizer
	)
	assert reloaded_optimizer.step_count == optimizer.step_count
	assert (
		_evaluate(
			engine,
			reloaded,
			test_features_host,
			test_labels_host,
			mel_bands,
			mel_frames,
			chunk,
		)[0]
		== accuracy
	)
	oa.Filesystem.remove_file(checkpoint)


if __name__ == "__main__":
	main()
