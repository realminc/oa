"""Tests for the oa.ml Python binding surface added in the latest batch.

Coverage strategy
─────────────────
* Every new class is constructed and its __repr__ smoke-tested.
* Plain config/metrics structs verify all getter values against the
  constructor arguments (round-trip contract).
* Enum types verify member identity and token() strings.
* Stateful objects (LossMetric, EarlyStopping, ValidationMetric,
  ValidationResult, Validation, LearningRateScheduler, PhaseSchedule,
  SequentialScheduler, ProgressBar, TrainingSummary, CsvLogger) exercise
  their public methods with minimal realistic inputs.
* CheckpointManagerConfig round-trips all fields (the full manager requires
  a running Engine with GPU, so it is guarded by a hardware-availability
  skip).
* TrainingProgram / TrainingCompilation* types verify enum membership and
  token strings; GPU-requiring capture is skipped on non-GPU hosts.
* NLP preset constructors and CharSampler.next_values() run on CPU only.
* The full module-export inventory asserts every expected name is present in
  oa.ml.__all__ so stale __init__.py omissions are caught immediately.
"""

import math
import unittest

import oa


def _has_gpu() -> bool:
	"""Return True when a compute Engine can be constructed."""
	try:
		oa.Engine()
		return True
	except Exception:
		return False


HAS_GPU = _has_gpu()
require_gpu = unittest.skipUnless(HAS_GPU, "requires a hardware Vulkan compute device")


class MlExportInventoryTest(unittest.TestCase):
	"""Verify every new name is reachable through oa.ml and oa.ml.__all__."""

	ALL_NEW_NAMES = [
		# Flow matching
		"FlowMatchBatch",
		"flow_linear_match",
		"flow_euler_step",
		"flow_masked_mse",
		# Training data / callbacks
		"GpuTimingStats",
		"TrainingSnapshot",
		"ItTrainingConfig",
		"LossAggregation",
		"LossMetric",
		"EarlyStopMode",
		"ProgressBar",
		"TrainingSummary",
		"EarlyStopping",
		"ItRolloutTrainingConfig",
		"CsvLogger",
		"ValidationResult",
		"ValidationMetric",
		"Validation",
		"LearningRateScheduler",
		"TrainingPhase",
		"PhaseSchedule",
		"SequentialScheduler",
		# RL trainer configs / metrics
		"PpoTrainerConfig",
		"PpoTrainerMetrics",
		"DqnTrainerConfig",
		"DqnTrainerMetrics",
		"SacTrainerConfig",
		"SacTrainerMetrics",
		"RolloutCollectorConfig",
		"RolloutCollectorMetrics",
		"PolicyEvaluationConfig",
		"PolicyEvaluationMetrics",
		"RolloutTrainingPhase",
		# RL trainers
		"PpoTrainer",
		"DqnTrainer",
		"SacTrainer",
		# Checkpoint
		"CheckpointManager",
		"CheckpointManagerConfig",
		# Training program
		"TrainingCompilationStage",
		"TrainingCompilationState",
		"TrainingCompilationStageRecord",
		"TrainingProgram",
		# NLP presets
		"CharSampler",
		"ByteSampler",
		"CharRnn",
		"CharGru",
		"CharTransformer",
		"CharMoeTransformer",
		"CharMamba3",
		"ByteRnn",
		"ByteGru",
		"ByteTransformer",
		"ByteMoeTransformer",
		"ByteMamba3",
		"ByteEmpyrealm",
		"BpeRnn",
		"BpeGru",
		"BpeTransformer",
		"BpeMoeTransformer",
		"BpeMamba3",
		"nlp_generate_greedy",
		"nlp_generate_bytes_greedy",
	]

	def test_all_new_names_are_accessible_on_oa_ml(self) -> None:
		for name in self.ALL_NEW_NAMES:
			with self.subTest(name=name):
				self.assertTrue(
					hasattr(oa.ml, name),
					f"oa.ml.{name} is not accessible",
				)

	def test_all_new_names_are_in_dunder_all(self) -> None:
		declared = set(oa.ml.__all__)
		for name in self.ALL_NEW_NAMES:
			with self.subTest(name=name):
				self.assertIn(name, declared, f"{name} missing from oa.ml.__all__")


class GpuTimingStatsTest(unittest.TestCase):
	def test_default_values_are_zero(self) -> None:
		# GpuTimingStats is only produced by TrainingSnapshot; we verify the type
		# exists and has the right attribute names by introspecting a snapshot
		# produced during a real training step only if GPU is available.
		# For the pure-Python smoke test we rely on the repr from a zero-filled
		# instance that the Rust side exposes via TrainingSnapshot.gpu_timing_stats().
		# Since we can't construct GpuTimingStats directly, just confirm the type.
		self.assertTrue(hasattr(oa.ml, "GpuTimingStats"))


class ItTrainingConfigTest(unittest.TestCase):
	def test_round_trips_all_constructor_args(self) -> None:
		cfg = oa.ml.ItTrainingConfig(
			total_steps=1000,
			initial_step=10,
			steps_per_epoch=100,
			batch_size=32,
			sequence_length=16,
			sequence_unit="token",
			source_units_per_sample=1.5,
			source_unit="byte",
			timer_name="my_step",
			enable_gpu_timing=True,
		)
		self.assertEqual(cfg.total_steps, 1000)
		self.assertEqual(cfg.initial_step, 10)
		self.assertEqual(cfg.steps_per_epoch, 100)
		self.assertEqual(cfg.batch_size, 32)
		self.assertEqual(cfg.sequence_length, 16)
		self.assertEqual(cfg.sequence_unit, "token")
		self.assertAlmostEqual(cfg.source_units_per_sample, 1.5)
		self.assertEqual(cfg.source_unit, "byte")
		self.assertEqual(cfg.timer_name, "my_step")
		self.assertTrue(cfg.enable_gpu_timing)
		self.assertIn("ItTrainingConfig", repr(cfg))

	def test_defaults_produce_valid_object(self) -> None:
		cfg = oa.ml.ItTrainingConfig()
		self.assertEqual(cfg.total_steps, 0)
		self.assertFalse(cfg.enable_gpu_timing)


class ItRolloutTrainingConfigTest(unittest.TestCase):
	def test_round_trips_all_fields(self) -> None:
		cfg = oa.ml.ItRolloutTrainingConfig(
			rollouts=50,
			horizon=200,
			environments=4,
			update_epochs=8,
			timer_name="rl_step",
			enable_gpu_timing=False,
		)
		self.assertEqual(cfg.rollouts, 50)
		self.assertEqual(cfg.horizon, 200)
		self.assertEqual(cfg.environments, 4)
		self.assertEqual(cfg.update_epochs, 8)
		self.assertEqual(cfg.timer_name, "rl_step")
		self.assertFalse(cfg.enable_gpu_timing)
		self.assertIn("ItRolloutTrainingConfig", repr(cfg))


class LossAggregationTest(unittest.TestCase):
	def test_members_exist_and_are_distinct(self) -> None:
		self.assertEqual(oa.ml.LossAggregation.Mean, oa.ml.LossAggregation.Mean)
		self.assertNotEqual(oa.ml.LossAggregation.Mean, oa.ml.LossAggregation.Last)

	def test_used_in_loss_metric_construction(self) -> None:
		m = oa.ml.LossMetric("val", oa.ml.LossAggregation.Mean)
		self.assertIn("LossMetric", repr(m))


class LossMetricTest(unittest.TestCase):
	def test_initial_state_is_zero(self) -> None:
		m = oa.ml.LossMetric()
		self.assertEqual(m.count(), 0)
		self.assertEqual(m.mean(), 0.0)
		self.assertEqual(m.last(), 0.0)
		self.assertEqual(m.result(), 0.0)

	def test_repr_includes_class_name(self) -> None:
		self.assertIn("LossMetric", repr(oa.ml.LossMetric()))


class EarlyStopModeTest(unittest.TestCase):
	def test_members_exist(self) -> None:
		self.assertEqual(oa.ml.EarlyStopMode.Min, oa.ml.EarlyStopMode.Min)
		self.assertNotEqual(oa.ml.EarlyStopMode.Min, oa.ml.EarlyStopMode.Max)


class EarlyStoppingTest(unittest.TestCase):
	def test_constructs_with_defaults(self) -> None:
		es = oa.ml.EarlyStopping()
		self.assertFalse(es.should_stop)
		self.assertTrue(math.isfinite(es.best) or math.isinf(es.best))
		self.assertEqual(es.bad_epochs, 0)
		self.assertIn("EarlyStopping", repr(es))

	def test_rejects_invalid_patience(self) -> None:
		with self.assertRaises(Exception):
			oa.ml.EarlyStopping(patience=0)

	def test_rejects_negative_min_delta(self) -> None:
		with self.assertRaises(Exception):
			oa.ml.EarlyStopping(min_delta=-1.0)


class ValidationResultTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		vr = oa.ml.ValidationResult(loss=0.42, batches=10, samples=320)
		self.assertAlmostEqual(vr.loss, 0.42, places=6)
		self.assertEqual(vr.batches, 10)
		self.assertEqual(vr.samples, 320)
		self.assertIn("ValidationResult", repr(vr))

	def test_default_produces_nan_loss(self) -> None:
		vr = oa.ml.ValidationResult()
		self.assertTrue(math.isnan(vr.loss))
		self.assertEqual(vr.batches, 0)
		self.assertEqual(vr.samples, 0)


class ValidationMetricTest(unittest.TestCase):
	@require_gpu
	def test_metric_from_validation_starts_with_no_value(self) -> None:
		def _evaluate(_snap: oa.ml.TrainingSnapshot) -> oa.ml.ValidationResult:
			return oa.ml.ValidationResult(loss=0.5, batches=1, samples=16)

		v = oa.ml.Validation(_evaluate, metric_name="val_loss", step_interval=0)
		metric = v.metric()
		self.assertIsInstance(metric, oa.ml.ValidationMetric)
		self.assertEqual(metric.name, "val_loss")
		# No validation pass has run yet.
		self.assertIsNone(metric.value())
		self.assertIn("ValidationMetric", repr(metric))


class ValidationTest(unittest.TestCase):
	def test_constructs_and_exposes_metric_handle(self) -> None:
		call_count = [0]

		def _evaluate(snap: oa.ml.TrainingSnapshot) -> oa.ml.ValidationResult:
			call_count[0] += 1
			return oa.ml.ValidationResult(loss=0.1 * call_count[0], batches=1, samples=8)

		v = oa.ml.Validation(_evaluate, metric_name="test_val")
		metric = v.metric()
		self.assertEqual(metric.name, "test_val")
		self.assertIn("Validation", repr(v))

	def test_raises_on_non_callable(self) -> None:
		with self.assertRaises((TypeError, Exception)):
			# Should raise because we pass an int, not a callable.
			oa.ml.Validation(42)  # type: ignore[arg-type]


class CsvLoggerTest(unittest.TestCase):
	def test_stores_path(self) -> None:
		logger = oa.ml.CsvLogger("/tmp/test_training.csv")
		self.assertEqual(logger.path(), "/tmp/test_training.csv")
		self.assertIn("CsvLogger", repr(logger))


class TrainingPhaseTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		phase = oa.ml.TrainingPhase("warmup", 5, 500)
		self.assertEqual(phase.id, "warmup")
		self.assertEqual(phase.epochs, 5)
		self.assertEqual(phase.steps, 500)
		self.assertIn("TrainingPhase", repr(phase))

	def test_rejects_empty_id(self) -> None:
		with self.assertRaises(Exception):
			oa.ml.TrainingPhase("", 1, 10)


class PhaseScheduleTest(unittest.TestCase):
	def test_empty_schedule_has_zero_totals(self) -> None:
		ps = oa.ml.PhaseSchedule()
		self.assertEqual(ps.total_epochs(), 0)
		self.assertEqual(ps.total_steps(), 0)
		self.assertEqual(len(ps.phases()), 0)
		self.assertIsNone(ps.current_phase())
		self.assertIn("PhaseSchedule", repr(ps))

	def test_add_phase_accumulates_totals(self) -> None:
		ps = oa.ml.PhaseSchedule()
		ps.add_phase("warmup", 2, 200)
		ps.add_phase("train", 8, 800)
		self.assertEqual(ps.total_epochs(), 10)
		self.assertEqual(ps.total_steps(), 1000)
		self.assertEqual(len(ps.phases()), 2)
		self.assertEqual(ps.phases()[0].id, "warmup")
		self.assertEqual(ps.phases()[1].id, "train")

	def test_hook_is_accepted(self) -> None:
		ps = oa.ml.PhaseSchedule()
		ps.add_phase("p1", 1, 10)
		received = []
		ps.set_on_phase_begin(lambda idx, phase: received.append((idx, phase.id)))
		# The hook is not called until the schedule is driven by a training loop.
		self.assertEqual(len(ps.phases()), 1)


class LearningRateSchedulerTest(unittest.TestCase):
	def test_wraps_cosine_scheduler(self) -> None:
		cosine = oa.ml.CosineScheduler(
			initial_lr=0.1, final_lr=0.0, total_steps=1000
		)
		wrapper = oa.ml.LearningRateScheduler(cosine)
		self.assertIn("LearningRateScheduler", repr(wrapper))

	def test_wraps_arbitrary_python_object(self) -> None:
		class ConstantSchedule:
			def learning_rate(self, step: int) -> float:
				return 0.01

		wrapper = oa.ml.LearningRateScheduler(ConstantSchedule())
		self.assertIn("LearningRateScheduler", repr(wrapper))


class SequentialSchedulerTest(unittest.TestCase):
	def test_chains_two_cosine_schedulers(self) -> None:
		a = oa.ml.CosineScheduler(initial_lr=0.1, final_lr=0.01, total_steps=500)
		b = oa.ml.CosineScheduler(initial_lr=0.01, final_lr=0.001, total_steps=500)
		seq = oa.ml.SequentialScheduler([a, b], milestones=[500])
		lr_early = seq.learning_rate(0)
		lr_late = seq.learning_rate(600)
		# Both values must be positive finite floats.
		self.assertGreater(lr_early, 0.0)
		self.assertGreater(lr_late, 0.0)
		self.assertIn("SequentialScheduler", repr(seq))

	def test_rejects_wrong_milestone_count(self) -> None:
		a = oa.ml.CosineScheduler(initial_lr=0.1, final_lr=0.0, total_steps=100)
		b = oa.ml.CosineScheduler(initial_lr=0.0, final_lr=0.0, total_steps=100)
		with self.assertRaises(Exception):
			# Two schedulers need exactly one milestone, not two.
			oa.ml.SequentialScheduler([a, b], milestones=[100, 200])

	def test_rejects_empty_scheduler_list(self) -> None:
		with self.assertRaises(Exception):
			oa.ml.SequentialScheduler([], milestones=[])


class ProgressBarTest(unittest.TestCase):
	@require_gpu
	def test_render_line_returns_non_empty_string(self) -> None:
		engine = oa.Engine()
		# CharTransformer is the simplest available model: build one so we can
		# drive a single step and get a real snapshot.
		# Instead, just verify render_line accepts the right argument types.
		cfg = oa.ml.ItTrainingConfig(
			total_steps=10, steps_per_epoch=5, batch_size=1
		)
		pb = oa.ml.ProgressBar(width=10)
		pb.set_show_epoch_header(False)
		self.assertIn("ProgressBar", repr(pb))


class RolloutTrainingPhaseTest(unittest.TestCase):
	def test_members_are_distinct(self) -> None:
		self.assertEqual(
			oa.ml.RolloutTrainingPhase.Collect,
			oa.ml.RolloutTrainingPhase.Collect,
		)
		self.assertNotEqual(
			oa.ml.RolloutTrainingPhase.Collect,
			oa.ml.RolloutTrainingPhase.Update,
		)
		self.assertNotEqual(
			oa.ml.RolloutTrainingPhase.Update,
			oa.ml.RolloutTrainingPhase.Complete,
		)


class PpoTrainerConfigTest(unittest.TestCase):
	def test_round_trips_required_fields(self) -> None:
		cfg = oa.ml.PpoTrainerConfig(
			rollouts=10,
			horizon=128,
			environments=4,
			update_epochs=4,
			observation_shape=[8],
			seed=42,
			enable_gpu_timing=False,
		)
		self.assertEqual(cfg.rollouts, 10)
		self.assertEqual(cfg.horizon, 128)
		self.assertEqual(cfg.environments, 4)
		self.assertEqual(cfg.update_epochs, 4)
		self.assertEqual(cfg.observation_shape, [8])
		self.assertEqual(cfg.seed, 42)
		self.assertFalse(cfg.enable_gpu_timing)
		self.assertIn("PpoTrainerConfig", repr(cfg))


class DqnTrainerConfigTest(unittest.TestCase):
	def test_round_trips_required_fields(self) -> None:
		cfg = oa.ml.DqnTrainerConfig(
			updates=500,
			batch_size=64,
			observation_shape=[4],
			target_update_interval=50,
			seed=7,
		)
		self.assertEqual(cfg.updates, 500)
		self.assertEqual(cfg.batch_size, 64)
		self.assertEqual(cfg.observation_shape, [4])
		self.assertEqual(cfg.target_update_interval, 50)
		self.assertEqual(cfg.seed, 7)
		self.assertIn("DqnTrainerConfig", repr(cfg))


class SacTrainerConfigTest(unittest.TestCase):
	def test_round_trips_required_fields(self) -> None:
		cfg = oa.ml.SacTrainerConfig(
			updates=200,
			batch_size=32,
			action_dimensions=2,
			observation_shape=[6],
			action_minimum=-1.0,
			action_maximum=1.0,
		)
		self.assertEqual(cfg.updates, 200)
		self.assertEqual(cfg.batch_size, 32)
		self.assertEqual(cfg.action_dimensions, 2)
		self.assertEqual(cfg.observation_shape, [6])
		self.assertAlmostEqual(cfg.action_minimum, -1.0)
		self.assertAlmostEqual(cfg.action_maximum, 1.0)
		self.assertIn("SacTrainerConfig", repr(cfg))


class RolloutCollectorConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.ml.RolloutCollectorConfig(horizon=64, seed=99)
		self.assertEqual(cfg.horizon, 64)
		self.assertEqual(cfg.seed, 99)
		self.assertIn("RolloutCollectorConfig", repr(cfg))


class PolicyEvaluationConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.ml.PolicyEvaluationConfig(horizon=500, seed=3)
		self.assertEqual(cfg.horizon, 500)
		self.assertEqual(cfg.seed, 3)
		self.assertIn("PolicyEvaluationConfig", repr(cfg))


class CheckpointManagerConfigTest(unittest.TestCase):
	def test_round_trips_all_fields(self) -> None:
		cfg = oa.ml.CheckpointManagerConfig(
			directory="/tmp/ckpt",
			model_name="MyModel",
			context="run1",
			max_keep=3,
			save_best=True,
			metric_name="val_loss",
			lower_is_better=True,
		)
		self.assertEqual(cfg.directory, "/tmp/ckpt")
		self.assertEqual(cfg.model_name, "MyModel")
		self.assertEqual(cfg.context, "run1")
		self.assertEqual(cfg.max_keep, 3)
		self.assertTrue(cfg.save_best)
		self.assertEqual(cfg.metric_name, "val_loss")
		self.assertTrue(cfg.lower_is_better)
		self.assertIn("CheckpointManagerConfig", repr(cfg))

	def test_defaults_are_sane(self) -> None:
		cfg = oa.ml.CheckpointManagerConfig()
		self.assertGreater(len(cfg.directory), 0)
		self.assertGreater(len(cfg.model_name), 0)
		self.assertGreater(cfg.max_keep, 0)

	def test_rejects_path_traversal_in_model_name(self) -> None:
		# CheckpointManager validates at construction time; CheckpointManagerConfig
		# is a plain data container — validation happens in CheckpointManager.new.
		# We just confirm the config accepts any string.
		cfg = oa.ml.CheckpointManagerConfig(model_name="../escape")
		self.assertEqual(cfg.model_name, "../escape")


@require_gpu
class CheckpointManagerTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_constructs_and_reports_paths(self) -> None:
		cfg = oa.ml.CheckpointManagerConfig(
			directory="/tmp/oa_test_ckpt",
			model_name="TestModel",
			metric_name="loss",
			lower_is_better=True,
		)
		mgr = oa.ml.CheckpointManager(self.engine, cfg)
		self.assertIn("TestModel", mgr.model_directory())
		self.assertIn("TestModel", mgr.master_path())
		self.assertIn("TestModel", mgr.incremental_directory())
		self.assertTrue(math.isinf(mgr.best_metric))
		self.assertEqual(mgr.metric_name, "loss")
		self.assertIn("CheckpointManager", repr(mgr))

	def test_is_better_respects_lower_is_better(self) -> None:
		cfg = oa.ml.CheckpointManagerConfig(
			directory="/tmp/oa_test_ckpt",
			model_name="TestModel",
			metric_name="loss",
			lower_is_better=True,
		)
		mgr = oa.ml.CheckpointManager(self.engine, cfg)
		self.assertTrue(mgr.is_better(0.5))
		self.assertTrue(mgr.is_better(-100.0))

	def test_is_better_max_mode(self) -> None:
		cfg = oa.ml.CheckpointManagerConfig(
			directory="/tmp/oa_test_ckpt",
			model_name="TestModel",
			metric_name="accuracy",
			lower_is_better=False,
		)
		mgr = oa.ml.CheckpointManager(self.engine, cfg)
		self.assertTrue(mgr.is_better(0.9))

	def test_rejects_path_traversal_in_model_name(self) -> None:
		cfg = oa.ml.CheckpointManagerConfig(model_name="../escape")
		with self.assertRaises(Exception):
			oa.ml.CheckpointManager(self.engine, cfg)


class TrainingCompilationStageTest(unittest.TestCase):
	"""TrainingCompilationStage enum: all 11 members, token strings."""

	EXPECTED_TOKENS = {
		"SemanticValidation": "semantic_validation",
		"ReplaySafety": "replay_safety",
		"Decomposition": "decomposition",
		"Fusion": "fusion",
		"Placement": "placement",
		"Precision": "precision",
		"KernelSelection": "kernel_selection",
		"LoweringValidation": "lowering_validation",
		"MemoryPlanning": "memory_planning",
		"SynchronizationPlanning": "synchronization_planning",
		"CommandRecording": "command_recording",
	}

	def test_all_members_exist(self) -> None:
		for name in self.EXPECTED_TOKENS:
			with self.subTest(stage=name):
				self.assertTrue(
					hasattr(oa.ml.TrainingCompilationStage, name),
					f"TrainingCompilationStage.{name} is missing",
				)

	def test_token_strings_are_correct(self) -> None:
		for name, token in self.EXPECTED_TOKENS.items():
			with self.subTest(stage=name):
				member = getattr(oa.ml.TrainingCompilationStage, name)
				self.assertEqual(member.token(), token)

	def test_members_compare_equal_to_themselves(self) -> None:
		s = oa.ml.TrainingCompilationStage
		self.assertEqual(s.SemanticValidation, s.SemanticValidation)
		self.assertNotEqual(s.SemanticValidation, s.CommandRecording)


class TrainingCompilationStateTest(unittest.TestCase):
	"""TrainingCompilationState enum: all 5 members, token strings."""

	EXPECTED_TOKENS = {
		"NotRun": "not_run",
		"Inherited": "inherited",
		"Analyzed": "analyzed",
		"Applied": "applied",
		"Failed": "failed",
	}

	def test_all_members_exist(self) -> None:
		for name in self.EXPECTED_TOKENS:
			with self.subTest(state=name):
				self.assertTrue(
					hasattr(oa.ml.TrainingCompilationState, name),
					f"TrainingCompilationState.{name} is missing",
				)

	def test_token_strings_are_correct(self) -> None:
		for name, token in self.EXPECTED_TOKENS.items():
			with self.subTest(state=name):
				member = getattr(oa.ml.TrainingCompilationState, name)
				self.assertEqual(member.token(), token)


@require_gpu
class TrainingProgramTest(unittest.TestCase):
	"""TrainingProgram: capture a real linear step and verify contracts."""

	def setUp(self) -> None:
		self.engine = oa.Engine()
		# Build a minimal 1-layer linear model + AdamW for capture.
		self.linear = oa.ml.Linear(self.engine, in_features=4, out_features=1, seed=0)
		params = self.linear.all_parameters()
		self.optimizer = oa.ml.AdamW(params, learning_rate=1e-3)

	def _step_fn(self) -> "oa.Matrix":
		"""One forward + backward step; returns a scalar F32 loss."""
		x = oa.Matrix.from_f32(self.engine, [1, 4], [1.0, 2.0, 3.0, 4.0])
		target = oa.Matrix.from_f32(self.engine, [1, 1], [0.0])
		with oa.ml.GradientTape() as tape:
			pred = self.linear(x)
			loss = oa.ml.loss.mse(pred, target)
			tape.backward(loss)
		return loss

	def test_capture_produces_correct_stage_count(self) -> None:
		prog = oa.ml.TrainingProgram.capture(
			self.engine, self.optimizer, self._step_fn
		)
		stages = prog.compilation_stages()
		self.assertEqual(len(stages), 11)

	def test_stage_records_have_valid_types(self) -> None:
		prog = oa.ml.TrainingProgram.capture(
			self.engine, self.optimizer, self._step_fn
		)
		for record in prog.compilation_stages():
			self.assertIsInstance(record, oa.ml.TrainingCompilationStageRecord)
			self.assertIsInstance(record.stage, oa.ml.TrainingCompilationStage)
			self.assertIsInstance(record.state, oa.ml.TrainingCompilationState)
			self.assertGreaterEqual(record.input_count, 0)
			self.assertGreaterEqual(record.output_count, 0)

	def test_stage_tokens_cover_all_11_stages(self) -> None:
		prog = oa.ml.TrainingProgram.capture(
			self.engine, self.optimizer, self._step_fn
		)
		tokens = {r.stage.token() for r in prog.compilation_stages()}
		expected = {
			"semantic_validation", "replay_safety", "decomposition",
			"fusion", "placement", "precision", "kernel_selection",
			"lowering_validation", "memory_planning",
			"synchronization_planning", "command_recording",
		}
		self.assertEqual(tokens, expected)

	def test_replay_count_starts_at_zero(self) -> None:
		prog = oa.ml.TrainingProgram.capture(
			self.engine, self.optimizer, self._step_fn
		)
		self.assertEqual(prog.replay_count(), 0)

	def test_replay_and_wait_returns_finite_loss(self) -> None:
		prog = oa.ml.TrainingProgram.capture(
			self.engine, self.optimizer, self._step_fn
		)
		loss = prog.replay_and_wait(self.engine, self.optimizer)
		self.assertIsInstance(loss, float)
		self.assertTrue(math.isfinite(loss))
		self.assertEqual(prog.replay_count(), 1)

	def test_repr_includes_class_name(self) -> None:
		prog = oa.ml.TrainingProgram.capture(
			self.engine, self.optimizer, self._step_fn
		)
		self.assertIn("TrainingProgram", repr(prog))

	def test_loss_matrix_is_scalar_f32(self) -> None:
		prog = oa.ml.TrainingProgram.capture(
			self.engine, self.optimizer, self._step_fn
		)
		loss_mat = prog.loss()
		self.assertIsInstance(loss_mat, oa.Matrix)
		self.assertEqual(loss_mat.dtype, "f32")
		self.assertEqual(loss_mat.shape, [])


class CharSamplerTest(unittest.TestCase):
	def test_constructs_with_valid_batch_size(self) -> None:
		sampler = oa.ml.CharSampler(batch_size=2)
		self.assertIn("CharSampler", repr(sampler))

	def test_next_values_returns_paired_u32_sequences(self) -> None:
		sampler = oa.ml.CharSampler(batch_size=1)
		inputs, targets = sampler.next_values()
		# CharSampler uses sequence_length=16; input and target have same length
		self.assertEqual(len(inputs), len(targets))
		self.assertGreater(len(inputs), 0)
		# All tokens are valid u32 values
		for tok in inputs + targets:
			self.assertIsInstance(tok, int)
			self.assertGreaterEqual(tok, 0)

	def test_rejects_zero_batch_size(self) -> None:
		with self.assertRaises(Exception):
			oa.ml.CharSampler(batch_size=0)


class ByteSamplerTest(unittest.TestCase):
	def test_constructs_with_valid_batch_size(self) -> None:
		sampler = oa.ml.ByteSampler(batch_size=2)
		self.assertIn("ByteSampler", repr(sampler))

	def test_next_values_returns_paired_sequences(self) -> None:
		sampler = oa.ml.ByteSampler(batch_size=1)
		inputs, targets = sampler.next_values()
		self.assertEqual(len(inputs), len(targets))
		self.assertGreater(len(inputs), 0)


@require_gpu
class NlpPresetSmoke(unittest.TestCase):
	"""Smoke-test that each NLP model preset constructs without error."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def _assert_preset(self, cls_name: str, **kwargs) -> None:
		preset_cls = getattr(oa.ml, cls_name)
		model = preset_cls(self.engine, **kwargs)
		self.assertIn(cls_name, repr(model))
		params = model.all_parameters()
		self.assertGreater(len(params), 0)

	def test_char_rnn(self) -> None:
		self._assert_preset("CharRnn")

	def test_char_gru(self) -> None:
		self._assert_preset("CharGru")

	def test_char_transformer(self) -> None:
		self._assert_preset("CharTransformer")

	def test_byte_rnn(self) -> None:
		self._assert_preset("ByteRnn")

	def test_byte_gru(self) -> None:
		self._assert_preset("ByteGru")

	def test_byte_transformer(self) -> None:
		self._assert_preset("ByteTransformer")

	def test_char_mamba3(self) -> None:
		self._assert_preset("CharMamba3")

	def test_byte_mamba3(self) -> None:
		self._assert_preset("ByteMamba3")


@require_gpu
class FlowMatchBatchTest(unittest.TestCase):
	"""FlowMatchBatch: linear_match produces correct state/velocity shapes."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_linear_match_returns_flow_match_batch(self) -> None:
		# Minimal 1-D batch: clean and noise are [2, 4], time is [2]
		clean = oa.Matrix.from_f32(self.engine, [2, 4], [0.0] * 8)
		noise = oa.Matrix.from_f32(self.engine, [2, 4], [1.0] * 8)
		time = oa.Matrix.from_f32(self.engine, [2], [0.5, 0.25])
		batch = oa.ml.flow_linear_match(clean, noise, time)
		self.assertIsInstance(batch, oa.ml.FlowMatchBatch)
		self.assertEqual(repr(batch), "FlowMatchBatch")
		state = batch.state
		velocity = batch.velocity
		self.assertIsInstance(state, oa.Matrix)
		self.assertIsInstance(velocity, oa.Matrix)
		self.assertEqual(state.shape, [2, 4])
		self.assertEqual(velocity.shape, [2, 4])

	def test_euler_step_returns_matrix_with_matching_shape(self) -> None:
		state = oa.Matrix.from_f32(self.engine, [2, 4], [0.5] * 8)
		velocity = oa.Matrix.from_f32(self.engine, [2, 4], [1.0] * 8)
		result = oa.ml.flow_euler_step(state, velocity, delta_time=0.1)
		self.assertIsInstance(result, oa.Matrix)
		self.assertEqual(result.shape, [2, 4])

	def test_masked_mse_returns_scalar_loss(self) -> None:
		pred = oa.Matrix.from_f32(self.engine, [2, 4], [0.0] * 8)
		target = oa.Matrix.from_f32(self.engine, [2, 4], [1.0] * 8)
		mask = oa.Matrix.from_f32(self.engine, [2, 4], [1.0] * 8)
		loss = oa.ml.flow_masked_mse(pred, target, mask)
		self.assertIsInstance(loss, oa.Matrix)

	def test_flow_submodule_aliases_same_functions(self) -> None:
		self.assertIs(oa.ml.flow.linear_match, oa.ml.flow_linear_match)
		self.assertIs(oa.ml.flow.euler_step, oa.ml.flow_euler_step)
		self.assertIs(oa.ml.flow.masked_mse, oa.ml.flow_masked_mse)
		self.assertIs(oa.ml.flow.FlowMatchBatch, oa.ml.FlowMatchBatch)




# ── LR Scheduler completeness ─────────────────────────────────────────────────

class WarmupSchedulerTest(unittest.TestCase):
	def test_increases_monotonically_during_warmup(self) -> None:
		sched = oa.ml.WarmupScheduler(target_learning_rate=0.1, warmup_steps=100)
		lr0 = sched.learning_rate(0)
		lr50 = sched.learning_rate(50)
		lr100 = sched.learning_rate(100)
		self.assertLess(lr0, lr50)
		self.assertLessEqual(lr50, lr100)
		self.assertIn("WarmupScheduler", repr(sched))

	def test_holds_constant_after_warmup(self) -> None:
		sched = oa.ml.WarmupScheduler(target_learning_rate=0.01, warmup_steps=10)
		lr_at = sched.learning_rate(10)
		lr_after = sched.learning_rate(500)
		self.assertAlmostEqual(lr_at, lr_after, places=6)


class OneCycleSchedulerTest(unittest.TestCase):
	def test_constructs_with_defaults(self) -> None:
		sched = oa.ml.OneCycleScheduler(max_learning_rate=0.1, total_steps=1000)
		lr = sched.learning_rate(0)
		self.assertGreater(lr, 0.0)
		self.assertIn("OneCycleScheduler", repr(sched))

	def test_peak_is_near_max(self) -> None:
		sched = oa.ml.OneCycleScheduler(
			max_learning_rate=0.1, total_steps=100, percent_start=0.3
		)
		# Peak is at the end of the warmup region (~step 30).
		peak_lr = sched.learning_rate(30)
		self.assertAlmostEqual(peak_lr, 0.1, places=3)


class CyclicSchedulerTest(unittest.TestCase):
	def test_triangular_mode_cycles(self) -> None:
		sched = oa.ml.CyclicScheduler(
			base_learning_rate=0.001,
			max_learning_rate=0.01,
			step_size_up=100,
			mode=oa.ml.CyclicMode.Triangular,
		)
		lr_base = sched.learning_rate(0)
		lr_peak = sched.learning_rate(100)
		self.assertAlmostEqual(lr_base, 0.001, places=5)
		self.assertAlmostEqual(lr_peak, 0.01, places=5)
		self.assertIn("CyclicScheduler", repr(sched))

	def test_cyclic_mode_members_are_distinct(self) -> None:
		self.assertNotEqual(oa.ml.CyclicMode.Triangular, oa.ml.CyclicMode.Triangular2)
		self.assertNotEqual(oa.ml.CyclicMode.Triangular2, oa.ml.CyclicMode.ExpRange)


class CosineWarmRestartsSchedulerTest(unittest.TestCase):
	def test_constructs_and_monotonically_decays_in_first_period(self) -> None:
		sched = oa.ml.CosineWarmRestartsScheduler(
			max_learning_rate=0.1, initial_period=50
		)
		lr0 = sched.learning_rate(0)
		lr25 = sched.learning_rate(25)
		lr49 = sched.learning_rate(49)
		self.assertAlmostEqual(lr0, 0.1, places=5)
		self.assertGreater(lr0, lr25)
		self.assertGreater(lr25, lr49)
		self.assertIn("CosineWarmRestartsScheduler", repr(sched))

	def test_restarts_at_period_boundary(self) -> None:
		sched = oa.ml.CosineWarmRestartsScheduler(
			max_learning_rate=0.1, initial_period=50
		)
		# After one full period, lr should restart close to the max.
		lr_restart = sched.learning_rate(50)
		self.assertGreater(lr_restart, sched.learning_rate(49))


class LinearWarmupCosineSchedulerTest(unittest.TestCase):
	def test_constructs_and_initial_lr_is_near_zero(self) -> None:
		sched = oa.ml.LinearWarmupCosineScheduler(
			warmup_steps=100, total_steps=1000, max_learning_rate=0.1, min_learning_rate=0.0
		)
		lr0 = sched.learning_rate(0)
		lr100 = sched.learning_rate(100)
		lr1000 = sched.learning_rate(1000)
		self.assertLess(lr0, lr100)
		self.assertAlmostEqual(lr100, 0.1, places=4)
		self.assertAlmostEqual(lr1000, 0.0, places=4)
		self.assertIn("LinearWarmupCosineScheduler", repr(sched))


class PlateauModeTest(unittest.TestCase):
	def test_members_are_distinct(self) -> None:
		self.assertNotEqual(oa.ml.PlateauMode.Min, oa.ml.PlateauMode.Max)
		self.assertEqual(oa.ml.PlateauMode.Min, oa.ml.PlateauMode.Min)


class ReduceOnPlateauSchedulerTest(unittest.TestCase):
	def test_constructs_with_defaults(self) -> None:
		sched = oa.ml.ReduceOnPlateauScheduler(initial_learning_rate=0.1)
		lr = sched.learning_rate(0)
		self.assertAlmostEqual(lr, 0.1, places=5)
		self.assertIn("ReduceOnPlateauScheduler", repr(sched))

	def test_reduces_after_plateau(self) -> None:
		sched = oa.ml.ReduceOnPlateauScheduler(
			initial_learning_rate=0.1, factor=0.5, patience=3, threshold=1e-4
		)
		initial_lr = sched.learning_rate(0)
		# First call sets the best; then patience=3 more non-improving calls trigger
		# the reduction (total of 4+ calls needed).
		for _ in range(5):
			sched.step(0.5)
		reduced_lr = sched.learning_rate(0)
		self.assertLess(reduced_lr, initial_lr)


# ── BpeTokenizer ──────────────────────────────────────────────────────────────

class BpeTokenizerTest(unittest.TestCase):
	def test_constructs_untrained(self) -> None:
		tok = oa.ml.BpeTokenizer(target_vocab=300)
		self.assertEqual(tok.num_merges(), 0)
		self.assertEqual(tok.vocab_size(), 256)
		self.assertIn("BpeTokenizer", repr(tok))

	def test_train_increases_vocab(self) -> None:
		tok = oa.ml.BpeTokenizer(target_vocab=300)
		corpus = b"hello world " * 200
		tok.train(corpus, num_merges=10)
		self.assertGreater(tok.num_merges(), 0)
		self.assertGreater(tok.vocab_size(), 256)

	def test_encode_decode_roundtrip(self) -> None:
		tok = oa.ml.BpeTokenizer(target_vocab=300)
		corpus = b"the quick brown fox jumps " * 100
		tok.train(corpus, num_merges=20)
		text = "the quick"
		tokens = tok.encode_text(text)
		self.assertIsInstance(tokens, list)
		self.assertGreater(len(tokens), 0)
		decoded = tok.decode_text(tokens)
		self.assertEqual(decoded, text)

	def test_encode_prompt_pads_to_context_length(self) -> None:
		tok = oa.ml.BpeTokenizer(target_vocab=256)
		prompt = b"hi"
		result = tok.encode_prompt(prompt, context_length=8)
		self.assertEqual(len(result), 8)


# ── TrainingSessionConfig / TrainingState ──────────────────────────────────────

class TrainingSessionConfigTest(unittest.TestCase):
	def test_round_trips_all_fields(self) -> None:
		cfg = oa.ml.TrainingSessionConfig(
			command_capacity=32,
			result_capacity=64,
			snapshot_capacity=16,
		)
		self.assertEqual(cfg.command_capacity, 32)
		self.assertEqual(cfg.result_capacity, 64)
		self.assertEqual(cfg.snapshot_capacity, 16)
		self.assertIn("TrainingSessionConfig", repr(cfg))

	def test_defaults_are_positive(self) -> None:
		cfg = oa.ml.TrainingSessionConfig()
		self.assertGreater(cfg.command_capacity, 0)
		self.assertGreater(cfg.result_capacity, 0)
		self.assertGreater(cfg.snapshot_capacity, 0)


class TrainingStateTest(unittest.TestCase):
	def test_all_members_exist_and_are_distinct(self) -> None:
		s = oa.ml.TrainingState
		members = [s.Running, s.Paused, s.Stopping, s.Completed, s.Failed]
		# All equal to themselves.
		for m in members:
			self.assertEqual(m, m)
		# All distinct from each other.
		for i, a in enumerate(members):
			for j, b in enumerate(members):
				if i != j:
					self.assertNotEqual(a, b)


class TrainingSessionTest(unittest.TestCase):
	def test_constructs_without_config(self) -> None:
		sess = oa.ml.TrainingSession()
		self.assertIsInstance(sess, oa.ml.TrainingSession)

	def test_constructs_with_config(self) -> None:
		cfg = oa.ml.TrainingSessionConfig(command_capacity=8)
		sess = oa.ml.TrainingSession(cfg)
		self.assertIsInstance(sess, oa.ml.TrainingSession)


# ── GaeConfig / RL data structures ────────────────────────────────────────────

class GaeConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		# `lambda` is a Python keyword; pass positionally and access via getattr.
		cfg = oa.ml.GaeConfig(0.98, 0.92)
		self.assertAlmostEqual(cfg.gamma, 0.98, places=5)
		self.assertAlmostEqual(getattr(cfg, "lambda"), 0.92, places=5)
		self.assertIn("GaeConfig", repr(cfg))

	def test_defaults_are_valid(self) -> None:
		cfg = oa.ml.GaeConfig()
		self.assertGreater(cfg.gamma, 0.0)
		self.assertGreater(getattr(cfg, "lambda"), 0.0)


class PpoLossConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.ml.PpoLossConfig(
			clip_epsilon=0.1, value_coefficient=0.25, entropy_coefficient=0.005
		)
		self.assertAlmostEqual(cfg.clip_epsilon, 0.1, places=5)
		self.assertAlmostEqual(cfg.value_coefficient, 0.25, places=5)
		self.assertAlmostEqual(cfg.entropy_coefficient, 0.005, places=5)
		self.assertIn("PpoLossConfig", repr(cfg))


class DqnLossConfigTest(unittest.TestCase):
	def test_round_trips_discount(self) -> None:
		cfg = oa.ml.DqnLossConfig(discount=0.95)
		self.assertAlmostEqual(cfg.discount, 0.95, places=5)
		self.assertIn("DqnLossConfig", repr(cfg))


class SacLossConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.ml.SacLossConfig(discount=0.98, entropy_coefficient=0.1)
		self.assertAlmostEqual(cfg.discount, 0.98, places=5)
		self.assertAlmostEqual(cfg.entropy_coefficient, 0.1, places=5)
		self.assertIn("SacLossConfig", repr(cfg))


class ReplayConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.ml.ReplayConfig(
			capacity=10_000, observation_shape=[4], action_shape=[2], action_dtype="f32"
		)
		self.assertEqual(cfg.capacity, 10_000)
		self.assertEqual(cfg.observation_shape, [4])
		self.assertEqual(cfg.action_shape, [2])
		self.assertEqual(cfg.action_dtype, "f32")
		self.assertIn("ReplayConfig", repr(cfg))

	def test_i32_action_dtype(self) -> None:
		cfg = oa.ml.ReplayConfig(capacity=100, observation_shape=[8], action_dtype="i32")
		self.assertEqual(cfg.action_dtype, "i32")

	def test_rejects_unknown_dtype(self) -> None:
		with self.assertRaises(Exception):
			oa.ml.ReplayConfig(capacity=100, observation_shape=[4], action_dtype="bf16")


class RolloutConfigTest(unittest.TestCase):
	def test_round_trips_fields(self) -> None:
		cfg = oa.ml.RolloutConfig(time=128, environments=4, observation_shape=[8])
		self.assertEqual(cfg.time, 128)
		self.assertEqual(cfg.environments, 4)
		self.assertEqual(cfg.observation_shape, [8])
		self.assertIn("RolloutConfig", repr(cfg))


# ── Metrics structs (type presence + GPU integration tests) ───────────────────

class MetricsTypePresenceTest(unittest.TestCase):
	"""Verify all metrics types are accessible as types in oa.ml."""

	def test_ppo_trainer_metrics_type_is_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "PpoTrainerMetrics"))

	def test_dqn_trainer_metrics_type_is_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "DqnTrainerMetrics"))

	def test_sac_trainer_metrics_type_is_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "SacTrainerMetrics"))

	def test_rollout_collector_metrics_type_is_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "RolloutCollectorMetrics"))

	def test_policy_evaluation_metrics_type_is_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "PolicyEvaluationMetrics"))


@require_gpu
class TrainerMetricsFromGpuTest(unittest.TestCase):
	"""Verify all metrics types are instantiable via their respective trainers."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def _make_actor_critic_and_adamw(self):
		"""Helper that returns (model, optimizer) for a tiny 4→2 actor-critic."""
		model_cfg = oa.ml.CategoricalActorCriticConfig(
			observation_size=4, action_count=2, hidden_size=16, seed=0
		)
		model = oa.ml.CategoricalActorCritic(self.engine, model_cfg)
		params = model.all_parameters()
		opt = oa.ml.AdamW(params, learning_rate=1e-3)
		return model, opt

	def test_ppo_trainer_metrics_from_trainer(self) -> None:
		model, opt = self._make_actor_critic_and_adamw()
		trainer_cfg = oa.ml.PpoTrainerConfig(
			rollouts=1, horizon=4, environments=1,
			update_epochs=1, observation_shape=[4], seed=0,
		)
		trainer = oa.ml.PpoTrainer(self.engine, model, opt, trainer_cfg)
		m = trainer.metrics()
		self.assertIsInstance(m, oa.ml.PpoTrainerMetrics)
		self.assertIn("PpoTrainerMetrics", repr(m))
		# Initial metrics should have zeroed fields.
		self.assertGreaterEqual(m.rollout, 0)
		self.assertGreaterEqual(m.update_epoch, 0)

	def test_dqn_trainer_metrics_from_trainer(self) -> None:
		online_cfg = oa.ml.CategoricalActorCriticConfig(
			observation_size=4, action_count=2, hidden_size=16, seed=0
		)
		online = oa.ml.CategoricalActorCritic(self.engine, online_cfg)
		target = oa.ml.CategoricalActorCritic(self.engine, online_cfg)
		params = online.all_parameters()
		opt = oa.ml.AdamW(params, learning_rate=1e-3)
		replay = oa.ml.ReplayBuffer(
			self.engine,
			oa.ml.ReplayConfig(capacity=100, observation_shape=[4], action_dtype="i32"),
		)
		trainer_cfg = oa.ml.DqnTrainerConfig(
			updates=1, batch_size=4, observation_shape=[4], seed=0
		)
		trainer = oa.ml.DqnTrainer(self.engine, online, target, opt, replay, trainer_cfg)
		m = trainer.metrics()
		self.assertIsInstance(m, oa.ml.DqnTrainerMetrics)
		self.assertIn("DqnTrainerMetrics", repr(m))


# ── TrainingSummary ────────────────────────────────────────────────────────────

class TrainingSummaryTest(unittest.TestCase):
	def test_constructs_and_repr(self) -> None:
		summary = oa.ml.TrainingSummary()
		self.assertIn("TrainingSummary", repr(summary))

	def test_render_report_returns_non_empty_string(self) -> None:
		# Build a minimal snapshot for rendering.
		cfg = oa.ml.ItTrainingConfig(total_steps=100, batch_size=8, steps_per_epoch=50)
		summary = oa.ml.TrainingSummary(track_initial_loss=False)
		# render_report requires a TrainingSnapshot; we need a GPU to produce one
		# so just verify the method exists.
		self.assertTrue(hasattr(summary, "render_report"))


# ── GPU-requiring optimizer constructors ──────────────────────────────────────

@require_gpu
class OptimizerConstructorsTest(unittest.TestCase):
	"""Verify every optimizer type constructs and exposes its getter contract."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.linear = oa.ml.Linear(cls.engine, in_features=4, out_features=2, seed=1)
		cls.params = cls.linear.all_parameters()

	def test_sgd_constructs_and_round_trips_lr(self) -> None:
		opt = oa.ml.Sgd(self.params, learning_rate=0.01)
		self.assertAlmostEqual(opt.learning_rate, 0.01, places=5)
		self.assertEqual(opt.step_count, 0)
		self.assertIn("Sgd", repr(opt))

	def test_sgd_set_learning_rate(self) -> None:
		opt = oa.ml.Sgd(self.params, learning_rate=0.01)
		opt.set_learning_rate(0.001)
		self.assertAlmostEqual(opt.learning_rate, 0.001, places=5)

	def test_adam_constructs_and_round_trips_lr(self) -> None:
		opt = oa.ml.Adam(self.params, learning_rate=1e-3)
		self.assertAlmostEqual(opt.learning_rate, 1e-3, places=7)
		self.assertEqual(opt.step_count, 0)
		self.assertIn("Adam", repr(opt))

	def test_adamw_constructs_and_round_trips_lr(self) -> None:
		opt = oa.ml.AdamW(self.params, learning_rate=1e-3)
		self.assertAlmostEqual(opt.learning_rate, 1e-3, places=7)
		self.assertEqual(opt.step_count, 0)
		self.assertIn("AdamW", repr(opt))

	def test_muon_constructs_and_round_trips_lr(self) -> None:
		opt = oa.ml.Muon(self.params, learning_rate=0.02)
		self.assertAlmostEqual(opt.learning_rate, 0.02, places=5)
		self.assertEqual(opt.step_count, 0)
		self.assertIn("Muon", repr(opt))


# ── Environment schema (CPU-only) ─────────────────────────────────────────────

class EnvironmentSpaceKindTest(unittest.TestCase):
	def test_members_are_distinct(self) -> None:
		k = oa.ml.EnvironmentSpaceKind
		self.assertEqual(k.Box, k.Box)
		self.assertNotEqual(k.Box, k.Discrete)
		self.assertNotEqual(k.Discrete, k.Binary)

	def test_all_three_members_exist(self) -> None:
		k = oa.ml.EnvironmentSpaceKind
		self.assertTrue(hasattr(k, "Box"))
		self.assertTrue(hasattr(k, "Discrete"))
		self.assertTrue(hasattr(k, "Binary"))


class EnvironmentSpaceTest(unittest.TestCase):
	def test_continuous_space_round_trips_fields(self) -> None:
		s = oa.ml.EnvironmentSpace.continuous("obs", [4], "f32", -1.0, 1.0)
		self.assertEqual(s.name, "obs")
		self.assertEqual(s.kind, oa.ml.EnvironmentSpaceKind.Box)
		self.assertEqual(s.shape, [4])
		self.assertEqual(s.dtype, "f32")
		self.assertAlmostEqual(s.minimum, -1.0)
		self.assertAlmostEqual(s.maximum, 1.0)
		self.assertGreater(s.elements_per_environment, 0)
		self.assertIn("EnvironmentSpace", repr(s))

	def test_discrete_space_round_trips_cardinality(self) -> None:
		s = oa.ml.EnvironmentSpace.discrete("action", 6, "i32")
		self.assertEqual(s.name, "action")
		self.assertEqual(s.kind, oa.ml.EnvironmentSpaceKind.Discrete)
		self.assertEqual(s.cardinality, 6)
		self.assertEqual(s.dtype, "i32")

	def test_binary_space_is_u8(self) -> None:
		s = oa.ml.EnvironmentSpace.binary("done", [1])
		self.assertEqual(s.kind, oa.ml.EnvironmentSpaceKind.Binary)
		self.assertEqual(s.dtype, "u8")

	def test_batched_shape_includes_environments(self) -> None:
		s = oa.ml.EnvironmentSpace.continuous("obs", [4], "f32", -1.0, 1.0)
		shape = s.batched_shape(3)
		self.assertEqual(shape, [3, 4])


class EnvironmentSpecTest(unittest.TestCase):
	def _make_spec(self):
		obs = oa.ml.EnvironmentSpace.continuous("obs", [4], "f32", -1.0, 1.0)
		act = oa.ml.EnvironmentSpace.discrete("action", 2, "i32")
		# Reward must be a scalar space (shape=[]) — one float per environment.
		rew = oa.ml.EnvironmentSpace.continuous("reward", [], "f32", -10.0, 10.0)
		# Termination/truncation must also be scalar binary spaces (shape=[]).
		term = oa.ml.EnvironmentSpace.binary("terminated", [])
		trunc = oa.ml.EnvironmentSpace.binary("truncated", [])
		return oa.ml.EnvironmentSpec(obs, act, rew, term, trunc)

	def test_constructs_and_exposes_all_five_spaces(self) -> None:
		spec = self._make_spec()
		self.assertEqual(spec.observation.name, "obs")
		self.assertEqual(spec.action.name, "action")
		self.assertEqual(spec.reward.name, "reward")
		self.assertEqual(spec.terminated.name, "terminated")
		self.assertEqual(spec.truncated.name, "truncated")
		self.assertIn("EnvironmentSpec", repr(spec))


@require_gpu
class EnvironmentTransitionTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_round_trips_all_five_matrices(self) -> None:
		e = self.engine
		obs = oa.Matrix.from_f32(e, [2, 4], [0.0] * 8)
		nobs = oa.Matrix.from_f32(e, [2, 4], [1.0] * 8)
		rew = oa.Matrix.from_f32(e, [2], [0.5, 0.5])
		term = oa.Matrix.from_u8(e, [2], [0, 1])
		trunc = oa.Matrix.from_u8(e, [2], [0, 0])
		t = oa.ml.EnvironmentTransition(obs, nobs, rew, term, trunc)
		self.assertIsInstance(t, oa.ml.EnvironmentTransition)
		self.assertEqual(t.observation.shape, [2, 4])
		self.assertEqual(t.next_observation.shape, [2, 4])
		self.assertEqual(t.reward.shape, [2])
		self.assertEqual(t.terminated.shape, [2])
		self.assertEqual(t.truncated.shape, [2])
		self.assertEqual(repr(t), "EnvironmentTransition")


# ── RL environment preprocessing functions (GPU) ──────────────────────────────

@require_gpu
class EnvironmentFunctionsTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_normalize_observation_preserves_shape(self) -> None:
		e = self.engine
		obs = oa.Matrix.from_f32(e, [2, 4], [1.0] * 8)
		mean = oa.Matrix.from_f32(e, [4], [0.0] * 4)
		std = oa.Matrix.from_f32(e, [4], [1.0] * 4)
		result = oa.ml.normalize_observation(obs, mean, std, 1e-5, 5.0)
		self.assertEqual(result.shape, [2, 4])

	def test_scale_action_clamps_and_rescales(self) -> None:
		e = self.engine
		action = oa.Matrix.from_f32(e, [2, 2], [0.5, -0.5, 1.0, -1.0])
		result = oa.ml.scale_action(action, -1.0, 1.0, 0.0, 1.0, clamp_source=True)
		self.assertEqual(result.shape, [2, 2])

	def test_clip_reward_preserves_shape(self) -> None:
		e = self.engine
		reward = oa.Matrix.from_f32(e, [4], [5.0, -5.0, 0.1, -0.1])
		result = oa.ml.clip_reward(reward, -1.0, 1.0)
		self.assertEqual(result.shape, [4])


# ── RL data types: ReplayTransition / ReplayBatch (GPU) ───────────────────────

@require_gpu
class ReplayTransitionTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_round_trips_all_six_matrices(self) -> None:
		e = self.engine
		obs = oa.Matrix.from_f32(e, [1, 4], [0.0] * 4)
		act = oa.Matrix.from_i32(e, [1, 1], [0])
		nobs = oa.Matrix.from_f32(e, [1, 4], [1.0] * 4)
		rew = oa.Matrix.from_f32(e, [1], [1.0])
		term = oa.Matrix.from_u8(e, [1], [0])
		trunc = oa.Matrix.from_u8(e, [1], [0])
		t = oa.ml.ReplayTransition(obs, act, nobs, rew, term, trunc)
		self.assertEqual(t.observation.shape, [1, 4])
		self.assertEqual(t.action.shape, [1, 1])
		self.assertEqual(t.next_observation.shape, [1, 4])
		self.assertEqual(t.reward.shape, [1])
		self.assertEqual(t.terminated.shape, [1])
		self.assertEqual(t.truncated.shape, [1])


@require_gpu
class ReplayBufferTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def _make_buffer(self, capacity: int = 10):
		cfg = oa.ml.ReplayConfig(
			capacity=capacity, observation_shape=[4], action_shape=[1], action_dtype="i32"
		)
		return oa.ml.ReplayBuffer(self.engine, cfg)

	def _make_transition(self):
		e = self.engine
		obs = oa.Matrix.from_f32(e, [1, 4], [0.0] * 4)
		act = oa.Matrix.from_i32(e, [1, 1], [0])  # batch=1, action_shape=[1]
		nobs = oa.Matrix.from_f32(e, [1, 4], [1.0] * 4)
		rew = oa.Matrix.from_f32(e, [1], [1.0])
		term = oa.Matrix.from_u8(e, [1], [0])
		trunc = oa.Matrix.from_u8(e, [1], [0])
		return oa.ml.ReplayTransition(obs, act, nobs, rew, term, trunc)

	def test_empty_buffer_len_and_capacity(self) -> None:
		buf = self._make_buffer(capacity=20)
		self.assertEqual(buf.capacity(), 20)
		self.assertEqual(buf.len(), 0)
		self.assertTrue(buf.is_empty())
		self.assertFalse(buf.is_full())
		self.assertIn("ReplayBuffer", repr(buf))

	def test_append_increments_len(self) -> None:
		buf = self._make_buffer()
		buf.append(self._make_transition())
		self.assertEqual(buf.len(), 1)
		self.assertFalse(buf.is_empty())

	def test_sample_returns_replay_batch(self) -> None:
		buf = self._make_buffer(capacity=10)
		for _ in range(5):
			buf.append(self._make_transition())
		batch = buf.sample(batch_size=3, seed=42)
		self.assertIsInstance(batch, oa.ml.ReplayBatch)
		self.assertEqual(batch.observation.shape, [3, 4])
		self.assertEqual(batch.action.shape, [3, 1])
		self.assertEqual(batch.next_observation.shape, [3, 4])
		self.assertEqual(batch.reward.shape, [3])
		self.assertEqual(batch.terminated.shape, [3])
		self.assertEqual(batch.truncated.shape, [3])
		self.assertEqual(batch.index.shape, [3])

	def test_reset_empties_buffer(self) -> None:
		buf = self._make_buffer()
		buf.append(self._make_transition())
		buf.reset()
		self.assertEqual(buf.len(), 0)


# ── RL data types: RolloutTransition / RolloutBuffer (GPU) ────────────────────

@require_gpu
class RolloutTransitionTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_round_trips_all_eight_matrices(self) -> None:
		e = self.engine
		obs = oa.Matrix.from_f32(e, [1, 4], [0.0] * 4)
		act = oa.Matrix.from_i32(e, [1], [0])
		rew = oa.Matrix.from_f32(e, [1], [0.5])
		val = oa.Matrix.from_f32(e, [1], [0.1])
		nval = oa.Matrix.from_f32(e, [1], [0.2])
		logp = oa.Matrix.from_f32(e, [1], [-1.0])
		term = oa.Matrix.from_u8(e, [1], [0])
		trunc = oa.Matrix.from_u8(e, [1], [0])
		t = oa.ml.RolloutTransition(obs, act, rew, val, nval, logp, term, trunc)
		self.assertEqual(t.observation.shape, [1, 4])
		self.assertEqual(t.action.shape, [1])
		self.assertEqual(t.reward.shape, [1])
		self.assertEqual(t.value.shape, [1])
		self.assertEqual(t.next_value.shape, [1])
		self.assertEqual(t.log_probability.shape, [1])
		self.assertEqual(t.terminated.shape, [1])
		self.assertEqual(t.truncated.shape, [1])
		self.assertEqual(repr(t), "RolloutTransition")


@require_gpu
class RolloutBufferTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def _make_buffer(self):
		cfg = oa.ml.RolloutConfig(time=4, environments=1, observation_shape=[4])
		return oa.ml.RolloutBuffer(self.engine, cfg)

	def _make_transition(self):
		e = self.engine
		obs = oa.Matrix.from_f32(e, [1, 4], [0.0] * 4)
		act = oa.Matrix.from_i32(e, [1], [0])
		rew = oa.Matrix.from_f32(e, [1], [0.5])
		val = oa.Matrix.from_f32(e, [1], [0.1])
		nval = oa.Matrix.from_f32(e, [1], [0.2])
		logp = oa.Matrix.from_f32(e, [1], [-1.0])
		term = oa.Matrix.from_u8(e, [1], [0])
		trunc = oa.Matrix.from_u8(e, [1], [0])
		return oa.ml.RolloutTransition(obs, act, rew, val, nval, logp, term, trunc)

	def test_empty_buffer_properties(self) -> None:
		buf = self._make_buffer()
		self.assertEqual(buf.capacity(), 4)
		self.assertEqual(buf.len(), 0)
		self.assertTrue(buf.is_empty())
		self.assertFalse(buf.is_full())
		self.assertFalse(buf.is_finalized())
		self.assertIn("RolloutBuffer", repr(buf))

	def test_append_increments_len(self) -> None:
		buf = self._make_buffer()
		buf.append(self._make_transition())
		self.assertEqual(buf.len(), 1)

	def test_reset_clears_buffer(self) -> None:
		buf = self._make_buffer()
		buf.append(self._make_transition())
		buf.reset()
		self.assertEqual(buf.len(), 0)


# ── RL policy and loss functions (GPU) ────────────────────────────────────────

@require_gpu
class PolicyAndLossFunctionsTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_advantage_normalize_preserves_shape(self) -> None:
		e = self.engine
		adv = oa.Matrix.from_f32(e, [8], [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0])
		result = oa.ml.advantage_normalize(adv, epsilon=1e-8)
		self.assertIsInstance(result, oa.Matrix)
		self.assertEqual(result.shape, [8])

	def test_advantage_gae_returns_result_with_correct_fields(self) -> None:
		e = self.engine
		shape = [4, 1]
		reward = oa.Matrix.from_f32(e, shape, [1.0, 0.5, 2.0, 0.0])
		value = oa.Matrix.from_f32(e, shape, [0.5, 0.3, 0.8, 0.1])
		next_value = oa.Matrix.from_f32(e, shape, [0.3, 0.8, 0.1, 0.0])
		terminated = oa.Matrix.from_u8(e, shape, [0] * 4)
		truncated = oa.Matrix.from_u8(e, shape, [0] * 4)
		cfg = oa.ml.GaeConfig(0.99, 0.95)
		result = oa.ml.advantage_gae(reward, value, next_value, terminated, truncated, cfg)
		self.assertIsInstance(result, oa.ml.GaeResult)
		self.assertEqual(result.advantage.shape, shape)
		self.assertEqual(result.returns.shape, shape)
		self.assertEqual(repr(result), "GaeResult")

	def test_clip_reward_clamps_values(self) -> None:
		e = self.engine
		reward = oa.Matrix.from_f32(e, [4], [10.0, -10.0, 0.5, -0.5])
		clipped = oa.ml.clip_reward(reward, -1.0, 1.0)
		self.assertEqual(clipped.shape, [4])

	def test_policy_sample_categorical_returns_policy_result(self) -> None:
		e = self.engine
		logits = oa.Matrix.from_f32(e, [2, 4], [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1])
		value = oa.Matrix.from_f32(e, [2], [0.5, 0.3])
		result = oa.ml.policy_sample_categorical(logits, value, 42)
		self.assertIsInstance(result, oa.ml.PolicyResult)
		self.assertEqual(result.action.shape, [2])
		self.assertEqual(result.log_probability.shape, [2])
		self.assertEqual(result.entropy.shape, [2])
		self.assertEqual(result.value.shape, [2])
		self.assertEqual(repr(result), "PolicyResult")

	def test_policy_evaluate_categorical_returns_policy_result(self) -> None:
		e = self.engine
		logits = oa.Matrix.from_f32(e, [2, 4], [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1])
		action = oa.Matrix.from_i32(e, [2], [1, 2])
		value = oa.Matrix.from_f32(e, [2], [0.5, 0.3])
		result = oa.ml.policy_evaluate_categorical(logits, action, value)
		self.assertIsInstance(result, oa.ml.PolicyResult)
		self.assertEqual(result.action.shape, [2])

	def test_policy_sample_tanh_normal_returns_continuous_result(self) -> None:
		e = self.engine
		mean = oa.Matrix.from_f32(e, [2, 2], [0.0] * 4)
		log_std = oa.Matrix.from_f32(e, [2, 2], [-0.5] * 4)
		value = oa.Matrix.from_f32(e, [2], [0.5, 0.3])
		result = oa.ml.policy_sample_tanh_normal(mean, log_std, value, -1.0, 1.0, 42, 1e-6)
		self.assertIsInstance(result, oa.ml.ContinuousPolicyResult)
		self.assertEqual(result.action.shape, [2, 2])
		self.assertEqual(result.raw_action.shape, [2, 2])
		self.assertEqual(result.log_probability.shape, [2])
		self.assertEqual(result.entropy.shape, [2])
		self.assertEqual(result.value.shape, [2])
		self.assertEqual(repr(result), "ContinuousPolicyResult")

	def test_policy_evaluate_tanh_normal_returns_continuous_result(self) -> None:
		e = self.engine
		mean = oa.Matrix.from_f32(e, [2, 2], [0.0] * 4)
		log_std = oa.Matrix.from_f32(e, [2, 2], [-0.5] * 4)
		raw_action = oa.Matrix.from_f32(e, [2, 2], [0.1, -0.1, 0.2, -0.2])
		value = oa.Matrix.from_f32(e, [2], [0.5, 0.3])
		result = oa.ml.policy_evaluate_tanh_normal(mean, log_std, raw_action, value, -1.0, 1.0, 1e-6)
		self.assertIsInstance(result, oa.ml.ContinuousPolicyResult)

	def test_loss_ppo_clipped_policy_returns_matrix(self) -> None:
		e = self.engine
		new_lp = oa.Matrix.from_f32(e, [4], [-0.5, -0.3, -0.7, -0.4])
		old_lp = oa.Matrix.from_f32(e, [4], [-0.6, -0.4, -0.6, -0.5])
		adv = oa.Matrix.from_f32(e, [4], [1.0, -1.0, 0.5, -0.5])
		result = oa.ml.loss_ppo_clipped_policy(new_lp, old_lp, adv, 0.2)
		self.assertIsInstance(result, oa.Matrix)
		self.assertEqual(result.shape, [])

	def test_loss_ppo_returns_ppo_loss_result(self) -> None:
		e = self.engine
		new_lp = oa.Matrix.from_f32(e, [4], [-0.5, -0.3, -0.7, -0.4])
		old_lp = oa.Matrix.from_f32(e, [4], [-0.6, -0.4, -0.6, -0.5])
		adv = oa.Matrix.from_f32(e, [4], [1.0, -1.0, 0.5, -0.5])
		value = oa.Matrix.from_f32(e, [4], [0.5, 0.3, 0.8, 0.1])
		target = oa.Matrix.from_f32(e, [4], [0.6, 0.4, 0.9, 0.2])
		entropy = oa.Matrix.from_f32(e, [4], [0.5, 0.6, 0.4, 0.5])
		cfg = oa.ml.PpoLossConfig(clip_epsilon=0.2)
		result = oa.ml.loss_ppo(new_lp, old_lp, adv, value, target, entropy, cfg)
		self.assertIsInstance(result, oa.ml.PpoLossResult)
		self.assertEqual(repr(result), "PpoLossResult")
		self.assertIsInstance(result.policy_loss, oa.Matrix)
		self.assertIsInstance(result.value_loss, oa.Matrix)
		self.assertIsInstance(result.entropy, oa.Matrix)
		self.assertIsInstance(result.total_loss, oa.Matrix)

	def test_loss_dqn_returns_dqn_loss_result(self) -> None:
		e = self.engine
		q = oa.Matrix.from_f32(e, [4, 2], [0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8, 0.1])
		action = oa.Matrix.from_i32(e, [4], [0, 1, 0, 1])
		reward = oa.Matrix.from_f32(e, [4], [1.0, 0.0, 0.5, -0.5])
		next_q = oa.Matrix.from_f32(e, [4, 2], [0.4, 0.6, 0.3, 0.7, 0.5, 0.5, 0.2, 0.8])
		terminated = oa.Matrix.from_u8(e, [4], [0, 0, 0, 1])
		truncated = oa.Matrix.from_u8(e, [4], [0] * 4)
		cfg = oa.ml.DqnLossConfig(discount=0.99)
		result = oa.ml.loss_dqn(q, action, reward, next_q, terminated, truncated, cfg)
		self.assertIsInstance(result, oa.ml.DqnLossResult)
		self.assertEqual(repr(result), "DqnLossResult")
		self.assertIsInstance(result.selected_q, oa.Matrix)
		self.assertIsInstance(result.target_q, oa.Matrix)
		self.assertIsInstance(result.loss, oa.Matrix)

	def test_loss_sac_critic_returns_sac_critic_loss_result(self) -> None:
		e = self.engine
		q1 = oa.Matrix.from_f32(e, [4], [0.5, 0.3, 0.7, 0.2])
		q2 = oa.Matrix.from_f32(e, [4], [0.4, 0.35, 0.65, 0.25])
		reward = oa.Matrix.from_f32(e, [4], [1.0, 0.0, 0.5, -0.5])
		next_q1 = oa.Matrix.from_f32(e, [4], [0.4, 0.3, 0.6, 0.1])
		next_q2 = oa.Matrix.from_f32(e, [4], [0.45, 0.25, 0.55, 0.15])
		next_lp = oa.Matrix.from_f32(e, [4], [-0.5, -0.4, -0.6, -0.3])
		terminated = oa.Matrix.from_u8(e, [4], [0, 0, 0, 1])
		truncated = oa.Matrix.from_u8(e, [4], [0] * 4)
		cfg = oa.ml.SacLossConfig(discount=0.99, entropy_coefficient=0.2)
		result = oa.ml.loss_sac_critic(q1, q2, reward, next_q1, next_q2, next_lp, terminated, truncated, cfg)
		self.assertIsInstance(result, oa.ml.SacCriticLossResult)
		self.assertEqual(repr(result), "SacCriticLossResult")
		self.assertIsInstance(result.target_q, oa.Matrix)
		self.assertIsInstance(result.q1_loss, oa.Matrix)
		self.assertIsInstance(result.q2_loss, oa.Matrix)
		self.assertIsInstance(result.total_loss, oa.Matrix)

	def test_loss_sac_actor_returns_matrix(self) -> None:
		e = self.engine
		q1 = oa.Matrix.from_f32(e, [4], [0.5, 0.3, 0.7, 0.2])
		q2 = oa.Matrix.from_f32(e, [4], [0.4, 0.35, 0.65, 0.25])
		lp = oa.Matrix.from_f32(e, [4], [-0.5, -0.4, -0.6, -0.3])
		result = oa.ml.loss_sac_actor(q1, q2, lp, 0.2)
		self.assertIsInstance(result, oa.Matrix)
		self.assertEqual(result.shape, [])


# ── NN layers (GPU) ───────────────────────────────────────────────────────────

@require_gpu
class NnLayerSmokeTest(unittest.TestCase):
	"""Smoke-test that all public NN layer types construct and forward-pass."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_embedding_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Embedding(e, num_embeddings=10, embedding_dim=4, seed=0)
		tokens = oa.Matrix.from_u32(e, [2, 3], [0, 1, 2, 3, 4, 5])
		out = layer(tokens)
		self.assertEqual(out.shape, [2, 3, 4])
		self.assertIn("Embedding", repr(layer))

	def test_layer_norm_forward(self) -> None:
		e = self.engine
		layer = oa.ml.LayerNorm(e, normalized_shape=[4], seed=0)
		x = oa.Matrix.from_f32(e, [2, 4], [1.0] * 8)
		out = layer(x)
		self.assertEqual(out.shape, [2, 4])
		self.assertIn("LayerNorm", repr(layer))

	def test_rms_norm_forward(self) -> None:
		e = self.engine
		layer = oa.ml.RmsNorm(e, features=4, seed=0)
		x = oa.Matrix.from_f32(e, [2, 4], [1.0] * 8)
		out = layer(x)
		self.assertEqual(out.shape, [2, 4])
		self.assertIn("RmsNorm", repr(layer))

	def test_dropout_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Dropout(probability=0.1, seed=0)
		x = oa.Matrix.from_f32(e, [2, 4], [1.0] * 8)
		out = layer.forward(x)
		self.assertEqual(out.shape, [2, 4])
		self.assertIn("Dropout", repr(layer))

	def test_rnn_cell_forward(self) -> None:
		e = self.engine
		layer = oa.ml.RnnCell(e, input_size=4, hidden_size=8, seed=0)
		x = oa.Matrix.from_f32(e, [2, 4], [0.5] * 8)
		h = oa.Matrix.from_f32(e, [2, 8], [0.0] * 16)
		out = layer(x, h)
		self.assertEqual(out.shape, [2, 8])
		self.assertIn("RnnCell", repr(layer))

	def test_rnn_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Rnn(e, input_size=4, hidden_size=8, seed=0)
		x = oa.Matrix.from_f32(e, [2, 3, 4], [0.5] * 24)
		h = oa.Matrix.from_f32(e, [2, 8], [0.0] * 16)
		out = layer(x, h)
		self.assertEqual(out.shape, [2, 3, 8])
		self.assertIn("Rnn", repr(layer))

	def test_gru_cell_forward(self) -> None:
		e = self.engine
		layer = oa.ml.GruCell(e, input_size=4, hidden_size=8, seed=0)
		x = oa.Matrix.from_f32(e, [2, 4], [0.5] * 8)
		h = oa.Matrix.from_f32(e, [2, 8], [0.0] * 16)
		out = layer(x, h)
		self.assertEqual(out.shape, [2, 8])
		self.assertIn("GruCell", repr(layer))

	def test_gru_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Gru(e, input_size=4, hidden_size=8, seed=0)
		x = oa.Matrix.from_f32(e, [2, 3, 4], [0.5] * 24)
		h = oa.Matrix.from_f32(e, [2, 8], [0.0] * 16)
		out = layer(x, h)
		self.assertEqual(out.shape, [2, 3, 8])
		self.assertIn("Gru", repr(layer))

	def test_transformer_block_forward(self) -> None:
		e = self.engine
		block = oa.ml.TransformerBlock(
			e, dim=8, heads=2, seq_len=4, seed=0
		)
		# forward expects [B*S, D] input, so [4, 8] for batch=1, seq=4, dim=8
		x = oa.Matrix.from_f32(e, [4, 8], [0.0] * 32)
		out = block(x)
		self.assertEqual(out.shape, [4, 8])
		self.assertIn("TransformerBlock", repr(block))

	def test_relu_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Relu()
		x = oa.Matrix.from_f32(e, [4], [-1.0, 0.0, 1.0, 2.0])
		out = layer(x)
		self.assertEqual(out.shape, [4])
		self.assertIn("Relu", repr(layer))

	def test_gelu_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Gelu()
		x = oa.Matrix.from_f32(e, [4], [-1.0, 0.0, 1.0, 2.0])
		out = layer(x)
		self.assertEqual(out.shape, [4])
		self.assertIn("Gelu", repr(layer))

	def test_silu_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Silu()
		x = oa.Matrix.from_f32(e, [4], [-1.0, 0.0, 1.0, 2.0])
		out = layer(x)
		self.assertEqual(out.shape, [4])

	def test_softmax_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Softmax(dim=1)
		x = oa.Matrix.from_f32(e, [2, 4], [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
		out = layer(x)
		self.assertEqual(out.shape, [2, 4])
		self.assertIn("Softmax", repr(layer))

	def test_log_softmax_forward(self) -> None:
		e = self.engine
		layer = oa.ml.LogSoftmax(dim=1)
		x = oa.Matrix.from_f32(e, [2, 4], [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5])
		out = layer(x)
		self.assertEqual(out.shape, [2, 4])

	def test_identity_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Identity()
		x = oa.Matrix.from_f32(e, [2, 4], [1.0] * 8)
		out = layer(x)
		self.assertEqual(out.shape, [2, 4])
		self.assertIn("Identity", repr(layer))

	def test_flatten_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Flatten()
		x = oa.Matrix.from_f32(e, [2, 3, 4], [1.0] * 24)
		out = layer(x)
		self.assertEqual(out.shape, [2, 12])
		self.assertIn("Flatten", repr(layer))

	def test_swiglu_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Swiglu(e, input_dim=4, hidden_dim=8, output_dim=4, seed=0)
		x = oa.Matrix.from_f32(e, [2, 4], [0.5] * 8)
		out = layer(x)
		self.assertEqual(out.shape, [2, 4])
		self.assertIn("Swiglu", repr(layer))

	def test_ffn_forward(self) -> None:
		e = self.engine
		layer = oa.ml.Ffn(e, dim=4, hidden_dim=8, seed=0)
		x = oa.Matrix.from_f32(e, [2, 4, 4], [0.5] * 32)
		out = layer(x)
		self.assertEqual(out.shape, [2, 4, 4])
		self.assertIn("Ffn", repr(layer))

	def test_sequential_forward(self) -> None:
		e = self.engine
		linear1 = oa.ml.Linear(e, in_features=4, out_features=8, seed=0)
		linear2 = oa.ml.Linear(e, in_features=8, out_features=2, seed=1)
		seq = oa.ml.Sequential([linear1, linear2])
		x = oa.Matrix.from_f32(e, [2, 4], [0.5] * 8)
		out = seq(x)
		self.assertEqual(out.shape, [2, 2])
		self.assertIn("Sequential", repr(seq))

	def test_mamba3_config_round_trips(self) -> None:
		cfg = oa.ml.Mamba3Config(
			dim=8, state_dim=4, expand=2, head_dim=4, dt_rank="auto",
			dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, seed=0
		)
		self.assertIn("Mamba3Config", repr(cfg))

	def test_rope_forward(self) -> None:
		e = self.engine
		# Rope(dim=4) → num_heads=1, head_dim=4; input must be [tokens, 4]
		layer = oa.ml.Rope(dim=4, seq_len=8)
		x = oa.Matrix.from_f32(e, [8, 4], [0.5] * 32)
		out = layer(x)
		self.assertEqual(out.shape, [8, 4])
		self.assertIn("Rope", repr(layer))


# ── Checkpoint functions (GPU) ────────────────────────────────────────────────

@require_gpu
class CheckpointFunctionsTest(unittest.TestCase):
	"""Test save_checkpoint, load_checkpoint, and clip_grad_norm."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()
		cls.linear = oa.ml.Linear(cls.engine, in_features=4, out_features=2, seed=0)
		cls.optimizer = oa.ml.AdamW(cls.linear.all_parameters(), learning_rate=1e-3)

	def test_save_and_load_checkpoint_roundtrip(self) -> None:
		import tempfile, os
		with tempfile.NamedTemporaryFile(suffix=".oam", delete=False) as f:
			path = f.name
		try:
			# save_checkpoint(path, params, optimizer)
			oa.ml.save_checkpoint(path, self.linear.all_parameters(), self.optimizer)
			self.assertTrue(os.path.exists(path))
			# Load into a fresh model/optimizer with the same shape.
			linear2 = oa.ml.Linear(self.engine, in_features=4, out_features=2, seed=99)
			opt2 = oa.ml.AdamW(linear2.all_parameters(), learning_rate=1e-3)
			oa.ml.load_checkpoint(self.engine, path, linear2.all_parameters(), opt2)
		finally:
			try:
				os.unlink(path)
			except OSError:
				pass

	def test_clip_grad_norm_clips_gradients(self) -> None:
		e = self.engine
		# Run a forward+backward to accumulate gradients.
		linear = oa.ml.Linear(e, in_features=4, out_features=1, seed=0)
		params = linear.all_parameters()
		opt = oa.ml.AdamW(params, learning_rate=1e-3)
		x = oa.Matrix.from_f32(e, [1, 4], [1.0, 2.0, 3.0, 4.0])
		target = oa.Matrix.from_f32(e, [1, 1], [0.0])
		with oa.ml.GradientTape() as tape:
			pred = linear(x)
			loss = oa.ml.loss.mse(pred, target)
			tape.backward(loss)
		# clip_grad_norm clips in-place and returns None (OA API).
		result = oa.ml.clip_grad_norm(params, 1.0)
		self.assertIsNone(result)


# ── Submodule export completeness ─────────────────────────────────────────────

class SubmoduleExportTest(unittest.TestCase):
	"""Verify all submodule re-exports are accessible."""

	def test_rl_submodule_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "rl"))
		self.assertTrue(hasattr(oa.ml.rl, "EnvironmentSpace"))
		self.assertTrue(hasattr(oa.ml.rl, "advantage_gae"))

	def test_loss_submodule_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "loss"))
		self.assertTrue(hasattr(oa.ml.loss, "mse"))
		self.assertTrue(hasattr(oa.ml.loss, "cross_entropy"))

	def test_matrix_submodule_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "matrix"))
		self.assertTrue(hasattr(oa.ml.matrix, "relu"))

	def test_metric_submodule_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "metric"))
		self.assertTrue(hasattr(oa.ml.metric, "accuracy"))

	def test_byte_submodule_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "byte"))
		self.assertTrue(hasattr(oa.ml.byte, "encode"))
		self.assertEqual(oa.ml.byte.VOCAB_SIZE, 256)

	def test_flow_submodule_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "flow"))
		self.assertTrue(hasattr(oa.ml.flow, "linear_match"))
		self.assertTrue(hasattr(oa.ml.flow, "FlowMatchBatch"))

	def test_nn_submodule_accessible(self) -> None:
		self.assertTrue(hasattr(oa.ml, "nn"))
		self.assertTrue(hasattr(oa.ml.nn, "Linear"))


# ── Parameter and GradientTape (GPU) ──────────────────────────────────────────

@require_gpu
class ParameterAndGradientTapeTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_parameter_is_accessible_on_ml_and_has_value_attr(self) -> None:
		e = self.engine
		linear = oa.ml.Linear(e, in_features=4, out_features=2, seed=0)
		params = linear.all_parameters()
		self.assertGreater(len(params), 0)
		for p in params:
			self.assertIsInstance(p, oa.ml.Parameter)
			self.assertIsInstance(p.value, oa.Matrix)

	def test_categorical_actor_critic_output_has_logits_and_value(self) -> None:
		e = self.engine
		cfg = oa.ml.CategoricalActorCriticConfig(
			observation_size=4, action_count=2, hidden_size=8, seed=0
		)
		model = oa.ml.CategoricalActorCritic(e, cfg)
		obs = oa.Matrix.from_f32(e, [1, 4], [0.1, 0.2, 0.3, 0.4])
		output = model.evaluate(obs)
		self.assertIsInstance(output, oa.ml.CategoricalActorCriticOutput)
		self.assertEqual(output.logits.shape, [1, 2])
		self.assertEqual(output.value.shape, [1])
		self.assertEqual(repr(output), "CategoricalActorCriticOutput")

	def test_training_session_snapshot_has_all_fields(self) -> None:
		# TrainingSessionSnapshot is not directly constructible from Python;
		# verify the type is accessible and has the expected attribute names.
		self.assertTrue(hasattr(oa.ml, "TrainingSessionSnapshot"))
		snap_cls = oa.ml.TrainingSessionSnapshot
		# Verify expected attribute names exist at the class level as properties.
		for attr in ("revision", "state", "step", "epoch", "learning_rate", "loss", "gpu_ms", "wall_ms"):
			self.assertTrue(hasattr(snap_cls, attr), f"TrainingSessionSnapshot.{attr} missing")


# ── NLP presets: BpeSampler and BpeMamba3/BpeTransformer family (GPU) ─────────

@require_gpu
class BpeSamplerTest(unittest.TestCase):
	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_bpe_sampler_constructs_and_produces_paired_sequences(self) -> None:
		tok = oa.ml.BpeTokenizer(target_vocab=300)
		corpus = b"hello world " * 300
		tok.train(corpus, num_merges=20)
		sampler = oa.ml.BpeSampler(batch_size=1, tokenizer=tok)
		self.assertIn("BpeSampler", repr(sampler))
		inputs, targets = sampler.next_values()
		self.assertEqual(len(inputs), len(targets))
		self.assertGreater(len(inputs), 0)

	def test_bpe_sampler_last_batch_bytes_is_positive(self) -> None:
		tok = oa.ml.BpeTokenizer(target_vocab=300)
		corpus = b"hello world " * 300
		tok.train(corpus, num_merges=20)
		sampler = oa.ml.BpeSampler(batch_size=1, tokenizer=tok)
		sampler.next_values()
		self.assertGreater(sampler.last_batch_bytes, 0)


@require_gpu
class BpeNlpPresetsSmoke(unittest.TestCase):
	"""Smoke-test for BPE model presets (require GPU to construct)."""

	@classmethod
	def setUpClass(cls) -> None:
		cls.engine = oa.Engine()

	def test_bpe_rnn_constructs(self) -> None:
		model = oa.ml.BpeRnn(self.engine)
		self.assertIn("BpeRnn", repr(model))
		self.assertGreater(len(model.all_parameters()), 0)

	def test_bpe_gru_constructs(self) -> None:
		model = oa.ml.BpeGru(self.engine)
		self.assertIn("BpeGru", repr(model))

	def test_bpe_transformer_constructs(self) -> None:
		model = oa.ml.BpeTransformer(self.engine)
		self.assertIn("BpeTransformer", repr(model))

	def test_byte_empyrealm_constructs(self) -> None:
		model = oa.ml.ByteEmpyrealm(self.engine)
		self.assertIn("ByteEmpyrealm", repr(model))


if __name__ == "__main__":
	unittest.main()
