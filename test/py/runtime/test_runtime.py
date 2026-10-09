"""Tests for the oa.runtime Python binding surface.

Coverage:
* LogLevel enum — all 7 members exist and are distinct.
* LogOptions — constructor, builder methods, repr.
* Engine — construction with explicit LogOptions, log_level getter/setter,
  log_path (None when file output is disabled), flush_log.
* Identity aliases: oa.LogLevel / oa.runtime.LogLevel, oa.LogOptions / oa.runtime.LogOptions.
"""

import unittest

import oa


class LogLevelTest(unittest.TestCase):
	EXPECTED = ["Trace", "Debug", "Info", "Warn", "Error", "Fatal", "Off"]

	def test_all_members_exist_on_oa_runtime(self) -> None:
		for name in self.EXPECTED:
			with self.subTest(level=name):
				self.assertTrue(
					hasattr(oa.runtime.LogLevel, name),
					f"LogLevel.{name} is missing from oa.runtime",
				)

	def test_members_are_accessible_at_root(self) -> None:
		for name in self.EXPECTED:
			with self.subTest(level=name):
				self.assertTrue(
					hasattr(oa.LogLevel, name),
					f"LogLevel.{name} is missing from oa root",
				)

	def test_identity_alias(self) -> None:
		self.assertIs(oa.LogLevel, oa.runtime.LogLevel)

	def test_members_compare_equal_to_themselves(self) -> None:
		ll = oa.LogLevel
		self.assertEqual(ll.Info, ll.Info)
		self.assertNotEqual(ll.Info, ll.Warn)
		self.assertNotEqual(ll.Trace, ll.Off)

	def test_all_members_are_distinct(self) -> None:
		ll = oa.LogLevel
		members = [ll.Trace, ll.Debug, ll.Info, ll.Warn, ll.Error, ll.Fatal, ll.Off]
		for i, a in enumerate(members):
			for j, b in enumerate(members):
				if i != j:
					self.assertNotEqual(a, b, f"LogLevel[{i}] == LogLevel[{j}]")


class LogOptionsTest(unittest.TestCase):
	def test_identity_alias(self) -> None:
		self.assertIs(oa.LogOptions, oa.runtime.LogOptions)

	def test_constructs_with_default_level_info(self) -> None:
		# Default minimum level is Info in release builds (or Debug in debug).
		opts = oa.LogOptions()
		level = opts.level()
		self.assertIn(level, (oa.LogLevel.Debug, oa.LogLevel.Info))

	def test_minimum_level_builder_returns_same_object(self) -> None:
		opts = oa.LogOptions()
		result = opts.minimum_level(oa.LogLevel.Warn)
		# Builder returns self for chaining.
		self.assertIs(opts, result)
		self.assertEqual(opts.level(), oa.LogLevel.Warn)

	def test_console_output_builder(self) -> None:
		opts = oa.LogOptions()
		opts.console_output(False)
		# No error — just verifying the call is accepted.

	def test_color_output_builder(self) -> None:
		opts = oa.LogOptions()
		opts.color_output(False)

	def test_prefix_builder(self) -> None:
		opts = oa.LogOptions()
		opts.prefix("test")

	def test_repr_includes_class_name(self) -> None:
		self.assertIn("LogOptions", repr(oa.LogOptions()))


class EngineLoggingTest(unittest.TestCase):
	def test_default_engine_construction_succeeds(self) -> None:
		# Basic construction without explicit LogOptions.
		engine = oa.Engine()
		self.assertIsInstance(engine, oa.Engine)

	def test_engine_with_explicit_log_options(self) -> None:
		opts = oa.LogOptions()
		opts.minimum_level(oa.LogLevel.Warn)
		opts.console_output(False)
		engine = oa.Engine(logging=opts)
		self.assertIsInstance(engine, oa.Engine)

	def test_log_level_getter_returns_log_level(self) -> None:
		engine = oa.Engine()
		level = engine.log_level()
		# Level must be one of the known enum members.
		known = [
			oa.LogLevel.Trace,
			oa.LogLevel.Debug,
			oa.LogLevel.Info,
			oa.LogLevel.Warn,
			oa.LogLevel.Error,
			oa.LogLevel.Fatal,
			oa.LogLevel.Off,
		]
		self.assertIn(level, known)

	def test_set_log_level_takes_effect(self) -> None:
		engine = oa.Engine()
		engine.set_log_level(oa.LogLevel.Off)
		self.assertEqual(engine.log_level(), oa.LogLevel.Off)
		# Restore to Info so subsequent tests are not silenced.
		engine.set_log_level(oa.LogLevel.Info)
		self.assertEqual(engine.log_level(), oa.LogLevel.Info)

	def test_log_path_is_none_without_file_output(self) -> None:
		# No directory configured → file output not enabled → path is None.
		opts = oa.LogOptions()
		opts.console_output(False)
		engine = oa.Engine(logging=opts)
		self.assertIsNone(engine.log_path())

	def test_flush_log_succeeds(self) -> None:
		engine = oa.Engine()
		# flush_log must not raise when console-only logging is active.
		engine.flush_log()

	def test_existing_aliases_still_work(self) -> None:
		self.assertIs(oa.Engine, oa.runtime.Engine)
		self.assertIs(oa.Event, oa.runtime.Event)

	def test_checkpoint_returns_a_waitable_event(self) -> None:
		engine = oa.Engine()
		oa.matrix.ones(engine, [1])
		event = engine.checkpoint()
		event.wait()
		self.assertTrue(event.is_complete())
		self.assertIn("Event", repr(event))


if __name__ == "__main__":
	unittest.main()
