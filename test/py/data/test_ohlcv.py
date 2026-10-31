"""Native bindings: exact time, immutable views and checked candle windows."""

import gc
import unittest

import oa


def frame(rows=6):
	value = oa.DataFrame()
	value.set_i64("timestamp", [1_700_000_000_000_000_001 + i for i in range(rows)])
	for name, number in (("open", 10), ("high", 12), ("low", 9), ("close", 11), ("volume", 100)):
		value.set_f64(name, [float(number)] * rows)
	return value


class OhlcvBindingTest(unittest.TestCase):
	def test_module_identity(self):
		for name in ("DataFrame", "DataFrameView", "DsOhlcv", "OhlcvWindows", "UnixTimeUnit"):
			self.assertIs(getattr(oa, name), getattr(oa.data, name))

	def test_exact_time_and_transactional_replacement(self):
		value = frame()
		self.assertEqual(value.column("timestamp")[0], 1_700_000_000_000_000_001)
		self.assertEqual(value.column_names(), ["timestamp", "open", "high", "low", "close", "volume"])
		with self.assertRaises(ValueError):
			value.set_f32("close", [1.0])
		self.assertEqual(value.column("close"), [11.0] * 6)
		with self.assertRaises(OverflowError):
			value.set_i64("timestamp", [2**63] * 6)
		with self.assertRaises(ValueError):
			value.column("missing")

	def test_views_retain_snapshot_across_edits_and_deletion(self):
		value = frame()
		view = value.slice(1, 4)
		windows = oa.OhlcvWindows.new_from_view(view, oa.UnixTimeUnit.Nanoseconds, 2, 1)
		value.set_f64("close", [10.0] * 6)
		self.assertEqual(value.column("close"), [10.0] * 6)
		del value, view
		gc.collect()
		history, future = windows.window(0)
		self.assertEqual(history.column("close"), [11.0, 11.0])
		self.assertEqual(future.column("timestamp"), [1_700_000_000_000_000_004])
		self.assertEqual(len(windows), 1)
		self.assertEqual(windows.time_unit, oa.UnixTimeUnit.Nanoseconds)
		with self.assertRaises(ValueError):
			windows.window(-1)
		with self.assertRaises(ValueError):
			windows.window(1)

	def test_partitioned_windows_and_validation(self):
		value = frame()
		train = oa.OhlcvWindows.new_from_view(value.slice(0, 3), oa.UnixTimeUnit.Nanoseconds, 2, 1)
		test = oa.OhlcvWindows.new_from_view(value.slice(3, 6), oa.UnixTimeUnit.Nanoseconds, 2, 1)
		self.assertEqual((len(train), len(test)), (1, 1))
		with self.assertRaises(ValueError):
			oa.OhlcvWindows(value, oa.UnixTimeUnit.Seconds, 0, 1)
		with self.assertRaises(ValueError):
			value.slice(4, 3)
		value.set_f64("volume", [-1.0] * 6)
		with self.assertRaises(ValueError):
			oa.OhlcvWindows(value, oa.UnixTimeUnit.Seconds, 2, 1)
		# Invalid rows outside a partition do not invalidate that partition.
		value.set_f64("volume", [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
		self.assertEqual(len(oa.OhlcvWindows.new_from_view(value.slice(3, 6), oa.UnixTimeUnit.Nanoseconds, 2, 1)), 1)

	def test_dataset_retains_engine_and_uploads_native_sample(self):
		value = frame()
		value.set_f32("close", [10.0 + row * 0.1 for row in range(6)])
		windows = oa.OhlcvWindows(value, oa.UnixTimeUnit.Nanoseconds, 2, 2)
		engine = oa.Engine(logging=oa.LogOptions().console_output(False))
		dataset = oa.DsOhlcv(engine, windows)
		del engine, windows, value
		gc.collect()
		self.assertEqual(len(dataset), 3)
		x, y = dataset.get_sample(1)
		self.assertEqual(x.shape, [2, 5])
		self.assertEqual(y.shape, [2])
		for got, expected in zip(x.read_f32(), [10, 12, 9, 10.1, 100, 10, 12, 9, 10.2, 100]):
			self.assertAlmostEqual(got, expected, places=5)
		for got, expected in zip(y.read_f32(), [10.3, 10.4]):
			self.assertAlmostEqual(got, expected, places=5)
		self.assertEqual(dataset.get_item(1).read_f32(), x.read_f32())
		with self.assertRaises(ValueError):
			dataset.get_sample(-1)
		with self.assertRaises(ValueError):
			dataset.get_item(3)


if __name__ == "__main__":
	unittest.main()
