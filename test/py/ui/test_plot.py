import unittest

import oa


class PlotBindingTest(unittest.TestCase):
	def test_retained_axes_keeps_figure_alive(self) -> None:
		figure = oa.plot.Figure(oa.plot.FigureConfig(rows=2, cols=3))
		axes = figure.ax(1, 2)
		del figure
		axes.title("retained")
		axes.plot([0.0, 1.0, 0.5])
		axes.scatter([0.0, 1.0], [1.0, 0.0])

	def test_figure_normalizes_grid_and_clamps_indices(self) -> None:
		figure = oa.plot.Figure(oa.plot.FigureConfig(rows=0, cols=-2, padding=-8))
		self.assertEqual(figure.rows(), 1)
		self.assertEqual(figure.cols(), 1)
		a = figure.cell_rect(-5, 9, 100, 80)
		b = figure.cell_rect(0, 0, 100, 80)
		self.assertEqual((a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h))

	def test_style_construction(self) -> None:
		self.assertIsInstance(oa.plot.LineStyle(label="train"), oa.plot.LineStyle)
		self.assertIsInstance(oa.plot.HeatmapStyle(colormap=2), oa.plot.HeatmapStyle)
		self.assertIsInstance(oa.plot.BoxStyle(width=0.4, whisk=1.5), oa.plot.BoxStyle)

	def test_stream_series_returns_handle_and_pushes(self) -> None:
		axes = oa.plot.Figure().ax(0, 0)
		series = axes.stream(8, oa.plot.LineStyle(step=True))
		self.assertIsInstance(series, int)
		axes.stream_push(series, [1.0, 2.0, 3.0])
		axes.stream_push(series, [4.0])
		self.assertEqual(axes.artist_count(), 1)

	def test_band_grouped_and_stacked_bars_record_artists(self) -> None:
		axes = oa.plot.Figure().ax(0, 0)
		axes.band([0.0, 1.0], [2.0, 3.0])
		self.assertEqual(axes.artist_count(), 1)
		axes.bar_grouped([[1.0, 2.0], [3.0, 4.0]], [oa.plot.BarStyle(), oa.plot.BarStyle()])
		self.assertEqual(axes.artist_count(), 3)
		axes.bar_stacked([[1.0, 2.0], [3.0, 4.0]])
		self.assertEqual(axes.artist_count(), 5)

	def test_boxplot_records_summary_artist(self) -> None:
		axes = oa.plot.Figure().ax(0, 0)
		axes.boxplot([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0]], oa.plot.BoxStyle())
		self.assertEqual(axes.artist_count(), 1)

	def test_scales_ticks_and_categories(self) -> None:
		axes = oa.plot.Figure().ax(0, 0)
		axes.xscale(oa.plot.PlotScale.Log10)
		axes.yscale(oa.plot.PlotScale.Symlog)
		axes.xticks([1.0, 10.0, 100.0], decimals=0)
		axes.yticks([-10.0, 0.0, 10.0], labels=["neg", "zero", "pos"])
		axes.categories(["a", "b", "c"])

	def test_limits_round_trip_and_artist_visibility(self) -> None:
		axes = oa.plot.Figure().ax(0, 0)
		axes.limits(0.0, 10.0, -1.0, 1.0)
		self.assertEqual(axes.explicit_limits(), (0.0, 10.0, -1.0, 1.0))
		axes.auto_limits()
		self.assertIsNone(axes.explicit_limits())
		axes.plot([1.0, 2.0, 3.0])
		limits = axes.data_limits()
		self.assertEqual(len(limits), 4)
		axes.set_artist_hidden(0, True)

	def test_figure_view_takes_figure_and_delegates_axes(self) -> None:
		figure = oa.plot.Figure(oa.plot.FigureConfig(rows=2, cols=2))
		figure.ax(0, 0).plot([0.0, 1.0, 0.5])
		view = oa.plot.FigureView(figure)
		# The view owns the recorded content; the source figure is left empty.
		self.assertEqual(figure.ax(0, 0).artist_count(), 0)
		axes = view.ax(0, 0)
		self.assertEqual(axes.artist_count(), 1)
		series = axes.stream(16)
		axes.stream_push(series, [1.0, 2.0, 3.0])
		self.assertEqual(axes.artist_count(), 2)
		view.crosshair(False)
		view.reset_view()


if __name__ == "__main__":
	unittest.main()
