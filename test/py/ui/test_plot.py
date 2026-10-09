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


if __name__ == "__main__":
	unittest.main()
