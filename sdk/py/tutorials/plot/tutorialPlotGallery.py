#!/usr/bin/env python3
"""One-to-one Python companion to sdk/cpp/tutorials/plot/tutorialPlotGallery.cpp.

Both tutorials compose the same retained figures, exercise the same public
artists, and write the same six presentation filenames. Use --headless for
automation; without it the dark gallery also opens in Viewer.
"""

# pyright: reportWildcardImportFromLibrary=false

from __future__ import annotations

import argparse
import math
from pathlib import Path as FilePath

from oa import *


pi = math.pi


def rgba(r: float, g: float, b: float, a: float = 1.0) -> int:
	def channel(value: float) -> int:
		return int(math.floor(max(0.0, min(1.0, value)) * 255.0 + 0.5))

	return (
		(channel(r) << 24)
		| (channel(g) << 16)
		| (channel(b) << 8)
		| channel(a)
	)


accent = rgba(0.388, 0.400, 0.945)
success = rgba(0.188, 0.820, 0.345)
warning = rgba(0.961, 0.620, 0.043)
cyan = rgba(0.133, 0.827, 0.933)
purple = rgba(0.659, 0.333, 0.969)
pink = rgba(0.925, 0.282, 0.600)
gray = rgba(0.565, 0.565, 0.565)


def line(
	colorRgba: int,
	label: str = "",
	width: float = 1.35,
	antialiasSamples: int = 4,
) -> plot.LineStyle:
	style = plot.LineStyle()
	style.colorRgba = colorRgba
	style.label = label
	style.width = width
	style.antialiasSamples = antialiasSamples
	return style


def points(
	colorRgba: int, label: str = "", radius: float = 3.0
) -> plot.ScatterStyle:
	style = plot.ScatterStyle()
	style.colorRgba = colorRgba
	style.label = label
	style.radius = radius
	return style


def bars(
	colorRgba: int, label: str = "", gap: float = 0.18
) -> plot.BarStyle:
	style = plot.BarStyle()
	style.colorRgba = colorRgba
	style.label = label
	style.gap = gap
	return style


def gradient(inA: tuple[float, float, float], inB: tuple[float, float, float], inT: float) -> int:
	t = max(0.0, min(1.0, inT))
	return rgba(*(
		inA[index] + (inB[index] - inA[index]) * t
		for index in range(3)
	))


# [oa-plot-intro-begin]
def introFigure() -> plot.Figure:
	config = plot.FigureConfig()
	config.title = "OA plot intro"
	config.rows, config.cols = 1, 2
	config.width, config.height = 1280, 560
	config.hSpacing, config.padding = 36, 34
	config.theme = plot.Theme.Dark
	figure = plot.Figure(config)
	figure.title("One retained figure - C++ and Python parity")

	steps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
	train = [1.00, 0.78, 0.61, 0.48, 0.37, 0.29, 0.23, 0.19]
	validation = [1.04, 0.83, 0.66, 0.53, 0.43, 0.35, 0.30, 0.27]

	curves = figure.ax(0, 0)
	curves.title("Training curves")
	curves.xLabel("optimizer step")
	curves.yLabel("cross entropy")
	curves.limits(0.0, 7.0, 0.0, 1.1)
	curves.plot(steps, train, line(accent, "train", 1.6))
	curves.plot(steps, validation, line(success, "validation", 1.6))

	quality = figure.ax(0, 1)
	quality.title("Calibration")
	quality.xLabel("confidence")
	quality.yLabel("observed accuracy")
	quality.limits(0.0, 1.0, 0.0, 1.0)
	ideal = [0.0, 0.25, 0.50, 0.75, 1.0]
	quality.plot(ideal, ideal, line(gray, "ideal"))
	quality.scatter(
		[0.10, 0.30, 0.50, 0.70, 0.90],
		[0.08, 0.34, 0.47, 0.74, 0.88],
		points(cyan, "model", 3.5),
	)
	return figure
# [oa-plot-intro-end]


def trainingPlot(axes: plot.axes) -> None:
	steps = list(range(72))
	train = [
		1.55 * math.exp(-step / 18.0)
		+ 0.035 * math.sin(step * 0.52) + 0.11
		for step in steps
	]
	validation = [
		1.45 * math.exp(-step / 21.0)
		+ 0.045 * math.sin(step * 0.39 + 0.8) + 0.17
		for step in steps
	]
	axes.title("Training curves")
	axes.xLabel("optimizer step")
	axes.yLabel("cross entropy")
	axes.plot(train, line(accent, "train"))
	axes.plot(validation, line(success, "validation"))


def rocPlot(axes: plot.axes) -> None:
	rate = [index / 32.0 for index in range(33)]
	vision = [max(0.0, min(1.0, 1.0 - math.exp(-4.3 * x))) for x in rate]
	fusion = [max(0.0, min(1.0, 1.0 - math.exp(-6.7 * x))) for x in rate]
	axes.title("ROC - threshold sweep")
	axes.xLabel("false positive rate")
	axes.yLabel("true positive rate")
	axes.limits(0.0, 1.0, 0.0, 1.0)
	axes.plot(rate, vision, line(cyan, "vision AUC 0.88"))
	axes.plot(rate, fusion, line(success, "fusion AUC 0.93"))
	axes.plot(rate, rate, line(gray, "chance"))


def precisionRecallPlot(axes: plot.axes) -> None:
	recall = [index / 32.0 for index in range(33)]
	vision = [
		max(0.0, min(1.0,
			0.98 - 0.40 * x ** 1.6 - 0.025 * math.sin(x * 5.0 * pi)))
		for x in recall
	]
	fusion = [
		max(0.0, min(1.0,
			0.995 - 0.27 * x ** 2.1 - 0.015 * math.sin(x * 4.0 * pi)))
		for x in recall
	]
	axes.title("Precision-recall")
	axes.xLabel("recall")
	axes.yLabel("precision")
	axes.limits(0.0, 1.0, 0.0, 1.0)
	axes.plot(recall, vision, line(purple, "vision AP 0.84"))
	axes.plot(recall, fusion, line(pink, "fusion AP 0.91"))


def scoreHistogram(axes: plot.axes) -> None:
	scores = []
	for index in range(192):
		phase = float(index)
		cluster = 0.28 if index % 3 == 0 else 0.74
		scores.append(max(0.0, min(1.0,
			cluster + 0.12 * math.sin(phase * 1.73)
			+ 0.045 * math.cos(phase * 0.37))))
	axes.title("confidence distribution")
	axes.xLabel("model score")
	axes.yLabel("samples")
	axes.histogram(scores, 20, bars(accent, "validation scores", 0.20))


def calibrationScatter(axes: plot.axes) -> None:
	confidence = [(index + 0.5) / 24.0 for index in range(24)]
	accuracy = [
		max(0.0, min(1.0,
			x + 0.055 * math.sin(8.0 * pi * x) - 0.025))
		for x in confidence
	]
	axes.title("Calibration")
	axes.xLabel("confidence")
	axes.yLabel("observed accuracy")
	axes.limits(0.0, 1.0, 0.0, 1.0)
	axes.plot(confidence, confidence, line(gray, "ideal"))
	axes.scatter(confidence, accuracy, points(cyan, "model", 3.5))


def throughputBars(axes: plot.axes) -> None:
	axes.title("Normalized throughput")
	axes.xLabel("kernel route")
	axes.yLabel("relative peak")
	axes.limits(-0.5, 5.5, 0.0, 1.0)
	axes.bar([0.42, 0.58, 0.71, 0.86, 0.79, 0.94],
		bars(warning, "device routes", 0.24))


def confusionHeatmap(axes: plot.axes) -> None:
	confusion = [
		42.0, 2.0, 0.0, 1.0, 0.0,
		3.0, 37.0, 2.0, 0.0, 1.0,
		0.0, 2.0, 40.0, 3.0, 0.0,
		1.0, 0.0, 2.0, 38.0, 2.0,
		0.0, 1.0, 0.0, 2.0, 43.0,
	]
	style = plot.HeatmapStyle()
	style.colormap = 1
	style.autoScale = True
	style.showGrid = True
	axes.title("confusion matrix")
	axes.xLabel("predicted class")
	axes.yLabel("reference class")
	axes.heatmap(confusion, 5, 5, style)


def wireframe(
	axes: plot.axes,
	title: str,
	height,
	near: tuple[float, float, float],
	far: tuple[float, float, float],
) -> None:
	lineCount, sampleCount = 13, 29
	for row in range(lineCount):
		v = -1.0 + 2.0 * row / (lineCount - 1)
		x, y = [], []
		for column in range(sampleCount):
			u = -1.0 + 2.0 * column / (sampleCount - 1)
			z = height(u, v)
			x.append(0.50 + 0.34 * (u - v))
			y.append(0.58 - 0.16 * (u + v) - 0.23 * z)
		t = row / (lineCount - 1)
		axes.plot(x, y, line(gradient(far, near, t), "", 1.15, 8))
	for column in range(lineCount):
		u = -1.0 + 2.0 * column / (lineCount - 1)
		x, y = [], []
		for row in range(sampleCount):
			v = -1.0 + 2.0 * row / (sampleCount - 1)
			z = height(u, v)
			x.append(0.50 + 0.34 * (u - v))
			y.append(0.58 - 0.16 * (u + v) - 0.23 * z)
		t = column / (lineCount - 1)
		color = gradient(far, near, t)
		color = (color & 0xFFFFFF00) | (rgba(0.0, 0.0, 0.0, 0.78) & 0xFF)
		axes.plot(x, y, line(color, "", 1.15, 8))
	axes.title(title)
	axes.limits(-0.20, 1.20, 0.02, 1.08)
	axes.grid(False)
	axes.legend(False)


def lossLandscape(axes: plot.axes) -> None:
	def height(x: float, y: float) -> float:
		return 0.42 * (x * x + 0.75 * y * y) \
			+ 0.18 * math.sin(2.7 * x) * math.cos(3.1 * y)

	wireframe(axes, "Projected loss landscape", height,
		(0.133, 0.827, 0.933), (0.659, 0.333, 0.969))


def gradientBasin(axes: plot.axes) -> None:
	def height(x: float, y: float) -> float:
		radius = math.sqrt(x * x + y * y)
		return 0.34 * radius + 0.17 * math.cos(9.0 * radius) \
			* math.exp(-1.8 * radius)

	wireframe(axes, "Projected optimizer basin", height,
		(0.188, 0.820, 0.345), (0.961, 0.620, 0.043))


def newFigure(
	title: str,
	rows: int,
	cols: int,
	width: int,
	height: int,
	theme: plot.Theme = plot.Theme.Dark,
	hSpacing: int = 32,
	vSpacing: int = 36,
	padding: int = 34,
) -> plot.Figure:
	config = plot.FigureConfig()
	config.title = title
	config.rows, config.cols = rows, cols
	config.width, config.height = width, height
	config.hSpacing, config.vSpacing = hSpacing, vSpacing
	config.padding, config.theme = padding, theme
	return plot.Figure(config)


def gallery(theme: plot.Theme) -> plot.Figure:
	figure = newFigure("OA plot gallery", 3, 3, 1600, 1050, theme)
	figure.title("OA plot - GPU-composed ML diagnostics")
	trainingPlot(figure.ax(0, 0))
	rocPlot(figure.ax(0, 1))
	precisionRecallPlot(figure.ax(0, 2))
	scoreHistogram(figure.ax(1, 0))
	calibrationScatter(figure.ax(1, 1))
	throughputBars(figure.ax(1, 2))
	confusionHeatmap(figure.ax(2, 0))
	lossLandscape(figure.ax(2, 1))
	gradientBasin(figure.ax(2, 2))
	return figure


def evaluationFigure() -> plot.Figure:
	figure = newFigure("OA model evaluation", 1, 2, 1280, 560, hSpacing=36)
	figure.title("Model evaluation - ROC and precision-recall")
	rocPlot(figure.ax(0, 0))
	precisionRecallPlot(figure.ax(0, 1))
	return figure


def diagnosticsFigure() -> plot.Figure:
	figure = newFigure("OA model diagnostics", 2, 2, 1200, 900)
	figure.title("Training and evaluation diagnostics")
	trainingPlot(figure.ax(0, 0))
	scoreHistogram(figure.ax(0, 1))
	calibrationScatter(figure.ax(1, 0))
	confusionHeatmap(figure.ax(1, 1))
	return figure


def landscapeFigure() -> plot.Figure:
	figure = newFigure("OA projected landscapes", 1, 2, 1280, 560, hSpacing=36)
	figure.title("Scalar-field projections - explicit-x wireframes")
	lossLandscape(figure.ax(0, 0))
	gradientBasin(figure.ax(0, 1))
	return figure


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--output",
		type=FilePath,
		default=FilePath(Paths.var("artifact/plot/python")),
	)
	parser.add_argument("--headless", action="store_true")
	arguments = parser.parse_args()
	arguments.output.mkdir(parents=True, exist_ok=True)

	intro = introFigure()
	dark = gallery(plot.Theme.Dark)
	figures = (
		("oa-plot-intro-dark.png", intro),
		("oa-plot-gallery-dark.png", dark),
		("oa-plot-gallery-light.png", gallery(plot.Theme.Light)),
		("oa-plot-evaluation-dark.png", evaluationFigure()),
		("oa-plot-diagnostics-dark.png", diagnosticsFigure()),
		("oa-plot-landscapes-dark.png", landscapeFigure()),
	)
	print("OA Python plot walkthrough artifacts:")
	for name, figure in figures:
		pathName = arguments.output / name
		figure.saveTo(str(pathName))
		print(f"  {pathName}")

	if not arguments.headless:
		dark.show()


if __name__ == "__main__":
	main()
