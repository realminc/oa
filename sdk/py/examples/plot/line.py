#!/usr/bin/env python3
"""Render one retained plot figure into a semantic image."""

# OA_DOC_BEGIN: plot-line
from oa import initComputeEngine, plot, shutdownComputeEngine


if not initComputeEngine():
	raise RuntimeError("OA could not create a Vulkan compute engine")

try:
	config = plot.FigureConfig()
	config.title = "Training loss"
	config.width = 360
	config.height = 240
	config.theme = plot.Theme.Dark

	figure = plot.Figure(config)
	figure.ax(0, 0).xLabel("step")
	figure.ax(0, 0).yLabel("loss")
	figure.ax(0, 0).plot([1.0, 0.72, 0.51, 0.38, 0.29, 0.23])

	image = figure.render()
	assert image.validate()
	assert image.width() == 360
	assert image.height() == 240
	print("rendered a 360x240 training-loss figure")
finally:
	shutdownComputeEngine()
# OA_DOC_END: plot-line
