#!/usr/bin/env python3
"""Render deterministic training metrics into an Image and display it.
"""

# pyright: reportWildcardImportFromLibrary=false

import math

from oa import *


steps = list(range(80))
trainingLoss = [
	1.8 * math.exp(-step / 19.0) + 0.035 * math.sin(step * 0.55) + 0.12
	for step in steps
]
validationAccuracy = [
	min(0.95, 0.48 + 0.47 * (1.0 - math.exp(-step / 24.0)) + 0.008 * math.sin(step * 0.35))
	for step in steps
]

config = plot.FigureConfig()
config.title = "Training metrics"
config.rows = 1
config.cols = 2
config.width = 960
config.height = 420
config.hSpacing = 20
config.padding = 20

figure = plot.Figure(config)
figure.ax(0, 0).title("Training loss")
figure.ax(0, 0).xLabel("step")
figure.ax(0, 0).plot(trainingLoss)
figure.ax(0, 1).title("Validation accuracy")
figure.ax(0, 1).xLabel("step")
figure.ax(0, 1).plot(validationAccuracy)

metricsImage = figure.render()
Viewer.show(metricsImage, title="Training metrics · loss | accuracy")
