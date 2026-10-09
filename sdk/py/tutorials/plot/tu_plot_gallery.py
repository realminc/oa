"""Build a retained plot and save it or show it interactively.

Usage: python sdk/py/tutorials/plot/tu_plot_gallery.py [output.png]
"""

import math
import sys

import oa


config = oa.plot.FigureConfig(title="OA Plot", rows=1, cols=2, width=1000, height=420)
figure = oa.plot.Figure(config)
steps = [float(i) for i in range(64)]
loss = [math.exp(-step / 15.0) + 0.03 * math.sin(step) for step in steps]
axes = figure.ax(0, 0)
axes.title("training loss")
axes.x_label("step")
axes.y_label("loss")
axes.plot_xy(steps, loss, oa.plot.LineStyle(label="train"))

heat = figure.ax(0, 1)
heat.title("confusion")
heat.heatmap([42.0, 2.0, 1.0, 37.0], 2, 2)

engine = oa.Engine(presentation=len(sys.argv) == 1)
if len(sys.argv) > 1:
	figure.save_to(engine, sys.argv[1])
else:
	figure.show(engine)
