"""Build a retained plot gallery and save it or show it interactively.

Covers the artist surface of ``sdk/rs/tutorials/plot/tu_plot_gallery/tu_plot_gallery.rs``
plus the newer artists: scatter, bar, histogram, heatmap, error band,
boxplot, categorical and explicit ticks.

Usage: python sdk/py/tutorials/plot/tu_plot_gallery.py [output.png]
"""

import math
import random
import sys

import oa


config = oa.plot.FigureConfig(title="OA Plot", rows=2, cols=3, width=1280, height=720)
figure = oa.plot.Figure(config)
random.seed(7)

steps = [float(i) for i in range(64)]
loss = [math.exp(-step / 15.0) + 0.03 * math.sin(step) for step in steps]
val = [math.exp(-step / 20.0) + 0.05 * math.cos(step * 0.7) for step in steps]

axes = figure.ax(0, 0)
axes.title("training loss")
axes.x_label("step")
axes.y_label("loss")
axes.legend(True)
axes.grid(True)
axes.plot_xy(steps, loss, oa.plot.LineStyle(label="train"))
axes.plot_xy(steps, val, oa.plot.LineStyle(label="val", step=True))

sc = figure.ax(0, 1)
sc.title("residuals")
sc.x_label("predicted")
sc.y_label("actual")
res_x = [random.uniform(0.0, 1.0) for _ in range(80)]
res_y = [x + random.gauss(0.0, 0.08) for x in res_x]
sc.scatter(res_x, res_y, oa.plot.ScatterStyle(label="batch", radius=2.5))
sc.limits(0.0, 1.2, 0.0, 1.2)

bars = figure.ax(0, 2)
bars.title("per-class recall")
bars.categories(["cat", "dog", "car", "bus"])
bars.y_label("recall")
bars.bar_grouped(
	[[0.91, 0.84, 0.77, 0.69], [0.88, 0.81, 0.72, 0.63]],
	[oa.plot.BarStyle(label="epoch 40"), oa.plot.BarStyle(label="epoch 20")],
)

hist = figure.ax(1, 0)
hist.title("grad norm")
hist.x_label("value")
hist.y_label("count")
hist.histogram([random.gauss(0.0, 0.5) for _ in range(400)], 24)

heat = figure.ax(1, 1)
heat.title("confusion")
heat.heatmap([42.0, 2.0, 1.0, 37.0], 2, 2)

stats = figure.ax(1, 2)
stats.title("uncertainty")
stats.x_label("step")
stats.y_label("loss")
stats.grid(True)
mid = [loss[i] for i in range(64)]
lo = [v - 0.06 for v in mid]
hi = [v + 0.06 for v in mid]
stats.band(lo, hi, oa.plot.LineStyle(label="±σ"))
stats.plot_xy(steps, mid, oa.plot.LineStyle(label="mean"))
stats.annotate("checkpoint", 32.0, mid[32])
stats.xticks([0.0, 16.0, 32.0, 48.0, 63.0])
stats.legend(True)

engine = oa.Engine(presentation=len(sys.argv) == 1)

oa.Engine.set_default(engine=engine)
if len(sys.argv) > 1:
	figure.save_to(sys.argv[1])
else:
	figure.show(engine=engine)
