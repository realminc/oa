# OA_DOC_BEGIN: plot-line
import sys

import oa

figureConfig = oa.plot.FigureConfig()
figureConfig.title = "Training loss"
figureConfig.width = 960
figureConfig.height = 540
figureConfig.theme = oa.plot.Theme.Dark
figure = oa.plot.Figure(figureConfig)
figure.ax(0, 0).xLabel("step")
figure.ax(0, 0).yLabel("loss")
figure.ax(0, 0).plot([1.0, 0.72, 0.51, 0.38, 0.29, 0.23])

rendered = figure.render()

output = oa.Paths.var("example/ui/trainingLoss.jpg")
oa.Filesystem.createDirectories(output.parentPath())
oa.FnImage.saveFile(output, rendered, 92)

assert rendered.validate()
assert rendered.width() == 960
assert rendered.height() == 540
assert rendered.format() == oa.ImageFormat.Rgba
assert oa.Filesystem.isFile(output)

print(f"Saved training-loss plot: {output}")

if "--preview" in sys.argv:
	oa.Viewer.preview(output, title="OA plot · training loss", width=1280, height=646)
# OA_DOC_END: plot-line
