# OA_DOC_BEGIN: vision-image
import sys

import oa

source = oa.FnImage.decodeFile(oa.Paths.asset("image/coverVision.jpg"), oa.ImageFormat.Rgb)

grayscale = oa.FnImage.grayscale(source)

output = oa.Paths.var("example/vision/coverVisionGrayscale.jpg")
oa.Filesystem.createDirectories(output.parentPath())
oa.FnImage.saveFile(output, grayscale, 92)

assert grayscale.validate()
assert grayscale.width() == 1672
assert grayscale.height() == 941
assert grayscale.format() == oa.ImageFormat.Gray
assert oa.Filesystem.isFile(output)

print(f"Saved grayscale image: {output}")

if "--preview" in sys.argv:
	oa.Viewer.preview(output, title="OA vision · grayscale cover", width=1280, height=646)
# OA_DOC_END: vision-image
