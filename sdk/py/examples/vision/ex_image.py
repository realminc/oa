# OA_DOC_BEGIN: vision-image
import os

import oa

engine = oa.Engine()

asset_root = os.environ.get("OA_ASSET", "asset")
path = os.path.join(asset_root, "image/visionTestPattern320x180.jpg")

image = oa.image.decode_file(engine, path, "jpeg")
small = oa.image.resize(image, 160, 90)
adjusted = oa.image.brightness_contrast(small, 0.05, 1.1)

matrix = adjusted.as_matrix()
values = matrix.read_f32()

assert matrix.shape == [1, 3, 90, 160]
assert len(values) == 3 * 90 * 160

print(matrix.shape, min(values), max(values))
# OA_DOC_END: vision-image
