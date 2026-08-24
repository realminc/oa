#!/usr/bin/env python3
"""Render six deterministic GPU augmentation views from one source image."""

from __future__ import annotations

import argparse

# pyright: reportWildcardImportFromLibrary=false
from oa import *


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--input",
		default=Paths.asset("image/visionTestPattern320x180.jpg"),
	)
	parser.add_argument(
		"--output",
		default="/tmp/oa-vision-data-augmentation.png",
	)
	args = parser.parse_args()

	if not initComputeEngine():
		raise RuntimeError("OA could not create a Vulkan compute engine")

	try:
		sourceImage = FnImage.decodeFile(args.input)
		source = sourceImage.asMatrix()

		# OA_DOC_BEGIN: vision-data-augmentation
		views = [sourceImage]
		views.append(Image(
			FnImage.flip(source, True, False),
			ImageLayout.Nchw,
			ImageFormat.Rgb,
		))
		crop = FnImage.centerCrop(source, 280, 150)
		views.append(Image(
			FnImage.resize(crop, 320, 180),
			ImageLayout.Nchw,
			ImageFormat.Rgb,
		))
		views.append(Image(
			FnImage.brightnessContrast(source, 0.08, 1.15),
			ImageLayout.Nchw,
			ImageFormat.Rgb,
		))
		noisy = FnImage.gaussianNoise(source, 0.0, 0.035, 2026)
		views.append(Image(
			FnImage.clamp(noisy, 0.0, 1.0),
			ImageLayout.Nchw,
			ImageFormat.Rgb,
		))
		views.append(Image(
			FnImage.solarize(source, 0.55, 1.0),
			ImageLayout.Nchw,
			ImageFormat.Rgb,
		))
		# OA_DOC_END: vision-data-augmentation

		titles = [
			"source",
			"horizontal flip",
			"crop + resize",
			"brightness + contrast",
			"seeded Gaussian noise",
			"solarize",
		]
		config = plot.FigureConfig()
		config.title = "OA GPU data augmentation"
		config.rows = 2
		config.cols = 3
		config.width = 1200
		config.height = 760
		config.hSpacing = 18
		config.vSpacing = 20
		config.padding = 20
		config.theme = plot.Theme.Dark
		figure = plot.Figure(config)
		for index, (title, image) in enumerate(zip(titles, views, strict=True)):
			axes = figure.ax(index // 3, index % 3)
			axes.imshow(image)
			axes.title(title)
			axes.grid(False)
			axes.legend(False)

		figure.saveTo(args.output)
		print(f"saved 6 GPU augmentation views to {args.output}")
	finally:
		shutdownComputeEngine()


if __name__ == "__main__":
	main()
