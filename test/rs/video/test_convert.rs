//! `oa::video::to_matrix` contract tests against an FP64 host oracle.
//!
//! The oracle mirrors the documented reference order — crop, per-plane
//! resize, color conversion, normalization, layout — so kernel agreement is
//! evidence for the fused implementation, not a second code path.

use oa::ImageLayout;
use oa::video::{
	MatrixChannels, VideoColorInfo, VideoColorMatrix, VideoColorRange, VideoFrameTiming,
	VideoInterpolation, VideoMatrixConfig, VideoPixelFormat, VideoRegion, VideoResize,
};

const TOLERANCE: f64 = 2.0 / 255.0;

fn timing() -> VideoFrameTiming {
	VideoFrameTiming::new(std::time::Duration::from_millis(33), None).expect("valid timing")
}

fn color(matrix: VideoColorMatrix, range: VideoColorRange) -> VideoColorInfo {
	VideoColorInfo::new(matrix, range)
}

/// Deterministic plane contents: value = (x + 3y + 11p) mod 251.
fn plane(width: usize, height: usize, plane: usize) -> Vec<u8> {
	(0..height)
		.flat_map(|y| (0..width).map(move |x| ((x + 3 * y + 11 * plane) % 251) as u8))
		.collect()
}

fn i420_planes(width: usize, height: usize) -> Vec<Vec<u8>> {
	vec![
		plane(width, height, 0),
		plane(width.div_ceil(2), height.div_ceil(2), 1),
		plane(width.div_ceil(2), height.div_ceil(2), 2),
	]
}

fn nv12_planes(width: usize, height: usize) -> Vec<Vec<u8>> {
	let chroma_width = width.div_ceil(2);
	let chroma_height = height.div_ceil(2);
	let u = plane(chroma_width, chroma_height, 1);
	let v = plane(chroma_width, chroma_height, 2);
	let mut uv = vec![0_u8; chroma_width * 2 * chroma_height];
	for y in 0..chroma_height {
		for x in 0..chroma_width {
			uv[(y * chroma_width + x) * 2] = u[y * chroma_width + x];
			uv[(y * chroma_width + x) * 2 + 1] = v[y * chroma_width + x];
		}
	}
	vec![plane(width, height, 0), uv]
}

test_vk!(
	p010_odd_plane_extents_and_color_match_fp64_oracle,
	runtime,
	{
		// Odd luma width and height leave the last sample in a half-full word.
		let luma = [0, 63, 64, 255, 511, 512, 767, 939, 940];
		let chroma = [0, 638, 402, 636, 404, 634, 406, 1023];
		let pack = |samples: &[u16]| {
			samples
				.iter()
				.flat_map(|code| ((code << 6) | 0x3f).to_le_bytes())
				.collect::<Vec<_>>()
		};
		for range in [VideoColorRange::Full, VideoColorRange::Limited] {
			let frame = oa::VideoFrame::upload_planes(
				VideoPixelFormat::P010,
				vec![pack(&luma), pack(&chroma)],
				3,
				3,
				timing(),
				color(VideoColorMatrix::Bt709, range),
			)?;
			let config = VideoMatrixConfig {
				interpolation: VideoInterpolation::Nearest,
				layout: ImageLayout::Hwc,
				..VideoMatrixConfig::default()
			};
			let output = frame.to_matrix(&config)?.read_f32()?;
			let (pixels, tail) = output.as_chunks::<3>();
			assert_eq!(pixels.len(), luma.len());
			assert!(tail.is_empty());
			for (index, (code, pixel)) in luma.iter().zip(pixels).enumerate() {
				let (y, x) = (index / 3, index % 3);
				let uv = ((y / 2) * 2 + x / 2) * 2;
				let expand_luma = |value: u16| match range {
					VideoColorRange::Full => f64::from(value) / 1023.0,
					_ => ((f64::from(value) - 64.0) / 876.0).clamp(0.0, 1.0),
				};
				let expand_chroma = |value: u16| {
					let scale = if range == VideoColorRange::Full {
						1023.0
					} else {
						896.0
					};
					((f64::from(value) - 512.0) / scale).clamp(-0.5, 0.5)
				};
				let (r, g, b) = rgb(
					(
						expand_luma(*code),
						expand_chroma(chroma[uv]),
						expand_chroma(chroma[uv + 1]),
					),
					VideoColorMatrix::Bt709,
				);
				for (actual, expected) in pixel.iter().zip([r, g, b]) {
					assert!((f64::from(*actual) - expected).abs() < 2e-6);
				}
			}
			let error = oa::video::crop(
				&frame,
				VideoRegion {
					x: 0,
					y: 0,
					width: 2,
					height: 2,
				},
			)
			.err()
			.expect("8-bit crop must decline P010 storage");
			assert_eq!(error.kind(), oa::ErrorKind::MissingCapability);
		}
		Ok(())
	}
);

// ── FP64 host oracle ─────────────────────────────────────────────────────

fn sample(plane: &[u8], width: usize, height: usize, x: f64, y: f64, nearest: bool) -> f64 {
	let at = |px: i64, py: i64| -> f64 {
		let px = px.clamp(0, width as i64 - 1) as usize;
		let py = py.clamp(0, height as i64 - 1) as usize;
		plane[py * width + px] as f64 / 255.0
	};
	if nearest {
		return at((x + 0.5).floor() as i64, (y + 0.5).floor() as i64);
	}
	let xc = x.clamp(0.0, width as f64 - 1.0);
	let yc = y.clamp(0.0, height as f64 - 1.0);
	let x0 = xc.floor().min(width as f64 - 1.0) as usize;
	let y0 = yc.floor().min(height as f64 - 1.0) as usize;
	let x1 = (x0 + 1).min(width - 1);
	let y1 = (y0 + 1).min(height - 1);
	let fx = xc - x0 as f64;
	let fy = yc - y0 as f64;
	let top = at(x0 as i64, y0 as i64) * (1.0 - fx) + at(x1 as i64, y0 as i64) * fx;
	let bottom = at(x0 as i64, y1 as i64) * (1.0 - fx) + at(x1 as i64, y1 as i64) * fx;
	top * (1.0 - fy) + bottom * fy
}

fn scale_luma(code: f64, full: bool) -> f64 {
	if full {
		code
	} else {
		((code * 255.0 - 16.0) / 219.0).clamp(0.0, 1.0)
	}
}

fn scale_chroma(code: f64, full: bool) -> f64 {
	if full {
		code - 0.5
	} else {
		((code * 255.0 - 128.0) / 224.0).clamp(-0.5, 0.5)
	}
}

fn rgb(yuv: (f64, f64, f64), space: VideoColorMatrix) -> (f64, f64, f64) {
	let (y, cb, cr) = yuv;
	// Components saturate in the kernel; mirror that here.
	let saturate = |value: f64| value.clamp(0.0, 1.0);
	let (r, g, b) = match space {
		VideoColorMatrix::Bt709 => (
			y + 1.5748 * cr,
			y - 0.187_324 * cb - 0.468_124 * cr,
			y + 1.8556 * cb,
		),
		VideoColorMatrix::Bt2020 => (
			y + 1.4746 * cr,
			y - 0.164_553 * cb - 0.571_353 * cr,
			y + 1.8814 * cb,
		),
		_ => (
			y + 1.402 * cr,
			y - 0.344_136 * cb - 0.714_136 * cr,
			y + 1.772 * cb,
		),
	};
	(saturate(r), saturate(g), saturate(b))
}

fn mapping(
	config: &VideoMatrixConfig,
	crop: VideoRegion,
	out_w: u32,
	out_h: u32,
) -> (f64, f64, f64, f64, [f64; 4], [f64; 3]) {
	let crop_w = f64::from(crop.width);
	let crop_h = f64::from(crop.height);
	let ow = f64::from(out_w);
	let oh = f64::from(out_h);
	match config.resize {
		VideoResize::Stretch => (
			crop_w / ow,
			crop_h / oh,
			f64::from(crop.x) - 0.5,
			f64::from(crop.y) - 0.5,
			[0.0, 0.0, ow, oh],
			[0.0; 3],
		),
		VideoResize::CenterCrop => {
			let scale = (ow / crop_w).max(oh / crop_h);
			let src_w = ow / scale;
			let src_h = oh / scale;
			(
				src_w / ow,
				src_h / oh,
				f64::from(crop.x) + (crop_w - src_w) * 0.5 - 0.5,
				f64::from(crop.y) + (crop_h - src_h) * 0.5 - 0.5,
				[0.0, 0.0, ow, oh],
				[0.0; 3],
			)
		}
		VideoResize::Letterbox { fill } => {
			let scale = (ow / crop_w).min(oh / crop_h);
			let map_w = crop_w * scale;
			let map_h = crop_h * scale;
			let ix0 = (ow - map_w) * 0.5;
			let iy0 = (oh - map_h) * 0.5;
			(
				crop_w / map_w,
				crop_h / map_h,
				f64::from(crop.x) - ix0 * (crop_w / map_w) - 0.5,
				f64::from(crop.y) - iy0 * (crop_h / map_h) - 0.5,
				[ix0, iy0, ix0 + map_w, iy0 + map_h],
				[fill[0] as f64, fill[1] as f64, fill[2] as f64],
			)
		}
	}
}

/// Oracle conversion for one frame; returns [C,H,W] channel-major values.
fn oracle(
	planes: &[Vec<u8>],
	nv12: bool,
	luma_w: usize,
	luma_h: usize,
	color: VideoColorInfo,
	config: &VideoMatrixConfig,
) -> Vec<f64> {
	let crop = config.crop.unwrap_or(VideoRegion {
		x: 0,
		y: 0,
		width: luma_w as u32,
		height: luma_h as u32,
	});
	let (out_w, out_h) = config.size.unwrap_or((crop.width, crop.height));
	let (sx, sy, ox, oy, valid, fill) = mapping(config, crop, out_w, out_h);
	let full = color.range() == VideoColorRange::Full;
	let channel_count = if config.channels == MatrixChannels::Luma {
		1
	} else {
		3
	};
	let chroma_w = luma_w.div_ceil(2);
	let chroma_h = luma_h.div_ceil(2);
	let nearest = config.interpolation == VideoInterpolation::Nearest;
	let mut output = vec![0.0; channel_count * out_h as usize * out_w as usize];
	for c in 0..channel_count {
		for y in 0..out_h as usize {
			for x in 0..out_w as usize {
				let px = x as f64 + 0.5;
				let py = y as f64 + 0.5;
				let inside = px >= valid[0] && px < valid[2] && py >= valid[1] && py < valid[3];
				let value = if !inside {
					fill[c]
				} else {
					let lx = px * sx + ox;
					let ly = py * sy + oy;
					let luma = scale_luma(sample(&planes[0], luma_w, luma_h, lx, ly, nearest), full);
					if config.channels == MatrixChannels::Luma {
						luma
					} else {
						let cx = (lx - 0.5) * 0.5;
						let cy = (ly - 0.5) * 0.5;
						let (cb, cr) = if nv12 {
							let uv = &planes[1];
							let at = |comp: usize, px2: i64, py2: i64| -> f64 {
								let px2 = px2.clamp(0, chroma_w as i64 - 1) as usize;
								let py2 = py2.clamp(0, chroma_h as i64 - 1) as usize;
								uv[(py2 * chroma_w + px2) * 2 + comp] as f64 / 255.0
							};
							let chroma = |cx2: f64, cy2: f64, comp: usize| -> f64 {
								if nearest {
									at(comp, (cx2 + 0.5).floor() as i64, (cy2 + 0.5).floor() as i64)
								} else {
									let xc = cx2.clamp(0.0, chroma_w as f64 - 1.0);
									let yc = cy2.clamp(0.0, chroma_h as f64 - 1.0);
									let x0 = xc.floor().min(chroma_w as f64 - 1.0) as usize;
									let y0 = yc.floor().min(chroma_h as f64 - 1.0) as usize;
									let x1 = (x0 + 1).min(chroma_w - 1);
									let y1 = (y0 + 1).min(chroma_h - 1);
									let fx = xc - x0 as f64;
									let fy = yc - y0 as f64;
									let top = at(comp, x0 as i64, y0 as i64) * (1.0 - fx)
										+ at(comp, x1 as i64, y0 as i64) * fx;
									let bottom = at(comp, x0 as i64, y1 as i64) * (1.0 - fx)
										+ at(comp, x1 as i64, y1 as i64) * fx;
									top * (1.0 - fy) + bottom * fy
								}
							};
							(
								scale_chroma(chroma(cx, cy, 0), full),
								scale_chroma(chroma(cx, cy, 1), full),
							)
						} else {
							(
								scale_chroma(
									sample(&planes[1], chroma_w, chroma_h, cx, cy, nearest),
									full,
								),
								scale_chroma(
									sample(&planes[2], chroma_w, chroma_h, cx, cy, nearest),
									full,
								),
							)
						};
						match config.channels {
							MatrixChannels::Yuv => {
								if c == 0 {
									luma
								} else if c == 1 {
									cb + 0.5
								} else {
									cr + 0.5
								}
							}
							_ => {
								let (r, g, b) = rgb((luma, cb, cr), color.matrix());
								let component = if config.channels == MatrixChannels::Bgr {
									2 - c
								} else {
									c
								};
								if component == 0 {
									r
								} else if component == 1 {
									g
								} else {
									b
								}
							}
						}
					}
				};
				output[(c * out_h as usize + y) * out_w as usize + x] =
					(value - config.mean[c] as f64) / config.std[c] as f64;
			}
		}
	}
	output
}

fn assert_close(actual: &[f32], expected: &[f64], context: &str) {
	assert_eq!(actual.len(), expected.len(), "{context} length");
	let mut worst = 0.0_f64;
	for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
		let delta = (*a as f64 - e).abs();
		assert!(
			delta <= TOLERANCE,
			"{context} index {index}: {a} vs {e} (delta {delta})"
		);
		worst = worst.max(delta);
	}
	eprintln!("{context}: max |delta| = {worst}");
}

// ── Vulkan contract tests ────────────────────────────────────────────────

fn upload_i420(
	_runtime: &oa::Runtime,
	width: usize,
	height: usize,
	color: VideoColorInfo,
) -> oa::Result<oa::VideoFrame> {
	oa::VideoFrame::upload_planes(
		VideoPixelFormat::Yuv420Planar8,
		i420_planes(width, height),
		width,
		height,
		timing(),
		color,
	)
}

fn run_oracle_check(
	actual: &oa::Matrix,
	planes: &[Vec<u8>],
	nv12: bool,
	luma_size: (usize, usize),
	color: VideoColorInfo,
	config: &VideoMatrixConfig,
	frame_index: usize,
) -> oa::Result<()> {
	let (luma_w, luma_h) = luma_size;
	let expected = oracle(planes, nv12, luma_w, luma_h, color, config);
	let channels = if config.channels == MatrixChannels::Luma {
		1
	} else {
		3
	};
	let (w, h) = config.size.unwrap_or_else(|| {
		config
			.crop
			.map(|c| (c.width, c.height))
			.unwrap_or((luma_w as u32, luma_h as u32))
	});
	let span = channels * w as usize * h as usize;
	let packed = matches!(config.layout, ImageLayout::Nhwc | ImageLayout::Hwc);
	let values = actual.read_f32()?;
	let frame = &values[frame_index * span..(frame_index + 1) * span];
	if packed {
		// Expand the oracle's channel-major result into the packed layout.
		let mut expanded = vec![0.0_f32; span];
		for c in 0..channels {
			for y in 0..h as usize {
				for x in 0..w as usize {
					expanded[(y * w as usize + x) * channels + c] =
						expected[(c * h as usize + y) * w as usize + x] as f32;
				}
			}
		}
		assert_close(
			frame,
			&expanded.iter().map(|v| *v as f64).collect::<Vec<_>>(),
			"packed output",
		);
	} else {
		assert_close(frame, &expected, "planar output");
	}
	Ok(())
}

test_vk!(to_matrix_bt709_limited_i420_matches_oracle, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let config = VideoMatrixConfig::default();
	let matrix = oa::video::to_matrix(std::slice::from_ref(&frame), &config)?;
	run_oracle_check(
		&matrix,
		&i420_planes(8, 8),
		false,
		(8, 8),
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
		&config,
		0,
	)
});

test_vk!(to_matrix_bt601_full_nv12_matches_oracle, runtime, {
	let frame = oa::VideoFrame::upload_planes(
		VideoPixelFormat::Nv12,
		nv12_planes(10, 6),
		10,
		6,
		timing(),
		color(VideoColorMatrix::Bt601, VideoColorRange::Full),
	)?;
	let config = VideoMatrixConfig::default();
	let matrix = oa::video::to_matrix(std::slice::from_ref(&frame), &config)?;
	run_oracle_check(
		&matrix,
		&nv12_planes(10, 6),
		true,
		(10, 6),
		color(VideoColorMatrix::Bt601, VideoColorRange::Full),
		&config,
		0,
	)
});

test_vk!(
	to_matrix_crop_letterbox_nearest_nhwc_matches_oracle,
	runtime,
	{
		let frame = upload_i420(
			&runtime,
			16,
			12,
			color(VideoColorMatrix::Bt2020, VideoColorRange::Limited),
		)?;
		let config = VideoMatrixConfig {
			crop: Some(VideoRegion {
				x: 2,
				y: 2,
				width: 10,
				height: 8,
			}),
			size: Some((12, 12)),
			resize: VideoResize::Letterbox {
				fill: [0.25, 0.5, 0.75],
			},
			interpolation: VideoInterpolation::Nearest,
			layout: ImageLayout::Nhwc,
			channels: MatrixChannels::Bgr,
			..VideoMatrixConfig::default()
		};
		let matrix = oa::video::to_matrix(std::slice::from_ref(&frame), &config)?;
		run_oracle_check(
			&matrix,
			&i420_planes(16, 12),
			false,
			(16, 12),
			color(VideoColorMatrix::Bt2020, VideoColorRange::Limited),
			&config,
			0,
		)
	}
);

test_vk!(to_matrix_batch_of_two_frames_matches_oracle, runtime, {
	let first = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let second = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let config = VideoMatrixConfig::default();
	let matrix = oa::video::to_matrix(&[first, second], &config)?;
	assert_eq!(matrix.shape(), &[2, 3, 8, 8]);
	run_oracle_check(
		&matrix,
		&i420_planes(8, 8),
		false,
		(8, 8),
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
		&config,
		0,
	)?;
	run_oracle_check(
		&matrix,
		&i420_planes(8, 8),
		false,
		(8, 8),
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
		&config,
		1,
	)
});

test_vk!(to_matrix_luma_channel_matches_oracle, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt601, VideoColorRange::Limited),
	)?;
	let config = VideoMatrixConfig {
		channels: MatrixChannels::Luma,
		..VideoMatrixConfig::default()
	};
	let matrix = oa::video::to_matrix(std::slice::from_ref(&frame), &config)?;
	assert_eq!(matrix.shape(), &[1, 1, 8, 8]);
	run_oracle_check(
		&matrix,
		&i420_planes(8, 8),
		false,
		(8, 8),
		color(VideoColorMatrix::Bt601, VideoColorRange::Limited),
		&config,
		0,
	)
});

test_vk!(to_matrix_rejects_invalid_arguments, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	assert!(oa::video::to_matrix(&[], &VideoMatrixConfig::default()).is_err());
	assert!(
		oa::video::to_matrix(
			std::slice::from_ref(&frame),
			&VideoMatrixConfig {
				crop: Some(VideoRegion {
					x: 6,
					y: 0,
					width: 8,
					height: 8
				}),
				..VideoMatrixConfig::default()
			}
		)
		.is_err()
	);
	assert!(
		oa::video::to_matrix(
			std::slice::from_ref(&frame),
			&VideoMatrixConfig {
				size: Some((0, 8)),
				..VideoMatrixConfig::default()
			}
		)
		.is_err()
	);
	let host = oa::VideoFrame::new_from_yuv420p(
		i420_planes(8, 8).concat(),
		8,
		8,
		timing(),
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	assert!(
		oa::video::to_matrix(std::slice::from_ref(&host), &VideoMatrixConfig::default()).is_err()
	);
	Ok(())
});

// ── Single-boundary enforcement ──────────────────────────────────────────

#[test]
fn shared_color_modules_have_single_importers() -> oa::Result<()> {
	let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/slang");
	let mut yuv_to_rgb_importers = Vec::new();
	let mut rgb_to_yuv_importers = Vec::new();
	for entry in std::fs::read_dir(&root)
		.expect("src/slang must exist")
		.flatten()
	{
		let mut stack = vec![entry.path()];
		while let Some(path) = stack.pop() {
			if path.is_dir() {
				stack.extend(
					std::fs::read_dir(&path)
						.expect("readable dir")
						.flatten()
						.map(|e| e.path()),
				);
				continue;
			}
			if path.extension().is_some_and(|ext| ext == "slang") {
				let text = std::fs::read_to_string(&path).expect("readable shader");
				let name = path.file_name().unwrap().to_string_lossy().into_owned();
				if name != "yuv_to_rgb.slang" && text.contains("import yuv_to_rgb") {
					yuv_to_rgb_importers.push(name.clone());
				}
				if name != "rgb_to_yuv.slang" && text.contains("import rgb_to_yuv") {
					rgb_to_yuv_importers.push(name);
				}
			}
		}
	}
	yuv_to_rgb_importers.sort();
	rgb_to_yuv_importers.sort();
	assert_eq!(
		yuv_to_rgb_importers,
		vec![
			"blit_yuv.frag.slang".to_string(),
			"cvt_yuv420_to_rgba8.slang".to_string(),
			"to_matrix_i420.slang".to_string(),
			"to_matrix_semiplanar.slang".to_string(),
		],
		"only the to_matrix family and video presentation may import yuv_to_rgb"
	);
	assert_eq!(
		rgb_to_yuv_importers,
		vec![
			"cvt_rgba64_to_p010.slang".to_string(),
			"cvt_rgba_to_nv12.slang".to_string(),
		],
		"only encoder-input conversion may import rgb_to_yuv"
	);
	Ok(())
}

// ── VNV-03 geometry ops ──────────────────────────────────────────────────

test_vk!(crop_extracts_exact_subregion_per_plane, runtime, {
	let frame = upload_i420(
		&runtime,
		16,
		12,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let cropped = oa::video::crop(
		&frame,
		VideoRegion {
			x: 4,
			y: 2,
			width: 8,
			height: 6,
		},
	)?;
	assert_eq!(cropped.width(), 8);
	assert_eq!(cropped.height(), 6);
	let planes = i420_planes(16, 12);
	let out = cropped.planes().expect("device-planar planes");
	let luma = out[0].read::<u8>()?;
	for y in 0..6usize {
		for x in 0..8usize {
			assert_eq!(
				luma[y * 8 + x],
				planes[0][(y + 2) * 16 + x + 4],
				"luma {x},{y}"
			);
		}
	}
	for (plane_index, out_plane) in out[1..].iter().enumerate() {
		let expected = &planes[plane_index + 1];
		let actual = out_plane.read::<u8>()?;
		for y in 0..3usize {
			for x in 0..4usize {
				assert_eq!(
					actual[y * 4 + x],
					expected[(y + 1) * 8 + x + 2],
					"plane {plane_index} {x},{y}"
				);
			}
		}
	}
	Ok(())
});

test_vk!(flip_horizontal_mirrors_each_plane, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let flipped = oa::video::flip(&frame, oa::video::VideoAxis::Horizontal)?;
	let planes = i420_planes(8, 8);
	let out = flipped.planes().expect("device-planar planes");
	let luma = out[0].read::<u8>()?;
	for y in 0..8usize {
		for x in 0..8usize {
			assert_eq!(luma[y * 8 + x], planes[0][y * 8 + (7 - x)], "luma {x},{y}");
		}
	}
	Ok(())
});

test_vk!(stack_places_frames_side_by_side, runtime, {
	let first = upload_i420(
		&runtime,
		8,
		4,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let second = upload_i420(
		&runtime,
		8,
		4,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let stacked = oa::video::stack(&[first, second], oa::video::VideoAxis::Horizontal)?;
	assert_eq!(stacked.width(), 16);
	assert_eq!(stacked.height(), 4);
	let planes = i420_planes(8, 4);
	let luma = stacked.planes().expect("device-planar planes")[0].read::<u8>()?;
	for y in 0..4usize {
		for x in 0..8usize {
			assert_eq!(luma[y * 16 + x], planes[0][y * 8 + x], "left {x},{y}");
			assert_eq!(luma[y * 16 + x + 8], planes[0][y * 8 + x], "right {x},{y}");
		}
	}
	Ok(())
});

test_vk!(rotate90_transposes_each_plane, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		4,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let rotated = oa::video::rotate90(&frame)?;
	assert_eq!(rotated.width(), 4);
	assert_eq!(rotated.height(), 8);
	let planes = i420_planes(8, 4);
	let luma = rotated.planes().expect("device-planar planes")[0].read::<u8>()?;
	for y in 0..8usize {
		for x in 0..4usize {
			assert_eq!(luma[y * 4 + x], planes[0][(3 - x) * 8 + y], "luma {x},{y}");
		}
	}
	Ok(())
});

test_vk!(pad_fills_border_and_preserves_content, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let padded = oa::video::pad(
		&frame,
		oa::video::VideoPadding {
			left: 2,
			top: 2,
			right: 2,
			bottom: 2,
		},
		[16, 128, 128],
	)?;
	assert_eq!(padded.width(), 12);
	assert_eq!(padded.height(), 12);
	let planes = i420_planes(8, 8);
	let luma = padded.planes().expect("device-planar planes")[0].read::<u8>()?;
	for y in 0..12usize {
		for x in 0..12usize {
			let expected = if (2..10).contains(&y) && (2..10).contains(&x) {
				planes[0][(y - 2) * 8 + (x - 2)]
			} else {
				16
			};
			assert_eq!(luma[y * 12 + x], expected, "luma {x},{y}");
		}
	}
	Ok(())
});

test_vk!(convert_format_nv12_i420_round_trip_is_exact, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let nv12 = oa::video::convert_format(&frame, VideoPixelFormat::Nv12)?;
	assert_eq!(nv12.pixel_format(), Some(VideoPixelFormat::Nv12));
	let back = oa::video::convert_format(&nv12, VideoPixelFormat::Yuv420Planar8)?;
	let planes = i420_planes(8, 8);
	let out = back.planes().expect("device-planar planes");
	for (index, expected) in planes.iter().enumerate() {
		let actual = out[index].read::<u8>()?;
		assert_eq!(&actual, expected, "plane {index}");
	}
	let uv = nv12.planes().expect("nv12 planes")[1].read::<u8>()?;
	for y in 0..4usize {
		for x in 0..4usize {
			assert_eq!(uv[(y * 4 + x) * 2], planes[1][y * 4 + x], "u {x},{y}");
			assert_eq!(uv[(y * 4 + x) * 2 + 1], planes[2][y * 4 + x], "v {x},{y}");
		}
	}
	Ok(())
});

// ── Per-plane resample oracle (mirrors resample.slang) ──────────────────

/// One element sample with the kernel's bilinear/nearest rule and
/// round-to-nearest u8 output.
fn kernel_sample(
	plane: &[u8],
	width: usize,
	height: usize,
	stride: usize,
	comp: usize,
	coords: (f64, f64),
	nearest: bool,
) -> u8 {
	let (x, y) = coords;
	let at = |px: i64, py: i64| -> f64 {
		let px = px.clamp(0, width as i64 - 1) as usize;
		let py = py.clamp(0, height as i64 - 1) as usize;
		plane[(py * width + px) * stride + comp] as f64
	};
	let value = if nearest {
		at((x + 0.5).floor() as i64, (y + 0.5).floor() as i64)
	} else {
		let xc = x.clamp(0.0, width as f64 - 1.0);
		let yc = y.clamp(0.0, height as f64 - 1.0);
		let x0 = xc.floor().min(width as f64 - 1.0) as usize;
		let y0 = yc.floor().min(height as f64 - 1.0) as usize;
		let x1 = (x0 + 1).min(width - 1);
		let y1 = (y0 + 1).min(height - 1);
		let fx = xc - x0 as f64;
		let fy = yc - y0 as f64;
		at(x0 as i64, y0 as i64) * (1.0 - fx) * (1.0 - fy)
			+ at(x1 as i64, y0 as i64) * fx * (1.0 - fy)
			+ at(x0 as i64, y1 as i64) * (1.0 - fx) * fy
			+ at(x1 as i64, y1 as i64) * fx * fy
	};
	value.round().clamp(0.0, 255.0) as u8
}

/// Expected plane contents under one resample map; chroma callers pass their
/// derived plane-space parameters (off_p = 0.5*off - 0.25, valid *= 0.5).
#[allow(clippy::too_many_arguments)]
fn resample_expected(
	src: &[u8],
	src_w: usize,
	src_h: usize,
	stride: usize,
	comp: usize,
	dst_w: usize,
	dst_h: usize,
	dst_elems: usize,
	sx: f64,
	sy: f64,
	ox: f64,
	oy: f64,
	valid: [f64; 4],
	fill: u8,
	nearest: bool,
) -> Vec<u8> {
	let mut out = vec![fill; dst_h * dst_elems];
	for y in 0..dst_h {
		for x in 0..dst_w {
			let px = x as f64 + 0.5;
			let py = y as f64 + 0.5;
			if px >= valid[0] && px < valid[2] && py >= valid[1] && py < valid[3] {
				out[(y * dst_w + x) * stride + comp] = kernel_sample(
					src,
					src_w,
					src_h,
					stride,
					comp,
					(px * sx + ox, py * sy + oy),
					nearest,
				);
			}
		}
	}
	out
}

// ── Additional operation contract tests ──────────────────────────────────

test_vk!(resize_downscale_nearest_subsamples_each_plane, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let resized = oa::video::resize(&frame, 4, 4, oa::video::VideoInterpolation::Nearest)?;
	assert_eq!(resized.width(), 4);
	assert_eq!(resized.height(), 4);
	let planes = i420_planes(8, 8);
	let out = resized.planes().expect("device-planar planes");
	for (index, expected) in planes.iter().enumerate() {
		let src_w = if index == 0 { 8 } else { 4 };
		let actual = out[index].read::<u8>()?;
		let oracle = resample_expected(
			expected,
			src_w,
			src_w,
			1,
			0,
			src_w / 2,
			src_w / 2,
			src_w / 2,
			2.0,
			2.0,
			-0.5,
			-0.5,
			[0.0, 0.0, src_w as f64 / 2.0, src_w as f64 / 2.0],
			0,
			true,
		);
		assert_eq!(&actual, &oracle, "plane {index}");
	}
	Ok(())
});

test_vk!(resize_bilinear_upscale_stays_within_one_code, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let resized = oa::video::resize(&frame, 12, 12, oa::video::VideoInterpolation::Bilinear)?;
	let planes = i420_planes(8, 8);
	let out = resized.planes().expect("device-planar planes");
	// Luma only: f32 shader lerp against the f64 oracle can differ by one
	// code at rounding boundaries.
	let actual = out[0].read::<u8>()?;
	let oracle = resample_expected(
		&planes[0],
		8,
		8,
		1,
		0,
		12,
		12,
		12,
		8.0 / 12.0,
		8.0 / 12.0,
		-0.5,
		-0.5,
		[0.0, 0.0, 12.0, 12.0],
		0,
		false,
	);
	for (index, (a, e)) in actual.iter().zip(oracle.iter()).enumerate() {
		let delta = (*a as i32 - *e as i32).abs();
		assert!(delta <= 1, "luma {index}: {a} vs {e}");
	}
	Ok(())
});

test_vk!(letterbox_fills_bars_and_maps_content, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let boxed = oa::video::letterbox(
		&frame,
		12,
		8,
		[16, 128, 128],
		oa::video::VideoInterpolation::Nearest,
	)?;
	assert_eq!(boxed.width(), 12);
	assert_eq!(boxed.height(), 8);
	let planes = i420_planes(8, 8);
	let out = boxed.planes().expect("device-planar planes");
	let luma = resample_expected(
		&planes[0],
		8,
		8,
		1,
		0,
		12,
		8,
		12,
		1.0,
		1.0,
		-2.5,
		-0.5,
		[2.0, 0.0, 10.0, 8.0],
		16,
		true,
	);
	assert_eq!(&out[0].read::<u8>()?, &luma, "luma");
	for (index, expected) in planes[1..].iter().enumerate() {
		let chroma = resample_expected(
			expected,
			4,
			4,
			1,
			0,
			6,
			4,
			6,
			1.0,
			1.0,
			-1.5,
			-0.5,
			[1.0, 0.0, 5.0, 4.0],
			128,
			true,
		);
		assert_eq!(
			&out[index + 1].read::<u8>()?,
			&chroma,
			"plane {}",
			index + 1
		);
	}
	Ok(())
});

test_vk!(stack_vertical_concatenates_each_plane, runtime, {
	let first = upload_i420(
		&runtime,
		8,
		4,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let second = upload_i420(
		&runtime,
		8,
		4,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let stacked = oa::video::stack(&[first, second], oa::video::VideoAxis::Vertical)?;
	assert_eq!(stacked.width(), 8);
	assert_eq!(stacked.height(), 8);
	let planes = i420_planes(8, 4);
	let out = stacked.planes().expect("device-planar planes");
	let luma = out[0].read::<u8>()?;
	assert_eq!(&luma[..32], &planes[0][..], "top half");
	assert_eq!(&luma[32..], &planes[0][..], "bottom half");
	for (index, expected) in planes[1..].iter().enumerate() {
		let chroma = out[index + 1].read::<u8>()?;
		assert_eq!(&chroma[..8], &expected[..], "plane {} top", index + 1);
		assert_eq!(&chroma[8..], &expected[..], "plane {} bottom", index + 1);
	}
	Ok(())
});

test_vk!(overlay_mask_stencils_fill_at_plane_resolution, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let mut mask_planes = i420_planes(8, 8);
	for byte in mask_planes[0].iter_mut() {
		*byte = 0;
	}
	for y in 2..6usize {
		for x in 2..6usize {
			mask_planes[0][y * 8 + x] = 255;
		}
	}
	let mask = oa::VideoFrame::upload_planes(
		VideoPixelFormat::Yuv420Planar8,
		mask_planes,
		8,
		8,
		timing(),
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let stenciled = oa::video::overlay_mask(&frame, &mask, [90, 20, 30])?;
	let planes = i420_planes(8, 8);
	let out = stenciled.planes().expect("device-planar planes");
	let luma = out[0].read::<u8>()?;
	for y in 0..8usize {
		for x in 0..8usize {
			let expected = if (2..6).contains(&y) && (2..6).contains(&x) {
				90
			} else {
				planes[0][y * 8 + x]
			};
			assert_eq!(luma[y * 8 + x], expected, "luma {x},{y}");
		}
	}
	// Chroma samples the mask at twice its coordinate: the 4x4 luma mask
	// region covers chroma cells (1..3, 1..3) via their top-left texel.
	let u = out[1].read::<u8>()?;
	for y in 0..4usize {
		for x in 0..4usize {
			let expected = if (1..3).contains(&y) && (1..3).contains(&x) {
				20
			} else {
				planes[1][y * 4 + x]
			};
			assert_eq!(u[y * 4 + x], expected, "u {x},{y}");
		}
	}
	Ok(())
});

test_vk!(composite_blends_each_component_with_rounding, runtime, {
	let back = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let front_planes: Vec<Vec<u8>> = i420_planes(8, 8)
		.iter()
		.map(|plane| plane.iter().map(|v| v.wrapping_add(97)).collect())
		.collect();
	let front = oa::VideoFrame::upload_planes(
		VideoPixelFormat::Yuv420Planar8,
		front_planes.clone(),
		8,
		8,
		timing(),
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let blended = oa::video::composite(&back, &front, 128)?;
	let planes = i420_planes(8, 8);
	let out = blended.planes().expect("device-planar planes");
	for (index, expected) in planes.iter().enumerate() {
		let actual = out[index].read::<u8>()?;
		for (position, (b, f)) in expected.iter().zip(front_planes[index].iter()).enumerate() {
			let oracle = ((*f as u32 * 128 + *b as u32 * 127 + 127) / 255) as u8;
			assert_eq!(actual[position], oracle, "plane {index} element {position}");
		}
	}
	Ok(())
});

/// Return whether a converted value sits on a .5 rounding boundary.
fn near_tie(converted: f32) -> bool {
	(converted - converted.floor() - 0.5).abs() < 1e-4
}

/// Assert one converted code against its f32 pre-round value. GPU `round` is
/// implementation-defined at exact .5 ties, so both neighbors are admitted
/// there and nowhere else.
fn assert_rounded(actual: u8, converted: f32, context: String) {
	let clamped = converted.clamp(0.0, 255.0);
	if near_tie(clamped) {
		let lo = clamped.floor() as i32;
		let hi = clamped.ceil() as i32;
		assert!(
			(lo..=hi).contains(&(actual as i32)),
			"{context}: {actual} outside {lo}..={hi}"
		);
	} else {
		assert_eq!(actual, clamped.round() as u8, "{context}");
	}
}

fn expand_limited(code: u8, luma: bool) -> f32 {
	if luma {
		(code as f32 - 16.0) * 255.0 / 219.0
	} else {
		(code as f32 - 128.0) * 255.0 / 224.0 + 128.0
	}
}

fn pack_full(code: u8, luma: bool) -> f32 {
	if luma {
		code as f32 * 219.0 / 255.0 + 16.0
	} else {
		(code as f32 - 128.0) * 224.0 / 255.0 + 128.0
	}
}

test_vk!(convert_range_expands_packs_and_copies, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let full = oa::video::convert_range(&frame, oa::video::VideoColorRange::Full)?;
	assert_eq!(full.color_info().range(), oa::video::VideoColorRange::Full);
	let planes = i420_planes(8, 8);
	let out = full.planes().expect("device-planar planes");
	for (index, expected) in planes.iter().enumerate() {
		let actual = out[index].read::<u8>()?;
		for (position, code) in expected.iter().enumerate() {
			assert_rounded(
				actual[position],
				expand_limited(*code, index == 0),
				format!("plane {index} element {position}"),
			);
		}
	}
	let back = oa::video::convert_range(&full, oa::video::VideoColorRange::Limited)?;
	assert_eq!(
		back.color_info().range(),
		oa::video::VideoColorRange::Limited
	);
	let round_trip = back.planes().expect("device-planar planes");
	for (index, expected) in planes.iter().enumerate() {
		let actual = round_trip[index].read::<u8>()?;
		for (position, code) in expected.iter().enumerate() {
			let expanded_pre = expand_limited(*code, index == 0);
			let expanded = expanded_pre.round().clamp(0.0, 255.0) as u8;
			let repacked = pack_full(expanded, index == 0);
			if near_tie(expanded_pre) {
				// The first-stage tie admits either neighbor, shifting the
				// repack result by one code.
				let lo = (repacked.floor() - 1.0).clamp(0.0, 255.0) as i32;
				let hi = (repacked.ceil() + 1.0).clamp(0.0, 255.0) as i32;
				assert!(
					(lo..=hi).contains(&(actual[position] as i32)),
					"plane {index} element {position}: {} outside {lo}..={hi}",
					actual[position]
				);
			} else {
				assert_rounded(
					actual[position],
					repacked,
					format!("plane {index} element {position}"),
				);
			}
		}
	}
	let copied = oa::video::convert_range(&frame, oa::video::VideoColorRange::Limited)?;
	assert_eq!(
		&copied.planes().expect("planes")[0].read::<u8>()?,
		&planes[0],
		"same-range conversion copies"
	);
	Ok(())
});

test_vk!(convert_range_covers_nv12_chroma_pair, runtime, {
	let frame = oa::VideoFrame::upload_planes(
		VideoPixelFormat::Nv12,
		nv12_planes(8, 8),
		8,
		8,
		timing(),
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let full = oa::video::convert_range(&frame, oa::video::VideoColorRange::Full)?;
	let planes = nv12_planes(8, 8);
	let uv = full.planes().expect("nv12 planes")[1].read::<u8>()?;
	for (position, code) in planes[1].iter().enumerate() {
		assert_rounded(
			uv[position],
			expand_limited(*code, false),
			format!("uv element {position}"),
		);
	}
	Ok(())
});

test_vk!(threshold_maps_luma_and_copies_chroma, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let mapped = oa::video::threshold(&frame, 100, 10, 235)?;
	let planes = i420_planes(8, 8);
	let out = mapped.planes().expect("device-planar planes");
	let luma = out[0].read::<u8>()?;
	for (position, code) in planes[0].iter().enumerate() {
		assert_eq!(
			luma[position],
			if *code >= 100 { 235 } else { 10 },
			"luma {position}"
		);
	}
	for (index, expected) in planes[1..].iter().enumerate() {
		assert_eq!(
			&out[index + 1].read::<u8>()?,
			expected,
			"plane {} copied",
			index + 1
		);
	}
	Ok(())
});

test_vk!(overlay_boxes_border_draws_a_hollow_ring, runtime, {
	let frame = upload_i420(
		&runtime,
		16,
		16,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let boxed = oa::video::overlay_boxes(
		&frame,
		&[oa::video::VideoBox {
			region: VideoRegion {
				x: 4,
				y: 4,
				width: 8,
				height: 8,
			},
			fill: [200, 100, 50],
		}],
		Some(1),
	)?;
	let planes = i420_planes(16, 16);
	let luma = boxed.planes().expect("device-planar planes")[0].read::<u8>()?;
	for y in 0..16usize {
		for x in 0..16usize {
			let inside = (4..12).contains(&x) && (4..12).contains(&y);
			let ring = inside && (x == 4 || x == 11 || y == 4 || y == 11);
			let expected = if ring { 200 } else { planes[0][y * 16 + x] };
			assert_eq!(luma[y * 16 + x], expected, "luma {x},{y}");
		}
	}
	Ok(())
});

test_vk!(geometry_ops_reject_invalid_arguments, runtime, {
	let frame = upload_i420(
		&runtime,
		8,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	assert!(
		oa::video::crop(
			&frame,
			VideoRegion {
				x: 1,
				y: 0,
				width: 4,
				height: 4
			}
		)
		.is_err(),
		"odd crop x"
	);
	assert!(
		oa::video::crop(
			&frame,
			VideoRegion {
				x: 6,
				y: 0,
				width: 8,
				height: 4
			}
		)
		.is_err(),
		"crop outside extent"
	);
	assert!(
		oa::video::pad(
			&frame,
			oa::video::VideoPadding {
				left: 1,
				top: 0,
				right: 0,
				bottom: 0
			},
			[0, 128, 128]
		)
		.is_err(),
		"odd pad left"
	);
	assert!(
		oa::video::resize(&frame, 0, 8, oa::video::VideoInterpolation::Bilinear).is_err(),
		"zero resize extent"
	);
	assert!(oa::video::stack(&[], oa::video::VideoAxis::Horizontal).is_err());
	let narrow = upload_i420(
		&runtime,
		4,
		8,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	assert!(
		oa::video::stack(&[frame.clone(), narrow], oa::video::VideoAxis::Vertical).is_err(),
		"stack cross-extent mismatch"
	);
	assert!(
		oa::video::overlay_boxes(
			&frame,
			&[oa::video::VideoBox {
				region: VideoRegion {
					x: 6,
					y: 0,
					width: 4,
					height: 4
				},
				fill: [0, 128, 128],
			}],
			None,
		)
		.is_err(),
		"box outside extent"
	);
	assert!(
		oa::video::overlay_mask(&frame, &frame, [0, 128, 128]).is_ok(),
		"same-extent mask is admissible"
	);
	let nv12 = oa::video::convert_format(&frame, VideoPixelFormat::Nv12)?;
	assert!(
		oa::video::composite(&frame, &nv12, 128).is_err(),
		"mixed-format composite"
	);
	assert!(
		oa::video::convert_format(&frame, VideoPixelFormat::Yuv420Planar8).is_err(),
		"same-format conversion"
	);
	let host = oa::VideoFrame::new_from_yuv420p(
		i420_planes(8, 8).concat(),
		8,
		8,
		VideoFrameTiming::new(std::time::Duration::from_millis(33), None)?,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	assert!(oa::video::resize(&host, 4, 4, oa::video::VideoInterpolation::Nearest).is_err());
	Ok(())
});

test_vk!(overlay_boxes_writes_fill_in_yuv, runtime, {
	let frame = upload_i420(
		&runtime,
		16,
		16,
		color(VideoColorMatrix::Bt709, VideoColorRange::Limited),
	)?;
	let boxed = oa::video::overlay_boxes(
		&frame,
		&[oa::video::VideoBox {
			region: VideoRegion {
				x: 2,
				y: 2,
				width: 4,
				height: 4,
			},
			fill: [200, 100, 50],
		}],
		None,
	)?;
	let planes = i420_planes(16, 16);
	let luma = boxed.planes().expect("device-planar planes")[0].read::<u8>()?;
	for y in 0..16usize {
		for x in 0..16usize {
			let expected = if (2..6).contains(&y) && (2..6).contains(&x) {
				200
			} else {
				planes[0][y * 16 + x]
			};
			assert_eq!(luma[y * 16 + x], expected, "luma {x},{y}");
		}
	}
	let u = boxed.planes().expect("planes")[1].read::<u8>()?;
	for y in 0..8usize {
		for x in 0..8usize {
			let expected = if (1..3).contains(&y) && (1..3).contains(&x) {
				100
			} else {
				planes[1][y * 8 + x]
			};
			assert_eq!(u[y * 8 + x], expected, "u {x},{y}");
		}
	}
	Ok(())
});
