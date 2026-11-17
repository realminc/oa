use std::{env, time::Instant};

use anyhow::{Context, bail, ensure};
use oa::video::{
	H264Profile, H265Profile, VideoChromaSubsampling, VideoComponentBitDepth, VideoEncodeProfile,
	VideoEncoder, VideoRateControl, query_encode_capabilities,
};
use oa::{DeviceSelection, Engine};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FRAMES: usize = 60;

struct Options {
	device: usize,
	warmup: usize,
	samples: usize,
}

fn main() -> anyhow::Result<()> {
	let options = parse_options()?;
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(options.device))
		.build()
		.context("create the OA Vulkan engine")?;
	let yuv420 = VideoChromaSubsampling::Yuv420;
	for (name, profile) in [
		(
			"h264-high-420-8bit",
			VideoEncodeProfile::h264_high_420_8bit(),
		),
		(
			"h264-high10-420-10bit",
			VideoEncodeProfile::H264 {
				profile: H264Profile::High10,
				chroma_subsampling: yuv420,
				luma_bit_depth: VideoComponentBitDepth::Ten,
				chroma_bit_depth: VideoComponentBitDepth::Ten,
			},
		),
		(
			"h265-main-420-8bit",
			VideoEncodeProfile::h265_main_420_8bit(),
		),
		(
			"h265-main10-420-10bit",
			VideoEncodeProfile::H265 {
				profile: H265Profile::Main10,
				chroma_subsampling: yuv420,
				luma_bit_depth: VideoComponentBitDepth::Ten,
				chroma_bit_depth: VideoComponentBitDepth::Ten,
			},
		),
		(
			"h265-rext-420-10bit",
			VideoEncodeProfile::H265 {
				profile: H265Profile::FormatRangeExtensions,
				chroma_subsampling: yuv420,
				luma_bit_depth: VideoComponentBitDepth::Ten,
				chroma_bit_depth: VideoComponentBitDepth::Ten,
			},
		),
		(
			"h265-rext-420-12bit",
			VideoEncodeProfile::H265 {
				profile: H265Profile::FormatRangeExtensions,
				chroma_subsampling: yuv420,
				luma_bit_depth: VideoComponentBitDepth::Twelve,
				chroma_bit_depth: VideoComponentBitDepth::Twelve,
			},
		),
	] {
		bench_profile(&engine, name, profile, &options)?;
	}
	Ok(())
}

fn bench_profile(
	engine: &Engine,
	name: &str,
	profile: VideoEncodeProfile,
	options: &Options,
) -> anyhow::Result<()> {
	let caps = match query_encode_capabilities(engine, profile) {
		Ok(caps) => caps,
		Err(error) => {
			println!("OARS_BENCH name={name} supported=no error={error:?}");
			return Ok(());
		}
	};
	let supports = |mode: VideoRateControl| match mode {
		VideoRateControl::ConstantQp => caps.supports_constant_qp(),
		VideoRateControl::Cbr => caps.supports_cbr(),
		VideoRateControl::Vbr => caps.supports_vbr(),
	};
	ensure!(
		supports(VideoRateControl::ConstantQp),
		"constant-qp unsupported"
	);

	let wide_input = matches!(
		profile,
		VideoEncodeProfile::H264 { luma_bit_depth, .. }
		| VideoEncodeProfile::H265 { luma_bit_depth, .. }
		if luma_bit_depth != VideoComponentBitDepth::Eight
	);

	// Deterministic synthetic input: a moving gradient whose per-frame
	// content changes encode-relevant coefficients without readback cost.
	let mut rgba8 = vec![0u8; WIDTH as usize * HEIGHT as usize * 4];
	let mut rgba16 = vec![0u16; WIDTH as usize * HEIGHT as usize * 4];

	// Correctness gate: one full pass must accept every frame and emit at
	// least one packet before any timing is reported.
	let packets = encode_pass(engine, profile, &mut rgba8, &mut rgba16, wide_input)?;
	ensure!(packets > 0, "correctness gate: pass emitted no packets");
	for _ in 0..options.warmup {
		encode_pass(engine, profile, &mut rgba8, &mut rgba16, wide_input)?;
	}
	let mut samples = Vec::with_capacity(options.samples);
	for _ in 0..options.samples {
		let started = Instant::now();
		encode_pass(engine, profile, &mut rgba8, &mut rgba16, wide_input)?;
		samples.push(started.elapsed().as_secs_f64() * 1_000.0);
	}
	samples.sort_by(f64::total_cmp);
	let median = samples[samples.len() / 2];
	let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
	let fps = FRAMES as f64 / (median / 1_000.0);
	println!(
		"OARS_BENCH name={name} {WIDTH}x{HEIGHT} frames={FRAMES} \
		 median_ms={median:.3} p95_ms={p95:.3} fps={fps:.1} \
		 samples={} correctness=pass",
		options.samples
	);
	Ok(())
}

/// Encode `FRAMES` synthetic frames through one hardware session and return
/// the total packet count. Input is host-generated — this measures the
/// RGBA→YUV420 conversion + encode + bitstream-readback path, matching the
/// `VideoEncoder::encode`/`encode_rgba16` synchronous contracts. Wide
/// sessions take packed RGBA64; 8-bit sessions take packed RGBA8.
fn encode_pass(
	engine: &Engine,
	profile: VideoEncodeProfile,
	rgba8: &mut [u8],
	rgba16: &mut [u16],
	wide_input: bool,
) -> anyhow::Result<usize> {
	let config = oa::video::VideoEncodeConfig::new(profile, WIDTH, HEIGHT);
	let mut encoder = VideoEncoder::new(engine, config).context("open encode session")?;
	let mut packets = 0_usize;
	for index in 0..FRAMES {
		let pts = (index as u64) * 33_333;
		if wide_input {
			fill_frame16(rgba16, index as u32);
			encoder.encode_rgba16(rgba16, pts)?;
		} else {
			fill_frame(rgba8, index as u32);
			encoder.encode(rgba8, pts)?;
		}
		packets += 1;
	}
	packets += encoder.flush()?.len();
	encoder.close();
	Ok(packets)
}

/// Moving RGBA8 gradient; deterministic per frame index.
fn fill_frame(rgba: &mut [u8], frame: u32) {
	let mut i = 0_usize;
	for y in 0..HEIGHT {
		for x in 0..WIDTH {
			rgba[i] = ((x + frame * 3) & 0xff) as u8;
			rgba[i + 1] = ((y + frame * 5) & 0xff) as u8;
			rgba[i + 2] = ((x + y + frame * 7) & 0xff) as u8;
			rgba[i + 3] = 255;
			i += 4;
		}
	}
}

/// Moving RGBA64 gradient spanning the mid 10-bit range; deterministic per
/// frame index.
fn fill_frame16(rgba: &mut [u16], frame: u32) {
	let mut i = 0_usize;
	for y in 0..HEIGHT {
		for x in 0..WIDTH {
			rgba[i] = (((x + frame * 3) & 0x3ff) << 6) as u16;
			rgba[i + 1] = (((y + frame * 5) & 0x3ff) << 6) as u16;
			rgba[i + 2] = (((x + y + frame * 7) & 0x3ff) << 6) as u16;
			rgba[i + 3] = 0xffff;
			i += 4;
		}
	}
}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options {
		device: 0,
		warmup: 1,
		samples: 5,
	};
	let mut args = env::args().skip(1);
	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--device" => options.device = args.next().context("--device needs a value")?.parse()?,
			"--warmup" => options.warmup = args.next().context("--warmup needs a value")?.parse()?,
			"--samples" => options.samples = args.next().context("--samples needs a value")?.parse()?,
			other => bail!("unknown argument {other}"),
		}
	}
	Ok(options)
}
