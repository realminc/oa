use std::{collections::VecDeque, env, path::PathBuf, time::Instant};

use anyhow::{Context, bail, ensure};
use oa::video::{VideoDecoder, VideoDemuxer, VideoFrame, VideoPixelFormat};
use oa::{DeviceSelection, Engine, ErrorKind, Event};

struct Options {
	device: usize,
	warmup: usize,
	samples: usize,
	readback: bool,
	present: bool,
	clips: Vec<PathBuf>,
}

fn main() -> anyhow::Result<()> {
	let options = parse_options()?;
	let engine = Engine::builder()
		.devices(DeviceSelection::Index(options.device))
		.build()
		.context("create the OA Vulkan engine")?;
	Engine::set_default(engine.clone());
	for clip in &options.clips {
		bench_clip(&engine, clip, &options)?;
	}
	Ok(())
}

fn bench_clip(engine: &Engine, clip: &PathBuf, options: &Options) -> anyhow::Result<()> {
	// Correctness gate first: one decode pass must produce exactly
	// `sample_count` native frames before any timing is reported.
	let expected = decode_pass(engine, clip, options, None)?.frames;
	let info = VideoDemuxer::open(clip)?.info();
	ensure!(
		expected == info.sample_count() as usize,
		"correctness gate: decoded {} frames, expected {}",
		expected,
		info.sample_count()
	);
	for _ in 0..options.warmup {
		decode_pass(engine, clip, options, None)?;
	}
	let mut samples = Vec::with_capacity(options.samples);
	for _ in 0..options.samples {
		let started = Instant::now();
		let _pass = decode_pass(engine, clip, options, None)?;
		samples.push(started.elapsed().as_secs_f64() * 1_000.0);
	}
	samples.sort_by(f64::total_cmp);
	let median = samples[samples.len() / 2];
	let p95 = samples[(samples.len() * 95 / 100).min(samples.len() - 1)];
	let fps = expected as f64 / (median / 1_000.0);
	let name = clip
		.file_name()
		.and_then(|name| name.to_str())
		.unwrap_or("clip");
	println!(
		"OARS_BENCH name={name} codec={:?} {}x{} frames={expected} \
		 median_ms={median:.3} p95_ms={p95:.3} fps={fps:.1} \
		 samples={} correctness=pass",
		info.codec(),
		info.width(),
		info.height(),
		options.samples
	);
	Ok(())
}

struct Pass {
	frames: usize,
}

/// One full demux+decode pass over a clip. Frames stay on the device —
/// no pixel readback unless `readback` is requested and no on-device
/// YUV→RGBA presentation conversion unless `present` is requested,
/// matching the decode-capacity methodology of the C++ sustained-fps
/// campaign.
fn decode_pass(
	engine: &Engine,
	clip: &PathBuf,
	options: &Options,
	_marker: Option<()>,
) -> anyhow::Result<Pass> {
	let mut demuxer = VideoDemuxer::open(clip)?;
	let mut decoder = VideoDecoder::new(engine, demuxer.info())?;
	let mut frames = 0_usize;
	let mut packets = Vec::new();
	while let Some(packet) = demuxer.read_next_packet()? {
		packets.push(packet);
	}
	// Decode-ahead is bounded by the codec DPB: a slot stays pinned until the
	// frame's consumption event completes. `ResourceExhausted` is the decoder's
	// backpressure signal, so on exhaustion wait on the oldest in-flight
	// consumer before retrying the packet.
	let mut pending: VecDeque<Event> = VecDeque::new();
	for packet in &packets {
		let frame = decode_with_backpressure(&mut decoder, packet, &mut pending)?;
		if let Some(frame) = frame {
			if options.readback {
				read_back(&mut decoder, &frame)?;
			}
			if options.present {
				let _texture = decoder.native_frame_texture(&frame)?;
			}
			let consumer = engine.checkpoint()?;
			frame.mark_consumed(&consumer)?;
			pending.push_back(consumer);
			drop(frame);
			frames += 1;
		}
	}
	for frame in decoder.flush()? {
		if options.readback {
			read_back(&mut decoder, &frame)?;
		}
		if options.present {
			let _texture = decoder.native_frame_texture(&frame)?;
		}
		let consumer = engine.checkpoint()?;
		frame.mark_consumed(&consumer)?;
		drop(frame);
		frames += 1;
	}
	decoder.close();
	Ok(Pass { frames })
}

/// Decode one packet, honoring the decoder's DPB backpressure contract: on
/// `ResourceExhausted`, wait on the oldest in-flight consumer event and retry
/// the same packet until it submits.
fn decode_with_backpressure(
	decoder: &mut VideoDecoder,
	packet: &oa::video::VideoPacket,
	pending: &mut VecDeque<Event>,
) -> anyhow::Result<Option<VideoFrame>> {
	loop {
		match decoder.decode_native(packet) {
			Ok(frame) => return Ok(frame),
			Err(error) if error.kind() == ErrorKind::ResourceExhausted => {
				let oldest = pending
					.pop_front()
					.context("decode backpressure with no in-flight consumer")?;
				oldest.wait()?;
			}
			Err(error) => return Err(error.into()),
		}
	}
}

/// Read one native frame back to host planes, selecting the 8-bit or wide
/// (10/12-bit) contract from the frame's pixel format.
fn read_back(decoder: &mut VideoDecoder, frame: &VideoFrame) -> anyhow::Result<()> {
	match frame.pixel_format() {
		Some(VideoPixelFormat::Yuv420Planar10 | VideoPixelFormat::P010 | VideoPixelFormat::P012) => {
			let _ = decoder.read_native_yuv420p10(frame)?;
		}
		_ => {
			let _ = decoder.read_native_yuv420p(frame)?;
		}
	}
	Ok(())
}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options {
		device: 0,
		warmup: 2,
		samples: 7,
		readback: false,
		present: false,
		clips: Vec::new(),
	};
	let mut args = env::args().skip(1);
	while let Some(arg) = args.next() {
		let (key, value) = arg
			.split_once('=')
			.map(|(key, value)| (key, value.to_string()))
			.unwrap_or((arg.as_str(), String::new()));
		let mut take = |flag: &str| -> anyhow::Result<String> {
			if !value.is_empty() {
				return Ok(value.clone());
			}
			args.next().with_context(|| format!("{flag} needs a value"))
		};
		match key {
			"--device" => options.device = take("--device")?.parse()?,
			"--warmup" => options.warmup = take("--warmup")?.parse()?,
			"--samples" => options.samples = take("--samples")?.parse()?,
			"--readback" => options.readback = true,
			"--present" => options.present = true,
			"--clip" => options.clips.push(PathBuf::from(take("--clip")?)),
			"--clip-dir" => {
				let dir = PathBuf::from(take("--clip-dir")?);
				for entry in std::fs::read_dir(&dir)? {
					let path = entry?.path();
					if path.extension().is_some_and(|ext| ext == "mp4") {
						options.clips.push(path);
					}
				}
				options.clips.sort();
			}
			other => bail!("unknown option {other}"),
		}
	}
	ensure!(!options.clips.is_empty(), "pass --clip or --clip-dir");
	Ok(options)
}
