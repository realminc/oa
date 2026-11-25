use super::*;
use crate::runtime::device::logical::vkcache;
use crate::runtime::dna::tuning::ObservedRoute;

#[test]
#[ignore = "explicit Vulkan device; proves unread timing pools do not form a Device cycle"]
fn unread_timing_pool_does_not_retain_owning_device() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);
	let weak = Arc::downgrade(&device);
	let timings = device.route_timings(vec![(1, ObservedRoute::Tape)])?;
	device.jit_observe_timings(timings.clone());
	drop(timings);
	drop(runtime);
	drop(device);
	assert!(
		weak.upgrade().is_none(),
		"unsubmitted queries must not form an ownership cycle"
	);
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device; bounds failed JIT admissions without invoking Slang"]
fn negative_pipeline_admissions_stay_bounded() -> crate::Result<()> {
	use crate::runtime::dna::ir::elementwise::{Instruction, Source};
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);
	let mut fusion = JitFusion::new();
	fusion.restored = true;
	fusion.compiler = Some(Err(jit_missing("deliberately absent test compiler")));
	for constant in 0..MAX_PIPELINE_ENTRIES + 16 {
		let program = Program {
			element_count: 1,
			input_bytes: vec![4],
			outputs: vec![0],
			instructions: vec![Instruction {
				opcode: 12,
				sources: vec![Source::Input(0), Source::Constant(constant as u32)],
				parameters: vec![],
			}],
		};
		assert!(fusion.admit(&device, &program, false).is_none());
	}
	assert_eq!(fusion.pipelines.len(), MAX_PIPELINE_ENTRIES);
	Ok(())
}

#[test]
#[ignore = "explicit trusted Slang library and spirv-val; cached artifact needs no compiler session"]
fn cached_artifact_is_usable_with_unavailable_compiler() -> crate::Result<()> {
	use crate::runtime::dna::ir::elementwise::{Instruction, Source};
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let compiler = SlangCompiler::open(std::path::Path::new(&library))?;
	let program = Program {
		element_count: 17,
		input_bytes: vec![68],
		outputs: vec![0],
		instructions: vec![Instruction {
			opcode: 12,
			sources: vec![Source::Input(0), Source::Constant(1.25f32.to_bits())],
			parameters: vec![],
		}],
	};
	let mut fusion = JitFusion::new();
	fusion.artifacts.chain(&compiler, &program, true)?;
	drop(compiler);
	fusion.compiler = Some(Err(jit_missing("deliberately absent test compiler")));
	// The cached artifact serves admissions without the compiler; the
	// descriptor profile remains part of artifact identity.
	assert!(fusion.artifacts.get(&program, true).is_some());
	assert!(fusion.artifacts.get(&program, false).is_none());
	let census = FusionCensus::new();
	fusion.queue_compile(&census, &program, false);
	assert_eq!(census.snapshot().jit_compiler_unavailable_count(), 1);
	assert!(
		fusion
			.pipelines
			.get(&program.artifact_key())
			.is_some_and(Option::is_none),
		"an unavailable compiler is remembered as a negative admission"
	);
	Ok(())
}

impl JitFusion {
	/// Completed observations `(signature, specialized samples, tape samples,
	/// selected)`, used by gated tests.
	pub(in crate::runtime::device) fn observations(&mut self) -> Vec<(u64, usize, usize, bool)> {
		self.drain_pending();
		self
			.observations
			.iter()
			.map(|(signature, observation)| {
				(
					*signature,
					observation.specialized_ns.len(),
					observation.tape_ns.len(),
					observation.specialized,
				)
			})
			.collect()
	}
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library + spirv-val; block artifact persistence"]
fn block_artifacts_restore_from_vkcache() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let var_dir = std::env::temp_dir().join(format!("oa-block-persist-{}", std::process::id()));
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let _var = ScopedEnv::set("OA_VAR", var_dir.as_os_str());
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);

	// Session one: admit compiles through the trusted path and the Runtime
	// snapshot serializes sections 7/8 into the scoped cache.
	let program = crate::runtime::dna::ir::tile::BlockProgram::matmul(64, 64, 16, 4, 4, 4)?;
	let key = device
		.jit_admit_tile(&program)
		.expect("block admission compiles under the trusted compiler");
	let scope = vkcache::VkCacheScope::new(device.physical());
	let updates = device.jit_section_updates();
	vkcache::update(&scope.path(), &scope, &updates)?;
	let file = vkcache::VkCacheFile::open(&scope.path(), &scope)?;
	let index_records = jit::decode_index(
		&file
			.section(vkcache::SECTION_BLOCK_ARTIFACT_INDEX)
			.expect("block index section")
			.bytes,
	)?;
	assert_eq!(index_records.len(), 1);
	assert_eq!(index_records[0].key, key);
	assert!(
		file
			.section(vkcache::SECTION_BLOCK_SPIRV_ARTIFACTS)
			.is_some()
	);
	drop(runtime);

	// Session two: a fresh JitFusion restores without admitting — the only
	// path that can populate `block_artifacts` — so a hit proves restore.
	let mut fusion = JitFusion::new();
	fusion.restore_blocks_once(&device);
	assert_eq!(fusion.block_artifacts.len(), 1);
	assert!(fusion.block_artifacts.contains_key(&key));
	// The restored words serve admission into a live pipeline — the
	// program's push ABI is re-verified, never the stale record's.
	assert_eq!(
		fusion.admit_block(&device, &program).as_deref(),
		Some(&key[..])
	);

	// Emitter-version invalidation: a record compiled by a different tile
	// emitter must restore nothing.
	let mut records = index_records;
	records[0].emitter_version = BLOCK_EMITTER_VERSION + 1;
	vkcache::update(
		&scope.path(),
		&scope,
		&[vkcache::SectionUpdate::replace(
			vkcache::SECTION_BLOCK_ARTIFACT_INDEX,
			1,
			jit::encode_index(&records)?,
		)],
	)?;
	let mut stale = JitFusion::new();
	stale.restore_blocks_once(&device);
	assert!(stale.block_artifacts.is_empty());
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library; block route records persist"]
fn block_route_observations_record_tile_dispatches() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let var_dir = std::env::temp_dir().join(format!("oa-block-obs-{}", std::process::id()));
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let _var = ScopedEnv::set("OA_VAR", var_dir.as_os_str());
	let _tile = ScopedEnv::set("OA_TILE_GEMM", std::ffi::OsStr::new("1"));
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);
	let left = crate::Matrix::new_from_f32([64, 64], &vec![0.25; 64 * 64])?;
	let right = crate::Matrix::new_from_f32([64, 64], &vec![0.5; 64 * 64])?;
	// Each flush rebuilds the graph: admission probes the specialized route
	// while recordings timestamp it under the block signature.
	for _ in 0..8 {
		crate::matrix::matmul(&left, &right)?.read_f32()?;
	}
	let updates = device.jit_section_updates();
	let observations = updates
		.iter()
		.find(|update| update.section_type == vkcache::SECTION_FUSION_ROUTE_OBSERVATIONS)
		.expect("tile dispatches must leave a route-observation update");
	let payload = observations.payload.as_deref().expect("section payload");
	let records = jit::decode_observations(payload)?;
	let program = crate::runtime::dna::ir::tile::BlockProgram::matmul(64, 64, 16, 4, 4, 4)?;
	let signature = block_signature(&program);
	let record = records
		.iter()
		.find(|record| record.signature_key == signature)
		.expect("the tile signature must own an observation record");
	assert!(record.selected_route <= jit::ROUTE_SPECIALIZED);
	// Timestamp pools are best-effort: where the device supports them the
	// specialized route accrues real GPU samples.
	if record.sample_count > 0 {
		assert!(record.specialized_p50_ns > 0);
	}
	Ok(())
}

/// Restore `OA_VAR`/`OA_SLANG_LIBRARY` after environment-sensitive runs.
struct ScopedEnv {
	name: &'static str,
	previous: Option<std::ffi::OsString>,
}

impl ScopedEnv {
	fn set(name: &'static str, value: &std::ffi::OsStr) -> Self {
		let previous = std::env::var_os(name);
		// SAFETY: ignored environment-mutating qualification runs are serial.
		unsafe { std::env::set_var(name, value) };
		Self { name, previous }
	}
}

impl Drop for ScopedEnv {
	fn drop(&mut self) {
		// SAFETY: same serial qualification scope as construction.
		unsafe {
			match &self.previous {
				Some(value) => std::env::set_var(self.name, value),
				None => std::env::remove_var(self.name),
			}
		}
	}
}

#[test]
fn tile_candidate_probes_then_holds_measured_minimum() {
	use crate::runtime::dna::tile::select_tile_candidate;
	// (specialized p50 ns, contributing samples, admission attempts)
	// in candidate order.
	assert_eq!(select_tile_candidate(&[]), None);
	// Fresh candidates probe least-samples-first, stable in order.
	assert_eq!(
		select_tile_candidate(&[(None, 0, 0), (None, 0, 0)]),
		Some(0)
	);
	assert_eq!(
		select_tile_candidate(&[(None, 1, 1), (None, 0, 1)]),
		Some(1)
	);
	// An under-sampled candidate is probed ahead of a measured winner.
	assert_eq!(
		select_tile_candidate(&[(Some(100), 4, 4), (None, 0, 0)]),
		Some(1)
	);
	// All measured: the median minimum holds.
	assert_eq!(
		select_tile_candidate(&[(Some(200), 8, 8), (Some(150), 6, 6)]),
		Some(1)
	);
	// A candidate that exhausted its probe budget without samples is
	// skipped; the measured winner holds.
	assert_eq!(
		select_tile_candidate(&[(Some(200), 8, 8), (None, 0, 16)]),
		Some(0)
	);
	// Everything declined still returns a candidate so the incumbent
	// hysteresis can run tape.
	assert_eq!(
		select_tile_candidate(&[(None, 0, 16), (None, 0, 16)]),
		Some(0)
	);
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library; audio normalize tile route admission and parity"]
fn audio_mel_normalize_tile_route_matches_incumbent() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);

	// The emitted program must compile, validate, and reflect its push ABI.
	let tile = ScopedEnv::set("OA_TILE_AUDIO", std::ffi::OsStr::new("1"));
	let program = crate::runtime::dna::ir::tile::BlockProgram::channel_normalize(256)?;
	assert!(
		device.jit_admit_tile_audio(&program).is_some(),
		"channel_normalize must admit"
	);
	let program = crate::runtime::dna::ir::tile::BlockProgram::audio_mel_filterbank();
	assert!(
		device.jit_admit_tile_audio(&program).is_some(),
		"audio_mel_filterbank must admit"
	);

	// 8 mels x 40 frames = 320 values per channel, striding the 256-lane
	// workgroup; two channels exercise the per-channel reduction.
	let samples: Vec<f32> = (0..4096)
		.map(|i| {
			(std::f32::consts::TAU * i as f32 / 37.0).sin()
				+ 0.3 * (std::f32::consts::TAU * i as f32 / 11.0).cos()
		})
		.collect();
	let input = crate::Audio::new_from_planar_f32(
		&runtime,
		&samples,
		2,
		16_000,
		crate::AudioChannelLayout::Stereo,
	)?;
	let config = crate::audio::MelConfig {
		fft_size: 64,
		hop_size: 32,
		num_mels: 8,
		log_scale: true,
		normalize: true,
		..crate::audio::MelConfig::default()
	};

	let before = device.fusion_census().snapshot();
	let specialized = crate::audio::mel_spectrogram(&input, config)?.read_f32()?;
	drop(tile);
	let after = device.fusion_census().snapshot();
	assert!(
		after.jit_admitted_count() + after.jit_reused_count()
			> before.jit_admitted_count() + before.jit_reused_count(),
		"the audio normalize specialization must be admitted: {after}"
	);
	let off = ScopedEnv::set("OA_TILE_AUDIO", std::ffi::OsStr::new("0"));
	let canonical = crate::audio::mel_spectrogram(&input, config)?.read_f32()?;
	drop(off);
	assert_eq!(canonical.len(), specialized.len());
	// Serial-donor vs lane-strided reduction order differs; allow float
	// accumulation tolerance instead of bitwise parity.
	for (i, (a, b)) in canonical.iter().zip(&specialized).enumerate() {
		assert!(
			(a - b).abs() <= 1e-4 * a.abs().max(1.0),
			"tile normalize diverges at {i}: canonical {a} vs tile {b}"
		);
	}

	// The mel filterbank rides the same audio gate — a serial per-lane
	// fold, so the tile route is bitwise-identical to the incumbent.
	// Cover both log_scale variants and an unnormalized output.
	for (log_scale, normalize) in [(true, false), (false, true), (false, false)] {
		let config = crate::audio::MelConfig {
			log_scale,
			normalize,
			..config
		};
		let tile = ScopedEnv::set("OA_TILE_AUDIO", std::ffi::OsStr::new("1"));
		let specialized = crate::audio::mel_spectrogram(&input, config)?.read_f32()?;
		drop(tile);
		let off = ScopedEnv::set("OA_TILE_AUDIO", std::ffi::OsStr::new("0"));
		let canonical = crate::audio::mel_spectrogram(&input, config)?.read_f32()?;
		drop(off);
		if normalize {
			// The lane-strided channel_normalize reduction order differs
			// from the serial donor; tolerance, not bitwise.
			for (i, (a, b)) in canonical.iter().zip(&specialized).enumerate() {
				assert!(
					(a - b).abs() <= 1e-4 * a.abs().max(1.0),
					"tile mel chain diverges at {i}: canonical {a} vs tile {b}"
				);
			}
		} else {
			assert_eq!(
				canonical, specialized,
				"tile mel filterbank must match the incumbent bitwise (log={log_scale})"
			);
		}
	}
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library; mel filterbank+normalize staged chain region and parity"]
fn audio_mel_chain_runs_as_one_post_fft_region() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);

	// The staged chain program must compile, validate, and admit.
	let tile = ScopedEnv::set("OA_TILE_AUDIO", std::ffi::OsStr::new("1"));
	let program = crate::runtime::dna::ir::tile::BlockProgram::audio_mel_chain(256, 512)?;
	assert!(
		device.jit_admit_tile_audio(&program).is_some(),
		"audio_mel_chain must admit"
	);

	// 8 mels x 40 frames = 320 values per channel; two channels.
	let samples: Vec<f32> = (0..4096)
		.map(|i| {
			(std::f32::consts::TAU * i as f32 / 37.0).sin()
				+ 0.3 * (std::f32::consts::TAU * i as f32 / 11.0).cos()
		})
		.collect();
	let input = crate::Audio::new_from_planar_f32(
		&runtime,
		&samples,
		2,
		16_000,
		crate::AudioChannelLayout::Stereo,
	)?;
	let config = crate::audio::MelConfig {
		fft_size: 64,
		hop_size: 32,
		num_mels: 8,
		log_scale: true,
		normalize: true,
		..crate::audio::MelConfig::default()
	};

	// The post-FFT side must be ONE dispatch: STFT + the staged
	// filterbank→normalize chain region.
	let mel = |input: &crate::Audio| -> crate::Result<crate::Matrix> {
		crate::audio::mel_spectrogram(input, config)
	};
	let (plan, fused_output) = runtime.capture(|| mel(&input))?;
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("mel chain")).unwrap();
	assert_eq!(
		report["stats"]["dispatches"], 2,
		"stft + staged mel chain must emit two dispatches: {report}"
	);
	runtime.submit(&plan)?.wait()?;
	let fused = fused_output.read_f32()?;
	drop(tile);

	// The incumbent chain runs filterbank and normalize separately.
	let canonical = {
		let _off = ScopedEnv::set("OA_TILE_AUDIO", std::ffi::OsStr::new("0"));
		mel(&input)?.read_f32()?
	};
	assert_eq!(canonical.len(), fused.len());
	// The lane-strided reduce order differs from the serial donor;
	// tolerance, not bitwise.
	for (i, (a, b)) in canonical.iter().zip(&fused).enumerate() {
		assert!(
			(a - b).abs() <= 1e-4 * a.abs().max(1.0),
			"staged mel chain diverges at {i}: canonical {a} vs tile {b}"
		);
	}
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library; image resize tile route admission and parity"]
fn image_resize_tile_route_matches_incumbent() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);

	// Both variants must compile, validate, and reflect their push ABI.
	let tile = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("1"));
	for nearest in [true, false] {
		let program = crate::runtime::dna::ir::tile::BlockProgram::image_resize(nearest)?;
		assert!(
			device.jit_admit_tile_image(&program).is_some(),
			"image_resize(nearest={nearest}) must admit"
		);
	}

	// NCHW batch 2 x 3 channels over an odd extent; non-integer scale
	// factors exercise the clamped gather's fractional and edge paths.
	let shape = [2_usize, 3, 5, 7];
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 17 % 31) as f32 - 15.0) * 0.125)
		.collect::<Vec<_>>();
	let image = crate::Image::new(
		crate::Matrix::new_from_f32(shape, &values)?,
		crate::ImageLayout::Nchw,
		crate::ImageFormat::Rgb,
	)?;

	let parity =
		|width: u32, height: u32, mode: crate::image::InterpolationMode| -> crate::Result<()> {
			let before = device.fusion_census().snapshot();
			let specialized = crate::image::resize(&image, width, height, mode)?
				.as_matrix()
				.read_f32()?;
			let after = device.fusion_census().snapshot();
			assert!(
				after.jit_admitted_count() + after.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"the image resize specialization must be admitted: {after}"
			);
			let off = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("0"));
			let canonical = crate::image::resize(&image, width, height, mode)?
				.as_matrix()
				.read_f32()?;
			drop(off);
			assert_eq!(
				canonical, specialized,
				"tile-emitted resize must match the incumbent bitwise ({mode:?} {width}x{height})"
			);
			Ok(())
		};
	for mode in [
		crate::image::InterpolationMode::Nearest,
		crate::image::InterpolationMode::Bilinear,
	] {
		for (width, height) in [(11, 9), (3, 3), (14, 10)] {
			parity(width, height, mode)?;
		}
	}

	// The flip remap rides the same gather vocabulary — both flag combos.
	let flip_parity = |horizontal: bool, vertical: bool| -> crate::Result<()> {
		let before = device.fusion_census().snapshot();
		let specialized = crate::image::flip(&image, horizontal, vertical)?
			.as_matrix()
			.read_f32()?;
		let after = device.fusion_census().snapshot();
		assert!(
			after.jit_admitted_count() + after.jit_reused_count()
				> before.jit_admitted_count() + before.jit_reused_count(),
			"the image flip specialization must be admitted: {after}"
		);
		let off = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("0"));
		let canonical = crate::image::flip(&image, horizontal, vertical)?
			.as_matrix()
			.read_f32()?;
		drop(off);
		assert_eq!(
			canonical, specialized,
			"tile-emitted flip must match the incumbent bitwise (h={horizontal} v={vertical})"
		);
		Ok(())
	};
	for horizontal in [false, true] {
		for vertical in [false, true] {
			flip_parity(horizontal, vertical)?;
		}
	}

	// The bordered-gather donors: pad, remap, and both warps across
	// border modes. Admission is asserted on each program's first use —
	// the route-observation policy may demote repeat dispatches to the
	// incumbent, so later calls prove parity only.
	let mut fresh = std::collections::BTreeSet::new();
	let admit = |fresh: &mut std::collections::BTreeSet<&'static str>,
	             label: &str,
	             program: &'static str,
	             run: &dyn Fn() -> crate::Result<Vec<f32>>|
	 -> crate::Result<Vec<f32>> {
		let first = fresh.insert(program);
		let before = device.fusion_census().snapshot();
		let specialized = run()?;
		let after = device.fusion_census().snapshot();
		if first {
			assert!(
				after.jit_admitted_count() + after.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"{label}: the tile specialization must be admitted: {after}"
			);
		}
		let off = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("0"));
		let canonical = run()?;
		drop(off);
		assert_eq!(
			canonical, specialized,
			"{label}: tile-emitted output must match the incumbent bitwise"
		);
		Ok(specialized)
	};

	let borders = [
		crate::image::BorderMode::Constant,
		crate::image::BorderMode::Replicate,
		crate::image::BorderMode::Reflect,
		crate::image::BorderMode::Reflect101,
		crate::image::BorderMode::Wrap,
	];
	for border in borders {
		admit(&mut fresh, &format!("pad {border:?}"), "pad", &|| {
			crate::image::pad(&image, 2, 1, 1, 3, border, -7.0)?
				.as_matrix()
				.read_f32()
		})?;
	}

	// remap over a fractional/edge map on a shared batch.
	let map = crate::Matrix::new_from_f32(
		[2, 2, 5, 7],
		&(0..140)
			.map(|index| (index * 7 % 29) as f32 * 0.25 - 1.5)
			.collect::<Vec<_>>(),
	)?;
	for border in borders {
		for interp in [
			crate::image::InterpolationMode::Nearest,
			crate::image::InterpolationMode::Bilinear,
		] {
			admit(
				&mut fresh,
				&format!("remap {border:?} {interp:?}"),
				"remap",
				&|| {
					crate::image::remap(&image, &map, interp, border, 0.25)?
						.as_matrix()
						.read_f32()
				},
			)?;
		}
	}

	// Both warp donors — rotation+shear affine and a true perspective.
	let affine = crate::Matrix::new_from_f32([2, 3], &[0.9, -0.2, 1.5, 0.1, 1.1, -0.5])?;
	let perspective = crate::Matrix::new_from_f32(
		[3, 3],
		&[1.0, 0.1, 0.0, -0.05, 1.0, 0.0, 0.001, 0.0005, 1.0],
	)?;
	for border in borders {
		for interp in [
			crate::image::InterpolationMode::Nearest,
			crate::image::InterpolationMode::Bilinear,
		] {
			admit(
				&mut fresh,
				&format!("warp_affine {border:?} {interp:?}"),
				"warp_affine",
				&|| {
					crate::image::warp_affine(&image, &affine, 9, 7, interp, border, -3.0)?
						.as_matrix()
						.read_f32()
				},
			)?;
			admit(
				&mut fresh,
				&format!("warp_perspective {border:?} {interp:?}"),
				"warp_perspective",
				&|| {
					crate::image::warp_perspective(&image, &perspective, 8, 6, interp, border, 1.0)?
						.as_matrix()
						.read_f32()
				},
			)?;
		}
	}

	// The index-remap donors: crop, center_crop, and all rotate degrees.
	admit(&mut fresh, "crop", "crop", &|| {
		crate::image::crop(&image, 1, 2, 4, 3)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "center_crop", "crop", &|| {
		crate::image::center_crop(&image, 3, 4)?
			.as_matrix()
			.read_f32()
	})?;
	for degrees in [0u32, 90, 180, 270] {
		admit(&mut fresh, &format!("rotate {degrees}"), "rotate", &|| {
			crate::image::rotate(&image, degrees)?
				.as_matrix()
				.read_f32()
		})?;
	}

	// The fold family on the bordered-window anchor: uniform blur,
	// morphology Min/Max, nested twice folds, hats, and separable.
	for (kw, kh) in [(3usize, 3usize), (5, 5), (3, 7)] {
		for border in borders {
			admit(
				&mut fresh,
				&format!("blur {kw}x{kh} {border:?}"),
				"blur",
				&|| {
					crate::image::average_blur(&image, kw as u32, kh as u32, border)?
						.as_matrix()
						.read_f32()
				},
			)?;
		}
	}
	for (kw, kh) in [(3usize, 3usize)] {
		for border in borders {
			admit(&mut fresh, &format!("erode {border:?}"), "erode", &|| {
				crate::image::erode(&image, kw as u32, kh as u32, border, 0.0)?
					.as_matrix()
					.read_f32()
			})?;
			admit(&mut fresh, &format!("dilate {border:?}"), "dilate", &|| {
				crate::image::dilate(&image, kw as u32, kh as u32, border, 0.0)?
					.as_matrix()
					.read_f32()
			})?;
			admit(&mut fresh, &format!("open {border:?}"), "open", &|| {
				crate::image::morphology_open(&image, kw as u32, kh as u32, border, 0.0)?
					.as_matrix()
					.read_f32()
			})?;
			admit(&mut fresh, &format!("close {border:?}"), "close", &|| {
				crate::image::morphology_close(&image, kw as u32, kh as u32, border, 0.0)?
					.as_matrix()
					.read_f32()
			})?;
			admit(
				&mut fresh,
				&format!("gradient {border:?}"),
				"gradient",
				&|| {
					crate::image::morphology_gradient(&image, kw as u32, kh as u32, border, 0.0)?
						.as_matrix()
						.read_f32()
				},
			)?;
			admit(
				&mut fresh,
				&format!("top_hat {border:?}"),
				"top_hat",
				&|| {
					crate::image::morphology_top_hat(&image, kw as u32, kh as u32, border, 0.0)?
						.as_matrix()
						.read_f32()
				},
			)?;
			admit(
				&mut fresh,
				&format!("black_hat {border:?}"),
				"black_hat",
				&|| {
					crate::image::morphology_black_hat(&image, kw as u32, kh as u32, border, 0.0)?
						.as_matrix()
						.read_f32()
				},
			)?;
		}
	}
	let kernel_x = crate::Matrix::new_from_f32([1, 3], &[0.5, 1.0, 0.5])?;
	let kernel_y = crate::Matrix::new_from_f32([3, 1], &[0.25, 0.5, 0.25])?;
	for border in borders {
		admit(
			&mut fresh,
			&format!("separable {border:?}"),
			"separable",
			&|| {
				crate::image::separable_convolve_2d(&image, &kernel_x, &kernel_y, border, 0.0)?
					.as_matrix()
					.read_f32()
			},
		)?;
	}

	// The fixed 3x3 coefficient folds and the weighted neighborhoods:
	// sobel/scharr directional pairs, laplacian, sharpen, gaussian,
	// bilateral, unsharp, and both adaptive thresholds.
	for border in borders {
		admit(
			&mut fresh,
			&format!("sobel_x {border:?}"),
			"sobel_x",
			&|| {
				crate::image::sobel(&image, 1, 0, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("sobel_y {border:?}"),
			"sobel_y",
			&|| {
				crate::image::sobel(&image, 0, 1, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("scharr_x {border:?}"),
			"scharr_x",
			&|| {
				crate::image::scharr(&image, 1, 0, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("scharr_y {border:?}"),
			"scharr_y",
			&|| {
				crate::image::scharr(&image, 0, 1, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("laplacian {border:?}"),
			"laplacian",
			&|| {
				crate::image::laplacian(&image, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("sharpen {border:?}"),
			"sharpen",
			&|| {
				crate::image::sharpen(&image, 1.5, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("gaussian {border:?}"),
			"gaussian",
			&|| {
				crate::image::gaussian_blur(&image, 1.4, 5)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("bilateral {border:?}"),
			"bilateral",
			&|| {
				crate::image::bilateral_filter(&image, 5, 0.35, 2.0, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("unsharp {border:?}"),
			"unsharp",
			&|| {
				crate::image::unsharp_mask(&image, 1.0, 0.8, 5)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("adp_mean {border:?}"),
			"adp_mean",
			&|| {
				crate::image::adaptive_threshold_mean(&image, 5, 2.0, 255.0, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
		admit(
			&mut fresh,
			&format!("adp_gauss {border:?}"),
			"adp_gauss",
			&|| {
				crate::image::adaptive_threshold_gaussian(&image, 5, 2.0, 255.0, 1.0, border)?
					.as_matrix()
					.read_f32()
			},
		)?;
	}
	admit(&mut fresh, "gaussian resolved", "gaussian", &|| {
		crate::image::gaussian_blur(&image, 0.7, 0)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "unsharp resolved", "unsharp", &|| {
		crate::image::unsharp_mask(&image, 0.7, 1.5, 0)?
			.as_matrix()
			.read_f32()
	})?;

	// The parameterized pixel map — all twelve `pointwise` donor
	// branches (thresholds, range, clamp, invert, affine, gamma,
	// solarize, posterize).
	admit(&mut fresh, "thresh_bin", "pointwise0", &|| {
		crate::image::threshold_binary(&image, 0.25, 1.0)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "thresh_inv", "pointwise1", &|| {
		crate::image::threshold_binary_inv(&image, 0.25, 1.0)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "truncate", "pointwise2", &|| {
		crate::image::threshold_truncate(&image, 0.5)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "to_zero", "pointwise3", &|| {
		crate::image::threshold_to_zero(&image, 0.5)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "to_zero_inv", "pointwise4", &|| {
		crate::image::threshold_to_zero_inv(&image, 0.5)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "in_range", "pointwise5", &|| {
		crate::image::in_range(&image, -0.5, 0.5, 1.0)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "clamp", "pointwise6", &|| {
		crate::image::clamp(&image, -0.5, 0.75)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "invert", "pointwise7", &|| {
		crate::image::invert(&image, 1.0)?.as_matrix().read_f32()
	})?;
	admit(&mut fresh, "brightness", "pointwise8", &|| {
		crate::image::brightness_contrast(&image, 1.25, 0.1)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "gamma", "pointwise9", &|| {
		crate::image::gamma_contrast(&image, 1.8, 1.1)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "solarize", "pointwise10", &|| {
		crate::image::solarize(&image, 0.0, 1.0)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "posterize", "pointwise11", &|| {
		crate::image::posterize(&image, 4, -1.0, 1.0)?
			.as_matrix()
			.read_f32()
	})?;

	// The label-indexed palette blend: i32 labels index [K,3] colors;
	// out-of-range, negative, and alpha-channel pixels pass through.
	let mask = crate::Matrix::new_from_slice(
		[2usize, 5, 7],
		&(0..70i32)
			.map(|i| match i % 11 {
				9 => -1,
				10 => 7,
				value => value % 4,
			})
			.collect::<Vec<_>>(),
	)?;
	let palette = crate::Matrix::new_from_f32(
		[4usize, 3],
		&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5],
	)?;
	admit(&mut fresh, "overlay", "overlay", &|| {
		crate::image::segmentation_overlay(&image, &mask, &palette, 0.4)?
			.as_matrix()
			.read_f32()
	})?;

	// The color family: per-channel `(v − m)/s` normalize, the RGB/BGR
	// plane permutation, and the half-pixel resize + normalize fused
	// donor with the swap_rb tail word.
	let parameters = crate::image::NormalizationConfig {
		mean: [0.125, -0.375, 0.5],
		std: [0.5, 1.0, 2.0],
	};
	admit(&mut fresh, "normalize", "normalize", &|| {
		crate::image::normalize(&image, parameters)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "convert to bgr", "convert_color", &|| {
		crate::image::convert_color(&image, crate::ImageFormat::Bgr)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "convert identity", "convert_color", &|| {
		crate::image::convert_color(&image, crate::ImageFormat::Rgb)?
			.as_matrix()
			.read_f32()
	})?;
	for (width, height) in [(11_u32, 9_u32), (4, 4), (14, 6)] {
		admit(
			&mut fresh,
			&format!("resize_normalize {width}x{height}"),
			"resize_normalize",
			&|| {
				crate::image::resize_normalize(&image, width, height, parameters)?
					.as_matrix()
					.read_f32()
			},
		)?;
	}

	// The elementwise/compositing pixel family: Rec.709 grayscale, the
	// order-array channel remap, the 3x4 color twist, and the three
	// composite-donor ops (alpha/mask blend and the erase rect).
	admit(&mut fresh, "grayscale", "grayscale", &|| {
		crate::image::grayscale(&image)?.as_matrix().read_f32()
	})?;
	admit(&mut fresh, "reorder", "reorder", &|| {
		crate::image::channel_reorder(&image, [2, 0, 1, 3], crate::ImageFormat::Rgb)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "reorder identity", "reorder", &|| {
		crate::image::channel_reorder(&image, [0, 1, 2, 3], crate::ImageFormat::Rgb)?
			.as_matrix()
			.read_f32()
	})?;
	let twist = crate::Matrix::new_from_f32(
		[3usize, 4],
		&[
			0.0, 1.0, 0.0, 0.25, //
			1.0, 0.0, 0.0, -0.5, //
			0.0, 0.0, 1.0, 0.125, //
		],
	)?;
	admit(&mut fresh, "color_twist", "color_twist", &|| {
		crate::image::color_twist(&image, &twist)?
			.as_matrix()
			.read_f32()
	})?;
	let right = crate::Image::new(
		crate::Matrix::new_from_f32(shape, &values.iter().rev().copied().collect::<Vec<_>>())?,
		crate::ImageLayout::Nchw,
		crate::ImageFormat::Rgb,
	)?;
	admit(&mut fresh, "alpha_blend", "composite", &|| {
		crate::image::alpha_blend(&image, &right, 0.3)?
			.as_matrix()
			.read_f32()
	})?;
	let mask_image = crate::Image::new(
		crate::Matrix::new_from_f32(
			[2usize, 1, 5, 7],
			&(0..70)
				.map(|i| ((i * 7) % 13) as f32 / 12.0)
				.collect::<Vec<_>>(),
		)?,
		crate::ImageLayout::Nchw,
		crate::ImageFormat::Gray,
	)?;
	admit(&mut fresh, "composite mask", "composite", &|| {
		crate::image::composite(&image, &right, &mask_image)?
			.as_matrix()
			.read_f32()
	})?;
	admit(&mut fresh, "erase", "composite", &|| {
		crate::image::erase(&image, 1, 2, 3, 2, -0.75)?
			.as_matrix()
			.read_f32()
	})?;

	// The bordered-window conv: runtime-K taps folded serially in the
	// donor's ky-outer/kx-inner order — 1x1, square, rectangular, and
	// wider-than-tall kernels across border modes.
	for (kh, kw) in [(1usize, 1usize), (3, 3), (5, 3), (3, 5), (7, 7)] {
		let coefficients = (0..kh * kw)
			.map(|index| ((index * 11 % 17) as f32 - 8.0) * 0.0625)
			.collect::<Vec<_>>();
		let kernel = crate::Matrix::new_from_f32([kh, kw], &coefficients)?;
		for border in borders {
			admit(
				&mut fresh,
				&format!("convolve {kh}x{kw} {border:?}"),
				"convolve",
				&|| {
					crate::image::convolve_2d(&image, &kernel, border, -2.0)?
						.as_matrix()
						.read_f32()
				},
			)?;
		}
	}
	drop(tile);
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library; video tile route admission and parity"]
fn video_to_matrix_tile_route_matches_incumbent() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let value = crate::matrix::from_f32([1], &[1.0])?;
	let device = value
		.storage()
		.buffer()
		.expect("device buffer")
		.device()
		.clone();
	drop(value);

	// The emitted program must compile, validate, and reflect its push ABI.
	for (planar, bits) in [(false, 8), (true, 8), (false, 10)] {
		let program = crate::runtime::dna::ir::tile::BlockProgram::video_to_matrix(planar, bits)?;
		assert!(
			device.jit_admit_tile(&program).is_some(),
			"video_to_matrix(planar={planar}, bits={bits}) must admit"
		);
	}

	// Run `to_matrix` on the tile route and the canonical kernel; the outputs
	// must match bitwise and the specialization must actually admit.
	let parity =
		|frames: &[crate::VideoFrame], config: &crate::video::VideoMatrixConfig| -> crate::Result<()> {
			let tile = ScopedEnv::set("OA_TILE_VIDEO", std::ffi::OsStr::new("1"));
			let before = device.fusion_census().snapshot();
			let specialized = crate::video::to_matrix(frames, config)?.read_f32()?;
			drop(tile);
			let after = device.fusion_census().snapshot();
			assert!(
				after.jit_admitted_count() + after.jit_reused_count()
					> before.jit_admitted_count() + before.jit_reused_count(),
				"the video tile specialization must be admitted: {after}"
			);
			let off = ScopedEnv::set("OA_TILE_VIDEO", std::ffi::OsStr::new("0"));
			let canonical = crate::video::to_matrix(frames, config)?.read_f32()?;
			drop(off);
			assert_eq!(
				canonical, specialized,
				"tile-emitted conversion must match the incumbent bitwise"
			);
			Ok(())
		};

	let timing = crate::video::VideoFrameTiming::new(std::time::Duration::from_millis(33), None)?;
	let color = |matrix: crate::video::VideoColorMatrix, range: crate::video::VideoColorRange| {
		crate::video::VideoColorInfo::new(matrix, range)
	};
	let fill = |w: usize, h: usize, salt: u8| {
		(0..h)
			.flat_map(|y| (0..w).map(move |x| ((x * 5 + y * 7 + salt as usize) % 251) as u8))
			.collect::<Vec<u8>>()
	};
	let (w, h) = (32usize, 24usize);
	let (cw, chh) = (w / 2, h / 2);
	let config = crate::video::VideoMatrixConfig::default();

	// NV12 + BT.709 limited + bilinear NCHW RGB.
	let uv: Vec<u8> = (0..chh)
		.flat_map(|y| (0..cw * 2).map(move |x| ((x * 3 + y * 11 + 40) % 251) as u8))
		.collect();
	let nv12 = crate::VideoFrame::upload_planes(
		crate::video::VideoPixelFormat::Nv12,
		vec![fill(w, h, 3), uv],
		w,
		h,
		timing,
		color(
			crate::video::VideoColorMatrix::Bt709,
			crate::video::VideoColorRange::Limited,
		),
	)?;
	parity(std::slice::from_ref(&nv12), &config)?;

	// I420 + crop + letterbox + nearest + luma-only channels.
	let i420 = crate::VideoFrame::upload_planes(
		crate::video::VideoPixelFormat::Yuv420Planar8,
		vec![fill(w, h, 1), fill(cw, chh, 2), fill(cw, chh, 3)],
		w,
		h,
		timing,
		color(
			crate::video::VideoColorMatrix::Bt601,
			crate::video::VideoColorRange::Full,
		),
	)?;
	let i420_config = crate::video::VideoMatrixConfig {
		channels: crate::video::MatrixChannels::Luma,
		layout: crate::ImageLayout::Hwc,
		crop: Some(crate::video::VideoRegion {
			x: 4,
			y: 2,
			width: 20,
			height: 12,
		}),
		size: Some((16, 16)),
		resize: crate::video::VideoResize::Letterbox {
			fill: [0.5, 0.5, 0.5],
		},
		interpolation: crate::video::VideoInterpolation::Nearest,
		..config
	};
	parity(std::slice::from_ref(&i420), &i420_config)?;

	// P010 10-bit extraction + full-range scaling + center-crop resize.
	let pack10 = |samples: &[u16]| {
		samples
			.iter()
			.flat_map(|code| (code << 6).to_le_bytes())
			.collect::<Vec<u8>>()
	};
	let luma10: Vec<u16> = (0..w * h).map(|i| (i * 9 % 1024) as u16).collect();
	let chroma10: Vec<u16> = (0..cw * 2 * chh).map(|i| (i * 13 % 1024) as u16).collect();
	let p010 = crate::VideoFrame::upload_planes(
		crate::video::VideoPixelFormat::P010,
		vec![pack10(&luma10), pack10(&chroma10)],
		w,
		h,
		timing,
		color(
			crate::video::VideoColorMatrix::Bt601,
			crate::video::VideoColorRange::Full,
		),
	)?;
	let p010_config = crate::video::VideoMatrixConfig {
		crop: Some(crate::video::VideoRegion {
			x: 8,
			y: 4,
			width: 16,
			height: 16,
		}),
		size: Some((12, 12)),
		resize: crate::video::VideoResize::CenterCrop,
		interpolation: crate::video::VideoInterpolation::Nearest,
		mean: [0.1, 0.2, 0.3],
		std: [0.9, 0.8, 0.7],
		..config
	};
	parity(std::slice::from_ref(&p010), &p010_config)?;

	// Batched conversion exercises the z/group dimension (same format
	// required by the batch contract).
	let nv12_second = crate::VideoFrame::upload_planes(
		crate::video::VideoPixelFormat::Nv12,
		vec![fill(w, h, 9), fill(w, chh, 5)],
		w,
		h,
		timing,
		color(
			crate::video::VideoColorMatrix::Bt709,
			crate::video::VideoColorRange::Limited,
		),
	)?;
	parity(&[nv12, nv12_second], &config)?;
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library; tile scalar-epilogue merge admission and parity"]
fn image_scalar_epilogue_folds_into_the_tile_route() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());
	let _tile = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("1"));

	let shape = [2_usize, 3, 5, 7];
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 17 % 31) as f32 - 15.0) * 0.125)
		.collect::<Vec<_>>();
	let image = crate::Image::new(
		crate::Matrix::new_from_f32(shape, &values)?,
		crate::ImageLayout::Nchw,
		crate::ImageFormat::Rgb,
	)?;
	let chain = |image: &crate::Image| -> crate::Result<crate::Matrix> {
		crate::image::resize(image, 14, 10, crate::image::InterpolationMode::Bilinear)?
			.as_matrix()
			.sub_scalar(0.25)?
			.div_scalar(0.5)
	};

	// The captured chain must fuse into ONE dispatch: the tile gather
	// carries both scalar ops in its store tail.
	let (plan, fused_output) = runtime.capture(|| chain(&image))?;
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("image normalize")).unwrap();
	assert_eq!(
		report["stats"]["dispatches"], 1,
		"resize + scalar normalize must emit one region: {report}"
	);
	runtime.submit(&plan)?.wait()?;
	let fused = fused_output.read_f32()?;

	// The unfused reference runs the same ops on separate dispatches.
	let unfused = {
		let _off = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("0"));
		chain(&image)?.read_f32()?
	};
	// The folded tail emits one expression tree; the incumbent chain
	// materializes each scalar op — codegen may differ by 1 ulp, so the
	// differential contract is a tight tolerance, not bitwise.
	let worst = fused
		.iter()
		.zip(&unfused)
		.map(|(a, b)| (a - b).abs())
		.fold(0.0f32, f32::max);
	assert!(
		worst <= 1.0e-6,
		"fused resize+normalize region diverges from the incumbent chain: worst |d|={worst}"
	);

	// Unary tails fold identically: crop → relu → sqrt stays one region.
	let unary = |image: &crate::Image| -> crate::Result<crate::Matrix> {
		crate::image::center_crop(image, 4, 4)?
			.as_matrix()
			.add_scalar(1.0)?
			.sqrt()?
			.neg()
	};
	let (plan, fused_output) = runtime.capture(|| unary(&image))?;
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("unary tail")).unwrap();
	assert_eq!(
		report["stats"]["dispatches"], 1,
		"crop + unary tail must emit one region: {report}"
	);
	runtime.submit(&plan)?.wait()?;
	let fused = fused_output.read_f32()?;
	let unfused = {
		let _off = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("0"));
		unary(&image)?.read_f32()?
	};
	let worst = fused
		.iter()
		.zip(&unfused)
		.map(|(a, b)| (a - b).abs())
		.fold(0.0f32, f32::max);
	assert!(
		worst <= 1.0e-6,
		"fused crop+unary region diverges from the incumbent chain: worst |d|={worst}"
	);

	// Prologue direction: a pointwise producer folds into an
	// elementwise consumer's input load — `log → flip` runs as one
	// region reading the producer's input directly.
	let positive = crate::Image::new(
		image.as_matrix().exp()?,
		crate::ImageLayout::Nchw,
		crate::ImageFormat::Rgb,
	)?;
	runtime.checkpoint()?.wait()?;
	let prologue = |image: &crate::Image| -> crate::Result<crate::Matrix> {
		let upstream = crate::Image::new(
			image.as_matrix().log()?,
			crate::ImageLayout::Nchw,
			crate::ImageFormat::Rgb,
		)?;
		crate::image::flip(&upstream, true, false)?
			.as_matrix()
			.copy()
	};
	let (plan, fused_output) = runtime.capture(|| prologue(&positive))?;
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("prologue")).unwrap();
	assert_eq!(
		report["stats"]["dispatches"], 1,
		"scalar producer + elementwise consumer must emit one region: {report}"
	);
	runtime.submit(&plan)?.wait()?;
	let fused = fused_output.read_f32()?;
	let unfused = {
		let _off = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("0"));
		prologue(&positive)?.read_f32()?
	};
	assert_eq!(
		fused, unfused,
		"fused log+flip region diverges from the incumbent chain"
	);

	// An interpolating gather must NOT fold a pointwise producer — the
	// op does not commute through bilinear taps, so `log → resize`
	// keeps both dispatches on their canonical routes.
	let gather = |image: &crate::Image| -> crate::Result<crate::Matrix> {
		let upstream = crate::Image::new(
			image.as_matrix().log()?,
			crate::ImageLayout::Nchw,
			crate::ImageFormat::Rgb,
		)?;
		crate::image::resize(&upstream, 14, 10, crate::image::InterpolationMode::Bilinear)?
			.as_matrix()
			.copy()
	};
	let (plan, gather_output) = runtime.capture(|| gather(&positive))?;
	let report: serde_json::Value =
		serde_json::from_str(&plan.debug_report_json("gather prologue")).unwrap();
	assert_eq!(
		report["stats"]["dispatches"], 2,
		"a bilinear gather must not fold a pointwise producer: {report}"
	);
	runtime.submit(&plan)?.wait()?;
	let unfused = {
		let _off = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new("0"));
		gather(&positive)?.read_f32()?
	};
	assert_eq!(
		gather_output.read_f32()?,
		unfused,
		"the unfused gather route diverges from the incumbent chain"
	);
	Ok(())
}

#[test]
#[ignore = "explicit Vulkan device + trusted Slang library; tile-vs-incumbent device medians for the CC-40 ledger"]
fn tile_image_audio_families_record_device_medians() -> crate::Result<()> {
	let index = std::env::var("OA_TEST_DEVICE_INDEX")
		.expect("explicit device")
		.parse::<usize>()
		.unwrap();
	let library = std::env::var_os("OA_TEST_SLANG_LIBRARY").expect("explicit compiler library");
	let _lib = ScopedEnv::set("OA_SLANG_LIBRARY", &library);
	let runtime = crate::Runtime::builder()
		.devices(crate::DeviceSelection::Index(index))
		.build()?;
	crate::Runtime::set_default(runtime.clone());

	let shape = [2_usize, 3, 64, 96];
	let values = (0..shape.iter().product::<usize>())
		.map(|index| ((index * 17 % 61) as f32 - 30.0) * 0.03125)
		.collect::<Vec<_>>();
	let image = crate::Image::new(
		crate::Matrix::new_from_f32(shape, &values)?,
		crate::ImageLayout::Nchw,
		crate::ImageFormat::Rgb,
	)?;
	let kernel = crate::Matrix::new_from_f32(
		[5usize, 5],
		&(0..25)
			.map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.0625)
			.collect::<Vec<_>>(),
	)?;
	let parameters = crate::image::NormalizationConfig {
		mean: [0.125, -0.375, 0.5],
		std: [0.5, 1.0, 2.0],
	};
	let samples: Vec<f32> = (0..16_384)
		.map(|i| {
			(std::f32::consts::TAU * i as f32 / 37.0).sin()
				+ 0.3 * (std::f32::consts::TAU * i as f32 / 11.0).cos()
		})
		.collect();
	let audio = crate::Audio::new_from_planar_f32(
		&runtime,
		&samples,
		2,
		16_000,
		crate::AudioChannelLayout::Stereo,
	)?;
	let mel = crate::audio::MelConfig {
		fft_size: 256,
		hop_size: 64,
		num_mels: 32,
		log_scale: true,
		normalize: true,
		..crate::audio::MelConfig::default()
	};

	let (vw, vh) = (1920_usize, 1080_usize);
	let (cw, ch) = (vw / 2, vh / 2);
	let fill = |w: usize, h: usize, salt: u8| {
		(0..h)
			.flat_map(|y| (0..w).map(move |x| ((x * 5 + y * 7 + salt as usize) % 251) as u8))
			.collect::<Vec<u8>>()
	};
	let uv: Vec<u8> = (0..ch)
		.flat_map(|y| (0..cw * 2).map(move |x| ((x * 3 + y * 11 + 40) % 251) as u8))
		.collect();
	let timing = crate::video::VideoFrameTiming::new(std::time::Duration::from_millis(33), None)?;
	let nv12 = crate::VideoFrame::upload_planes(
		crate::video::VideoPixelFormat::Nv12,
		vec![fill(vw, vh, 3), uv],
		vw,
		vh,
		timing,
		crate::video::VideoColorInfo::new(
			crate::video::VideoColorMatrix::Bt709,
			crate::video::VideoColorRange::Limited,
		),
	)?;
	let matrix_config = crate::video::VideoMatrixConfig::default();

	// OA-CC-35 multimodal workload: demuxed video → to_matrix → head,
	// audio → mel → head, softmax join — the full captured plan.
	let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
		.join("test/fixtures/media/embedded_av.mp4");
	let mut demuxer = crate::video::VideoDemuxer::open(&fixture)?;
	let mut packets = Vec::new();
	while let Some(packet) = demuxer.read_next_packet()? {
		packets.push(packet);
	}
	let decoder = crate::video::VideoDecoder::new(&runtime, demuxer.info())?;
	let mm_audio = crate::audio::decode_file(&fixture)?;
	let mm_config = crate::video::VideoMatrixConfig {
		size: Some((16, 16)),
		..crate::video::VideoMatrixConfig::default()
	};
	let mm_mel = crate::audio::MelConfig {
		fft_size: 512,
		hop_size: 128,
		num_mels: 16,
		..crate::audio::MelConfig::default()
	};
	let mel_probe = crate::audio::mel_spectrogram(&mm_audio, mm_mel)?;
	let audio_features = mel_probe.shape().iter().product::<usize>();
	let _ = mel_probe.read_f32()?;
	let video_features = 3 * 16 * 16;
	let class_count = 2_usize;
	let make_weight = |features: usize, salt: u32| -> crate::Result<crate::Matrix> {
		let values = (0..class_count * features)
			.map(|index| {
				let value = (index as u32).wrapping_mul(1103515245).wrapping_add(salt);
				(value % 251) as f32 / 1000.0 - 0.125
			})
			.collect::<Vec<_>>();
		crate::Matrix::new_from_f32([class_count, features], &values)
	};
	let video_weight = make_weight(video_features, 7)?;
	let audio_weight = make_weight(audio_features, 13)?;

	type Row<'a> = (&'static str, Box<dyn Fn() -> crate::Result<()> + 'a>);
	let mut rows: Vec<Row<'_>> = Vec::new();
	rows.push((
		"multimodal_plan_decode_to_softmax",
		Box::new(|| {
			let frames = decoder.record_to_planes(&packets)?;
			let video_score = crate::video::to_matrix(&frames, &mm_config)?
				.reshape(vec![frames.len(), video_features])?
				.matmul(&video_weight)?
				.mean(0)?
				.reshape(vec![1, class_count])?;
			let audio_score = crate::audio::mel_spectrogram(&mm_audio, mm_mel)?
				.reshape(vec![1, audio_features])?
				.matmul(&audio_weight)?;
			video_score.add(&audio_score)?.softmax(-1)?;
			Ok(())
		}),
	));
	rows.push((
		"to_matrix_nv12_1920x1080",
		Box::new(|| {
			crate::video::to_matrix(std::slice::from_ref(&nv12), &matrix_config)?;
			Ok(())
		}),
	));
	rows.push((
		"resize_bilinear_192x128",
		Box::new(|| {
			crate::image::resize(&image, 192, 128, crate::image::InterpolationMode::Bilinear)?;
			Ok(())
		}),
	));
	rows.push((
		"convolve_5x5_replicate",
		Box::new(|| {
			crate::image::convolve_2d(&image, &kernel, crate::image::BorderMode::Replicate, 0.0)?;
			Ok(())
		}),
	));
	rows.push((
		"gaussian_5_mirror",
		Box::new(|| {
			crate::image::gaussian_blur(&image, 1.2, 5)?;
			Ok(())
		}),
	));
	rows.push((
		"resize_normalize_64x64",
		Box::new(|| {
			crate::image::resize_normalize(&image, 64, 64, parameters)?;
			Ok(())
		}),
	));
	rows.push((
		"grayscale",
		Box::new(|| {
			crate::image::grayscale(&image)?;
			Ok(())
		}),
	));
	rows.push((
		"mel_256_32_norm",
		Box::new(|| {
			crate::audio::mel_spectrogram(&audio, mel)?;
			Ok(())
		}),
	));

	for (name, run) in rows {
		let mut medians = [f64::NAN; 2];
		for (slot, enabled) in [(0_usize, "1"), (1, "0")] {
			let gate = ScopedEnv::set("OA_TILE_IMAGE", std::ffi::OsStr::new(enabled));
			let audio_gate = ScopedEnv::set("OA_TILE_AUDIO", std::ffi::OsStr::new(enabled));
			let video_gate = ScopedEnv::set("OA_TILE_VIDEO", std::ffi::OsStr::new(enabled));
			let (plan, ()) = runtime.capture(&*run)?;
			for _ in 0..3 {
				runtime.submit(&plan)?.wait()?;
			}
			let mut samples = Vec::new();
			for _ in 0..11 {
				samples.push(
					runtime
						.submit_timed(&plan)?
						.device_duration()?
						.as_secs_f64()
						* 1000.0,
				);
			}
			samples.sort_by(f64::total_cmp);
			medians[slot] = samples[5];
			drop(video_gate);
			drop(audio_gate);
			drop(gate);
		}
		eprintln!(
			"TILE_FAMILY_BENCH {name} incumbent_ms={:.6} tile_ms={:.6} delta_pct={:+.2}",
			medians[1],
			medians[0],
			(medians[0] - medians[1]) / medians[1] * 100.0
		);
	}
	Ok(())
}
