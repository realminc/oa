//! Matched OA C++/OARS host-memory benchmark.

use std::{
	alloc::{Layout, alloc_zeroed, dealloc},
	env,
	hint::black_box,
	ptr::NonNull,
	sync::atomic::{Ordering, compiler_fence},
	time::{Duration, Instant},
};

use anyhow::{Context, bail, ensure};
use oa::core::memory;

const KIB: usize = 1024;
const MIB: usize = 1024 * KIB;
const WARMUP_SAMPLES: usize = 5;
const MEASURED_SAMPLES: usize = 21;

#[derive(Clone, Copy)]
#[repr(usize)]
enum Implementation {
	OaRuntime,
	RustRuntime,
	OaFixed,
	RustFixed,
	OaStream,
}

impl Implementation {
	const COUNT: usize = 5;

	const fn name(self) -> &'static str {
		match self {
			Self::OaRuntime => "oa_rust",
			Self::RustRuntime => "rust_std",
			Self::OaFixed => "oa_rust_fixed",
			Self::RustFixed => "rust_fixed",
			Self::OaStream => "oa_rust_stream",
		}
	}
}

#[derive(Clone, Copy)]
struct CopyCase {
	size: usize,
	source_offset: usize,
	destination_offset: usize,
}

#[derive(Clone, Copy)]
struct Stats {
	median_ns: f64,
	p10_ns: f64,
	p90_ns: f64,
}

#[derive(Default)]
struct Options {
	quick: bool,
	copy_only: bool,
	streaming_only: bool,
	min_bytes: usize,
	max_bytes: Option<usize>,
}

struct AlignedArena {
	pointer: NonNull<u8>,
	layout: Layout,
}

impl AlignedArena {
	fn new(length: usize) -> anyhow::Result<Self> {
		let layout = Layout::from_size_align(length, 64).context("invalid arena layout")?;
		// SAFETY: `layout` has a non-zero size and valid 64-byte alignment.
		// Initialize before constructing any byte slice, including mutable ones.
		let pointer =
			NonNull::new(unsafe { alloc_zeroed(layout) }).context("arena allocation failed")?;
		Ok(Self { pointer, layout })
	}

	fn as_slice(&self) -> &[u8] {
		// SAFETY: the allocation remains live for `self`, and every byte is
		// initialized by the benchmark before this view is read.
		unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.layout.size()) }
	}

	fn as_mut_slice(&mut self) -> &mut [u8] {
		// SAFETY: `&mut self` proves exclusive access to the live allocation.
		unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
	}
}

impl Drop for AlignedArena {
	fn drop(&mut self) {
		// SAFETY: `pointer` was returned by `alloc_zeroed` for this exact `layout` and
		// has not been deallocated.
		unsafe { dealloc(self.pointer.as_ptr(), self.layout) }
	}
}

fn main() -> anyhow::Result<()> {
	let options = parse_options()?;
	pin_to_first_allowed_cpu();
	warm_cpu();
	if options.streaming_only {
		run_streaming(&options)
	} else {
		run_copy(&options)
	}
}

fn run_copy(options: &Options) -> anyhow::Result<()> {
	println!(
		"operation,size_bytes,src_offset,dst_offset,implementation,iterations,median_ns,p10_ns,p90_ns,GB_per_s"
	);
	for case in copy_cases(options.quick)
		.into_iter()
		.filter(|case| options.includes(case.size))
	{
		let allocation = case
			.size
			.checked_add(128)
			.context("copy allocation overflow")?;
		let mut source_arena = AlignedArena::new(allocation)?;
		let mut destination_arena = AlignedArena::new(allocation)?;
		for (index, byte) in source_arena.as_mut_slice().iter_mut().enumerate() {
			*byte = index.wrapping_mul(131).wrapping_add(17) as u8;
		}
		destination_arena.as_mut_slice().fill(0);
		let source = &source_arena.as_slice()[case.source_offset..case.source_offset + case.size];
		let destination = &mut destination_arena.as_mut_slice()
			[case.destination_offset..case.destination_offset + case.size];
		let mut implementations = vec![Implementation::OaRuntime, Implementation::RustRuntime];
		if supports_fixed(case.size) {
			implementations.push(Implementation::OaFixed);
			implementations.push(Implementation::RustFixed);
		}
		implementations.push(Implementation::OaStream);
		for &implementation in &implementations {
			verify(implementation, destination, source)?;
		}
		let iterations = iterations_for(case.size, options.quick);
		let measurements = measure(&implementations, destination, source, iterations)?;
		for &implementation in &implementations {
			let stats = measurements[implementation as usize];
			println!(
				"copy,{},{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
				case.size,
				case.source_offset,
				case.destination_offset,
				implementation.name(),
				iterations,
				stats.median_ns,
				stats.p10_ns,
				stats.p90_ns,
				case.size as f64 / stats.median_ns,
			);
		}
	}
	Ok(())
}

fn run_streaming(options: &Options) -> anyhow::Result<()> {
	let quick = options.quick;
	const FULL_CHUNKS: [usize; 15] = [
		256,
		512,
		KIB,
		2 * KIB,
		4 * KIB,
		8 * KIB,
		16 * KIB,
		64 * KIB,
		256 * KIB,
		MIB,
		2 * MIB,
		4 * MIB,
		8 * MIB,
		16 * MIB,
		64 * MIB,
	];
	const QUICK_CHUNKS: [usize; 13] = [
		256,
		512,
		KIB,
		2 * KIB,
		4 * KIB,
		8 * KIB,
		16 * KIB,
		64 * KIB,
		256 * KIB,
		MIB,
		4 * MIB,
		8 * MIB,
		16 * MIB,
	];
	const IMPLEMENTATIONS: [Implementation; 3] = [
		Implementation::OaRuntime,
		Implementation::RustRuntime,
		Implementation::OaStream,
	];
	let chunks = if quick {
		&QUICK_CHUNKS[..]
	} else {
		&FULL_CHUNKS[..]
	};
	let working_set = if quick { 64 * MIB } else { 256 * MIB };
	let mut source_arena = AlignedArena::new(working_set)?;
	let mut destination_arena = AlignedArena::new(working_set)?;
	source_arena.as_mut_slice().fill(0x5a);
	destination_arena.as_mut_slice().fill(0);
	let source = source_arena.as_slice();
	let destination = destination_arena.as_mut_slice();

	println!(
		"operation,chunk_bytes,working_set_bytes,implementation,passes,median_ns,p10_ns,p90_ns,GB_per_s"
	);
	for &chunk_bytes in chunks {
		if !options.includes(chunk_bytes) {
			continue;
		}
		ensure!(
			working_set % chunk_bytes == 0,
			"chunk does not divide working set"
		);
		for &implementation in &IMPLEMENTATIONS {
			destination.fill(0xa5);
			run_streaming_passes(implementation, destination, source, chunk_bytes)?;
			ensure!(destination == source, "streaming copy oracle failed");
		}
		let measurements = measure_streaming(&IMPLEMENTATIONS, destination, source, chunk_bytes)?;
		for &implementation in &IMPLEMENTATIONS {
			let stats = measurements[implementation as usize];
			println!(
				"stream_copy,{chunk_bytes},{working_set},{},1,{:.3},{:.3},{:.3},{:.3}",
				implementation.name(),
				stats.median_ns,
				stats.p10_ns,
				stats.p90_ns,
				working_set as f64 / stats.median_ns,
			);
		}
	}
	Ok(())
}

#[inline(never)]
fn run_copies(
	implementation: Implementation,
	destination: &mut [u8],
	source: &[u8],
	iterations: usize,
) -> anyhow::Result<()> {
	if matches!(
		implementation,
		Implementation::OaFixed | Implementation::RustFixed
	) {
		return run_fixed_dispatch(implementation, destination, source, iterations);
	}
	match implementation {
		Implementation::OaRuntime => copy_loop::<0>(destination, source, iterations),
		Implementation::RustRuntime => copy_loop::<1>(destination, source, iterations),
		Implementation::OaStream => copy_loop::<2>(destination, source, iterations),
		Implementation::OaFixed | Implementation::RustFixed => unreachable!(),
	}
}

// Monomorphize a direct call for each implementation, selecting it outside the
// timed inner loop. Keep lengths dynamic and identical for both public APIs.
#[inline(never)]
fn copy_loop<const POLICY: usize>(
	destination: &mut [u8],
	source: &[u8],
	iterations: usize,
) -> anyhow::Result<()> {
	for _ in 0..iterations {
		let destination = black_box(&mut *destination);
		let source = black_box(source);
		match POLICY {
			0 => memory::copy(destination, source)?,
			1 => destination.copy_from_slice(source),
			2 => memory::copy_streaming(destination, source)?,
			_ => unreachable!(),
		}
		compiler_barrier();
	}
	Ok(())
}

fn run_fixed_dispatch(
	implementation: Implementation,
	destination: &mut [u8],
	source: &[u8],
	iterations: usize,
) -> anyhow::Result<()> {
	match destination.len() {
		8 => run_fixed::<8>(implementation, destination, source, iterations),
		16 => run_fixed::<16>(implementation, destination, source, iterations),
		32 => run_fixed::<32>(implementation, destination, source, iterations),
		64 => run_fixed::<64>(implementation, destination, source, iterations),
		128 => run_fixed::<128>(implementation, destination, source, iterations),
		256 => run_fixed::<256>(implementation, destination, source, iterations),
		_ => bail!("fixed copy requires a supported size"),
	}
}

#[inline(never)]
fn run_fixed<const N: usize>(
	implementation: Implementation,
	destination: &mut [u8],
	source: &[u8],
	iterations: usize,
) -> anyhow::Result<()> {
	let destination: &mut [u8; N] = destination.try_into().expect("fixed destination length");
	let source: &[u8; N] = source.try_into().expect("fixed source length");
	for _ in 0..iterations {
		let destination = black_box(&mut *destination);
		let source = black_box(source);
		match implementation {
			Implementation::OaFixed => memory::copy(destination, source)?,
			Implementation::RustFixed => destination.copy_from_slice(source),
			_ => unreachable!(),
		}
		compiler_barrier();
	}
	Ok(())
}

#[inline(never)]
fn run_streaming_passes(
	implementation: Implementation,
	destination: &mut [u8],
	source: &[u8],
	chunk_bytes: usize,
) -> anyhow::Result<()> {
	match implementation {
		Implementation::OaRuntime => streaming_loop::<0>(destination, source, chunk_bytes),
		Implementation::RustRuntime => streaming_loop::<1>(destination, source, chunk_bytes),
		Implementation::OaStream => streaming_loop::<2>(destination, source, chunk_bytes),
		Implementation::OaFixed | Implementation::RustFixed => unreachable!(),
	}
}

#[inline(never)]
fn streaming_loop<const POLICY: usize>(
	destination: &mut [u8],
	source: &[u8],
	chunk_bytes: usize,
) -> anyhow::Result<()> {
	for (destination_chunk, source_chunk) in destination
		.chunks_exact_mut(chunk_bytes)
		.zip(source.chunks_exact(chunk_bytes))
	{
		let destination_chunk = black_box(destination_chunk);
		let source_chunk = black_box(source_chunk);
		match POLICY {
			0 => memory::copy(destination_chunk, source_chunk)?,
			1 => destination_chunk.copy_from_slice(source_chunk),
			2 => memory::copy_streaming(destination_chunk, source_chunk)?,
			_ => unreachable!(),
		}
		compiler_barrier();
	}
	Ok(())
}

fn verify(
	implementation: Implementation,
	destination: &mut [u8],
	source: &[u8],
) -> anyhow::Result<()> {
	destination.fill(0xa5);
	run_copies(implementation, destination, source, 1)?;
	ensure!(
		destination == source,
		"copy oracle failed for {}",
		implementation.name()
	);
	Ok(())
}

fn measure(
	implementations: &[Implementation],
	destination: &mut [u8],
	source: &[u8],
	iterations: usize,
) -> anyhow::Result<[Stats; Implementation::COUNT]> {
	for &implementation in implementations {
		for _ in 0..WARMUP_SAMPLES {
			run_copies(implementation, destination, source, iterations)?;
		}
	}
	let mut samples: [Vec<f64>; Implementation::COUNT] =
		std::array::from_fn(|_| Vec::with_capacity(MEASURED_SAMPLES));
	for sample in 0..MEASURED_SAMPLES {
		for order in 0..implementations.len() {
			let implementation = implementations[(order + sample) % implementations.len()];
			let started = Instant::now();
			run_copies(implementation, destination, source, iterations)?;
			let total_ns = started.elapsed().as_secs_f64() * 1.0e9;
			samples[implementation as usize].push(total_ns / iterations as f64);
		}
	}
	Ok(std::array::from_fn(|index| summarize(&samples[index])))
}

fn measure_streaming(
	implementations: &[Implementation; 3],
	destination: &mut [u8],
	source: &[u8],
	chunk_bytes: usize,
) -> anyhow::Result<[Stats; Implementation::COUNT]> {
	for &implementation in implementations {
		for _ in 0..WARMUP_SAMPLES {
			run_streaming_passes(implementation, destination, source, chunk_bytes)?;
		}
	}
	let mut samples: [Vec<f64>; Implementation::COUNT] =
		std::array::from_fn(|_| Vec::with_capacity(MEASURED_SAMPLES));
	for sample in 0..MEASURED_SAMPLES {
		for order in 0..implementations.len() {
			let implementation = implementations[(order + sample) % implementations.len()];
			let started = Instant::now();
			run_streaming_passes(implementation, destination, source, chunk_bytes)?;
			samples[implementation as usize].push(started.elapsed().as_secs_f64() * 1.0e9);
		}
	}
	Ok(std::array::from_fn(|index| summarize(&samples[index])))
}

fn summarize(samples: &[f64]) -> Stats {
	if samples.is_empty() {
		return Stats {
			median_ns: 0.0,
			p10_ns: 0.0,
			p90_ns: 0.0,
		};
	}
	let mut ordered = samples.to_vec();
	ordered.sort_by(f64::total_cmp);
	Stats {
		median_ns: percentile(&ordered, 0.5),
		p10_ns: percentile(&ordered, 0.1),
		p90_ns: percentile(&ordered, 0.9),
	}
}

fn percentile(ordered: &[f64], fraction: f64) -> f64 {
	let position = fraction * (ordered.len() - 1) as f64;
	let low = position.floor() as usize;
	let high = position.ceil() as usize;
	let weight = position - low as f64;
	ordered[low] * (1.0 - weight) + ordered[high] * weight
}

fn supports_fixed(size: usize) -> bool {
	matches!(size, 8 | 16 | 32 | 64 | 128 | 256)
}

fn iterations_for(size: usize, quick: bool) -> usize {
	if size <= 256 {
		return if quick { 200_000 } else { 1_000_000 };
	}
	let target_bytes = if quick { 64 * MIB } else { 256 * MIB };
	(target_bytes / size).clamp(4, if quick { 100_000 } else { 500_000 })
}

fn copy_cases(quick: bool) -> Vec<CopyCase> {
	const FULL: [usize; 34] = [
		1,
		2,
		3,
		4,
		7,
		8,
		15,
		16,
		17,
		24,
		31,
		32,
		33,
		48,
		63,
		64,
		65,
		96,
		127,
		128,
		129,
		192,
		255,
		256,
		257,
		512,
		KIB,
		4 * KIB,
		64 * KIB,
		MIB,
		2 * MIB,
		4 * MIB,
		16 * MIB,
		64 * MIB,
	];
	const QUICK: [usize; 14] = [
		8,
		16,
		32,
		64,
		128,
		256,
		512,
		4 * KIB,
		64 * KIB,
		MIB,
		2 * MIB,
		4 * MIB,
		16 * MIB,
		64 * MIB,
	];
	let sizes = if quick { &QUICK[..] } else { &FULL[..] };
	sizes
		.iter()
		.flat_map(|&size| {
			[
				CopyCase {
					size,
					source_offset: 0,
					destination_offset: 0,
				},
				CopyCase {
					size,
					source_offset: 1,
					destination_offset: 3,
				},
			]
		})
		.collect()
}

fn compiler_barrier() {
	compiler_fence(Ordering::SeqCst);
	black_box(());
}

fn warm_cpu() {
	let until = Instant::now() + Duration::from_millis(300);
	let mut value = 1_u64;
	while Instant::now() < until {
		value = value
			.wrapping_mul(6_364_136_223_846_793_005)
			.wrapping_add(1_442_695_040_888_963_407);
		black_box(value);
	}
}

#[cfg(target_os = "linux")]
fn pin_to_first_allowed_cpu() {
	// SAFETY: libc initializes and accesses `cpu_set_t` through its matching
	// macros, and the set remains live for both scheduler calls.
	unsafe {
		let mut allowed: libc::cpu_set_t = std::mem::zeroed();
		if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed) != 0 {
			return;
		}
		for cpu in 0..libc::CPU_SETSIZE as usize {
			if !libc::CPU_ISSET(cpu, &allowed) {
				continue;
			}
			let mut selected: libc::cpu_set_t = std::mem::zeroed();
			libc::CPU_ZERO(&mut selected);
			libc::CPU_SET(cpu, &mut selected);
			let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &selected);
			return;
		}
	}
}

#[cfg(not(target_os = "linux"))]
fn pin_to_first_allowed_cpu() {}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options::default();
	let mut arguments = env::args().skip(1);
	while let Some(argument) = arguments.next() {
		match argument.as_str() {
			"--quick" => options.quick = true,
			"--copy" => options.copy_only = true,
			"--streaming" => options.streaming_only = true,
			"--min-bytes" => {
				options.min_bytes = arguments.next().context("missing --min-bytes")?.parse()?
			}
			"--max-bytes" => {
				options.max_bytes = Some(arguments.next().context("missing --max-bytes")?.parse()?)
			}
			"--help" | "-h" => {
				println!(
					"usage: core_memory_bench [--quick] [--copy | --streaming] [--min-bytes N] [--max-bytes N]"
				);
				std::process::exit(0);
			}
			_ => bail!("unknown argument: {argument}"),
		}
	}
	ensure!(
		!(options.copy_only && options.streaming_only),
		"--copy and --streaming are mutually exclusive"
	);
	ensure!(
		options
			.max_bytes
			.is_none_or(|maximum| maximum >= options.min_bytes),
		"empty size range"
	);
	Ok(options)
}

impl Options {
	fn includes(&self, bytes: usize) -> bool {
		bytes >= self.min_bytes && self.max_bytes.is_none_or(|maximum| bytes <= maximum)
	}
}
