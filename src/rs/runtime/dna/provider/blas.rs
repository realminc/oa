//! Private matmul problem description, candidate registry, and planning.
//!
//! Port provenance: `oacpp/source/cpp/lib/oa/runtime/matmulTypes.h`,
//! `oacpp/source/cpp/lib/oa/runtime/gemmTypes.h`, and
//! `oacpp/source/cpp/lib/oa/runtime/gemm/{registry,router}.cpp`. Candidate
//! selection is a private lowering detail: semantic matmul operations reach
//! candidates only through the immutable [`MatmulPlan`] — never by kernel
//! identity or tile policy.
//!
//! Deliberate differences from the donor, all ownership-preserving:
//!
//! - Kernel references are typed [`KernelId`] values instead of dispatch
//!   strings; exact identity is checked, never name-prefix heuristics.
//! - Tile geometry, workgroup invocations, and shared-memory requirements are
//!   reflected from the embedded SPIR-V artifact rather than restated in the
//!   registry row, keeping one manifest per compiled contract.
//! - Route-cache persistence is ported (versioned fail-closed format under
//!   `var/gemm_route_cache.bin`); the tuning driver that produces new
//!   measurements is test-gated, so production caches are read-only.
//! - `OA_DISABLE_GEMM_ROUTE_CACHE` is honored; `OA_GEMM_FORCE_FP32`,
//!   `OA_FORCE_COOPVEC`, and `OA_GEMM_AUTOTUNE` are not ported.

use std::collections::HashMap;
use std::fmt;

use crate::{Error, Result};

use crate::runtime::device::NativeFlagBits;
use crate::runtime::{
	BufferAccess, BufferBinding, Device, EngineHandle, KernelId, PushConstant, Storage,
};

// ─────────────────────────────────────────────────────────────────────────────
// Capability bits — populated from the admitted device, AND-checked per variant
// ─────────────────────────────────────────────────────────────────────────────
//
// Bit ordering matches the donor exactly; these positions are serialized
// route/cache identities. New bits append to the end.

/// `VK_KHR_cooperative_matrix` usable in compute shaders.
const CAP_COOP_MAT1_KHR: u64 = 1 << 0;
/// Cooperative-matrix tuple accepting BF16 inputs.
const CAP_COOP_MAT1_BF16_INPUT: u64 = 1 << 1;
/// Cooperative-matrix tuple accumulating in BF16.
const CAP_COOP_MAT1_BF16_ACC: u64 = 1 << 2;
/// Cooperative-matrix tuple accepting F16 inputs.
const CAP_COOP_MAT1_F16_INPUT: u64 = 1 << 3;
/// Cooperative-matrix tuple accumulating in F32.
const CAP_COOP_MAT1_F32_ACC: u64 = 1 << 4;
/// NVIDIA cooperative-vector extension. OA does not query it yet; the bit
/// exists so future candidates share the donor's serialized identity space.
const CAP_COOP_VEC: u64 = 1 << 11;
/// Portable shared-memory tiled F32 family. Always available on an admitted
/// compute device.
const CAP_TILED_F32: u64 = 1 << 12;
/// Scalar F32 fallback family. Always available on an admitted compute
/// device; registered when its candidate shader lands.
const CAP_NAIVE_F32: u64 = 1 << 13;
/// Workgroup-scope cooperative-matrix tuple accepting BF16 inputs.
const CAP_COOP_MAT1_WORKGROUP_BF16: u64 = 1 << 14;
/// Small-M workgroup-reduction F32 family. Always available on an admitted
/// compute device.
const CAP_SMALLM_F32: u64 = 1 << 15;
// Bits 5–10, 14 (donor numbering) are retired; 16..63 are reserved.

const fn caps_satisfy(available: u64, required: u64) -> bool {
	(available & required) == required
}

// ─────────────────────────────────────────────────────────────────────────────
// Contract vocabulary — serialized identities match the donor values
// ─────────────────────────────────────────────────────────────────────────────

/// Shared mixed-precision/matmul storage vocabulary.
///
/// `Bf16` is the only 16-bit tensor-core storage precision; value 1 stays
/// retired so `Bf16` preserves its serialized route identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub(in crate::runtime) enum StoragePrecision {
	F32 = 0,
	Bf16 = 2,
}

/// Coarse kernel family used by routing policy. Explicit values are the
/// donor's serialized identities; retired values remain gaps.
///
/// The full family vocabulary is declared so serialized plan/cache identities
/// stay stable; families are constructed as their candidates register.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(in crate::runtime) enum GemmKernel {
	Auto = 0,
	TiledF32 = 4,
	Naive = 5,
	CoopVec = 6,
	GemmCmSgBf16 = 11,
	GemmCmWgBf16 = 12,
	StridedF32 = 13,
	SmallMF32 = 14,
	StridedTiledF32 = 15,
	SplitKF32 = 16,
}

/// Dispatch path shape.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(in crate::runtime) enum GemmPath {
	Standard = 0,
	CoopVec = 1,
}

/// Requested/actual compute precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub(in crate::runtime) enum GemmPrecision {
	Auto = 0,
	F32 = 1,
	Bf16 = 2,
}

/// Logical post-GEMM contract. Explicit in routing so a fused epilogue never
/// replays a shader with a different buffer contract.
///
/// The full epilogue vocabulary is declared for stable serialized identity;
/// rows land with their fused candidates.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub(in crate::runtime) enum GemmEpilogue {
	None = 0,
	Bias,
	BiasRelu,
	BiasGelu,
	BiasSilu,
	SiluDual,
}

/// How the selected winner was classified relative to the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(in crate::runtime) enum KernelSelectionKind {
	Unspecified,
	Direct,
	PrecisionFallback,
	LayoutFallback,
	NaiveFallback,
}

// ─────────────────────────────────────────────────────────────────────────────
// Problem description — caller-built, trivially copyable
// ─────────────────────────────────────────────────────────────────────────────

/// Address contract for one logical operand.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(in crate::runtime) struct MatmulLayout {
	pub offset: u32,
	pub row_stride: u32,
	pub col_stride: u32,
	pub batch_stride: u32,
}

/// Full problem description handed to the planner.
///
/// `b_transposed = true` means B is stored `[N, K]` (the OA weight convention)
/// and the chosen kernel must read it transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::runtime) struct MatmulProblem {
	pub m: u32,
	pub n: u32,
	pub k: u32,
	pub batch_count: u32,
	pub a: MatmulLayout,
	pub b: MatmulLayout,
	pub c: MatmulLayout,
	pub a_master: StoragePrecision,
	pub b_master: StoragePrecision,
	pub requested_output: StoragePrecision,
	pub a_contiguous: bool,
	pub b_contiguous: bool,
	pub b_transposed: bool,
	pub epilogue: GemmEpilogue,
	pub requires_pre_activation: bool,
	pub training: bool,
	pub precision_hint: GemmPrecision,
}

impl MatmulProblem {
	/// Canonical contiguous `[M, K] × [N, K]` problem. Matches the donor's
	/// `problemForRaw` defaults: F32 storage, transposed B, training enabled.
	pub(in crate::runtime) fn contiguous(m: u32, n: u32, k: u32) -> Self {
		let batch_stride =
			|rows: u32, cols: u32| u32::try_from(u64::from(rows) * u64::from(cols)).unwrap_or(0);
		Self {
			m,
			n,
			k,
			batch_count: 1,
			a: MatmulLayout {
				row_stride: k,
				col_stride: 1,
				batch_stride: batch_stride(m, k),
				..MatmulLayout::default()
			},
			b: MatmulLayout {
				row_stride: k,
				col_stride: 1,
				batch_stride: batch_stride(n, k),
				..MatmulLayout::default()
			},
			c: MatmulLayout {
				row_stride: n,
				col_stride: 1,
				batch_stride: batch_stride(m, n),
				..MatmulLayout::default()
			},
			a_master: StoragePrecision::F32,
			b_master: StoragePrecision::F32,
			requested_output: StoragePrecision::F32,
			a_contiguous: true,
			b_contiguous: true,
			b_transposed: true,
			epilogue: GemmEpilogue::None,
			requires_pre_activation: false,
			training: true,
			precision_hint: GemmPrecision::Auto,
		}
	}

	/// True when every operand uses the canonical contiguous layout the
	/// shared-memory kernels require.
	fn is_canonical(&self) -> bool {
		let expected_batch = |layout: MatmulLayout, elements: u64| {
			self.batch_count == 1 || u64::from(layout.batch_stride) == elements
		};
		let canonical_a = self.a.offset == 0
			&& self.a.row_stride == self.k
			&& self.a.col_stride == 1
			&& expected_batch(self.a, u64::from(self.m) * u64::from(self.k));
		let canonical_b = self.b.offset == 0
			&& self.b.row_stride == self.k
			&& self.b.col_stride == 1
			&& expected_batch(self.b, u64::from(self.n) * u64::from(self.k));
		let canonical_c = self.c.offset == 0
			&& self.c.row_stride == self.n
			&& self.c.col_stride == 1
			&& expected_batch(self.c, u64::from(self.m) * u64::from(self.n));
		self.a_contiguous && self.b_contiguous && canonical_a && canonical_b && canonical_c
	}
}

/// Selection policy for one plan request.
///
/// Smaller than the mathematical problem on purpose: a preference may change
/// the chosen implementation, never the result contract.
#[derive(Clone, Copy, Debug)]
pub(in crate::runtime) struct MatmulPreference {
	/// Private diagnostic/tuning constraint. When set, planning succeeds only
	/// when this exact variant is legal for the complete problem and device
	/// contract. Never process-global router state.
	pub required_variant: u64,
	/// Explicit workspace budget; candidates exceeding it are illegal.
	pub max_workspace_bytes: u64,
	/// Replay the engine's measured winner for this exact problem when one is
	/// recorded. Structural selection still decides unmeasured problems, and a
	/// stale winner is revalidated before every replay.
	pub use_measured_cache: bool,
}

impl Default for MatmulPreference {
	fn default() -> Self {
		Self {
			required_variant: INVALID_MATMUL_VARIANT_ID,
			max_workspace_bytes: 0,
			use_measured_cache: true,
		}
	}
}

/// Exact measured-route key: device identity, registry build, and the
/// complete problem record. Field-exact like the donor — a winner recorded
/// for one problem can never replay for a different problem because the full
/// record is compared, not a hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::runtime) struct MatmulRouteKey {
	vendor_id: u32,
	device_id: u32,
	driver_id: u32,
	driver_version: u32,
	registry_build_id: u64,
	problem: MatmulProblem,
}

/// One measured qualification record. Timings are evidence captured by the
/// tuner, never inputs to structural legality.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::runtime) struct MatmulRouteValue {
	/// Stable variant identity of the measured winner.
	pub winner_variant: u64,
	/// Median GPU dispatch time across accepted samples.
	pub median_gpu_time_ms: f32,
	/// 95th-percentile GPU dispatch time across accepted samples.
	pub p95_gpu_time_ms: f32,
	/// Accepted timing samples; a record with zero samples is invalid.
	pub sample_count: u32,
	/// Publication step of the qualification run that wrote this record.
	pub last_updated_step: u64,
}

/// Persisted file identity — `"OAGEMMRC"` matches the donor's magic.
const MATMUL_ROUTE_CACHE_MAGIC: u64 = 0x4F41_4745_4D4D_5243;
/// Versioned field-wise format; bump only when the record layout changes.
const MATMUL_ROUTE_CACHE_VERSION: u32 = 6;
/// Upper bound on accepted records, matching the donor's corrupt-file guard.
const MATMUL_ROUTE_CACHE_MAX_RECORDS: u64 = 1_000_000;

/// Canonical measured-route cache location under the `var/` tree.
pub(in crate::runtime) fn route_cache_path() -> crate::Path {
	crate::Path::var_rel("gemm_route_cache.bin")
}

/// Cached `OA_DISABLE_GEMM_ROUTE_CACHE` opt-out. Mirrors the donor: the env
/// flag is read once into process-local immutable state; it disables replay
/// only, never the structural router.
pub(in crate::runtime) fn measured_route_cache_disabled() -> bool {
	static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*DISABLED.get_or_init(|| crate::EnvFlag::is_set("OA_DISABLE_GEMM_ROUTE_CACHE"))
}

/// Engine-owned measured-winner cache with a monotonic publication step.
///
/// Records are appended by the tuning driver after legality and numerical
/// qualification, persisted field-wise at engine teardown, and loaded
/// best-effort at engine init. `EngineState` is thread-affine, so no locking
/// is required.
#[derive(Debug, Default)]
pub(in crate::runtime) struct MatmulRouteCache {
	records: HashMap<MatmulRouteKey, MatmulRouteValue>,
	publication_step: u64,
}

impl MatmulRouteCache {
	pub(in crate::runtime) fn new() -> Self {
		Self::default()
	}

	/// Measured winner for one exact route key. Mirrors the donor's `query`:
	/// a record written with zero samples is not a winner.
	pub(in crate::runtime) fn query(&self, key: &MatmulRouteKey) -> Option<u64> {
		self
			.records
			.get(key)
			.filter(|value| value.sample_count > 0)
			.map(|value| value.winner_variant)
	}

	/// Versioned field-wise encoding. Never dump Rust structs directly:
	/// padding, bool size, and enum layout are not a persistent contract.
	/// Records are sorted by their serialized key so the output bytes are
	/// deterministic for a given cache content.
	fn serialize(&self) -> Vec<u8> {
		let mut writer = RouteCacheWriter(Vec::new());
		writer.u64(MATMUL_ROUTE_CACHE_MAGIC);
		writer.u32(MATMUL_ROUTE_CACHE_VERSION);
		writer.u64(self.records.len() as u64);

		let mut records: Vec<(&MatmulRouteKey, &MatmulRouteValue)> = self.records.iter().collect();
		records.sort_by_key(|(key, _)| serialized_route_key(key));
		for (key, value) in records {
			writer.bytes(&serialized_route_key(key));
			writer.u64(value.winner_variant);
			writer.f32(value.median_gpu_time_ms);
			writer.f32(value.p95_gpu_time_ms);
			writer.u32(value.sample_count);
			writer.u64(value.last_updated_step);
		}
		writer.0
	}

	/// Load one serialized cache. Fail closed like the donor: any malformed
	/// header, record, enum, timing, dimension, duplicate key, or trailing
	/// byte rejects the whole file — a partially accepted map is never
	/// returned.
	fn deserialize(bytes: &[u8]) -> Result<Self> {
		let corrupt = || Error::invalid_argument("persisted matmul route cache is malformed");
		let mut reader = RouteCacheReader { bytes, offset: 0 };
		let magic = reader.u64().ok_or_else(corrupt)?;
		let version = reader.u32().ok_or_else(corrupt)?;
		if magic != MATMUL_ROUTE_CACHE_MAGIC || version != MATMUL_ROUTE_CACHE_VERSION {
			return Err(corrupt());
		}
		let count = reader.u64().ok_or_else(corrupt)?;
		if count > MATMUL_ROUTE_CACHE_MAX_RECORDS {
			return Err(corrupt());
		}

		let mut cache = Self::new();
		for _ in 0..count {
			let (key, value) = read_route_record(&mut reader).ok_or_else(corrupt)?;
			if cache.records.insert(key, value).is_some() {
				return Err(corrupt());
			}
			cache.publication_step = cache.publication_step.max(value.last_updated_step);
		}
		// Reject trailing bytes as a corrupt or incompatible cache rather than
		// accepting records written under a different contract.
		if reader.offset != bytes.len() {
			return Err(corrupt());
		}
		Ok(cache)
	}

	/// Load a persisted cache from disk. Missing files and corrupt content
	/// both fail; callers decide whether an empty cache is the fallback.
	pub(in crate::runtime) fn load(path: &crate::Path) -> Result<Self> {
		let bytes = crate::Filesystem::read_binary(path)?;
		Self::deserialize(&bytes)
	}

	/// Decode a `.vkcache` section payload through the same strict parser as
	/// `load`; the envelope adds integrity and bounds but never replaces these
	/// per-record checks.
	pub(in crate::runtime) fn from_bytes(bytes: &[u8]) -> Result<Self> {
		Self::deserialize(bytes)
	}

	/// Serialize only records whose key carries this device/driver identity —
	/// one `.vkcache` artifact holds records for its own scope. Returns `None`
	/// when no record matches, so callers remove rather than write an empty
	/// section.
	pub(in crate::runtime) fn serialize_scope(
		&self,
		vendor_id: u32,
		device_id: u32,
		driver_id: u32,
		driver_version: u32,
	) -> Option<Vec<u8>> {
		let filtered: HashMap<MatmulRouteKey, MatmulRouteValue> = self
			.records
			.iter()
			.filter(|(key, _)| {
				key.vendor_id == vendor_id
					&& key.device_id == device_id
					&& key.driver_id == driver_id
					&& key.driver_version == driver_version
			})
			.map(|(key, value)| (*key, *value))
			.collect();
		if filtered.is_empty() {
			return None;
		}
		Some(
			Self {
				records: filtered,
				publication_step: self.publication_step,
			}
			.serialize(),
		)
	}

	/// Merge records from `other` whose keys carry this device/driver identity.
	/// An existing record is replaced only by a strictly higher publication
	/// step, so a stale legacy file can never roll back qualified tuning.
	pub(in crate::runtime) fn merge_scope(
		&mut self,
		other: &MatmulRouteCache,
		vendor_id: u32,
		device_id: u32,
		driver_id: u32,
		driver_version: u32,
	) {
		for (key, value) in &other.records {
			if key.vendor_id != vendor_id
				|| key.device_id != device_id
				|| key.driver_id != driver_id
				|| key.driver_version != driver_version
			{
				continue;
			}
			let replace = self
				.records
				.get(key)
				.is_none_or(|existing| existing.last_updated_step < value.last_updated_step);
			if replace {
				self.records.insert(*key, *value);
				self.publication_step = self.publication_step.max(value.last_updated_step);
			}
		}
	}
}

/// Little-endian sink mirroring `RouteCacheReader`; the field-wise order is
/// the persisted contract, not Rust layout.
struct RouteCacheWriter(Vec<u8>);

impl RouteCacheWriter {
	fn bytes(&mut self, value: &[u8]) {
		self.0.extend_from_slice(value);
	}
	fn u32(&mut self, value: u32) {
		self.0.extend_from_slice(&value.to_le_bytes());
	}
	fn u64(&mut self, value: u64) {
		self.0.extend_from_slice(&value.to_le_bytes());
	}
	fn f32(&mut self, value: f32) {
		self.0.extend_from_slice(&value.to_le_bytes());
	}
}

/// Serialized route-key bytes — one canonical encoding shared by the file
/// writer and the deterministic ordering key.
fn serialized_route_key(key: &MatmulRouteKey) -> Vec<u8> {
	let problem = &key.problem;
	let mut writer = RouteCacheWriter(Vec::with_capacity(106));
	writer.u32(key.vendor_id);
	writer.u64(u64::from(key.device_id));
	writer.u32(key.driver_id);
	writer.u64(u64::from(key.driver_version));
	writer.u64(key.registry_build_id);
	writer.u32(problem.m);
	writer.u32(problem.n);
	writer.u32(problem.k);
	writer.u32(problem.batch_count);
	for layout in [&problem.a, &problem.b, &problem.c] {
		writer.u32(layout.offset);
		writer.u32(layout.row_stride);
		writer.u32(layout.col_stride);
		writer.u32(layout.batch_stride);
	}
	writer.bytes(&[
		problem.a_master as u8,
		problem.b_master as u8,
		problem.requested_output as u8,
		problem.precision_hint as u8,
		problem.epilogue as u8,
		u8::from(problem.a_contiguous),
		u8::from(problem.b_contiguous),
		u8::from(problem.b_transposed),
		u8::from(problem.requires_pre_activation),
		u8::from(problem.training),
	]);
	writer.0
}

/// Decode and strictly validate one record, mirroring the donor's load
/// checks: enum ranges, boolean bounds, nonzero dimensions and row/column
/// strides, finite nonnegative timings, a registered winner identity, and a
/// nonzero sample count. `None` marks the record corrupt.
fn read_route_record(
	reader: &mut impl RouteRecordReader,
) -> Option<(MatmulRouteKey, MatmulRouteValue)> {
	let key = MatmulRouteKey {
		vendor_id: reader.u32()?,
		device_id: u32::try_from(reader.u64()?).ok()?,
		driver_id: reader.u32()?,
		driver_version: u32::try_from(reader.u64()?).ok()?,
		registry_build_id: reader.u64()?,
		problem: MatmulProblem {
			m: reader.u32()?,
			n: reader.u32()?,
			k: reader.u32()?,
			batch_count: reader.u32()?,
			a: MatmulLayout {
				offset: reader.u32()?,
				row_stride: reader.u32()?,
				col_stride: reader.u32()?,
				batch_stride: reader.u32()?,
			},
			b: MatmulLayout {
				offset: reader.u32()?,
				row_stride: reader.u32()?,
				col_stride: reader.u32()?,
				batch_stride: reader.u32()?,
			},
			c: MatmulLayout {
				offset: reader.u32()?,
				row_stride: reader.u32()?,
				col_stride: reader.u32()?,
				batch_stride: reader.u32()?,
			},
			a_master: storage_precision_from_u8(reader.u8()?)?,
			b_master: storage_precision_from_u8(reader.u8()?)?,
			requested_output: storage_precision_from_u8(reader.u8()?)?,
			precision_hint: gemm_precision_from_u8(reader.u8()?)?,
			epilogue: gemm_epilogue_from_u8(reader.u8()?)?,
			a_contiguous: read_bool(reader)?,
			b_contiguous: read_bool(reader)?,
			b_transposed: read_bool(reader)?,
			requires_pre_activation: read_bool(reader)?,
			training: read_bool(reader)?,
		},
	};
	let value = MatmulRouteValue {
		winner_variant: reader.u64()?,
		median_gpu_time_ms: reader.f32()?,
		p95_gpu_time_ms: reader.f32()?,
		sample_count: reader.u32()?,
		last_updated_step: reader.u64()?,
	};
	let problem = &key.problem;
	let invalid_value = value.winner_variant == INVALID_MATMUL_VARIANT_ID
		|| !value.median_gpu_time_ms.is_finite()
		|| !value.p95_gpu_time_ms.is_finite()
		|| value.median_gpu_time_ms < 0.0
		|| value.p95_gpu_time_ms < 0.0
		|| value.sample_count == 0;
	let invalid_problem = problem.m == 0
		|| problem.n == 0
		|| problem.k == 0
		|| problem.batch_count == 0
		|| problem.a.row_stride == 0
		|| problem.a.col_stride == 0
		|| problem.b.row_stride == 0
		|| problem.b.col_stride == 0
		|| problem.c.row_stride == 0
		|| problem.c.col_stride == 0;
	if invalid_value || invalid_problem {
		return None;
	}
	Some((key, value))
}

/// Byte-reading surface the record decoder consumes; isolated so the
/// deserialize loop keeps a single reader implementation.
trait RouteRecordReader {
	fn u8(&mut self) -> Option<u8>;
	fn u32(&mut self) -> Option<u32>;
	fn u64(&mut self) -> Option<u64>;
	fn f32(&mut self) -> Option<f32>;
}

/// Bounds-checked little-endian cursor over the persisted cache bytes.
struct RouteCacheReader<'a> {
	bytes: &'a [u8],
	offset: usize,
}

impl RouteCacheReader<'_> {
	fn take(&mut self, count: usize) -> Option<&'_ [u8]> {
		let end = self.offset.checked_add(count)?;
		let slice = self.bytes.get(self.offset..end)?;
		self.offset = end;
		Some(slice)
	}
}

impl RouteRecordReader for RouteCacheReader<'_> {
	fn u8(&mut self) -> Option<u8> {
		self.take(1).map(|b| b[0])
	}
	fn u32(&mut self) -> Option<u32> {
		self
			.take(4)
			.map(|b| u32::from_le_bytes(b.try_into().unwrap()))
	}
	fn u64(&mut self) -> Option<u64> {
		self
			.take(8)
			.map(|b| u64::from_le_bytes(b.try_into().unwrap()))
	}
	fn f32(&mut self) -> Option<f32> {
		self
			.take(4)
			.map(|b| f32::from_le_bytes(b.try_into().unwrap()))
	}
}

fn read_bool(reader: &mut impl RouteRecordReader) -> Option<bool> {
	match reader.u8()? {
		0 => Some(false),
		1 => Some(true),
		_ => None,
	}
}

fn storage_precision_from_u8(value: u8) -> Option<StoragePrecision> {
	match value {
		0 => Some(StoragePrecision::F32),
		2 => Some(StoragePrecision::Bf16),
		_ => None,
	}
}

fn gemm_precision_from_u8(value: u8) -> Option<GemmPrecision> {
	match value {
		0 => Some(GemmPrecision::Auto),
		1 => Some(GemmPrecision::F32),
		2 => Some(GemmPrecision::Bf16),
		_ => None,
	}
}

fn gemm_epilogue_from_u8(value: u8) -> Option<GemmEpilogue> {
	match value {
		0 => Some(GemmEpilogue::None),
		1 => Some(GemmEpilogue::Bias),
		2 => Some(GemmEpilogue::BiasRelu),
		3 => Some(GemmEpilogue::BiasGelu),
		4 => Some(GemmEpilogue::BiasSilu),
		5 => Some(GemmEpilogue::SiluDual),
		_ => None,
	}
}

/// Route-cache key for one problem on one device contract. Device identity
/// and the registry build ride along so a driver, device, or registry change
/// cannot replay a stale winner.
fn route_key(contract: &MatmulDeviceContract, problem: &MatmulProblem) -> MatmulRouteKey {
	MatmulRouteKey {
		vendor_id: contract.vendor_id,
		device_id: contract.device_id,
		driver_id: contract.driver_id,
		driver_version: contract.driver_version,
		registry_build_id: registry_build_id(),
		problem: *problem,
	}
}

/// Device facts the planner reads, snapshotted once per plan request.
///
/// `caps_mask` records *enabled* capabilities — a feature queried but not
/// admitted by device configuration must not satisfy a candidate.
#[derive(Clone, Copy, Debug)]
pub(in crate::runtime) struct MatmulDeviceContract {
	pub vendor_id: u32,
	pub device_id: u32,
	pub driver_id: u32,
	pub driver_version: u32,
	pub capability_identity: u64,
	pub caps_mask: u64,
	pub max_compute_workgroup_invocations: u32,
	pub max_compute_workgroup_size: [u32; 3],
	pub max_compute_shared_memory_bytes: u32,
}

impl MatmulDeviceContract {
	/// Snapshot the queried limits and enabled feature surface of one
	/// admitted device.
	pub(in crate::runtime) fn from_device(device: &Device) -> Self {
		let physical = device.physical();
		Self {
			vendor_id: physical.info.vendor_id,
			device_id: physical.info.device_id,
			driver_id: u32::try_from(physical.info.driver_id.as_raw()).unwrap_or(0),
			driver_version: physical.info.driver_version,
			capability_identity: physical.info.capability_identity,
			caps_mask: compute_caps_mask(device),
			max_compute_workgroup_invocations: physical.limits.max_compute_work_group_invocations,
			max_compute_workgroup_size: physical.limits.max_compute_work_group_size,
			max_compute_shared_memory_bytes: physical.limits.max_compute_shared_memory_size,
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Variant registry — one row per compiled matmul contract
// ─────────────────────────────────────────────────────────────────────────────

/// Stable identity for one registered variant. Derived from the canonical
/// variant name so generated and handwritten rows share the identity space.
pub(in crate::runtime) const INVALID_MATMUL_VARIANT_ID: u64 = 0;

const fn matmul_variant_id_from_name(name: &str) -> u64 {
	let mut hash = 0xcbf2_9ce4_8422_2325_u64;
	let bytes = name.as_bytes();
	let mut index = 0;
	while index < bytes.len() {
		hash ^= bytes[index] as u64;
		hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
		index += 1;
	}
	hash
}

/// One registry row: kernel identity, required capabilities, tile contract,
/// and legality metadata. Geometry facts resolve from the embedded artifact;
/// the row declares only what the artifact does not carry.
#[derive(Debug)]
pub(in crate::runtime) struct MatmulVariant {
	/// Stable generated variant identity.
	pub id: u64,
	/// Logical complete-plan identity.
	pub name: &'static str,
	/// Exact kernel manifest identity.
	pub kernel_id: KernelId,
	/// Coarse family used by routing policy.
	pub kernel: GemmKernel,
	pub path: GemmPath,
	pub epilogue: GemmEpilogue,
	pub a_precision: StoragePrecision,
	pub b_precision: StoragePrecision,
	pub output_precision: StoragePrecision,
	pub accumulator_precision: StoragePrecision,
	/// Inner-dimension tile. The M/N tile extents live in the artifact's
	/// `dispatch_tile_size`; tile K is a shader-source constant.
	pub tile_k: u32,
	/// M and K must divide the tile extents.
	pub requires_aligned: bool,
	/// B must be stored `[N, K]`.
	pub requires_transposed_b: bool,
	/// Derived contract metadata for tooling.
	pub supports_bias: bool,
	pub supports_activation: bool,
	/// Emits pre-activation alongside the post-activation output.
	pub dual_output: bool,
	/// AND-checked against the device cap mask at planning time.
	pub required_caps_mask: u64,
	pub supports_arbitrary_layout: bool,
	pub supports_batch: bool,
	/// Zero means unbounded; specialized schedules carry an explicit bound so
	/// forced variants, tuning, and normal routing share one legality rule.
	pub max_m: u32,
	/// `split_k == 1` is the ordinary one-dispatch contract. A larger value
	/// requires a registered reduction stage and an explicit workspace budget.
	pub split_k: u32,
	pub reduction_kernel_id: Option<KernelId>,
	pub reduction_workgroup_invocations: u32,
}

impl MatmulVariant {
	/// M tile extent from the owning artifact.
	fn tile_m(&self) -> u32 {
		self.kernel_id.artifact().dispatch_tile_size[0]
	}

	/// N tile extent from the owning artifact.
	fn tile_n(&self) -> u32 {
		self.kernel_id.artifact().dispatch_tile_size[1]
	}

	/// `local_size_x` of the owning artifact.
	fn workgroup_invocations(&self) -> u32 {
		self.kernel_id.artifact().workgroup_size[0]
	}

	/// Statically declared workgroup storage of the owning artifact.
	fn shared_memory_bytes(&self) -> Result<u32> {
		self.kernel_id.artifact().workgroup_memory_size()
	}

	/// Exact embedded SPIR-V identity of the owning artifact.
	fn shader_content_id(&self) -> u64 {
		self.kernel_id.artifact().content_id()
	}
}

/// Registered matmul candidates. Ordered by family preference is the
/// planner's job, not the table's.
const VARIANTS: &[MatmulVariant] = &[
	MatmulVariant {
		id: matmul_variant_id_from_name("GemmTiled"),
		name: "GemmTiled",
		kernel_id: KernelId::MatrixMatMulNtTiledF32,
		kernel: GemmKernel::TiledF32,
		path: GemmPath::Standard,
		epilogue: GemmEpilogue::None,
		a_precision: StoragePrecision::F32,
		b_precision: StoragePrecision::F32,
		output_precision: StoragePrecision::F32,
		accumulator_precision: StoragePrecision::F32,
		tile_k: 16,
		requires_aligned: false,
		requires_transposed_b: true,
		supports_bias: false,
		supports_activation: false,
		dual_output: false,
		required_caps_mask: CAP_TILED_F32,
		supports_arbitrary_layout: false,
		supports_batch: false,
		max_m: 0,
		split_k: 1,
		reduction_kernel_id: None,
		reduction_workgroup_invocations: 0,
	},
	MatmulVariant {
		id: matmul_variant_id_from_name("GemmNaive"),
		name: "GemmNaive",
		kernel_id: KernelId::MatrixMatMulNtNaiveF32,
		kernel: GemmKernel::Naive,
		path: GemmPath::Standard,
		epilogue: GemmEpilogue::None,
		a_precision: StoragePrecision::F32,
		b_precision: StoragePrecision::F32,
		output_precision: StoragePrecision::F32,
		accumulator_precision: StoragePrecision::F32,
		tile_k: 1,
		requires_aligned: false,
		requires_transposed_b: true,
		supports_bias: false,
		supports_activation: false,
		dual_output: false,
		required_caps_mask: CAP_NAIVE_F32,
		supports_arbitrary_layout: false,
		supports_batch: false,
		max_m: 0,
		split_k: 1,
		reduction_kernel_id: None,
		reduction_workgroup_invocations: 0,
	},
	MatmulVariant {
		id: matmul_variant_id_from_name("GemmSmallM"),
		name: "GemmSmallM",
		kernel_id: KernelId::MatrixMatMulNtGemvF32,
		kernel: GemmKernel::SmallMF32,
		path: GemmPath::Standard,
		epilogue: GemmEpilogue::None,
		a_precision: StoragePrecision::F32,
		b_precision: StoragePrecision::F32,
		output_precision: StoragePrecision::F32,
		accumulator_precision: StoragePrecision::F32,
		tile_k: 1,
		requires_aligned: false,
		requires_transposed_b: true,
		supports_bias: false,
		supports_activation: false,
		dual_output: false,
		required_caps_mask: CAP_SMALLM_F32,
		supports_arbitrary_layout: false,
		supports_batch: false,
		max_m: 4,
		split_k: 1,
		reduction_kernel_id: None,
		reduction_workgroup_invocations: 0,
	},
	MatmulVariant {
		id: matmul_variant_id_from_name("GemmBiasTiled"),
		name: "GemmBiasTiled",
		kernel_id: KernelId::MatrixMatMulNtBiasF32,
		kernel: GemmKernel::TiledF32,
		path: GemmPath::Standard,
		epilogue: GemmEpilogue::Bias,
		a_precision: StoragePrecision::F32,
		b_precision: StoragePrecision::F32,
		output_precision: StoragePrecision::F32,
		accumulator_precision: StoragePrecision::F32,
		tile_k: 16,
		requires_aligned: false,
		requires_transposed_b: true,
		supports_bias: true,
		supports_activation: false,
		dual_output: false,
		required_caps_mask: CAP_TILED_F32,
		supports_arbitrary_layout: false,
		supports_batch: false,
		max_m: 0,
		split_k: 1,
		reduction_kernel_id: None,
		reduction_workgroup_invocations: 0,
	},
	MatmulVariant {
		id: matmul_variant_id_from_name("GemmBiasReluTiled"),
		name: "GemmBiasReluTiled",
		kernel_id: KernelId::MatrixMatMulNtBiasReluF32,
		kernel: GemmKernel::TiledF32,
		path: GemmPath::Standard,
		epilogue: GemmEpilogue::BiasRelu,
		a_precision: StoragePrecision::F32,
		b_precision: StoragePrecision::F32,
		output_precision: StoragePrecision::F32,
		accumulator_precision: StoragePrecision::F32,
		tile_k: 16,
		requires_aligned: false,
		requires_transposed_b: true,
		supports_bias: true,
		supports_activation: true,
		dual_output: false,
		required_caps_mask: CAP_TILED_F32,
		supports_arbitrary_layout: false,
		supports_batch: false,
		max_m: 0,
		split_k: 1,
		reduction_kernel_id: None,
		reduction_workgroup_invocations: 0,
	},
	MatmulVariant {
		id: matmul_variant_id_from_name("GemmBiasGeluTiled"),
		name: "GemmBiasGeluTiled",
		kernel_id: KernelId::MatrixMatMulNtBiasGeluF32,
		kernel: GemmKernel::TiledF32,
		path: GemmPath::Standard,
		epilogue: GemmEpilogue::BiasGelu,
		a_precision: StoragePrecision::F32,
		b_precision: StoragePrecision::F32,
		output_precision: StoragePrecision::F32,
		accumulator_precision: StoragePrecision::F32,
		tile_k: 16,
		requires_aligned: false,
		requires_transposed_b: true,
		supports_bias: true,
		supports_activation: true,
		dual_output: false,
		required_caps_mask: CAP_TILED_F32,
		supports_arbitrary_layout: false,
		supports_batch: false,
		max_m: 0,
		split_k: 1,
		reduction_kernel_id: None,
		reduction_workgroup_invocations: 0,
	},
	MatmulVariant {
		id: matmul_variant_id_from_name("GemmBiasSiluTiled"),
		name: "GemmBiasSiluTiled",
		kernel_id: KernelId::MatrixMatMulNtBiasSiluF32,
		kernel: GemmKernel::TiledF32,
		path: GemmPath::Standard,
		epilogue: GemmEpilogue::BiasSilu,
		a_precision: StoragePrecision::F32,
		b_precision: StoragePrecision::F32,
		output_precision: StoragePrecision::F32,
		accumulator_precision: StoragePrecision::F32,
		tile_k: 16,
		requires_aligned: false,
		requires_transposed_b: true,
		supports_bias: true,
		supports_activation: true,
		dual_output: false,
		required_caps_mask: CAP_TILED_F32,
		supports_arbitrary_layout: false,
		supports_batch: false,
		max_m: 0,
		split_k: 1,
		reduction_kernel_id: None,
		reduction_workgroup_invocations: 0,
	},
];

fn all_variants() -> &'static [MatmulVariant] {
	VARIANTS
}

/// Exact stable-id lookup; `None` for unknown or retired variants.
fn find_variant(id: u64) -> Option<&'static MatmulVariant> {
	VARIANTS.iter().find(|variant| variant.id == id)
}

/// Stable hash of the registered variant contracts. Persisted route entries
/// are ignored when this changes, preventing shader/registry updates from
/// replaying stale winners.
fn registry_build_id() -> u64 {
	let mut hash = 0xcbf2_9ce4_8422_2325_u64;
	let mix = |hash: &mut u64, value: u64| {
		*hash ^= value;
		*hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
	};
	for variant in VARIANTS {
		mix(&mut hash, variant.id);
		for byte in variant.name.as_bytes() {
			mix(&mut hash, u64::from(*byte));
		}
		mix(&mut hash, u64::from(variant.kernel_id as u16));
		mix(&mut hash, variant.kernel as u64);
		mix(&mut hash, variant.path as u64);
		mix(&mut hash, variant.epilogue as u64);
		mix(&mut hash, variant.a_precision as u64);
		mix(&mut hash, variant.b_precision as u64);
		mix(&mut hash, variant.output_precision as u64);
		mix(&mut hash, variant.accumulator_precision as u64);
		mix(&mut hash, u64::from(variant.tile_m()));
		mix(&mut hash, u64::from(variant.tile_n()));
		mix(&mut hash, u64::from(variant.tile_k));
		mix(&mut hash, u64::from(variant.workgroup_invocations()));
		mix(&mut hash, u64::from(variant.requires_aligned));
		mix(&mut hash, u64::from(variant.requires_transposed_b));
		mix(&mut hash, u64::from(variant.supports_bias));
		mix(&mut hash, u64::from(variant.supports_activation));
		mix(&mut hash, u64::from(variant.dual_output));
		mix(&mut hash, u64::from(variant.supports_arbitrary_layout));
		mix(&mut hash, u64::from(variant.supports_batch));
		mix(&mut hash, u64::from(variant.max_m));
		mix(&mut hash, u64::from(variant.split_k));
		mix(
			&mut hash,
			variant
				.reduction_kernel_id
				.map_or(0, |id| u64::from(id as u16)),
		);
		mix(
			&mut hash,
			u64::from(variant.reduction_workgroup_invocations),
		);
		mix(&mut hash, variant.required_caps_mask);
	}
	hash
}

/// Device cap mask from the admitted device contract.
///
/// Fallback bits are always available. Cooperative-matrix bits require the
/// extension, compute-stage support, the *enabled* feature, and at least one
/// legal tuple of the right shape.
fn compute_caps_mask(device: &Device) -> u64 {
	let mut mask = CAP_TILED_F32 | CAP_NAIVE_F32 | CAP_SMALLM_F32;
	let physical = device.physical();
	let cooperative = &physical.cooperative_matrix;
	let cooperative_enabled = physical.features.cooperative_matrix
		&& device.configuration().features().cooperative_matrix
		&& cooperative
			.supported_stages
			.contains(vk::ShaderStageFlagBits::COMPUTE_BIT);
	if !cooperative_enabled || cooperative.tuples.is_empty() {
		return mask;
	}
	mask |= CAP_COOP_MAT1_KHR;
	for tuple in &cooperative.tuples {
		if tuple.a_type == vk::ComponentTypeKHR::BFLOAT16_KHR
			|| tuple.b_type == vk::ComponentTypeKHR::BFLOAT16_KHR
		{
			mask |= CAP_COOP_MAT1_BF16_INPUT;
			if tuple.scope == vk::ScopeKHR::WORKGROUP_KHR {
				mask |= CAP_COOP_MAT1_WORKGROUP_BF16;
			}
		}
		if tuple.accumulator_type == vk::ComponentTypeKHR::BFLOAT16_KHR {
			mask |= CAP_COOP_MAT1_BF16_ACC;
		}
		if tuple.a_type == vk::ComponentTypeKHR::FLOAT16_KHR
			|| tuple.b_type == vk::ComponentTypeKHR::FLOAT16_KHR
		{
			mask |= CAP_COOP_MAT1_F16_INPUT;
		}
		if tuple.accumulator_type == vk::ComponentTypeKHR::FLOAT32_KHR {
			mask |= CAP_COOP_MAT1_F32_ACC;
		}
	}
	mask
}

// ─────────────────────────────────────────────────────────────────────────────
// Plan — immutable result of planning one exact problem on one device contract
// ─────────────────────────────────────────────────────────────────────────────

/// First legality rule a variant fails, in the donor's check order. Kept
/// explicit so planning evidence and route diagnostics name the exact gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::runtime) enum MatmulRejection {
	EmptyExtents,
	EpilogueMismatch,
	OutputPrecisionMismatch,
	NonCanonicalLayout,
	BatchUnsupported,
	MaxMExceeded,
	RequiresTransposedB,
	InputPrecisionMismatch,
	MissingDeviceCapability,
	WorkgroupInvocationLimit,
	WorkgroupSizeLimit,
	SharedMemoryLimit,
	SharedMemoryReflection,
	RequiresAligned,
	FamilyConstraint,
}

impl fmt::Display for MatmulRejection {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		let text = match self {
			Self::EmptyExtents => "a problem extent or the batch count is zero",
			Self::EpilogueMismatch => "the variant epilogue does not match the problem contract",
			Self::OutputPrecisionMismatch => {
				"the variant output precision does not match the problem contract"
			}
			Self::NonCanonicalLayout => {
				"the problem is not contiguous and the variant lacks arbitrary-layout support"
			}
			Self::BatchUnsupported => "the problem is batched and the variant lacks batch support",
			Self::MaxMExceeded => "the problem M exceeds the variant bound",
			Self::RequiresTransposedB => "the variant requires B stored as [N, K]",
			Self::InputPrecisionMismatch => {
				"the variant input precision does not match the resolved precision"
			}
			Self::MissingDeviceCapability => {
				"the device cap mask lacks a capability the variant requires"
			}
			Self::WorkgroupInvocationLimit => "the variant exceeds the device workgroup invocation limit",
			Self::WorkgroupSizeLimit => "the variant exceeds a device workgroup size limit",
			Self::SharedMemoryLimit => "the variant exceeds the device shared-memory limit",
			Self::SharedMemoryReflection => "the variant's shared-memory requirement is unknown",
			Self::RequiresAligned => "the variant requires tile-aligned extents",
			Self::FamilyConstraint => "the problem violates a family-specific rule",
		};
		f.write_str(text)
	}
}

/// Immutable result of planning one exact problem on one device contract.
///
/// Records every fact required to reproduce selection: the pinned variant,
/// launch geometry, workspace, and the contract identities a replay or cache
/// validation must recheck.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatmulPlan {
	/// Stable variant identity; `0` is never a valid plan.
	pub(in crate::runtime) variant: u64,
	/// Pinned kernel manifest identity for the dispatch record.
	pub(crate) kernel_id: KernelId,
	pub(in crate::runtime) kernel: GemmKernel,
	pub(in crate::runtime) path: GemmPath,
	pub(in crate::runtime) actual_precision: GemmPrecision,
	pub(in crate::runtime) selection: KernelSelectionKind,
	/// `vkCmdDispatch` workgroup grid for the pinned kernel.
	pub(crate) grid: [u32; 3],
	pub(in crate::runtime) workspace_bytes: u64,
	pub(in crate::runtime) problem_contract_hash: u64,
	pub(in crate::runtime) device_contract_hash: u64,
	pub(in crate::runtime) registry_build_id: u64,
	pub(in crate::runtime) shader_content_hash: u64,
	pub(in crate::runtime) split_k: u32,
	pub(in crate::runtime) reduction_kernel_id: Option<KernelId>,
	pub(in crate::runtime) reduction_grid: [u32; 3],
	pub(in crate::runtime) reduction_shader_content_hash: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ABI lowering — plan → kernel launch, shared by every execution frontend
// ─────────────────────────────────────────────────────────────────────────────

/// Kernel buffer-slot ceiling: A, B, C, the optional split-K workspace, and
/// the fused-epilogue operand — the donor's `MaxBuffers`.
const MATMUL_MAX_BUFFERS: usize = 5;
/// Push-constant ceiling in words — the donor's 64-byte launch cap.
const MATMUL_MAX_PUSH_WORDS: usize = 16;

/// Concrete kernel launch lowered from one already-validated immutable
/// plan — the donor's `MatmulKernelLaunch`.
///
/// `buffer_order` maps kernel binding slots to mathematical problem buffers
/// (0=A, 1=B, 2=C, 3=split-K workspace, 4=epilogue operand), so every
/// execution frontend — eager, captured, and the tuning driver — consumes
/// identical push constants and buffer ordering. This is not a second
/// planner.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatmulKernelLaunch {
	pub kernel_id: KernelId,
	pub grid: [u32; 3],
	pub buffer_order: [u8; MATMUL_MAX_BUFFERS],
	pub buffer_accesses: [BufferAccess; MATMUL_MAX_BUFFERS],
	pub buffer_count: usize,
	pub push_constants: [PushConstant; MATMUL_MAX_PUSH_WORDS],
	pub push_count: usize,
}

impl MatmulKernelLaunch {
	/// Bind problem-order storages into kernel-slot order.
	///
	/// `problem_buffers` is the mathematical list in the documented slot
	/// order; the returned bindings cover exactly `buffer_count` slots.
	pub(crate) fn bindings<'a>(
		&self,
		problem_buffers: &[&'a Storage],
	) -> Result<Vec<BufferBinding<'a>>> {
		let mut bindings = Vec::with_capacity(self.buffer_count);
		for slot in 0..self.buffer_count {
			let index = usize::from(self.buffer_order[slot]);
			let storage = problem_buffers.get(index).copied().ok_or_else(|| {
				Error::invalid_argument(format!(
					"matmul launch slot {slot} binds missing problem buffer {index}"
				))
			})?;
			bindings.push(BufferBinding {
				storage,
				access: self.buffer_accesses[slot],
			});
		}
		Ok(bindings)
	}
}

/// Pure ABI lowering for a plan already accepted by the planner — the
/// donor's `describeValidatedPlan`.
///
/// Performs no heuristic or route-cache lookup: callers must validate the
/// plan against the live device contract immediately before recording.
/// Kernel families without registered candidates have no lowering arm —
/// a family lands with its shader, schema record, registry row, and launch
/// contract in one checkpoint.
pub(in crate::runtime) fn lower_mat_mul_nt_plan(
	plan: &MatmulPlan,
	problem: &MatmulProblem,
) -> Result<MatmulKernelLaunch> {
	if plan.variant == INVALID_MATMUL_VARIANT_ID {
		return Err(Error::invalid_argument("cannot lower an empty matmul plan"));
	}
	// Epilogue launches bind kernel slots [A, B, bias, C] — problem buffers
	// 0, 1, 4, 2 — matching the donor's `EpilogueBias*` binding order.
	let (buffer_order, buffer_accesses, buffer_count) = if problem.epilogue == GemmEpilogue::None {
		(
			[0, 1, 2, 3, 4],
			[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Write,
				BufferAccess::Read,
			],
			3,
		)
	} else {
		(
			[0, 1, 4, 2, 3],
			[
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Read,
				BufferAccess::Write,
				BufferAccess::Write,
			],
			4,
		)
	};
	let mut launch = MatmulKernelLaunch {
		kernel_id: plan.kernel_id,
		grid: plan.grid,
		buffer_order,
		buffer_accesses,
		buffer_count,
		push_constants: [PushConstant::U32(0); MATMUL_MAX_PUSH_WORDS],
		push_count: 0,
	};
	match plan.kernel {
		GemmKernel::TiledF32 | GemmKernel::Naive | GemmKernel::SmallMF32 => {
			let constants = [
				PushConstant::U32(problem.m),
				PushConstant::U32(problem.n),
				PushConstant::U32(problem.k),
			];
			launch.push_constants[..constants.len()].copy_from_slice(&constants);
			launch.push_count = constants.len();
		}
		kernel => {
			return Err(Error::missing_capability(format!(
				"no matmul launch lowering for {kernel:?}"
			)));
		}
	}
	Ok(launch)
}

/// Lower a validated canonical `[M, K] × [N, K]` plan — the
/// `pub(crate)`-visible entry for callers outside `crate::runtime`, which
/// rebuilds the same contiguous problem the plan was produced against.
pub(crate) fn lower_mat_mul_nt(
	plan: &MatmulPlan,
	m: u32,
	n: u32,
	k: u32,
) -> Result<MatmulKernelLaunch> {
	lower_mat_mul_nt_plan(plan, &MatmulProblem::contiguous(m, n, k))
}

fn div_ceil(a: u32, b: u32) -> u32 {
	a.div_ceil(b)
}

/// Overflow-checked planner workspace: `m * n * split_k` F32 partials.
fn calculate_workspace_bytes(variant: &MatmulVariant, problem: &MatmulProblem) -> u64 {
	if variant.split_k <= 1 || problem.m == 0 || problem.n == 0 {
		return 0;
	}
	u64::from(problem.m)
		.checked_mul(u64::from(problem.n))
		.and_then(|elements| elements.checked_mul(u64::from(variant.split_k)))
		.and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
		.unwrap_or(u64::MAX)
}

fn problem_contract_hash(problem: &MatmulProblem) -> u64 {
	let mut hash = 0xcbf2_9ce4_8422_2325_u64;
	let mix = |hash: &mut u64, value: u64| {
		*hash ^= value;
		*hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
	};
	mix(&mut hash, u64::from(problem.m));
	mix(&mut hash, u64::from(problem.n));
	mix(&mut hash, u64::from(problem.k));
	mix(&mut hash, u64::from(problem.batch_count));
	for layout in [problem.a, problem.b, problem.c] {
		mix(&mut hash, u64::from(layout.offset));
		mix(&mut hash, u64::from(layout.row_stride));
		mix(&mut hash, u64::from(layout.col_stride));
		mix(&mut hash, u64::from(layout.batch_stride));
	}
	mix(&mut hash, problem.a_master as u64);
	mix(&mut hash, problem.b_master as u64);
	mix(&mut hash, problem.requested_output as u64);
	mix(&mut hash, u64::from(problem.a_contiguous));
	mix(&mut hash, u64::from(problem.b_contiguous));
	mix(&mut hash, u64::from(problem.b_transposed));
	mix(&mut hash, problem.epilogue as u64);
	mix(&mut hash, u64::from(problem.requires_pre_activation));
	mix(&mut hash, u64::from(problem.training));
	mix(&mut hash, problem.precision_hint as u64);
	hash
}

fn device_contract_hash(contract: &MatmulDeviceContract) -> u64 {
	let mut hash = 0xcbf2_9ce4_8422_2325_u64;
	let mix = |hash: &mut u64, value: u64| {
		*hash ^= value;
		*hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
	};
	mix(&mut hash, u64::from(contract.vendor_id));
	mix(&mut hash, u64::from(contract.device_id));
	mix(&mut hash, u64::from(contract.driver_id));
	mix(&mut hash, u64::from(contract.driver_version));
	mix(&mut hash, contract.capability_identity);
	mix(&mut hash, contract.caps_mask);
	mix(
		&mut hash,
		u64::from(contract.max_compute_workgroup_invocations),
	);
	for value in contract.max_compute_workgroup_size {
		mix(&mut hash, u64::from(value));
	}
	mix(
		&mut hash,
		u64::from(contract.max_compute_shared_memory_bytes),
	);
	hash
}

fn resolve_precision(problem: &MatmulProblem) -> GemmPrecision {
	if problem.precision_hint != GemmPrecision::Auto {
		return problem.precision_hint;
	}
	if problem.a_master == StoragePrecision::Bf16 || problem.b_master == StoragePrecision::Bf16 {
		GemmPrecision::Bf16
	} else {
		GemmPrecision::Auto
	}
}

/// First legality failure for one variant, in the donor's check order.
fn variant_legal(
	contract: &MatmulDeviceContract,
	variant: &MatmulVariant,
	problem: &MatmulProblem,
	precision: GemmPrecision,
) -> Option<MatmulRejection> {
	if problem.m == 0 || problem.n == 0 || problem.k == 0 || problem.batch_count == 0 {
		return Some(MatmulRejection::EmptyExtents);
	}
	if variant.epilogue != problem.epilogue || variant.dual_output != problem.requires_pre_activation
	{
		return Some(MatmulRejection::EpilogueMismatch);
	}
	if variant.output_precision != problem.requested_output {
		return Some(MatmulRejection::OutputPrecisionMismatch);
	}
	if !problem.is_canonical() && !variant.supports_arbitrary_layout {
		return Some(MatmulRejection::NonCanonicalLayout);
	}
	if problem.batch_count > 1 && !variant.supports_batch {
		return Some(MatmulRejection::BatchUnsupported);
	}
	if variant.max_m != 0 && problem.m > variant.max_m {
		return Some(MatmulRejection::MaxMExceeded);
	}
	if variant.requires_transposed_b && !problem.b_transposed {
		return Some(MatmulRejection::RequiresTransposedB);
	}
	if precision == GemmPrecision::F32
		&& (variant.a_precision != StoragePrecision::F32
			|| variant.b_precision != StoragePrecision::F32)
	{
		return Some(MatmulRejection::InputPrecisionMismatch);
	}
	if precision == GemmPrecision::Bf16
		&& contract.caps_mask & (CAP_COOP_MAT1_BF16_INPUT | CAP_COOP_MAT1_BF16_ACC) != 0
		&& (variant.a_precision != StoragePrecision::Bf16
			|| variant.b_precision != StoragePrecision::Bf16)
	{
		return Some(MatmulRejection::InputPrecisionMismatch);
	}
	if !caps_satisfy(contract.caps_mask, variant.required_caps_mask) {
		return Some(MatmulRejection::MissingDeviceCapability);
	}
	let workgroup_size = variant.kernel_id.artifact().workgroup_size;
	let invocations = workgroup_size
		.iter()
		.fold(1_u64, |total, dim| total * u64::from(*dim));
	if contract.max_compute_workgroup_invocations != 0
		&& invocations > u64::from(contract.max_compute_workgroup_invocations)
	{
		return Some(MatmulRejection::WorkgroupInvocationLimit);
	}
	for (axis, dim) in workgroup_size.iter().enumerate() {
		if contract.max_compute_workgroup_size[axis] != 0
			&& *dim > contract.max_compute_workgroup_size[axis]
		{
			return Some(MatmulRejection::WorkgroupSizeLimit);
		}
	}
	let shared_memory = match variant.shared_memory_bytes() {
		Ok(bytes) => bytes,
		Err(_) => return Some(MatmulRejection::SharedMemoryReflection),
	};
	if contract.max_compute_shared_memory_bytes != 0
		&& shared_memory > contract.max_compute_shared_memory_bytes
	{
		return Some(MatmulRejection::SharedMemoryLimit);
	}
	if variant.requires_aligned
		&& (!problem.m.is_multiple_of(variant.tile_m())
			|| !problem.n.is_multiple_of(variant.tile_n())
			|| !problem.k.is_multiple_of(variant.tile_k))
	{
		return Some(MatmulRejection::RequiresAligned);
	}
	match variant.kernel {
		GemmKernel::TiledF32 | GemmKernel::Naive | GemmKernel::SmallMF32 => {
			if problem.batch_count == 1 {
				None
			} else {
				Some(MatmulRejection::FamilyConstraint)
			}
		}
		GemmKernel::SplitKF32 => {
			const MAX_U32: u64 = u32::MAX as u64;
			let max_addressed_elements = MAX_U32 / size_of::<f32>() as u64 + 1;
			let a_elements = u64::from(problem.m) * u64::from(problem.k);
			let b_elements = u64::from(problem.n) * u64::from(problem.k);
			let output_elements = u64::from(problem.m) * u64::from(problem.n);
			let reduction_legal = variant
				.reduction_kernel_id
				.is_some_and(|_| variant.reduction_workgroup_invocations > 0)
				&& (contract.max_compute_workgroup_invocations == 0
					|| variant.reduction_workgroup_invocations <= contract.max_compute_workgroup_invocations)
				&& (contract.max_compute_workgroup_size[0] == 0
					|| variant.reduction_workgroup_invocations <= contract.max_compute_workgroup_size[0]);
			if problem.batch_count == 1
				&& variant.split_k > 1
				&& u64::from(problem.k) * u64::from(variant.split_k) <= MAX_U32
				&& a_elements <= max_addressed_elements
				&& b_elements <= max_addressed_elements
				&& output_elements <= max_addressed_elements / u64::from(variant.split_k)
				&& reduction_legal
				&& u64::from(problem.k) >= u64::from(variant.split_k) * u64::from(variant.tile_k)
			{
				None
			} else {
				Some(MatmulRejection::FamilyConstraint)
			}
		}
		GemmKernel::StridedF32 | GemmKernel::StridedTiledF32 => {
			if precision != GemmPrecision::Bf16 && problem.epilogue == GemmEpilogue::None {
				None
			} else {
				Some(MatmulRejection::FamilyConstraint)
			}
		}
		GemmKernel::GemmCmSgBf16 | GemmKernel::GemmCmWgBf16 => {
			if precision != GemmPrecision::F32 {
				None
			} else {
				Some(MatmulRejection::FamilyConstraint)
			}
		}
		GemmKernel::CoopVec => {
			const VENDOR_NVIDIA: u32 = 0x10DE;
			if precision != GemmPrecision::F32
				&& problem.epilogue == GemmEpilogue::None
				&& problem.m == 1
				&& contract.vendor_id == VENDOR_NVIDIA
				&& caps_satisfy(contract.caps_mask, CAP_COOP_VEC)
			{
				None
			} else {
				Some(MatmulRejection::FamilyConstraint)
			}
		}
		GemmKernel::Auto => Some(MatmulRejection::FamilyConstraint),
	}
}

/// Launch geometry and route facts for one legal variant.
fn result_for_variant(
	variant: &MatmulVariant,
	problem: &MatmulProblem,
) -> (GemmPrecision, [u32; 3]) {
	let precision = match variant.a_precision {
		StoragePrecision::Bf16 => GemmPrecision::Bf16,
		StoragePrecision::F32 => GemmPrecision::F32,
	};
	let grid = match variant.path {
		GemmPath::Standard => {
			if variant.kernel == GemmKernel::SplitKF32 {
				if variant.tile_m() == 1 && variant.tile_n() == 1 {
					[
						div_ceil(
							problem.m.saturating_mul(problem.n),
							variant.workgroup_invocations(),
						),
						variant.split_k,
						1,
					]
				} else {
					[
						div_ceil(problem.m, variant.tile_m()),
						div_ceil(problem.n, variant.tile_n()),
						variant.split_k,
					]
				}
			} else if variant.kernel == GemmKernel::Naive || variant.kernel == GemmKernel::StridedF32 {
				[
					div_ceil(problem.m.saturating_mul(problem.n), 256),
					1,
					problem.batch_count,
				]
			} else {
				[
					div_ceil(problem.m, variant.tile_m()),
					div_ceil(problem.n, variant.tile_n()),
					if variant.kernel == GemmKernel::StridedTiledF32 {
						problem.batch_count
					} else {
						1
					},
				]
			}
		}
		GemmPath::CoopVec => [div_ceil(problem.n, variant.tile_n()), 1, 1],
	};
	(precision, grid)
}

fn plan_for_variant(
	contract: &MatmulDeviceContract,
	variant: &'static MatmulVariant,
	problem: &MatmulProblem,
) -> MatmulPlan {
	let (actual_precision, grid) = result_for_variant(variant, problem);
	MatmulPlan {
		variant: variant.id,
		kernel_id: variant.kernel_id,
		kernel: variant.kernel,
		path: variant.path,
		actual_precision,
		selection: KernelSelectionKind::Unspecified,
		grid,
		workspace_bytes: calculate_workspace_bytes(variant, problem),
		problem_contract_hash: problem_contract_hash(problem),
		device_contract_hash: device_contract_hash(contract),
		registry_build_id: registry_build_id(),
		shader_content_hash: variant.shader_content_id(),
		split_k: variant.split_k,
		reduction_kernel_id: variant.reduction_kernel_id,
		reduction_grid: if variant.split_k > 1 {
			[
				div_ceil(
					problem.m.saturating_mul(problem.n),
					variant.reduction_workgroup_invocations,
				),
				1,
				1,
			]
		} else {
			[0, 0, 0]
		},
		reduction_shader_content_hash: variant
			.reduction_kernel_id
			.map_or(0, |id| id.artifact().content_id()),
	}
}

fn classify_selection(plan: &MatmulPlan, problem: &MatmulProblem) -> KernelSelectionKind {
	let requested = resolve_precision(problem);
	if requested == GemmPrecision::Bf16 && plan.actual_precision != GemmPrecision::Bf16 {
		return KernelSelectionKind::PrecisionFallback;
	}
	if plan.kernel == GemmKernel::StridedF32 || plan.kernel == GemmKernel::StridedTiledF32 {
		return KernelSelectionKind::LayoutFallback;
	}
	if plan.kernel == GemmKernel::Naive {
		return KernelSelectionKind::NaiveFallback;
	}
	KernelSelectionKind::Direct
}

/// Plan one exact problem against one device contract.
///
/// Order mirrors the donor router: an explicit required variant first, then
/// a measured winner recorded for this exact route key, the M=1
/// cooperative-vector route, tensor-core families, strided families for
/// non-canonical layouts, the small-M schedule, and finally the portable
/// F32 fallback order.
///
/// `measured_winner` is the caller-resolved cache value — the engine looks
/// its route cache up before calling, keeping `plan` free of engine state.
pub(in crate::runtime) fn plan(
	contract: &MatmulDeviceContract,
	problem: &MatmulProblem,
	preference: &MatmulPreference,
	measured_winner: Option<u64>,
) -> Result<MatmulPlan> {
	let precision = resolve_precision(problem);
	let mut rejections: Vec<(&'static str, MatmulRejection)> = Vec::new();
	let legal = |variant: &'static MatmulVariant| {
		variant_legal(contract, variant, problem, precision).is_none()
			&& calculate_workspace_bytes(variant, problem) <= preference.max_workspace_bytes
	};
	let finish = |variant: &'static MatmulVariant| {
		let mut plan = plan_for_variant(contract, variant, problem);
		plan.selection = classify_selection(&plan, problem);
		plan
	};

	// Explicit diagnostic/tuning isolation fails closed.
	if preference.required_variant != INVALID_MATMUL_VARIANT_ID {
		return match find_variant(preference.required_variant) {
			Some(variant) if legal(variant) => Ok(finish(variant)),
			Some(variant) => Err(Error::missing_capability(format!(
				"required matmul variant {} is not legal: {}",
				variant.name,
				variant_legal(contract, variant, problem, precision).map_or_else(
					|| "workspace budget exceeded".to_string(),
					|r| r.to_string()
				),
			))),
			None => Err(Error::invalid_argument(format!(
				"required matmul variant id {:#x} is not registered",
				preference.required_variant
			))),
		};
	}

	// A recorded measured winner is replayed only when the caller opted in,
	// and is always revalidated against the live registry, the complete
	// problem, and the device contract — never trusted blindly.
	if preference.use_measured_cache
		&& let Some(winner) = measured_winner
		&& let Some(variant) = find_variant(winner)
		&& legal(variant)
	{
		return Ok(finish(variant));
	}

	// M=1 decode is the only distinct path; the cooperative-vector candidate is
	// registered only when its capability qualification lands.
	if problem.epilogue == GemmEpilogue::None
		&& problem.m == 1
		&& precision != GemmPrecision::F32
		&& let Some(variant) = find_variant(matmul_variant_id_from_name("GemmCoopVec"))
		&& legal(variant)
	{
		return Ok(finish(variant));
	}

	// Tensor-core families in measured-policy order.
	if precision != GemmPrecision::F32 {
		for family in [GemmKernel::GemmCmWgBf16, GemmKernel::GemmCmSgBf16] {
			for variant in all_variants() {
				if variant.kernel == family && legal(variant) {
					return Ok(finish(variant));
				}
			}
		}
	}

	// Non-canonical views and strided batches use the arbitrary-layout
	// families when they register.
	if !problem.is_canonical() {
		let work = u64::from(problem.m) * u64::from(problem.n) * u64::from(problem.k);
		let first = if problem.m >= 16 && problem.n >= 16 && work >= 32_768 {
			GemmKernel::StridedTiledF32
		} else {
			GemmKernel::StridedF32
		};
		let second = if first == GemmKernel::StridedTiledF32 {
			GemmKernel::StridedF32
		} else {
			GemmKernel::StridedTiledF32
		};
		for family in [first, second] {
			for variant in all_variants() {
				if variant.kernel == family && legal(variant) {
					return Ok(finish(variant));
				}
			}
		}
	}

	// Portable F32 path. Tiny raw GEMMs prefer the scalar route when it
	// registers; every other problem tries the tiled family first.
	let prefer_naive =
		problem.epilogue == GemmEpilogue::None && u64::from(problem.m) * u64::from(problem.n) < 64;
	// Avoid the measured tiny-decode regression without relying on a tuning
	// cache populated only by the explicit qualification harness.
	if !prefer_naive && problem.m == 1 && problem.n >= 512 && problem.k >= 512 {
		for variant in all_variants() {
			if variant.kernel == GemmKernel::SmallMF32 && legal(variant) {
				return Ok(finish(variant));
			}
		}
	}
	let fallback_families = if prefer_naive {
		[GemmKernel::Naive, GemmKernel::TiledF32]
	} else {
		[GemmKernel::TiledF32, GemmKernel::Naive]
	};
	for family in fallback_families {
		for variant in all_variants() {
			if variant.kernel == family && variant.a_precision == StoragePrecision::F32 && legal(variant)
			{
				return Ok(finish(variant));
			}
			if variant.kernel == family
				&& let Some(rejection) = variant_legal(contract, variant, problem, precision)
			{
				rejections.push((variant.name, rejection));
			}
		}
	}

	for variant in all_variants() {
		if let Some(rejection) = variant_legal(contract, variant, problem, precision)
			&& !rejections.iter().any(|(name, _)| *name == variant.name)
		{
			rejections.push((variant.name, rejection));
		}
	}
	let evidence = rejections
		.iter()
		.map(|(name, reason)| format!("{name}: {reason}"))
		.collect::<Vec<_>>()
		.join("; ");
	Err(Error::missing_capability(format!(
		"no legal matmul variant for M={} N={} K={} batch={} epilogue={:?} ({evidence})",
		problem.m, problem.n, problem.k, problem.batch_count, problem.epilogue,
	)))
}

/// Recheck a recorded plan against the live registry, problem, and device
/// contract — the replay validation a persisted plan must pass.
pub(in crate::runtime) fn validate_plan(
	contract: &MatmulDeviceContract,
	plan: &MatmulPlan,
	problem: &MatmulProblem,
) -> bool {
	if plan.variant == INVALID_MATMUL_VARIANT_ID
		|| plan.registry_build_id != registry_build_id()
		|| plan.problem_contract_hash != problem_contract_hash(problem)
		|| plan.device_contract_hash != device_contract_hash(contract)
	{
		return false;
	}
	let Some(variant) = find_variant(plan.variant) else {
		return false;
	};
	if variant.kernel_id != plan.kernel_id
		|| variant.kernel != plan.kernel
		|| variant.path != plan.path
		|| plan.shader_content_hash == 0
		|| plan.shader_content_hash != variant.shader_content_id()
		|| variant_legal(contract, variant, problem, resolve_precision(problem)).is_some()
	{
		return false;
	}
	let (expected_precision, expected_grid) = result_for_variant(variant, problem);
	plan.actual_precision == expected_precision
		&& plan.grid == expected_grid
		&& plan.workspace_bytes == calculate_workspace_bytes(variant, problem)
		&& plan.split_k == variant.split_k
		&& plan.reduction_kernel_id == variant.reduction_kernel_id
		&& (variant.split_k == 1
			|| (plan.reduction_grid
				== [
					div_ceil(
						problem.m.saturating_mul(problem.n),
						variant.reduction_workgroup_invocations,
					),
					1,
					1,
				] && plan.reduction_shader_content_hash != 0
				&& plan.reduction_shader_content_hash
					== variant
						.reduction_kernel_id
						.map_or(0, |id| id.artifact().content_id())))
}

/// Plan the canonical contiguous `[M, K] × [N, K]` semantic matmul on an
/// engine's device contract.
pub(crate) fn plan_mat_mul_nt(engine: &EngineHandle, m: u32, n: u32, k: u32) -> Result<MatmulPlan> {
	let problem = MatmulProblem::contiguous(m, n, k);
	plan_mat_mul_nt_problem(engine, &problem, &MatmulPreference::default())
}

/// Plan one caller-constructed problem through the identical admission path —
/// the donor's `GemmRouter::plan`. Private providers build epilogue-bearing
/// problems; measured replay stays opt-in through the preference.
pub(in crate::runtime) fn plan_mat_mul_nt_problem(
	engine: &EngineHandle,
	problem: &MatmulProblem,
	preference: &MatmulPreference,
) -> Result<MatmulPlan> {
	let contract = engine.matmul_device_contract();
	let measured = engine.matmul_measured_route(&route_key(&contract, problem));
	let plan = plan(&contract, problem, preference, measured)?;
	debug_assert!(validate_plan(&contract, &plan, problem));
	Ok(plan)
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/gemm_tuner_unit.rs"]
mod gemm_tuner;

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/matmul_unit.rs"]
mod matmul_unit;
