//! Versioned OA `.vkcache` envelope — one scoped artifact per device, driver
//! and descriptor profile.
//!
//! Layout (all integers little-endian; every offset/length a byte count):
//!
//! ```text
//! header   48 bytes  — magic/version/size/scope/directory checksum
//! directory  count * 48 bytes — typed section extents + payload checksums
//! payloads 16-byte-aligned sections, zero padding between
//! ```
//!
//! The Vulkan driver pipeline-cache bytes remain an opaque payload in their
//! own section; the envelope never modifies or appends to them. Readers fail
//! closed: any malformed header, directory entry, checksum or scope mismatch
//! yields an empty cache. Writers hold an advisory lock, merge against the
//! last committed artifact and commit atomically through a same-directory
//! temporary file, fsync and rename. Persistence is best-effort — a cache is
//! disposable qualified state, never a correctness dependency.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path as StdPath;

use crate::{Error, Path, Result};

use super::features::ExecutionProfile;
use super::physical::DevicePhysical;

/// Envelope format identity — `"OAVK"` little-endian.
const VKCACHE_MAGIC: u32 = 0x4B56_414F;
/// Incompatible-layout generation; bump when the header/directory contract changes.
const VKCACHE_FORMAT_MAJOR: u16 = 1;
/// Section-compatible extension generation.
const VKCACHE_FORMAT_MINOR: u16 = 0;
/// Fixed header size in bytes.
const VKCACHE_HEADER_SIZE: u64 = 48;
/// Directory entry size in bytes.
const VKCACHE_ENTRY_SIZE: u64 = 48;
/// Integrity algorithm identifier: FNV-1a 64 over each decoded payload.
const VKCACHE_CHECKSUM_FNV1A64: u32 = 1;
/// Hard ceiling on an accepted artifact.
const VKCACHE_MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
/// Maximum typed sections in one artifact.
const VKCACHE_MAX_SECTIONS: u32 = 16;
/// Maximum decoded payload per section. Format 1 stores payloads
/// uncompressed, so this is also the stored-size ceiling.
const VKCACHE_MAX_SECTION_BYTES: u64 = 192 * 1024 * 1024;
/// Section payloads start at 16-byte-aligned offsets.
const VKCACHE_ALIGNMENT: u64 = 16;

/// Scope record section — required; identifies the owning device contract.
pub(in crate::runtime) const SECTION_SCOPE: u32 = 1;
/// Opaque `vkGetPipelineCacheData` bytes, verbatim.
pub(in crate::runtime) const SECTION_DRIVER_PIPELINE_CACHE: u32 = 2;
/// Existing version-6 field-wise BLAS measured-route payload, verbatim.
pub(in crate::runtime) const SECTION_GEMM_MEASURED_ROUTES: u32 = 3;
/// Bounded JIT artifact identity/ABI records.
pub(in crate::runtime) const SECTION_JIT_ARTIFACT_INDEX: u32 = 4;
/// Bounded JIT SPIR-V word table keyed by artifact identity.
pub(in crate::runtime) const SECTION_JIT_SPIRV_ARTIFACTS: u32 = 5;
/// Route-choice statistics; the producer arrives with adaptive routing.
pub(in crate::runtime) const SECTION_FUSION_ROUTE_OBSERVATIONS: u32 = 6;

/// Directory flag marking a section required for every reader.
const SECTION_FLAG_REQUIRED: u32 = 1;

/// Highest section type the current implementation understands.
const SECTION_TYPE_MAX_KNOWN: u32 = SECTION_FUSION_ROUTE_OBSERVATIONS;

/// Cache scope derived from one selected physical device.
///
/// `key` is the deterministic filename component; the remaining fields are
/// embedded in the `SCOPE` section so a renamed or foreign artifact can never
/// masquerade as this device's cache.
#[derive(Clone)]
pub(in crate::runtime) struct VkCacheScope {
	key: u64,
	vendor_id: u32,
	device_id: u32,
	driver_id: u32,
	driver_version: u32,
	api_version: u32,
	profile: u32,
	capability_identity: u64,
	pipeline_cache_uuid: [u8; 16],
}

impl VkCacheScope {
	/// Derive the cache scope for one admitted physical device.
	pub(in crate::runtime) fn new(physical: &DevicePhysical) -> Self {
		let profile = match physical.profile {
			ExecutionProfile::Strict => 0u32,
			ExecutionProfile::Compatibility => 1u32,
		};
		let mut hash = StableHash::new();
		hash.bytes(b"oavk.scope\x01");
		hash.u32(u32::from(VKCACHE_FORMAT_MAJOR));
		hash.u64(physical.info.capability_identity);
		hash.u32(profile);
		Self {
			key: hash.finish(),
			vendor_id: physical.info.vendor_id,
			device_id: physical.info.device_id,
			driver_id: u32::try_from(physical.info.driver_id.as_raw()).unwrap_or(0),
			driver_version: physical.info.driver_version,
			api_version: physical.api_version,
			profile,
			capability_identity: physical.info.capability_identity,
			pipeline_cache_uuid: physical.info.pipeline_cache_uuid,
		}
	}

	/// Device/driver identity shared with measured-route keys —
	/// `(vendor_id, device_id, driver_id, driver_version)`.
	pub(in crate::runtime) const fn device_route_scope(&self) -> (u32, u32, u32, u32) {
		(
			self.vendor_id,
			self.device_id,
			self.driver_id,
			self.driver_version,
		)
	}

	/// Scoped artifact path below `var/vk`.
	pub(in crate::runtime) fn path(&self) -> Path {
		Path::var()
			.join("vk")
			.join(format!("oa-{:016x}.vkcache", self.key))
	}

	/// Serialize the fixed 64-byte SCOPE record.
	fn encode(&self) -> [u8; 64] {
		let mut record = [0u8; 64];
		let mut writer = SectionWriter::new(&mut record);
		writer.u64(self.key);
		writer.u64(self.capability_identity);
		writer.u32(self.vendor_id);
		writer.u32(self.device_id);
		writer.u32(self.driver_id);
		writer.u32(self.driver_version);
		writer.u32(self.api_version);
		writer.u32(self.profile);
		writer.bytes(&self.pipeline_cache_uuid);
		record
	}

	/// Verify one decoded SCOPE record against this scope.
	fn matches(&self, bytes: &[u8]) -> bool {
		let mut reader = SectionReader::new(bytes);
		reader.u64() == Some(self.key)
			&& reader.u64() == Some(self.capability_identity)
			&& reader.u32() == Some(self.vendor_id)
			&& reader.u32() == Some(self.device_id)
			&& reader.u32() == Some(self.driver_id)
			&& reader.u32() == Some(self.driver_version)
			&& reader.u32() == Some(self.api_version)
			&& reader.u32() == Some(self.profile)
			&& reader.bytes(16) == Some(&self.pipeline_cache_uuid[..])
			&& reader.bytes(8) == Some(&[0u8; 8][..])
			&& reader.offset == bytes.len()
	}
}

/// One validated typed section.
#[derive(Clone)]
pub(in crate::runtime) struct VkCacheSection {
	/// Payload schema version understood by this reader.
	pub(in crate::runtime) version: u32,
	/// Decoded payload bytes (format 1: stored bytes verbatim).
	pub(in crate::runtime) bytes: Vec<u8>,
}

/// A parsed, integrity-checked `.vkcache` artifact.
#[derive(Default)]
pub(in crate::runtime) struct VkCacheFile {
	sections: BTreeMap<u32, VkCacheSection>,
}

impl VkCacheFile {
	/// Load and fully validate the artifact at `path` for `scope`.
	///
	/// Missing files, malformed bytes, integrity failures and scope mismatches
	/// all fail the same way; callers decide whether an empty cache is the
	/// fallback. No partially accepted artifact is ever returned.
	pub(in crate::runtime) fn open(path: &Path, scope: &VkCacheScope) -> Result<Self> {
		// Bound allocation before reading, including a file that grows after
		// metadata inspection. Decode repeats the limit for in-memory callers.
		let input = std::fs::File::open(path).map_err(|error| Error::io("vkcache open", error))?;
		if input
			.metadata()
			.map_err(|error| Error::io("vkcache metadata", error))?
			.len()
			> VKCACHE_MAX_FILE_BYTES
		{
			return Err(Error::invalid_argument(
				"vkcache artifact exceeds file limit",
			));
		}
		let mut bytes = Vec::new();
		input
			.take(VKCACHE_MAX_FILE_BYTES + 1)
			.read_to_end(&mut bytes)
			.map_err(|error| Error::io("vkcache read", error))?;
		let file = Self::decode(&bytes)?;
		match file.sections.get(&SECTION_SCOPE) {
			Some(section) if section.version == 1 && scope.matches(&section.bytes) => {}
			_ => {
				return Err(Error::invalid_argument(
					"vkcache artifact scope does not match the selected device",
				));
			}
		}
		Ok(file)
	}

	/// Look up one typed section.
	pub(in crate::runtime) fn section(&self, section_type: u32) -> Option<&VkCacheSection> {
		self.sections.get(&section_type)
	}

	/// Parse and validate an artifact without scope or filesystem I/O.
	fn decode(bytes: &[u8]) -> Result<Self> {
		let corrupt = || Error::invalid_argument("vkcache artifact is malformed");
		if bytes.len() < VKCACHE_HEADER_SIZE as usize || bytes.len() as u64 > VKCACHE_MAX_FILE_BYTES {
			return Err(corrupt());
		}
		let mut reader = SectionReader::new(&bytes[..VKCACHE_HEADER_SIZE as usize]);
		if reader.u32() != Some(VKCACHE_MAGIC)
			|| reader.u16() != Some(VKCACHE_FORMAT_MAJOR)
			|| reader.u16() != Some(VKCACHE_FORMAT_MINOR)
			|| reader.u32() != Some(VKCACHE_HEADER_SIZE as u32)
			|| reader.u32() != Some(VKCACHE_CHECKSUM_FNV1A64)
		{
			return Err(corrupt());
		}
		let _scope_key = reader.u64().ok_or_else(corrupt)?;
		if reader.u64() != Some(bytes.len() as u64) {
			return Err(corrupt());
		}
		let directory_checksum = reader.u64().ok_or_else(corrupt)?;
		let section_count = reader.u32().ok_or_else(corrupt)? as u64;
		if reader.u32() != Some(0) || section_count > VKCACHE_MAX_SECTIONS as u64 {
			return Err(corrupt());
		}
		let directory_bytes = section_count
			.checked_mul(VKCACHE_ENTRY_SIZE)
			.ok_or_else(corrupt)?;
		let directory_end = VKCACHE_HEADER_SIZE
			.checked_add(directory_bytes)
			.ok_or_else(corrupt)?;
		if directory_end > bytes.len() as u64 {
			return Err(corrupt());
		}
		let directory = &bytes[VKCACHE_HEADER_SIZE as usize..directory_end as usize];
		if fnv1a64(directory) != directory_checksum {
			return Err(corrupt());
		}

		struct Entry {
			section_type: u32,
			version: u32,
			offset: u64,
			length: u64,
			checksum: u64,
			required: bool,
		}

		let mut entries = Vec::with_capacity(section_count as usize);
		let mut entries_reader = SectionReader::new(directory);
		let mut last_type = 0u32;
		for index in 0..section_count {
			let section_type = entries_reader.u32().ok_or_else(corrupt)?;
			let version = entries_reader.u32().ok_or_else(corrupt)?;
			let offset = entries_reader.u64().ok_or_else(corrupt)?;
			let stored = entries_reader.u64().ok_or_else(corrupt)?;
			let decoded = entries_reader.u64().ok_or_else(corrupt)?;
			let checksum = entries_reader.u64().ok_or_else(corrupt)?;
			let flags = entries_reader.u32().ok_or_else(corrupt)?;
			let reserved = entries_reader.u32().ok_or_else(corrupt)?;
			if index > 0 && section_type <= last_type {
				return Err(corrupt());
			}
			last_type = section_type;
			if stored != decoded
				|| decoded > VKCACHE_MAX_SECTION_BYTES
				|| offset < directory_end
				|| offset % VKCACHE_ALIGNMENT != 0
				|| offset.checked_add(stored).ok_or_else(corrupt)? > bytes.len() as u64
				|| flags & !SECTION_FLAG_REQUIRED != 0
				|| reserved != 0
			{
				return Err(corrupt());
			}
			entries.push(Entry {
				section_type,
				version,
				offset,
				length: stored,
				checksum,
				required: flags & SECTION_FLAG_REQUIRED != 0,
			});
		}

		// Extents must not overlap; entries arrive sorted by type, so check
		// payload ranges independently of directory order.
		let mut extents: Vec<(u64, u64)> = entries
			.iter()
			.map(|entry| (entry.offset, entry.offset + entry.length))
			.collect();
		extents.sort_unstable();
		for pair in extents.windows(2) {
			if pair[1].0 < pair[0].1 {
				return Err(corrupt());
			}
		}

		let mut sections = BTreeMap::new();
		for entry in entries {
			if entry.section_type > SECTION_TYPE_MAX_KNOWN && entry.required {
				return Err(corrupt());
			}
			let payload = &bytes[entry.offset as usize..(entry.offset + entry.length) as usize];
			if fnv1a64(payload) != entry.checksum {
				return Err(corrupt());
			}
			// Unknown optional payloads stay opaque so a merge write preserves
			// them verbatim for newer readers.
			sections.insert(
				entry.section_type,
				VkCacheSection {
					version: entry.version,
					bytes: payload.to_vec(),
				},
			);
		}
		Ok(Self { sections })
	}
}

/// One writer-owned section replacement inside a merge commit.
pub(in crate::runtime) struct SectionUpdate {
	/// Section type this update owns.
	pub(in crate::runtime) section_type: u32,
	/// Payload schema version for the new bytes.
	pub(in crate::runtime) version: u32,
	/// `Some` replaces the section; `None` or an empty payload removes it.
	pub(in crate::runtime) payload: Option<Vec<u8>>,
}

impl SectionUpdate {
	/// Construct a replacement update.
	pub(in crate::runtime) fn replace(section_type: u32, version: u32, payload: Vec<u8>) -> Self {
		Self {
			section_type,
			version,
			payload: if payload.is_empty() {
				None
			} else {
				Some(payload)
			},
		}
	}

	/// Construct a section removal update.
	pub(in crate::runtime) fn remove(section_type: u32) -> Self {
		Self {
			section_type,
			version: 0,
			payload: None,
		}
	}
}

/// Merge `updates` into the artifact at `path` and commit atomically.
///
/// Under the advisory lock the previously committed artifact is re-read and
/// every section not owned by `updates` is preserved verbatim — including
/// optional sections newer than this implementation. A prior artifact with a
/// different scope is discarded entirely rather than merged across scopes.
/// Returns `Ok(())` without writing when another writer holds a live lock or
/// the updates produce no sections; I/O and serialization failures propagate
/// with the committed artifact left untouched.
pub(in crate::runtime) fn update(
	path: &Path,
	scope: &VkCacheScope,
	updates: &[SectionUpdate],
) -> Result<()> {
	if let Some(dir) = path.parent() {
		crate::Filesystem::create_directories(&dir)?;
	}
	let _lock = match CacheLock::acquire(path)? {
		Some(lock) => lock,
		None => return Ok(()),
	};

	// Merge against the last committed artifact only when it belongs to this
	// scope; a foreign or corrupt artifact contributes no preserved sections.
	let mut file = VkCacheFile::open(path, scope).unwrap_or_default();
	for update in updates {
		debug_assert!(update.section_type != SECTION_SCOPE);
		match &update.payload {
			Some(payload) => {
				file.sections.insert(
					update.section_type,
					VkCacheSection {
						version: update.version,
						bytes: payload.clone(),
					},
				);
			}
			None => {
				file.sections.remove(&update.section_type);
			}
		}
	}
	// The scope record is always re-emitted by the committing writer.
	file.sections.insert(
		SECTION_SCOPE,
		VkCacheSection {
			version: 1,
			bytes: scope.encode().to_vec(),
		},
	);
	let bytes = serialize(&file.sections)?;
	atomic_commit(path, &bytes)
}

/// Serialize a section map into one complete artifact.
fn serialize(sections: &BTreeMap<u32, VkCacheSection>) -> Result<Vec<u8>> {
	if sections.len() > VKCACHE_MAX_SECTIONS as usize {
		return Err(Error::invalid_argument(
			"vkcache section count exceeds limit",
		));
	}
	for (section_type, section) in sections {
		if *section_type == 0 {
			return Err(Error::invalid_argument(
				"vkcache section type zero is invalid",
			));
		}
		if section.bytes.len() as u64 > VKCACHE_MAX_SECTION_BYTES {
			return Err(Error::invalid_argument(
				"vkcache section exceeds size limit",
			));
		}
	}

	// Each payload begins 16-aligned relative to the file start; `cursor`
	// tracks the next free file offset while directory entries are emitted.
	let directory_bytes = VKCACHE_ENTRY_SIZE * sections.len() as u64;
	let mut cursor =
		(VKCACHE_HEADER_SIZE + directory_bytes).div_ceil(VKCACHE_ALIGNMENT) * VKCACHE_ALIGNMENT;
	let mut directory = Vec::with_capacity(directory_bytes as usize);
	let mut body = vec![0; (cursor - VKCACHE_HEADER_SIZE - directory_bytes) as usize];
	for (section_type, section) in sections {
		let mut writer = SectionWriter::unbound();
		writer.u32(*section_type);
		writer.u32(section.version);
		writer.u64(cursor);
		writer.u64(section.bytes.len() as u64);
		writer.u64(section.bytes.len() as u64);
		writer.u64(fnv1a64(&section.bytes));
		writer.u32(if *section_type == SECTION_SCOPE {
			SECTION_FLAG_REQUIRED
		} else {
			0
		});
		writer.u32(0);
		directory.extend_from_slice(&writer.into_bytes());
		body.extend_from_slice(&section.bytes);
		let end = cursor + section.bytes.len() as u64;
		cursor = end.div_ceil(VKCACHE_ALIGNMENT) * VKCACHE_ALIGNMENT;
		body.resize((cursor - VKCACHE_HEADER_SIZE - directory_bytes) as usize, 0);
	}

	let mut header = SectionWriter::unbound();
	header.u32(VKCACHE_MAGIC);
	header.u16(VKCACHE_FORMAT_MAJOR);
	header.u16(VKCACHE_FORMAT_MINOR);
	header.u32(VKCACHE_HEADER_SIZE as u32);
	header.u32(VKCACHE_CHECKSUM_FNV1A64);
	// scope_key mirrors the SCOPE payload's first field for index lookups.
	header.u64(scope_key_of(sections));
	header.u64(VKCACHE_HEADER_SIZE + directory_bytes + body.len() as u64);
	header.u64(fnv1a64(&directory));
	header.u32(sections.len() as u32);
	header.u32(0);

	let mut out = header.into_bytes();
	out.extend_from_slice(&directory);
	out.extend_from_slice(&body);
	if out.len() as u64 > VKCACHE_MAX_FILE_BYTES {
		return Err(Error::invalid_argument(
			"vkcache artifact exceeds file limit",
		));
	}
	Ok(out)
}

/// Scope key recorded in the header — the first field of the SCOPE payload.
fn scope_key_of(sections: &BTreeMap<u32, VkCacheSection>) -> u64 {
	sections
		.get(&SECTION_SCOPE)
		.and_then(|section| section.bytes.get(..8))
		.map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap_or_default()))
		.unwrap_or(0)
}

/// OS-owned advisory lock on a persistent sibling inode. Never unlink it:
/// another process may already have opened that inode. The OS releases the
/// lock after a crash; elapsed wall time cannot revoke a live writer's lock.
struct CacheLock {
	_file: std::fs::File,
}

impl CacheLock {
	fn acquire(artifact: &StdPath) -> Result<Option<Self>> {
		let file = std::fs::File::options()
			.write(true)
			.create(true)
			.truncate(false)
			.open(artifact.with_extension("lock"))
			.map_err(|error| Error::io("vkcache lock open", error))?;
		match file.try_lock() {
			Ok(()) => Ok(Some(Self { _file: file })),
			Err(std::fs::TryLockError::WouldBlock) => Ok(None),
			Err(std::fs::TryLockError::Error(error)) => Err(Error::io("vkcache lock", error)),
		}
	}
}

/// Commit `bytes` to `path` atomically: same-directory temporary file,
/// `sync_all`, rename, then a best-effort directory sync.
fn atomic_commit(path: &StdPath, bytes: &[u8]) -> Result<()> {
	let tmp = path.with_extension("tmp");
	{
		let mut file = std::fs::File::create(&tmp).map_err(|e| Error::io("vkcache tmp create", e))?;
		file
			.write_all(bytes)
			.map_err(|e| Error::io("vkcache tmp write", e))?;
		file
			.sync_all()
			.map_err(|e| Error::io("vkcache tmp sync", e))?;
	}
	std::fs::rename(&tmp, path).map_err(|e| Error::io("vkcache rename", e))?;
	if let Some(dir) = path.parent()
		&& let Ok(dir_file) = std::fs::File::open(dir)
	{
		let _ = dir_file.sync_all();
	}
	Ok(())
}

/// Validate the standard `VkPipelineCacheHeaderVersionOne` prefix of a driver
/// blob against this scope without passing foreign bytes to the driver.
///
/// The header is `header_size:u32, header_version:u32, vendor_id:u32,
/// device_id:u32, pipeline_cache_uuid:[u8;16]`; only version 1 blobs whose
/// device identity matches are trusted as seeds.
pub(in crate::runtime) fn driver_blob_matches_scope(bytes: &[u8], scope: &VkCacheScope) -> bool {
	if bytes.len() < 32 {
		return false;
	}
	let mut reader = SectionReader::new(bytes);
	let header_size = reader.u32();
	let header_version = reader.u32();
	let vendor_id = reader.u32();
	let device_id = reader.u32();
	let uuid = reader.bytes(16);
	header_version == Some(1)
		&& header_size.is_some_and(|size| (32..=bytes.len() as u32).contains(&size))
		&& vendor_id == Some(scope.vendor_id)
		&& device_id == Some(scope.device_id)
		&& uuid == Some(&scope.pipeline_cache_uuid[..])
}

// ── Bounded cursor codecs ────────────────────────────────────────────────────

/// Little-endian reader; every read is bounds-checked and advancing.
struct SectionReader<'a> {
	bytes: &'a [u8],
	offset: usize,
}

impl<'a> SectionReader<'a> {
	fn new(bytes: &'a [u8]) -> Self {
		Self { bytes, offset: 0 }
	}

	fn bytes(&mut self, count: usize) -> Option<&'a [u8]> {
		let end = self.offset.checked_add(count)?;
		let out = self.bytes.get(self.offset..end)?;
		self.offset = end;
		Some(out)
	}

	fn u16(&mut self) -> Option<u16> {
		Some(u16::from_le_bytes(self.bytes(2)?.try_into().ok()?))
	}

	fn u32(&mut self) -> Option<u32> {
		Some(u32::from_le_bytes(self.bytes(4)?.try_into().ok()?))
	}

	fn u64(&mut self) -> Option<u64> {
		Some(u64::from_le_bytes(self.bytes(8)?.try_into().ok()?))
	}
}

/// Little-endian writer for fixed buffers and owned vectors.
struct SectionWriter<'a> {
	fixed: Option<&'a mut [u8]>,
	offset: usize,
	owned: Vec<u8>,
}

impl<'a> SectionWriter<'a> {
	fn new(fixed: &'a mut [u8]) -> Self {
		Self {
			fixed: Some(fixed),
			offset: 0,
			owned: Vec::new(),
		}
	}

	fn unbound() -> Self {
		Self {
			fixed: None,
			offset: 0,
			owned: Vec::new(),
		}
	}

	fn bytes(&mut self, bytes: &[u8]) {
		match &mut self.fixed {
			Some(fixed) => {
				fixed[self.offset..self.offset + bytes.len()].copy_from_slice(bytes);
				self.offset += bytes.len();
			}
			None => self.owned.extend_from_slice(bytes),
		}
	}

	fn u16(&mut self, value: u16) {
		self.bytes(&value.to_le_bytes());
	}

	fn u32(&mut self, value: u32) {
		self.bytes(&value.to_le_bytes());
	}

	fn u64(&mut self, value: u64) {
		self.bytes(&value.to_le_bytes());
	}

	fn into_bytes(self) -> Vec<u8> {
		self.owned
	}
}

/// FNV-1a 64 — the existing repository-stable checksum vocabulary.
fn fnv1a64(bytes: &[u8]) -> u64 {
	let mut hash = StableHash::new();
	hash.bytes(bytes);
	hash.finish()
}

struct StableHash(u64);

impl StableHash {
	const fn new() -> Self {
		Self(0xcbf2_9ce4_8422_2325)
	}

	fn bytes(&mut self, bytes: &[u8]) {
		for byte in bytes {
			self.0 ^= u64::from(*byte);
			self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
		}
	}

	fn u32(&mut self, value: u32) {
		self.bytes(&value.to_le_bytes());
	}

	fn u64(&mut self, value: u64) {
		self.bytes(&value.to_le_bytes());
	}

	const fn finish(self) -> u64 {
		self.0
	}
}

// ── JIT artifact sections ────────────────────────────────────────────────────

/// Record codecs for the JIT artifact index and SPIR-V table sections.
///
/// `artifact_id` is FNV-1a64 over the artifact key, descriptor profile and
/// emitter version and joins each index record to its SPIR-V payload. These
/// sections are consumed by the compiler laboratory; production publication
/// remains gated, so the codec is exercised under tests.
pub(in crate::runtime) mod jit {
	use super::{Error, Result, SectionReader, SectionWriter, StableHash};

	/// Hard bounds for one JIT index.
	const MAX_JIT_ARTIFACTS: u32 = 4096;
	const MAX_JIT_KEY_BYTES: u32 = 4096;
	const MAX_JIT_COMPILER_BYTES: u32 = 256;
	/// Hard bounds for the SPIR-V table.
	const MAX_JIT_SPIRV_WORDS: u32 = 1_048_576;

	/// Identity/ABI record for one specialized artifact.
	#[derive(Clone, Debug, PartialEq)]
	pub(in crate::runtime) struct JitIndexRecord {
		/// Deterministic identity joining index and SPIR-V table.
		pub artifact_id: u64,
		/// Exact `Program::artifact_key` bytes — the complete identity.
		pub key: Vec<u8>,
		/// Descriptor profile the artifact was compiled against.
		pub descriptor_profile: u32,
		/// SPIR-V target environment identifier.
		pub spirv_environment: u32,
		/// Emit-time workgroup size.
		pub workgroup_size: u32,
		/// Storage-buffer binding count.
		pub binding_count: u32,
		/// Push-constant block size.
		pub push_bytes: u32,
		/// Emitter generation that produced the artifact.
		pub emitter_version: u32,
		/// Compiler identity (name + version), bounded.
		pub compiler: Vec<u8>,
		/// Record flags; format 1 uses none.
		pub flags: u32,
	}

	/// SPIR-V payload record for one artifact.
	#[derive(Clone, Debug, PartialEq)]
	pub(in crate::runtime) struct JitSpirvRecord {
		/// Identity matching one index record.
		pub artifact_id: u64,
		/// Complete validated SPIR-V words.
		pub words: Vec<u32>,
	}

	/// Deterministic artifact identity for one emitted program.
	pub(in crate::runtime) fn artifact_id(
		key: &[u8],
		descriptor_profile: u32,
		emitter_version: u32,
	) -> u64 {
		let mut hash = StableHash::new();
		hash.bytes(b"oavk.jit\x01");
		hash.bytes(key);
		hash.u32(descriptor_profile);
		hash.u32(emitter_version);
		hash.finish()
	}

	/// Encode the index section payload.
	pub(in crate::runtime) fn encode_index(records: &[JitIndexRecord]) -> Result<Vec<u8>> {
		let invalid = || Error::invalid_argument("JIT artifact index exceeds serialized limits");
		if records.len() > MAX_JIT_ARTIFACTS as usize {
			return Err(invalid());
		}
		let mut ids = std::collections::BTreeSet::new();
		for record in records {
			if record.key.len() > MAX_JIT_KEY_BYTES as usize
				|| record.compiler.len() > MAX_JIT_COMPILER_BYTES as usize
				|| !ids.insert(record.artifact_id)
			{
				return Err(invalid());
			}
		}
		let mut writer = SectionWriter::unbound();
		writer.u32(records.len() as u32);
		writer.u32(0);
		for record in records {
			writer.u64(record.artifact_id);
			writer.u32(record.key.len() as u32);
			writer.bytes(&record.key);
			writer.u32(record.descriptor_profile);
			writer.u32(record.spirv_environment);
			writer.u32(record.workgroup_size);
			writer.u32(record.binding_count);
			writer.u32(record.push_bytes);
			writer.u32(record.emitter_version);
			writer.u32(record.compiler.len() as u32);
			writer.bytes(&record.compiler);
			writer.u32(record.flags);
		}
		Ok(writer.into_bytes())
	}

	/// Decode the index section payload with strict bounds.
	pub(in crate::runtime) fn decode_index(bytes: &[u8]) -> Result<Vec<JitIndexRecord>> {
		let invalid = || Error::invalid_argument("JIT artifact index is malformed");
		let mut reader = SectionReader::new(bytes);
		let count = reader.u32().ok_or_else(invalid)?;
		if reader.u32() != Some(0) || count > MAX_JIT_ARTIFACTS {
			return Err(invalid());
		}
		let mut records = Vec::with_capacity(count as usize);
		let mut ids = std::collections::BTreeSet::new();
		for _ in 0..count {
			let artifact_id = reader.u64().ok_or_else(invalid)?;
			let key_len = reader.u32().ok_or_else(invalid)?;
			if key_len == 0 || key_len > MAX_JIT_KEY_BYTES {
				return Err(invalid());
			}
			let key = reader.bytes(key_len as usize).ok_or_else(invalid)?.to_vec();
			let descriptor_profile = reader.u32().ok_or_else(invalid)?;
			let spirv_environment = reader.u32().ok_or_else(invalid)?;
			let workgroup_size = reader.u32().ok_or_else(invalid)?;
			let binding_count = reader.u32().ok_or_else(invalid)?;
			let push_bytes = reader.u32().ok_or_else(invalid)?;
			let emitter_version = reader.u32().ok_or_else(invalid)?;
			let compiler_len = reader.u32().ok_or_else(invalid)?;
			if compiler_len > MAX_JIT_COMPILER_BYTES {
				return Err(invalid());
			}
			let compiler = reader
				.bytes(compiler_len as usize)
				.ok_or_else(invalid)?
				.to_vec();
			let flags = reader.u32().ok_or_else(invalid)?;
			if !ids.insert(artifact_id) {
				return Err(invalid());
			}
			records.push(JitIndexRecord {
				artifact_id,
				key,
				descriptor_profile,
				spirv_environment,
				workgroup_size,
				binding_count,
				push_bytes,
				emitter_version,
				compiler,
				flags,
			});
		}
		if reader.offset != bytes.len() {
			return Err(invalid());
		}
		Ok(records)
	}

	/// Encode the SPIR-V table section payload.
	pub(in crate::runtime) fn encode_spirv(records: &[JitSpirvRecord]) -> Result<Vec<u8>> {
		let invalid = || Error::invalid_argument("JIT SPIR-V table exceeds serialized limits");
		if records.len() > MAX_JIT_ARTIFACTS as usize {
			return Err(invalid());
		}
		let mut ids = std::collections::BTreeSet::new();
		for record in records {
			if record.words.len() > MAX_JIT_SPIRV_WORDS as usize || !ids.insert(record.artifact_id) {
				return Err(invalid());
			}
		}
		let mut writer = SectionWriter::unbound();
		writer.u32(records.len() as u32);
		writer.u32(0);
		for record in records {
			writer.u64(record.artifact_id);
			writer.u32(record.words.len() as u32);
			writer.u32(0);
			for word in &record.words {
				writer.u32(*word);
			}
		}
		Ok(writer.into_bytes())
	}

	/// Decode the SPIR-V table section payload with strict bounds.
	pub(in crate::runtime) fn decode_spirv(bytes: &[u8]) -> Result<Vec<JitSpirvRecord>> {
		let invalid = || Error::invalid_argument("JIT SPIR-V table is malformed");
		let mut reader = SectionReader::new(bytes);
		let count = reader.u32().ok_or_else(invalid)?;
		if reader.u32() != Some(0) || count > MAX_JIT_ARTIFACTS {
			return Err(invalid());
		}
		let mut records = Vec::with_capacity(count as usize);
		let mut ids = std::collections::BTreeSet::new();
		for _ in 0..count {
			let artifact_id = reader.u64().ok_or_else(invalid)?;
			let word_count = reader.u32().ok_or_else(invalid)?;
			if reader.u32() != Some(0)
				|| word_count == 0
				|| word_count > MAX_JIT_SPIRV_WORDS
				|| !ids.insert(artifact_id)
			{
				return Err(invalid());
			}
			let mut words = Vec::with_capacity(word_count as usize);
			for _ in 0..word_count {
				words.push(reader.u32().ok_or_else(invalid)?);
			}
			records.push(JitSpirvRecord { artifact_id, words });
		}
		if reader.offset != bytes.len() {
			return Err(invalid());
		}
		Ok(records)
	}
	/// Hard bound for one route-observation section.
	const MAX_ROUTE_OBSERVATIONS: u32 = 65_536;

	/// Selected-route vocabulary for `FUSION_ROUTE_OBSERVATIONS` records;
	/// the wire also admits `0` (unfused) for future producers.
	pub(in crate::runtime) const ROUTE_TAPE: u32 = 1;
	pub(in crate::runtime) const ROUTE_SPECIALIZED: u32 = 2;

	/// Route statistics for one fusion signature on this device scope.
	#[derive(Clone, Debug, PartialEq)]
	pub(in crate::runtime) struct RouteObservationRecord {
		/// `signature_key` of the observed `Program::canonical_key`.
		pub signature_key: u64,
		/// Currently selected route (`ROUTE_*`).
		pub selected_route: u32,
		/// Timing samples contributing to the recorded medians.
		pub sample_count: u32,
		/// Median GPU duration of the unfused route; 0 when unmeasured.
		pub unfused_p50_ns: u64,
		/// Median GPU duration of the tape route; 0 when unmeasured.
		pub tape_p50_ns: u64,
		/// Median GPU duration of the specialized route; 0 when unmeasured.
		pub specialized_p50_ns: u64,
		/// Producer's monotonic observation counter at write time.
		pub last_step: u64,
	}

	/// Deterministic observation identity for one canonical program signature.
	pub(in crate::runtime) fn signature_key(canonical_key: &[u8]) -> u64 {
		let mut hash = StableHash::new();
		hash.bytes(b"oavk.fusion\x01");
		hash.bytes(canonical_key);
		hash.finish()
	}

	/// Encode the route-observation section payload: fixed 48-byte records.
	pub(in crate::runtime) fn encode_observations(
		records: &[RouteObservationRecord],
	) -> Result<Vec<u8>> {
		if records.len() > MAX_ROUTE_OBSERVATIONS as usize {
			return Err(Error::invalid_argument(
				"route observations exceed serialized limits",
			));
		}
		let mut writer = SectionWriter::unbound();
		writer.u32(records.len() as u32);
		writer.u32(0);
		for record in records {
			writer.u64(record.signature_key);
			writer.u32(record.selected_route);
			writer.u32(record.sample_count);
			writer.u64(record.unfused_p50_ns);
			writer.u64(record.tape_p50_ns);
			writer.u64(record.specialized_p50_ns);
			writer.u64(record.last_step);
		}
		Ok(writer.into_bytes())
	}

	/// Decode the route-observation section payload with strict bounds.
	pub(in crate::runtime) fn decode_observations(
		bytes: &[u8],
	) -> Result<Vec<RouteObservationRecord>> {
		let invalid = || Error::invalid_argument("route observation section is malformed");
		let mut reader = SectionReader::new(bytes);
		let count = reader.u32().ok_or_else(invalid)?;
		if reader.u32() != Some(0) || count > MAX_ROUTE_OBSERVATIONS {
			return Err(invalid());
		}
		let mut records = Vec::with_capacity(count as usize);
		let mut signatures = std::collections::BTreeSet::new();
		for _ in 0..count {
			let signature_key = reader.u64().ok_or_else(invalid)?;
			let selected_route = reader.u32().ok_or_else(invalid)?;
			let sample_count = reader.u32().ok_or_else(invalid)?;
			let unfused_p50_ns = reader.u64().ok_or_else(invalid)?;
			let tape_p50_ns = reader.u64().ok_or_else(invalid)?;
			let specialized_p50_ns = reader.u64().ok_or_else(invalid)?;
			let last_step = reader.u64().ok_or_else(invalid)?;
			if selected_route > ROUTE_SPECIALIZED || !signatures.insert(signature_key) {
				return Err(invalid());
			}
			records.push(RouteObservationRecord {
				signature_key,
				selected_route,
				sample_count,
				unfused_p50_ns,
				tape_p50_ns,
				specialized_p50_ns,
				last_step,
			});
		}
		if reader.offset != bytes.len() {
			return Err(invalid());
		}
		Ok(records)
	}
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/vkcache_unit.rs"]
mod tests;
