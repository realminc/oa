//! Content-addressed dynamic artifacts and existing .vkcache codecs.
//! Donor: dna/kernel/jit.rs; extraction only, with unchanged keys and wire format.
use super::{
	CompiledChain, Program, SlangCompiler, jit_error, validate_program_abi, validate_with_spirv_tools,
};
use crate::Result;
use std::{collections::BTreeMap, sync::Arc};

/// Content-addressed artifact index: one artifact per key/profile.
/// Extent-free keys let one artifact serve every legal element count.
pub(in crate::runtime) struct JitArtifacts {
	chains: BTreeMap<(bool, Vec<u8>), Arc<CompiledChain>>,
	compiler_version: String,
}

impl JitArtifacts {
	pub(in crate::runtime) const MAX_CHAINS: usize = 64;

	pub(in crate::runtime) fn new() -> Self {
		Self {
			chains: BTreeMap::new(),
			compiler_version: String::new(),
		}
	}

	pub(in crate::runtime) fn len(&self) -> usize {
		self.chains.len()
	}

	pub(in crate::runtime) fn get(
		&self,
		program: &Program,
		bounded: bool,
	) -> Option<Arc<CompiledChain>> {
		self.chains.get(&(bounded, program.artifact_key())).cloned()
	}

	/// Look up a compiled artifact by its emission key when the owning
	/// `Program` is not available, as at record-time pipeline publication.
	pub(in crate::runtime) fn get_key(
		&self,
		bounded: bool,
		key: &[u8],
	) -> Option<Arc<CompiledChain>> {
		self.chains.get(&(bounded, key.to_vec())).cloned()
	}

	/// Insert a chain compiled off the admission path; dedup keeps the first.
	pub(in crate::runtime) fn insert(
		&mut self,
		compiler_version: &str,
		chain: CompiledChain,
	) -> Result<Arc<CompiledChain>> {
		let key = (chain.bounded, chain.key.clone());
		if let Some(chain) = self.chains.get(&key) {
			return Ok(chain.clone());
		}
		if self.chains.len() >= Self::MAX_CHAINS {
			return Err(jit_error("specialization artifact bound reached"));
		}
		self.compiler_version = compiler_version.to_owned();
		let chain = Arc::new(chain);
		self.chains.insert(key, chain.clone());
		Ok(chain)
	}

	/// Returns the artifact and whether this call compiled it (a cache miss).
	#[cfg_attr(
		not(test),
		expect(
			dead_code,
			reason = "production misses compile on the background admission worker"
		)
	)]
	pub(in crate::runtime) fn chain(
		&mut self,
		compiler: &SlangCompiler,
		program: &Program,
		bounded: bool,
	) -> Result<(Arc<CompiledChain>, bool)> {
		let key = (bounded, program.artifact_key());
		if let Some(chain) = self.chains.get(&key) {
			return Ok((chain.clone(), false));
		}
		if self.chains.len() >= Self::MAX_CHAINS {
			return Err(jit_error("specialization artifact bound reached"));
		}
		self.compiler_version = compiler.version.clone();
		let chain = Arc::new(compiler.compile_program(program, bounded)?);
		self.chains.insert(key, chain.clone());
		Ok((chain, true))
	}

	/// Emitter generation matching `Program::artifact_key`'s `v1` prefix.
	const EMITTER_VERSION: u32 = 1;

	/// Encode every artifact into `.vkcache` section updates for the Engine's
	/// single atomic commit. The index is useless without its SPIR-V table and
	/// vice versa, so both are produced or neither is.
	pub(in crate::runtime) fn section_updates(
		&self,
	) -> Result<Vec<crate::runtime::device::vkcache::SectionUpdate>> {
		use crate::runtime::device::vkcache::{SectionUpdate, jit};
		if self.chains.is_empty() {
			return Ok(Vec::new());
		}
		let mut index = Vec::with_capacity(self.chains.len());
		let mut spirv = Vec::with_capacity(self.chains.len());
		for ((bounded, key), chain) in &self.chains {
			let descriptor_profile = u32::from(*bounded);
			let artifact_id = jit::artifact_id(key, descriptor_profile, Self::EMITTER_VERSION);
			index.push(jit::JitIndexRecord {
				artifact_id,
				key: key.clone(),
				descriptor_profile,
				spirv_environment: u32::from(!*bounded),
				workgroup_size: 256,
				binding_count: chain.bindings,
				push_bytes: (chain.bindings + 1) * 4,
				emitter_version: Self::EMITTER_VERSION,
				compiler: self.compiler_version.clone().into_bytes(),
				flags: 0,
			});
			spirv.push(jit::JitSpirvRecord {
				artifact_id,
				words: chain.words().to_vec(),
			});
		}
		Ok(vec![
			SectionUpdate::replace(
				crate::runtime::device::vkcache::SECTION_JIT_ARTIFACT_INDEX,
				1,
				jit::encode_index(&index)?,
			),
			SectionUpdate::replace(
				crate::runtime::device::vkcache::SECTION_JIT_SPIRV_ARTIFACTS,
				1,
				jit::encode_spirv(&spirv)?,
			),
		])
	}

	/// Restore persisted artifacts for this scope.
	///
	/// Persisted bytes are qualified state, never trusted code: every restored
	/// module passes `spirv-val` and the ABI reflection checks again before it
	/// may serve a dispatch. Malformed sections decode to an error; individual
	/// records that fail revalidation are skipped so one damaged entry cannot
	/// discard unrelated artifacts.
	pub(in crate::runtime) fn restore(
		path: &crate::Path,
		scope: &crate::runtime::device::vkcache::VkCacheScope,
	) -> Result<Self> {
		use crate::runtime::device::vkcache::{VkCacheFile, jit};
		let mut restored = Self::new();
		let Ok(file) = VkCacheFile::open(path, scope) else {
			return Ok(restored);
		};
		let index = match file.section(crate::runtime::device::vkcache::SECTION_JIT_ARTIFACT_INDEX) {
			Some(section) if section.version == 1 => jit::decode_index(&section.bytes)?,
			_ => return Ok(restored),
		};
		let table = match file.section(crate::runtime::device::vkcache::SECTION_JIT_SPIRV_ARTIFACTS) {
			Some(section) if section.version == 1 => jit::decode_spirv(&section.bytes)?,
			_ => return Ok(restored),
		};
		let table: BTreeMap<u64, Vec<u32>> = table
			.into_iter()
			.map(|record| (record.artifact_id, record.words))
			.collect();
		for record in index {
			if restored.chains.len() >= Self::MAX_CHAINS {
				break;
			}
			let bounded = match record.descriptor_profile {
				0 => false,
				1 => true,
				_ => continue,
			};
			if bounded != cfg!(oa_profile = "compatibility")
				|| !(2..=31).contains(&record.binding_count)
				|| record.emitter_version != Self::EMITTER_VERSION
				|| record.spirv_environment != u32::from(!bounded)
				|| record.workgroup_size != 256
				|| record.push_bytes != (record.binding_count + 1) * 4
				|| jit::artifact_id(
					&record.key,
					record.descriptor_profile,
					record.emitter_version,
				) != record.artifact_id
			{
				continue;
			}
			let Some(words) = table.get(&record.artifact_id) else {
				continue;
			};
			if validate_with_spirv_tools(words, bounded).is_err()
				|| validate_program_abi(words, bounded, record.binding_count as usize).is_err()
			{
				continue;
			}
			if restored.compiler_version.is_empty() && !record.compiler.is_empty() {
				restored.compiler_version = String::from_utf8_lossy(&record.compiler).into_owned();
			}
			restored.chains.insert(
				(bounded, record.key.clone()),
				Arc::new(CompiledChain {
					words: words.clone(),
					key: record.key,
					bindings: record.binding_count,
					compile_time: std::time::Duration::ZERO,
					validation_time: std::time::Duration::ZERO,
					bounded,
				}),
			);
		}
		Ok(restored)
	}
}
