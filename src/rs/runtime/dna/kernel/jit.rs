//! Runtime Slang specialization: in-memory compile, admission validation and
//! `.vkcache` persistence for discovered elementwise `Program`s.
//!
//! Ordinary execution has no compiler dependency; the opt-in `OA_JIT_FUSION`
//! route opens a trusted local Slang library through `OA_SLANG_LIBRARY` or the
//! platform loader path. The deprecated Slang C request API is retained from
//! the Phase 1 laboratory; a production session API remains a separate
//! migration. Every admitted module is validated by `spirv-val` and a strict
//! ABI reflection check before it may serve a dispatch, and the fused tape
//! route remains the canonical fallback whenever admission fails.

use std::{
	collections::{BTreeMap, BTreeSet},
	ffi::{CStr, CString, c_char, c_int, c_void},
	io::Write,
	marker::PhantomData,
	process::{Command, Stdio},
	ptr::NonNull,
	sync::Arc,
};

use crate::{Error, Result};

use crate::runtime::dna::elementwise::Program;

type Handle = *mut c_void;

struct Api {
	create_session: unsafe extern "C" fn(*const c_char) -> Handle,
	destroy_session: unsafe extern "C" fn(Handle),
	create_request: unsafe extern "C" fn(Handle) -> Handle,
	destroy_request: unsafe extern "C" fn(Handle),
	build_tag: unsafe extern "C" fn() -> *const c_char,
	target: unsafe extern "C" fn(Handle, c_int),
	profile: unsafe extern "C" fn(Handle, c_int, u32),
	target_flags: unsafe extern "C" fn(Handle, c_int, u32),
	find_profile: unsafe extern "C" fn(Handle, *const c_char) -> u32,
	find_capability: unsafe extern "C" fn(Handle, *const c_char) -> c_int,
	add_capability: unsafe extern "C" fn(Handle, c_int, c_int),
	add_unit: unsafe extern "C" fn(Handle, c_int, *const c_char) -> c_int,
	add_source: unsafe extern "C" fn(Handle, c_int, *const c_char, *const c_char),
	add_entry: unsafe extern "C" fn(Handle, c_int, *const c_char, c_int) -> c_int,
	compile: unsafe extern "C" fn(Handle) -> c_int,
	diagnostics: unsafe extern "C" fn(Handle) -> *const c_char,
	code: unsafe extern "C" fn(Handle, c_int, *mut usize) -> *const c_void,
}

pub(in crate::runtime) struct SlangCompiler {
	api: Api,
	session: NonNull<c_void>,
	_library: libloading::Library,
	_thread: PhantomData<std::rc::Rc<()>>,
	pub(in crate::runtime) version: String,
}

// SAFETY: the compiler is opened under the device `JitFusion` mutex, then
// moved once into the single compile worker, which serializes every
// subsequent Slang API call; no session or request object is ever used
// concurrently. The retained library outlives all session/request owners
// inside this value.
unsafe impl Send for SlangCompiler {}

pub(in crate::runtime) struct CompiledChain {
	words: Vec<u32>,
	/// Emitted-code identity; empty for the manual Phase 1 recipe artifact.
	pub(in crate::runtime) key: Vec<u8>,
	/// Storage-buffer binding count baked into the descriptor ABI.
	pub(in crate::runtime) bindings: u32,
	pub(in crate::runtime) compile_time: std::time::Duration,
	#[cfg_attr(
		not(test),
		expect(
			dead_code,
			reason = "route observations report these only when adaptive routing is admitted"
		)
	)]
	pub(in crate::runtime) validation_time: std::time::Duration,
	pub(in crate::runtime) bounded: bool,
}

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
			if !(2..=31).contains(&record.binding_count)
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

fn jit_error(message: impl std::fmt::Display) -> Error {
	Error::backend_failure(
		"Slang",
		"runtime specialization",
		std::io::Error::other(message.to_string()),
	)
}

impl SlangCompiler {
	/// Open the Slang library from an explicit trusted path.
	pub(in crate::runtime) fn open(path: &std::path::Path) -> Result<Self> {
		// SAFETY: this explicit entry loads a trusted local Slang library,
		// never a path supplied by a model or an external request.
		let library = unsafe { libloading::Library::new(path) }
			.map_err(|error| jit_error(format!("load {}: {error}", path.display())))?;
		Self::from_library(library)
	}

	/// Resolve the Slang library for the opt-in runtime route:
	/// `OA_SLANG_LIBRARY` names an explicit trusted path, otherwise the
	/// platform loader search finds a system-installed `libslang`.
	pub(in crate::runtime) fn open_default() -> Result<Self> {
		if let Some(path) = std::env::var_os("OA_SLANG_LIBRARY") {
			return Self::open(std::path::Path::new(&path));
		}
		let mut last = None;
		for name in [
			"libslang.so",
			"libslang.so.2",
			"libslang.dylib",
			"slang.dll",
		] {
			// SAFETY: fixed OA-selected library names under the platform loader
			// search; no caller-controlled path is admitted here.
			match unsafe { libloading::Library::new(name) } {
				Ok(library) => return Self::from_library(library),
				Err(error) => last = Some(format!("{name}: {error}")),
			}
		}
		Err(jit_error(format!(
			"no Slang library found ({})",
			last.unwrap_or_default()
		)))
	}

	fn from_library(library: libloading::Library) -> Result<Self> {
		macro_rules! symbol {
			($name:literal) => {
				// SAFETY: signatures match Slang's installed slang[-deprecated].h.
				// `_library` outlives all copied symbols and session/request owners.
				unsafe {
					*library
						.get(concat!($name, "\0").as_bytes())
						.map_err(jit_error)?
				}
			};
		}
		let api = Api {
			create_session: symbol!("spCreateSession"),
			destroy_session: symbol!("spDestroySession"),
			create_request: symbol!("spCreateCompileRequest"),
			destroy_request: symbol!("spDestroyCompileRequest"),
			build_tag: symbol!("spGetBuildTagString"),
			target: symbol!("spSetCodeGenTarget"),
			profile: symbol!("spSetTargetProfile"),
			target_flags: symbol!("spSetTargetFlags"),
			find_profile: symbol!("spFindProfile"),
			find_capability: symbol!("spFindCapability"),
			add_capability: symbol!("spAddTargetCapability"),
			add_unit: symbol!("spAddTranslationUnit"),
			add_source: symbol!("spAddTranslationUnitSourceString"),
			add_entry: symbol!("spAddEntryPoint"),
			compile: symbol!("spCompile"),
			diagnostics: symbol!("spGetDiagnosticOutput"),
			code: symbol!("spGetEntryPointCode"),
		};
		// SAFETY: no deprecated search path is supplied; the library is retained.
		let session = NonNull::new(unsafe { (api.create_session)(std::ptr::null()) })
			.ok_or_else(|| jit_error("cannot create compiler session"))?;
		// SAFETY: Slang owns this null-terminated build tag for the library lifetime.
		let tag = unsafe { (api.build_tag)() };
		let version = if tag.is_null() {
			"unknown".to_owned()
		} else {
			unsafe { CStr::from_ptr(tag) }
				.to_string_lossy()
				.into_owned()
		};
		Ok(Self {
			api,
			session,
			_library: library,
			_thread: PhantomData,
			version,
		})
	}

	pub(in crate::runtime) fn compile_program(
		&self,
		program: &Program,
		bounded: bool,
	) -> Result<CompiledChain> {
		let source = program.slang_source(bounded)?;
		let started = std::time::Instant::now();
		let words = self.compile_source(&source, bounded)?;
		let compile_time = started.elapsed();
		let started = std::time::Instant::now();
		validate_with_spirv_tools(&words, bounded)?;
		let bindings = (program.input_bytes.len() + program.outputs.len()) as u32;
		validate_program_abi(&words, bounded, bindings as usize)?;
		Ok(CompiledChain {
			words,
			key: program.artifact_key(),
			bindings,
			compile_time,
			validation_time: started.elapsed(),
			bounded,
		})
	}

	fn compile_source(&self, source: &str, bounded: bool) -> Result<Vec<u32>> {
		if source.len() > 1024 * 1024 {
			return Err(jit_error("source exceeds specialization limit"));
		}
		let source = CString::new(source).map_err(jit_error)?;
		// SAFETY: the caller serializes session access; Request drops before self.
		let handle = NonNull::new(unsafe { (self.api.create_request)(self.session.as_ptr()) })
			.ok_or_else(|| jit_error("cannot create compile request"))?;
		let request = Request {
			compiler: self,
			handle,
		};
		let raw = request.handle.as_ptr();
		// SAFETY: all functions use the same live request/session. Strings remain
		// valid through compilation. Only in-memory source and code APIs are used.
		let entry = unsafe {
			(self.api.target)(raw, 6); // SLANG_SPIRV
			(self.api.target_flags)(raw, 0, 1 << 10); // direct SPIR-V, no glslang
			let profile = (self.api.find_profile)(self.session.as_ptr(), c"glsl_460".as_ptr());
			let capability = (self.api.find_capability)(
				self.session.as_ptr(),
				if bounded { c"spirv_1_5" } else { c"spirv_1_6" }.as_ptr(),
			);
			if profile == 0 || capability == 0 {
				return Err(jit_error("compiler lacks requested profile/capability"));
			}
			(self.api.profile)(raw, 0, profile);
			(self.api.add_capability)(raw, 0, capability);
			let unit = (self.api.add_unit)(raw, 1, c"oa_jit_chain".as_ptr());
			if unit < 0 {
				return Err(jit_error("translation unit rejected"));
			}
			(self.api.add_source)(raw, unit, c"oa_jit_chain.slang".as_ptr(), source.as_ptr());
			(self.api.add_entry)(raw, unit, c"main".as_ptr(), 6) // SLANG_STAGE_COMPUTE
		};
		if entry < 0 {
			return Err(jit_error("entry point rejected"));
		}
		if unsafe { (self.api.compile)(raw) } < 0 {
			let diagnostics = unsafe { (self.api.diagnostics)(raw) };
			return Err(jit_error(if diagnostics.is_null() {
				"compilation failed without diagnostics".into()
			} else {
				unsafe { CStr::from_ptr(diagnostics) }
					.to_string_lossy()
					.into_owned()
			}));
		}
		let mut byte_count = 0;
		let pointer = unsafe { (self.api.code)(raw, entry, &mut byte_count) };
		if pointer.is_null()
			|| !(20..=16 * 1024 * 1024).contains(&byte_count)
			|| !byte_count.is_multiple_of(4)
		{
			return Err(jit_error("invalid or oversized SPIR-V result"));
		}
		// SAFETY: Slang owns byte_count bytes until the request is destroyed.
		// Copy into aligned owned words before Request::drop.
		let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), byte_count) };
		super::decode_spirv(bytes)
	}
}

impl Drop for SlangCompiler {
	fn drop(&mut self) {
		unsafe { (self.api.destroy_session)(self.session.as_ptr()) };
	}
}

struct Request<'a> {
	compiler: &'a SlangCompiler,
	handle: NonNull<c_void>,
}

impl Drop for Request<'_> {
	fn drop(&mut self) {
		unsafe { (self.compiler.api.destroy_request)(self.handle.as_ptr()) };
	}
}

impl CompiledChain {
	pub(in crate::runtime) fn words(&self) -> &[u32] {
		&self.words
	}
}

fn validate_with_spirv_tools(words: &[u32], bounded: bool) -> Result<()> {
	let executable = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
	let mut child = Command::new(executable)
		.args([
			"--target-env",
			if bounded { "vulkan1.2" } else { "vulkan1.3" },
			"-",
		])
		.stdin(Stdio::piped())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.map_err(jit_error)?;
	let bytes = words
		.iter()
		.flat_map(|word| word.to_le_bytes())
		.collect::<Vec<_>>();
	let write = child.stdin.take().expect("piped stdin").write_all(&bytes);
	let output = child.wait_with_output().map_err(jit_error)?;
	write.map_err(jit_error)?;
	if !output.status.success() {
		return Err(jit_error(format!(
			"SPIR-V validation: {}",
			String::from_utf8_lossy(&output.stderr)
		)));
	}
	Ok(())
}

// Fail closed rather than treating the general static artifact reflector as
// complete validation of arbitrary generated shaders.
pub(in crate::runtime) fn validate_program_abi(
	words: &[u32],
	bounded: bool,
	expected_bindings: usize,
) -> Result<()> {
	if !(2..=31).contains(&expected_bindings)
		|| words.len() < 5
		|| words[0] != 0x0723_0203
		|| words[1] != if bounded { 0x0001_0500 } else { 0x0001_0600 }
		|| super::reflect_push_constant_size(words)? != ((expected_bindings + 1) * 4) as u32
	{
		return Err(jit_error("unexpected module header or push ABI"));
	}
	let mut entries = 0;
	let mut entry_id = None;
	let mut local_sizes = 0;
	let mut local_size_id = None;
	let mut bindings = Vec::new();
	let mut sets = Vec::new();
	let mut arrays = BTreeMap::new();
	let mut pointers = BTreeMap::new();
	let mut variables = BTreeMap::new();
	let mut constants = BTreeMap::new();
	let mut unsigned_words = BTreeSet::new();
	let mut structures = BTreeMap::new();
	let mut offsets = BTreeMap::new();
	let mut cursor = 5;
	while cursor < words.len() {
		let count = (words[cursor] >> 16) as usize;
		if count == 0 || count > words.len() - cursor {
			return Err(jit_error("invalid instruction"));
		}
		let instruction = &words[cursor..cursor + count];
		match instruction[0] as u16 {
			15 => {
				// OpEntryPoint: GLCompute, main
				if count < 5
					|| instruction[1] != 5
					|| instruction[3] != u32::from_le_bytes(*b"main")
					|| instruction[4] != 0
				{
					return Err(jit_error("unexpected entry point"));
				}
				entries += 1;
				entry_id = Some(instruction[2]);
			}
			16 if count == 6 && instruction[2..] == [17, 256, 1, 1] => {
				local_sizes += 1;
				local_size_id = Some(instruction[1]);
			}
			17 => {
				// Capability Shader, plus RuntimeDescriptorArray only on Strict.
				if count != 2 || !(instruction[1] == 1 || (!bounded && instruction[1] == 5302)) {
					return Err(jit_error("unexpected shader capability"));
				}
			}
			71 if count == 4 && instruction[2] == 33 => bindings.push((instruction[1], instruction[3])),
			71 if count == 4 && instruction[2] == 34 => sets.push((instruction[1], instruction[3])),
			28 if count == 4 => {
				arrays.insert(instruction[1], Some(instruction[3]));
			}
			29 if count == 3 => {
				arrays.insert(instruction[1], None);
			}
			32 if count == 4 => {
				pointers.insert(instruction[1], (instruction[2], instruction[3]));
			}
			43 if count == 4 => {
				constants.insert(instruction[2], instruction[3]);
			}
			59 if count >= 4 => {
				variables.insert(instruction[2], (instruction[1], instruction[3]));
			}
			21 if count == 4 && instruction[2..] == [32, 0] => {
				unsigned_words.insert(instruction[1]);
			}
			30 if count >= 2 => {
				structures.insert(instruction[1], instruction[2..].to_vec());
			}
			72 if count == 5 && instruction[3] == 35 => {
				offsets.insert((instruction[1], instruction[2]), instruction[4]);
			}
			_ => {}
		}
		cursor += count;
	}
	if entries != 1
		|| local_sizes != 1
		|| entry_id != local_size_id
		|| bindings.len() != 1
		|| sets != bindings
		|| bindings[0].1 != 0
	{
		return Err(jit_error("unexpected entry/workgroup/descriptor ABI"));
	}
	let &(pointer, storage_class) = variables
		.get(&bindings[0].0)
		.ok_or_else(|| jit_error("missing descriptor variable"))?;
	let &(pointer_class, array) = pointers
		.get(&pointer)
		.ok_or_else(|| jit_error("missing descriptor pointer"))?;
	let length = arrays
		.get(&array)
		.ok_or_else(|| jit_error("descriptor is not an array"))?;
	if storage_class != 12
		|| pointer_class != 12
		|| if bounded {
			length.and_then(|id| constants.get(&id).copied()) != Some(expected_bindings as u32)
		} else {
			length.is_some()
		} {
		return Err(jit_error(
			"unexpected descriptor storage class or array extent",
		));
	}
	let pushes = variables
		.values()
		.filter(|(_, class)| *class == 9)
		.collect::<Vec<_>>();
	if pushes.len() != 1 {
		return Err(jit_error("expected one push block"));
	}
	let &(class, structure) = pointers
		.get(&pushes[0].0)
		.ok_or_else(|| jit_error("missing push pointer"))?;
	let fields = structures
		.get(&structure)
		.ok_or_else(|| jit_error("missing push structure"))?;
	if class != 9
		|| fields.len() != expected_bindings + 1
		|| fields.iter().any(|field| !unsigned_words.contains(field))
		|| (0..expected_bindings as u32 + 1)
			.any(|field| offsets.get(&(structure, field)) != Some(&(field * 4)))
	{
		return Err(jit_error("unexpected packed U32 push fields"));
	}
	Ok(())
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/jit_compiler_unit.rs"]
pub(in crate::runtime) mod tests;
