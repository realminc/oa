//! Trusted local Slang library adapter; source-to-SPIR-V compilation only.
//! Donor: dna/kernel/jit.rs; preserve the session/request FFI and shader options.
use super::{CompiledChain, Program, jit_error, validate_program_abi, validate_with_spirv_tools};
use crate::Result;
use std::{
	ffi::{CStr, CString, c_char, c_int, c_void},
	marker::PhantomData,
	ptr::NonNull,
};

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
	/// `OA_SLANG_LIBRARY` names an explicit trusted path; otherwise OA probes
	/// the executable's staged `lib/` sibling, the `slangc` toolchain's
	/// `bin/../lib` sibling (dev installs keep the pair together), and finally
	/// the platform loader search. Compiler-specific `libslang-compiler`
	/// names are preferred over the legacy `libslang` names, which collide
	/// with the unrelated S-Lang interpreter library on some distributions.
	pub(in crate::runtime) fn open_default() -> Result<Self> {
		if let Some(path) = std::env::var_os("OA_SLANG_LIBRARY") {
			return Self::open(std::path::Path::new(&path));
		}
		let mut errors = Vec::new();
		for path in Self::candidate_paths() {
			match Self::open(&path) {
				Ok(compiler) => return Ok(compiler),
				Err(error) => errors.push(format!("{}: {error}", path.display())),
			}
		}
		for name in [
			"libslang-compiler.so",
			"libslang-compiler.so.0",
			"libslang-compiler.dylib",
			"slang-compiler.dll",
			"libslang.so",
			"libslang.so.2",
			"libslang.dylib",
			"slang.dll",
		] {
			// SAFETY: fixed OA-selected library names under the platform loader
			// search; no caller-controlled path is admitted here. A candidate
			// without the Slang compile API (for example the S-Lang
			// interpreter's `libslang`) fails at symbol lookup and the search
			// continues.
			match unsafe { libloading::Library::new(name) } {
				Ok(library) => match Self::from_library(library) {
					Ok(compiler) => return Ok(compiler),
					Err(error) => errors.push(format!("{name}: {error}")),
				},
				Err(error) => errors.push(format!("{name}: {error}")),
			}
		}
		Err(jit_error(format!(
			"no Slang compiler library found ({})",
			errors.join("; ")
		)))
	}

	/// Compiler-library candidates beside the executable and beside `slangc`
	/// (`SLANGC` first, then a PATH walk), preferring the compiler-specific
	/// name and accepting a single versioned soname per directory.
	fn candidate_paths() -> Vec<std::path::PathBuf> {
		use std::path::{Path, PathBuf};
		const LIB_DIRS: [&str; 2] = ["lib", "../lib"];
		const COMPILER_STEM: &str = "libslang-compiler.so";
		fn dir_candidates(dir: &Path, out: &mut Vec<PathBuf>) {
			for lib_dir in LIB_DIRS {
				let base = dir.join(lib_dir);
				let exact = base.join(COMPILER_STEM);
				if exact.is_file() {
					out.push(exact);
					continue;
				}
				if let Ok(entries) = std::fs::read_dir(&base) {
					let mut versioned: Vec<_> = entries
						.flatten()
						.map(|entry| entry.path())
						.filter(|path| {
							path
								.file_name()
								.and_then(|name| name.to_str())
								.is_some_and(|name| name.starts_with(COMPILER_STEM))
						})
						.collect();
					versioned.sort();
					out.extend(versioned.into_iter().take(1));
				}
			}
		}
		let mut candidates = Vec::new();
		if let Ok(exe) = std::env::current_exe()
			&& let Some(dir) = exe.parent()
		{
			dir_candidates(dir, &mut candidates);
		}
		let mut slangc_dirs = Vec::new();
		if let Some(path) = std::env::var_os("SLANGC") {
			slangc_dirs.push(PathBuf::from(path));
		}
		if let Some(paths) = std::env::var_os("PATH") {
			for dir in std::env::split_paths(&paths) {
				let candidate = dir.join("slangc");
				if candidate.is_file() {
					slangc_dirs.push(candidate);
				}
			}
		}
		for slangc in slangc_dirs {
			let dir = std::fs::canonicalize(&slangc)
				.ok()
				.and_then(|path| path.parent().map(Path::to_path_buf))
				.or_else(|| slangc.parent().map(Path::to_path_buf));
			if let Some(dir) = dir {
				dir_candidates(&dir, &mut candidates);
			}
		}
		candidates
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

	pub(in crate::runtime) fn compile_source(&self, source: &str, bounded: bool) -> Result<Vec<u32>> {
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
		crate::runtime::dna::kernel::decode_spirv(bytes)
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
