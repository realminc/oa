//! Host-callable Slang component oracles for production crypto formulas.
//!
//! Slang's explicit host target is a test laboratory, never an OA CPU
//! fallback. These checks prove formulas and compile both Vulkan SPIR-V
//! targets; they do not prove GPU execution, conformance, or constant-time
//! machine code. Rust port of `test/py/crypto/_pqc_harness.py`.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, MutexGuard, OnceLock};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

/// Entry point → test slang module, mirroring ENTRY_SOURCES.
const ENTRY_SOURCES: &[(&str, &str)] = &[
	("bigint_bit_components", "bigint/bits.slang"),
	("bigint_arithmetic_components", "bigint/arithmetic.slang"),
	("bigint_modular_components", "bigint/modular.slang"),
	("bigint_inverse_components", "bigint/modular.slang"),
	("bigint_montgomery_components", "bigint/montgomery.slang"),
	("bigint_barrett_components", "bigint/barrett.slang"),
	(
		"mldsa_prehash_digest_components",
		"mldsa/prehash_digest.slang",
	),
	("sha2_u32_pair_components", "common/sha2_pair_digest.slang"),
	(
		"sha2_u32_pair_arithmetic_components",
		"common/sha2_pair_digest.slang",
	),
	("sha2_u32_components", "common/sha2_digest.slang"),
	("sha2_u32_length_components", "common/sha2_digest.slang"),
	("sha3_components", "common/sha3_digest.slang"),
	("sha3_range_components", "common/sha3_digest.slang"),
	(
		"mldsa_sign_workspace_components",
		"mldsa/sign_workspace.slang",
	),
	(
		"hash_message_verification_components",
		"mldsa/verification.slang",
	),
	(
		"prehashed_verification_components",
		"mldsa/verification.slang",
	),
	("mldsa_prehash_components", "mldsa/prehash.slang"),
	("mldsa_sign_message_components", "mldsa/sign_message.slang"),
	(
		"mldsa_sign_message_workspace_components",
		"mldsa/sign_message_workspace.slang",
	),
	(
		"mldsa_prehash_workspace_components",
		"mldsa/prehash_workspace.slang",
	),
	(
		"mldsa_hash_message_workspace_components",
		"mldsa/hash_message_workspace.slang",
	),
	("mldsa_sign_components", "mldsa/sign.slang"),
	("mldsa_mask_components", "mldsa/sign.slang"),
	("mldsa_secret_sampling_components", "mldsa/keygen.slang"),
	("mldsa_keygen_components", "mldsa/keygen.slang"),
	("keccak_components", "common/keccak_permutation.slang"),
	(
		"parameter_verification_components",
		"mldsa/verification.slang",
	),
	("bit_packing_components", "mldsa/packing.slang"),
	("field_components", "mldsa/field.slang"),
	("mlkem_compression_components", "mlkem/compression.slang"),
	("mlkem_encaps_components", "mlkem/kem.slang"),
	("mlkem_field_components", "mlkem/field.slang"),
	("mlkem_kem_components", "mlkem/kem.slang"),
	(
		"mlkem_kem_workspace_components",
		"mlkem/kem_workspace.slang",
	),
	(
		"mlkem_decaps_workspace_components",
		"mlkem/kem_workspace.slang",
	),
	("mlkem_kem_decaps_components", "mlkem/kem.slang"),
	("mlkem_key_check_components", "mlkem/kem.slang"),
	("mlkem_keygen_components", "mlkem/kem.slang"),
	("mlkem_kpke_components", "mlkem/kpke.slang"),
	("mlkem_kpke_decrypt_components", "mlkem/kpke.slang"),
	(
		"mlkem_kpke_workspace_components",
		"mlkem/kpke_workspace.slang",
	),
	("mlkem_packing_components", "mlkem/packing.slang"),
	("mlkem_polynomial_components", "mlkem/ntt.slang"),
	("mlkem_rejection_components", "mlkem/sampling.slang"),
	("mlkem_sampling_components", "mlkem/sampling.slang"),
	("packing_components", "mldsa/packing.slang"),
	("polynomial_components", "mldsa/ntt.slang"),
	("range_components", "mldsa/verification.slang"),
	("sampling_components", "mldsa/sampling.slang"),
	("streaming_components", "common/shake_streaming.slang"),
	("verification_components", "mldsa/verification.slang"),
	("wide_components", "mldsa/field.slang"),
];

/// Entries whose input buffer is a ByteAddressBuffer: counts are bytes.
const BYTE_ADDRESS_ENTRIES: &[&str] = &[
	"mldsa_prehash_digest_components",
	"sha2_u32_pair_components",
	"sha2_u32_pair_arithmetic_components",
	"sha2_u32_components",
	"sha2_u32_length_components",
	"sha3_components",
	"sha3_range_components",
	"mldsa_sign_message_workspace_components",
	"mldsa_prehash_workspace_components",
	"mldsa_hash_message_workspace_components",
	"hash_message_verification_components",
	"prehashed_verification_components",
	"mldsa_prehash_components",
	"mldsa_sign_message_components",
	"packing_components",
	"bit_packing_components",
	"streaming_components",
	"verification_components",
	"parameter_verification_components",
	"range_components",
];

/// Entries whose output buffer is a ByteAddressBuffer.
const OUTPUT_BYTE_ADDRESS_ENTRIES: &[&str] = &[
	"mlkem_kpke_workspace_components",
	"mlkem_kem_workspace_components",
	"mlkem_decaps_workspace_components",
	"mldsa_sign_workspace_components",
	"mldsa_sign_message_workspace_components",
	"mldsa_prehash_workspace_components",
	"mldsa_hash_message_workspace_components",
];

#[repr(C)]
struct BufferView {
	data: *mut u32,
	count: usize,
}

#[repr(C)]
struct Parameters {
	inputs: BufferView,
	outputs: BufferView,
}

#[repr(C)]
struct VaryingInput {
	start: [u32; 3],
	end: [u32; 3],
}

type EntryFn = unsafe extern "C" fn(*mut VaryingInput, *mut c_void, *mut Parameters);

struct SlangHost {
	directory: PathBuf,
	libraries: HashMap<&'static str, libloading::Library>,
}

fn tool(name: &str) -> PathBuf {
	if name == "slangc"
		&& let Ok(configured) = std::env::var("SLANGC")
		&& !configured.is_empty()
	{
		return PathBuf::from(configured);
	}
	PathBuf::from(name)
}

fn run(command: &mut Command) -> String {
	let output = command
		.output()
		.unwrap_or_else(|error| panic!("failed to launch {command:?}: {error}"));
	if !output.status.success() {
		panic!(
			"component compilation/validation failed: {command:?}\n{}{}",
			String::from_utf8_lossy(&output.stdout),
			String::from_utf8_lossy(&output.stderr)
		);
	}
	String::from_utf8_lossy(&output.stdout).into_owned()
}

impl SlangHost {
	fn compile_entry(&mut self, entry: &'static str) {
		if self.libraries.contains_key(entry) {
			return;
		}
		let source = ENTRY_SOURCES
			.iter()
			.find(|(name, _)| *name == entry)
			.map(|(_, source)| *source)
			.unwrap_or_else(|| panic!("unknown slang host entry {entry}"));
		let mut args = vec![
			format!("{ROOT}/test/slang/crypto/{source}"),
			"-entry".into(),
			entry.into(),
			"-stage".into(),
			"compute".into(),
		];
		for include in [
			"src/slang/crypto/pqc/mldsa",
			"src/slang/crypto/pqc/mlkem",
			"src/slang/crypto/pqc/common",
			"src/slang/crypto/hash",
			"src/slang/core/math",
		] {
			args.push("-I".into());
			args.push(format!("{ROOT}/{include}"));
		}
		let slangc = tool("slangc");
		let library_path = self.directory.join(format!("{entry}.so"));
		let mut command = Command::new(&slangc);
		command
			.args(&args)
			.args(["-target", "shader-sharedlib", "-o"])
			.arg(&library_path);
		run(&mut command);
		// Compile and validate both production Vulkan targets; crypto
		// arithmetic must never silently require 64-bit shader integers.
		for (capability, environment) in [("spirv_1_5", "vulkan1.2"), ("spirv_1_6", "vulkan1.3")] {
			let artifact = self.directory.join(format!("{entry}-{capability}.spv"));
			let mut compile = Command::new(&slangc);
			compile.args(&args).args([
				"-target",
				"spirv",
				"-profile",
				"glsl_460",
				"-capability",
				capability,
				"-o",
			]);
			compile.arg(&artifact);
			run(&mut compile);
			let mut validate = Command::new(tool("spirv-val"));
			validate.args(["--target-env", environment]).arg(&artifact);
			run(&mut validate);
			let mut disassemble = Command::new(tool("spirv-dis"));
			disassemble.arg(&artifact);
			let assembly = run(&mut disassemble);
			assert!(
				!assembly.contains("OpCapability Int64"),
				"{entry} unexpectedly requires shaderInt64"
			);
		}
		// SAFETY: the path names a slangc shader-sharedlib artifact just
		// produced for this entry; the library stays loaded for the process.
		let library = unsafe {
			libloading::Library::new(&library_path)
				.unwrap_or_else(|error| panic!("dlopen {} failed: {error}", library_path.display()))
		};
		self.libraries.insert(entry, library);
	}
}

fn host() -> MutexGuard<'static, SlangHost> {
	static HOST: OnceLock<Mutex<SlangHost>> = OnceLock::new();
	HOST
		.get_or_init(|| {
			let directory = std::env::temp_dir().join(format!("oa-slang-host-{}", std::process::id()));
			std::fs::create_dir_all(&directory)
				.unwrap_or_else(|error| panic!("cannot create {}: {error}", directory.display()));
			Mutex::new(SlangHost {
				directory,
				libraries: HashMap::new(),
			})
		})
		.lock()
		.unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Execute a host-callable component entry over `values`, producing
/// `rows * width` output words with guard words checked either side.
pub fn execute(entry: &'static str, values: &[u32], rows: u32, width: usize) -> Vec<u32> {
	let mut host = host();
	host.compile_entry(entry);
	let library = &host.libraries[entry];
	// SAFETY: the symbol is the slangc-emitted host entry for this kernel,
	// which has the (VaryingInput*, uniform, GlobalParams*) call ABI.
	let function: libloading::Symbol<EntryFn> = unsafe {
		library
			.get(entry.as_bytes())
			.unwrap_or_else(|error| panic!("entry {entry} missing from shared library: {error}"))
	};
	let output_count = rows as usize * width;
	let mut outputs = vec![0xA5A5A5A5_u32; output_count + 2];
	let input_count = if BYTE_ADDRESS_ENTRIES.contains(&entry) {
		values.len() * 4
	} else {
		values.len()
	};
	let output_view_count = if OUTPUT_BYTE_ADDRESS_ENTRIES.contains(&entry) {
		output_count * 4
	} else {
		output_count
	};
	let mut inputs = values.to_vec();
	let mut parameters = Parameters {
		inputs: BufferView {
			data: inputs.as_mut_ptr(),
			count: input_count,
		},
		outputs: BufferView {
			data: outputs[1..].as_mut_ptr(),
			count: output_view_count,
		},
	};
	let mut varying = VaryingInput {
		start: [0, 0, 0],
		end: [rows, 1, 1],
	};
	// SAFETY: all three ABI structs outlive this synchronous call; the shader
	// writes only within the admitted row range between the guard words.
	unsafe { function(&mut varying, std::ptr::null_mut(), &mut parameters) };
	assert_eq!(outputs[0], 0xA5A5A5A5, "{entry} overwrote leading guard");
	assert_eq!(
		outputs[output_count + 1],
		0xA5A5A5A5,
		"{entry} overwrote trailing guard"
	);
	outputs[1..=output_count].to_vec()
}
