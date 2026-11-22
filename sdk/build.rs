//! SDK-owned shader staging.
//!
//! Compiles `slang/` compute sources into SPIR-V under `OUT_DIR` and exports
//! `OA_SHADER_DIR` so SDK modules embed their artifacts with `include_bytes!`.
//! Mirrors the OA build's profile selection and slangc recipe; extension
//! kernels validate their ABI at `oa::compute::Kernel::compile`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// (source, artifact stem, declared storage-buffer count).
const COMPUTE_SHADERS: &[(&str, &str, u32)] = &[
	(
		"slang/ml/rl/cart_pole/reset.slang",
		"ml_cart_pole_reset_f32",
		4,
	),
	(
		"slang/ml/rl/cart_pole/reset_cursor.slang",
		"ml_cart_pole_reset_cursor_f32",
		5,
	),
	(
		"slang/ml/rl/cart_pole/step.slang",
		"ml_cart_pole_step_f32",
		8,
	),
	(
		"slang/ml/rl/lunar_lander/reset.slang",
		"ml_lunar_lander_reset_f32",
		7,
	),
	(
		"slang/ml/rl/lunar_lander/reset_cursor.slang",
		"ml_lunar_lander_reset_cursor_f32",
		8,
	),
	(
		"slang/ml/rl/lunar_lander/step.slang",
		"ml_lunar_lander_step_f32",
		13,
	),
];

fn main() {
	if let Err(error) = build_shaders() {
		panic!("OA SDK shader build failed: {error}");
	}
}

fn build_shaders() -> Result<(), Box<dyn std::error::Error>> {
	println!("cargo:rerun-if-env-changed=OA_BUILD_PROFILE");
	println!("cargo:rerun-if-env-changed=SLANGC");
	println!("cargo:rerun-if-env-changed=SPIRV_VAL");
	let compatibility = matches!(env::var("OA_BUILD_PROFILE").as_deref(), Ok("compatibility"));
	let (spirv_capability, vulkan_target) = if compatibility {
		("spirv_1_5", "vulkan1.2")
	} else {
		("spirv_1_6", "vulkan1.3")
	};
	let output_directory =
		PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is unavailable")?).join("shaders");
	fs::create_dir_all(&output_directory)?;
	for entry in fs::read_dir(&output_directory)? {
		let path = entry?.path();
		if matches!(
			path.extension().and_then(|ext| ext.to_str()),
			Some("spv" | "slang")
		) {
			fs::remove_file(path)?;
		}
	}
	println!(
		"cargo:rustc-env=OA_SHADER_DIR={}",
		output_directory.display()
	);
	let slangc = env::var_os("SLANGC").unwrap_or_else(|| "slangc".into());
	let spirv_val = env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
	for support in [
		"../src/slang/core/math/storage.slang",
		"../src/slang/core/attributes.slang",
	] {
		println!("cargo:rerun-if-changed={support}");
	}
	for (source, stem, buffers) in COMPUTE_SHADERS {
		println!("cargo:rerun-if-changed={source}");
		let selected = if compatibility {
			let text = fs::read_to_string(source)?;
			const UNSIZED: &str = "[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];";
			if text.matches(UNSIZED).count() != 1 {
				return Err(format!("{source} has no unique storage declaration").into());
			}
			let bounded = text.replace(
				UNSIZED,
				&format!("[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[{buffers}];"),
			);
			let path = output_directory.join(format!("{stem}.slang"));
			fs::write(&path, bounded)?;
			path
		} else {
			PathBuf::from(source)
		};
		let spirv = output_directory.join(format!("{stem}.spv"));
		let mut command = Command::new(&slangc);
		if let Some(parent) = std::path::Path::new(source).parent() {
			command.arg("-I").arg(parent);
		}
		let output = command
			.arg(&selected)
			.args([
				"-entry",
				"main",
				"-stage",
				"compute",
				"-target",
				"spirv",
				"-profile",
				"glsl_460",
				"-capability",
				spirv_capability,
				"-fvk-use-entrypoint-name",
				"-warnings-as-errors",
				"all",
				// OA core slang support modules shared with the lib build.
				"-I",
				"../src/slang/core",
				"-I",
				"../src/slang/core/math",
				"-o",
			])
			.arg(&spirv)
			.output()
			.map_err(|e| format!("could not execute {slangc:?}: {e}"))?;
		if !output.status.success() {
			return Err(
				format!(
					"{slangc:?} failed for sdk.{stem} with {}\n{}",
					output.status,
					String::from_utf8_lossy(&output.stderr)
				)
				.into(),
			);
		}
		let validation = Command::new(&spirv_val)
			.args(["--target-env", vulkan_target])
			.arg(&spirv)
			.output()
			.map_err(|e| format!("could not execute {spirv_val:?}: {e}"))?;
		if !validation.status.success() {
			return Err(
				format!(
					"sdk.{stem} SPIR-V validation failed: {}",
					String::from_utf8_lossy(&validation.stderr)
				)
				.into(),
			);
		}
	}
	Ok(())
}
