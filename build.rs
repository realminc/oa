use std::{
	env, fs,
	path::{Path, PathBuf},
	process::Command,
};

use serde_json::{Value, json};

const SCHEMA: &str = "tool/gen/fn/schema/matrix/matrix_elemwise.json";
const BLAS_SCHEMA: &str = "tool/gen/fn/schema/matrix/matrix_blas.json";
const REDUCE_SCHEMA: &str = "tool/gen/fn/schema/matrix/matrix_reduce.json";
const RNG_SCHEMA: &str = "tool/gen/fn/schema/matrix/matrix_rng.json";
const INDEX_SCHEMA: &str = "tool/gen/fn/schema/matrix/matrix_index.json";
const ML_SCHEMA: &str = "tool/gen/fn/schema/ml/ml_training.json";
const AUDIO_SCHEMA: &str = "tool/gen/fn/schema/audio/audio.json";
const CRYPTOGRAPHY_HASH_SCHEMA: &str = "tool/gen/fn/schema/cryptography/cryptography_hash.json";
const CRYPTOGRAPHY_PQC_SCHEMA: &str = "tool/gen/fn/schema/cryptography/cryptography_pqc.json";
const IMAGE_SCHEMA: &str = "tool/gen/fn/schema/image/image.json";
const VISION_SCHEMA: &str = "tool/gen/fn/schema/vision/vision_detection.json";
const GENERATOR: &str = "tool/gen/fn/generate.py";
const STORAGE: &str = "src/slang/core/math/storage.slang";
const ATTRIBUTES: &str = "src/slang/core/attributes.slang";
const ACTIVATIONS: &str = "src/slang/core/math/activations.slang";
const SSM_MATH: &str = "src/slang/core/math/ssm_math.slang";
const PHILOX: &str = "src/slang/core/rng/philox.slang";
const DROPOUT_RNG: &str = "src/slang/matrix/rng/dropout_rng.slang";
const CRYPTOGRAPHY_KECCAK: &str = "src/slang/cryptography/hash/keccak.slang";
const CRYPTOGRAPHY_PQC_MLDSA_FIELD: &str = "src/slang/cryptography/pqc/mldsa/mldsa_field.slang";
const CRYPTOGRAPHY_PQC_MLDSA_NTT: &str = "src/slang/cryptography/pqc/mldsa/mldsa_ntt.slang";
const CRYPTOGRAPHY_PQC_MLDSA_PACKING: &str = "src/slang/cryptography/pqc/mldsa/mldsa_packing.slang";
const CRYPTOGRAPHY_PQC_MLDSA_SAMPLING: &str =
	"src/slang/cryptography/pqc/mldsa/mldsa_sampling.slang";
const CRYPTOGRAPHY_PQC_SHAKE: &str = "src/slang/cryptography/pqc/common/shake.slang";
const CRYPTOGRAPHY_PQC_VERIFICATION: &str =
	"src/slang/cryptography/pqc/mldsa/mldsa_verification.slang";
const CRYPTOGRAPHY_PQC_SHA3: &str = "src/slang/cryptography/pqc/common/sha3.slang";
const ENTRY_POINT: &str = "main";

struct ShaderBuild<'a> {
	output_directory: &'a Path,
	slangc: &'a std::ffi::OsStr,
	spirv_val: &'a std::ffi::OsStr,
	spirv_capability: &'static str,
	vulkan_target: &'static str,
}

fn shader_target() -> Result<(&'static str, &'static str), Box<dyn std::error::Error>> {
	match env::var("OA_SPIRV_TARGET").as_deref() {
		Err(env::VarError::NotPresent) | Ok("1.6") => Ok(("spirv_1_6", "vulkan1.3")),
		Ok("1.5") => Ok(("spirv_1_5", "vulkan1.2")),
		Ok(value) => Err(format!("unsupported OA_SPIRV_TARGET={value:?}; expected 1.5 or 1.6").into()),
		Err(error) => Err(format!("invalid OA_SPIRV_TARGET: {error}").into()),
	}
}

fn main() {
	if let Err(error) = build_shaders() {
		panic!("OA shader build failed: {error}");
	}
}

const RENDER_VERT: &str = "src/slang/render/vertex_color_lit.vert.slang";
const RENDER_FRAG: &str = "src/slang/render/vertex_color_lit.frag.slang";
const RENDER_MATERIAL_COMMON: &str = "src/slang/render/material_common.slang";
const RENDER_BRDF: &str = "src/slang/render/brdf.slang";
const RENDER_LIGHTS: &str = "src/slang/render/lights.slang";
const RENDER_FLAT_COLOR_VERT: &str = "src/slang/render/flat_color.vert.slang";
const RENDER_FLAT_COLOR_FRAG: &str = "src/slang/render/flat_color.frag.slang";
const RENDER_UNLIT_VERT: &str = "src/slang/render/unlit.vert.slang";
const RENDER_UNLIT_FRAG: &str = "src/slang/render/unlit.frag.slang";
const RENDER_SS_VERT: &str = "src/slang/render/standard_surface.vert.slang";
const RENDER_SS_FRAG: &str = "src/slang/render/standard_surface.frag.slang";

const UI_CLEAR_COMPOSE: &str = "src/slang/ui/clear_compose.slang";
const UI_BLIT_RGBA: &str = "src/slang/ui/blit_rgba.slang";
const UI_DRAW_RECT: &str = "src/slang/ui/draw_rect.slang";
const UI_DRAW_RECT_OUTLINE: &str = "src/slang/ui/draw_rect_outline.slang";
const UI_DRAW_LINE: &str = "src/slang/ui/draw_line.slang";
const UI_DRAW_PLOT_LINE: &str = "src/slang/ui/draw_plot_line.slang";
const UI_DRAW_GLYPHS: &str = "src/slang/ui/draw_glyphs.slang";
const UI_COMPOSITOR_COMMON: &str = "src/slang/ui/compositor_common.slang";

fn build_shaders() -> Result<(), Box<dyn std::error::Error>> {
	for source in [
		SCHEMA,
		BLAS_SCHEMA,
		REDUCE_SCHEMA,
		RNG_SCHEMA,
		INDEX_SCHEMA,
		ML_SCHEMA,
		AUDIO_SCHEMA,
		CRYPTOGRAPHY_HASH_SCHEMA,
		CRYPTOGRAPHY_PQC_SCHEMA,
		IMAGE_SCHEMA,
		VISION_SCHEMA,
		GENERATOR,
		"tool/gen/fn/template/matrix",
		"tool/gen/fn/template/audio",
		"tool/gen/fn/template/image",
		"tool/gen/fn/template/vision",
		"tool/gen/fn/template/cryptography",
		STORAGE,
		ATTRIBUTES,
		ACTIVATIONS,
		SSM_MATH,
		PHILOX,
		DROPOUT_RNG,
		CRYPTOGRAPHY_KECCAK,
		CRYPTOGRAPHY_PQC_MLDSA_FIELD,
		CRYPTOGRAPHY_PQC_MLDSA_NTT,
		CRYPTOGRAPHY_PQC_MLDSA_PACKING,
		CRYPTOGRAPHY_PQC_MLDSA_SAMPLING,
		CRYPTOGRAPHY_PQC_SHA3,
		CRYPTOGRAPHY_PQC_SHAKE,
		CRYPTOGRAPHY_PQC_VERIFICATION,
		RENDER_VERT,
		RENDER_FRAG,
		RENDER_MATERIAL_COMMON,
		RENDER_BRDF,
		RENDER_LIGHTS,
		RENDER_FLAT_COLOR_VERT,
		RENDER_FLAT_COLOR_FRAG,
		RENDER_UNLIT_VERT,
		RENDER_UNLIT_FRAG,
		RENDER_SS_VERT,
		RENDER_SS_FRAG,
		UI_COMPOSITOR_COMMON,
		UI_CLEAR_COMPOSE,
		UI_BLIT_RGBA,
		UI_DRAW_RECT,
		UI_DRAW_RECT_OUTLINE,
		UI_DRAW_LINE,
		UI_DRAW_PLOT_LINE,
		UI_DRAW_GLYPHS,
	] {
		println!("cargo:rerun-if-changed={source}");
	}
	println!("cargo:rerun-if-env-changed=SLANGC");
	println!("cargo:rerun-if-env-changed=SPIRV_VAL");
	println!("cargo:rerun-if-env-changed=PYTHON");
	println!("cargo:rerun-if-env-changed=OA_SPIRV_TARGET");
	verify_generated_sources()?;

	let schema: Value = serde_json::from_slice(&fs::read(SCHEMA)?)?;
	let operations = array_at(&schema, "operations")?;
	let default_dtype = string_at(&schema, "dtype")?;
	let workgroup_size = &schema["workgroup_size"];
	if operations.is_empty() {
		return Err("matrix elementwise schema contains no operations".into());
	}
	let output_directory = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is unavailable")?);
	let slangc = env::var_os("SLANGC").unwrap_or_else(|| "slangc".into());
	let spirv_val = env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
	let (spirv_capability, vulkan_target) = shader_target()?;
	let build = ShaderBuild {
		output_directory: &output_directory,
		slangc: &slangc,
		spirv_val: &spirv_val,
		spirv_capability,
		vulkan_target,
	};
	for operation in operations {
		build_shader(
			operation,
			operation,
			default_dtype,
			workgroup_size,
			"src/slang/matrix/elemwise",
			&build,
		)?;
		if let Some(variants) = operation["additional_dtype_variants"].as_array() {
			for variant in variants {
				build_shader(
					operation,
					variant,
					default_dtype,
					workgroup_size,
					"src/slang/matrix/elemwise",
					&build,
				)?;
			}
		}
		if let Some(lowerings) = operation["additional_lowering_variants"].as_array() {
			for lowering in lowerings {
				build_schema_shader(
					lowering,
					"matrix",
					string_at(lowering, "dtype")?,
					lowering
						.get("workgroup_size")
						.filter(|value| value.is_array())
						.unwrap_or(workgroup_size),
					&build,
				)?;
			}
		}
	}

	let blas_schema: Value = serde_json::from_slice(&fs::read(BLAS_SCHEMA)?)?;
	let blas_operations = array_at(&blas_schema, "operations")?;
	let blas_dtype = string_at(&blas_schema, "dtype")?;
	let blas_workgroup_size = &blas_schema["workgroup_size"];
	if blas_operations.is_empty() {
		return Err("matrix BLAS schema contains no operations".into());
	}
	for operation in blas_operations {
		build_shader(
			operation,
			operation,
			blas_dtype,
			blas_workgroup_size,
			"src/slang/matrix/blas",
			&build,
		)?;
	}

	let reduce_schema: Value = serde_json::from_slice(&fs::read(REDUCE_SCHEMA)?)?;
	let reduce_operations = array_at(&reduce_schema, "operations")?;
	let reduce_dtype = string_at(&reduce_schema, "dtype")?;
	let reduce_workgroup_size = &reduce_schema["workgroup_size"];
	if reduce_operations.is_empty() {
		return Err("matrix Reduce schema contains no operations".into());
	}
	for operation in reduce_operations {
		build_schema_shader(
			operation,
			"matrix",
			reduce_dtype,
			reduce_workgroup_size,
			&build,
		)?;
	}

	let rng_schema: Value = serde_json::from_slice(&fs::read(RNG_SCHEMA)?)?;
	let rng_operations = array_at(&rng_schema, "operations")?;
	let rng_dtype = string_at(&rng_schema, "dtype")?;
	let rng_workgroup_size = &rng_schema["workgroup_size"];
	for operation in rng_operations {
		let operation_dtype = operation["dtype"].as_str().unwrap_or(rng_dtype);
		let operation_workgroup_size = operation
			.get("workgroup_size")
			.filter(|value| value.is_array())
			.unwrap_or(rng_workgroup_size);
		build_schema_shader(
			operation,
			"matrix",
			operation_dtype,
			operation_workgroup_size,
			&build,
		)?;
	}

	let index_schema: Value = serde_json::from_slice(&fs::read(INDEX_SCHEMA)?)?;
	let index_operations = array_at(&index_schema, "operations")?;
	let index_dtype = string_at(&index_schema, "dtype")?;
	let index_workgroup_size = &index_schema["workgroup_size"];
	if index_operations.is_empty() {
		return Err("matrix Index schema contains no operations".into());
	}
	for operation in index_operations {
		let operation_dtype = operation["dtype"].as_str().unwrap_or(index_dtype);
		let operation_workgroup_size = operation
			.get("workgroup_size")
			.filter(|value| value.is_array())
			.unwrap_or(index_workgroup_size);
		build_schema_shader(
			operation,
			"matrix",
			operation_dtype,
			operation_workgroup_size,
			&build,
		)?;
	}

	let ml_schema: Value = serde_json::from_slice(&fs::read(ML_SCHEMA)?)?;
	let ml_operations = array_at(&ml_schema, "operations")?;
	let ml_dtype = string_at(&ml_schema, "dtype")?;
	let ml_workgroup_size = &ml_schema["workgroup_size"];
	if ml_operations.is_empty() {
		return Err("ML training schema contains no operations".into());
	}
	for operation in ml_operations {
		let operation_dtype = operation["dtype"].as_str().unwrap_or(ml_dtype);
		let operation_workgroup_size = operation
			.get("workgroup_size")
			.filter(|value| value.is_array())
			.unwrap_or(ml_workgroup_size);
		build_schema_shader(
			operation,
			"ml",
			operation_dtype,
			operation_workgroup_size,
			&build,
		)?;
	}

	let audio_schema: Value = serde_json::from_slice(&fs::read(AUDIO_SCHEMA)?)?;
	let audio_kernels = array_at(&audio_schema, "kernels")?;
	let audio_dtype = string_at(&audio_schema, "dtype")?;
	if audio_kernels.is_empty() {
		return Err("Audio schema contains no kernels".into());
	}
	for kernel in audio_kernels {
		build_schema_shader(
			kernel,
			"audio",
			audio_dtype,
			&kernel["workgroup_size"],
			&build,
		)?;
	}

	let cryptography_schema: Value = serde_json::from_slice(&fs::read(CRYPTOGRAPHY_HASH_SCHEMA)?)?;
	let cryptography_kernels = array_at(&cryptography_schema, "kernels")?;
	let cryptography_dtype = string_at(&cryptography_schema, "dtype")?;
	if cryptography_kernels.is_empty() {
		return Err("Cryptography Hash schema contains no kernels".into());
	}
	for kernel in cryptography_kernels {
		build_schema_shader(
			kernel,
			"cryptography",
			cryptography_dtype,
			&kernel["workgroup_size"],
			&build,
		)?;
	}

	// Imported private formulas are build inputs too. Track each family and
	// directory membership so additions/removals cannot leave stale artifacts.
	for directory in [
		"src/slang/cryptography/pqc/mlkem",
		"src/slang/cryptography/pqc/mldsa",
	] {
		println!("cargo:rerun-if-changed={directory}");
		for entry in fs::read_dir(directory)? {
			let path = entry?.path();
			if path
				.extension()
				.is_some_and(|extension| extension == "slang")
			{
				println!("cargo:rerun-if-changed={}", path.display());
			}
		}
	}
	let cryptography_pqc_schema: Value = serde_json::from_slice(&fs::read(CRYPTOGRAPHY_PQC_SCHEMA)?)?;
	let cryptography_pqc_kernels = array_at(&cryptography_pqc_schema, "kernels")?;
	let cryptography_pqc_dtype = string_at(&cryptography_pqc_schema, "dtype")?;
	if cryptography_pqc_kernels.is_empty() {
		return Err("Cryptography PQC schema contains no kernels".into());
	}
	for kernel in cryptography_pqc_kernels.iter().chain(
		cryptography_pqc_schema["private_kernels"]
			.as_array()
			.into_iter()
			.flatten(),
	) {
		build_schema_pqc_shader(
			kernel,
			"cryptography",
			cryptography_pqc_dtype,
			&kernel["workgroup_size"],
			&build,
		)?;
	}

	let image_schema: Value = serde_json::from_slice(&fs::read(IMAGE_SCHEMA)?)?;
	let image_kernels = array_at(&image_schema, "kernels")?;
	let image_dtype = string_at(&image_schema, "dtype")?;
	if image_kernels.is_empty() {
		return Err("Image schema contains no kernels".into());
	}
	for kernel in image_kernels {
		build_schema_shader(
			kernel,
			"image",
			image_dtype,
			&kernel["workgroup_size"],
			&build,
		)?;
	}

	let vision_schema: Value = serde_json::from_slice(&fs::read(VISION_SCHEMA)?)?;
	let vision_kernels = array_at(&vision_schema, "kernels")?;
	let vision_dtype = string_at(&vision_schema, "dtype")?;
	if vision_kernels.is_empty() {
		return Err("Vision schema contains no kernels".into());
	}
	for kernel in vision_kernels {
		let kernel_dtype = kernel["dtype"].as_str().unwrap_or(vision_dtype);
		build_schema_shader(
			kernel,
			"vision",
			kernel_dtype,
			&kernel["workgroup_size"],
			&build,
		)?;
	}
	build_graphics_shader("vertex_color_lit_vert", RENDER_VERT, "vertex", &build)?;
	build_graphics_shader("vertex_color_lit_frag", RENDER_FRAG, "fragment", &build)?;
	build_graphics_shader("flat_color_vert", RENDER_FLAT_COLOR_VERT, "vertex", &build)?;
	build_graphics_shader(
		"flat_color_frag",
		RENDER_FLAT_COLOR_FRAG,
		"fragment",
		&build,
	)?;
	build_graphics_shader("unlit_vert", RENDER_UNLIT_VERT, "vertex", &build)?;
	build_graphics_shader("unlit_frag", RENDER_UNLIT_FRAG, "fragment", &build)?;
	// Standard Surface — opaque base variant (no texture defines).
	build_graphics_shader("standard_surface_vert", RENDER_SS_VERT, "vertex", &build)?;
	build_graphics_shader("standard_surface_frag", RENDER_SS_FRAG, "fragment", &build)?;
	// UI compute shaders.
	build_ui_shader("clear_compose", UI_CLEAR_COMPOSE, &build)?;
	build_ui_shader("blit_rgba", UI_BLIT_RGBA, &build)?;
	build_ui_shader("draw_rect", UI_DRAW_RECT, &build)?;
	build_ui_shader("draw_rect_outline", UI_DRAW_RECT_OUTLINE, &build)?;
	build_ui_shader("draw_line", UI_DRAW_LINE, &build)?;
	build_ui_shader("draw_plot_line", UI_DRAW_PLOT_LINE, &build)?;
	build_ui_shader("draw_glyphs", UI_DRAW_GLYPHS, &build)?;
	Ok(())
}

fn build_ui_shader(
	stem: &str,
	source: &str,
	build: &ShaderBuild<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
	let spirv = build.output_directory.join(format!("ui_{stem}.spv"));
	let output = Command::new(build.slangc)
		.arg(source)
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
			build.spirv_capability,
			"-fvk-use-entrypoint-name",
			"-warnings-as-errors",
			"all",
			"-I",
			"src/slang/core",
			"-I",
			"src/slang/core/math",
			"-I",
			"src/slang/ui",
		])
		.arg("-o")
		.arg(&spirv)
		.output()
		.map_err(|e| format!("could not execute {:?}: {e}", build.slangc))?;
	if !output.status.success() {
		return Err(
			format!(
				"{:?} failed for ui.{stem} with {}\n{}",
				build.slangc,
				output.status,
				String::from_utf8_lossy(&output.stderr)
			)
			.into(),
		);
	}
	let val = Command::new(build.spirv_val)
		.args(["--target-env", build.vulkan_target])
		.arg(&spirv)
		.output()
		.map_err(|e| format!("could not execute {:?}: {e}", build.spirv_val))?;
	if !val.status.success() {
		return Err(
			format!(
				"{:?} failed for ui.{stem} with {}\n{}",
				build.spirv_val,
				val.status,
				String::from_utf8_lossy(&val.stderr)
			)
			.into(),
		);
	}
	Ok(())
}

fn build_graphics_shader(
	stem: &str,
	source: &str,
	stage: &str,
	build: &ShaderBuild<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
	let spirv = build.output_directory.join(format!("render_{stem}.spv"));
	let output = Command::new(build.slangc)
		.arg(source)
		.args([
			"-entry",
			"main",
			"-stage",
			stage,
			"-target",
			"spirv",
			"-profile",
			"glsl_460",
			"-capability",
			build.spirv_capability,
			"-fvk-use-entrypoint-name",
			"-warnings-as-errors",
			"all",
		])
		.arg("-o")
		.arg(&spirv)
		.output()
		.map_err(|e| format!("could not execute {:?}: {e}", build.slangc))?;
	if !output.status.success() {
		return Err(
			format!(
				"{:?} failed for render.{stem} with {}\n{}",
				build.slangc,
				output.status,
				String::from_utf8_lossy(&output.stderr)
			)
			.into(),
		);
	}
	let val = Command::new(build.spirv_val)
		.args(["--target-env", build.vulkan_target])
		.arg(&spirv)
		.output()
		.map_err(|e| format!("could not execute {:?}: {e}", build.spirv_val))?;
	if !val.status.success() {
		return Err(
			format!(
				"{:?} failed for render.{stem} with {}\n{}",
				build.spirv_val,
				val.status,
				String::from_utf8_lossy(&val.stderr)
			)
			.into(),
		);
	}
	Ok(())
}

fn verify_generated_sources() -> Result<(), Box<dyn std::error::Error>> {
	let python = env::var_os("PYTHON").unwrap_or_else(|| "python3".into());
	let output = Command::new(&python)
		.args([GENERATOR, "--check"])
		.output()
		.map_err(|source| format!("could not execute {python:?}: {source}"))?;
	if !output.status.success() {
		return Err(
			format!(
				"generated operation sources are stale; run `python3 {GENERATOR}`\n{}",
				String::from_utf8_lossy(&output.stderr)
			)
			.into(),
		);
	}
	Ok(())
}

fn build_shader(
	operation: &Value,
	variant: &Value,
	default_dtype: &str,
	workgroup_size: &Value,
	source_directory: &str,
	build: &ShaderBuild<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
	let name = string_at(operation, "name")?;
	let dtype = variant["dtype"].as_str().unwrap_or(default_dtype);
	let kernel_name = variant["kernel_name"].as_str().unwrap_or(name);
	let source_stem = variant["source_stem"].as_str().unwrap_or(name);
	let source = format!("{source_directory}/{source_stem}.gen.slang");
	println!("cargo:rerun-if-changed={source}");
	let spirv = build
		.output_directory
		.join(format!("matrix_{name}_{dtype}.spv"));
	let reflection = build
		.output_directory
		.join(format!("matrix_{name}_{dtype}.reflection.json"));

	let slang_output = Command::new(build.slangc)
		.arg(&source)
		.args([
			"-entry",
			ENTRY_POINT,
			"-stage",
			"compute",
			"-target",
			"spirv",
			"-profile",
			"glsl_460",
			"-capability",
			build.spirv_capability,
			"-fvk-use-entrypoint-name",
			"-warnings-as-errors",
			"all",
			"-I",
			"src/slang/core",
			"-I",
			"src/slang/core/math",
			"-I",
			"src/slang/core/rng",
			"-I",
			"src/slang/matrix/rng",
			"-I",
			"src/slang/cryptography/hash",
			"-reflection-json",
		])
		.arg(&reflection)
		.arg("-o")
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.slangc))?;
	if !slang_output.status.success() {
		return Err(
			format!(
				"{:?} failed for matrix.{name} with {}\n{}",
				build.slangc,
				slang_output.status,
				String::from_utf8_lossy(&slang_output.stderr)
			)
			.into(),
		);
	}

	validate_reflection(
		&reflection,
		operation,
		kernel_name,
		dtype,
		workgroup_size,
		0,
	)?;

	let validation_output = Command::new(build.spirv_val)
		.args(["--target-env", build.vulkan_target])
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.spirv_val))?;
	if !validation_output.status.success() {
		return Err(
			format!(
				"{:?} failed for matrix.{name} with {}\n{}",
				build.spirv_val,
				validation_output.status,
				String::from_utf8_lossy(&validation_output.stderr)
			)
			.into(),
		);
	}
	build_bounded_shader(
		&source,
		&format!("matrix_{name}_{dtype}"),
		&reflection,
		variant,
		build,
	)?;
	Ok(())
}

// Mechanical binding ABI adaptation: preserve the schema-owned shader body,
// push fields, and workgroup geometry while fixing descriptor-array extent.
fn build_bounded_shader(
	source_path: &str,
	stem: &str,
	strict_reflection: &Path,
	metadata: &Value,
	build: &ShaderBuild<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
	const UNSIZED: &str = "[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];";
	let source = fs::read_to_string(source_path)?;
	if source.matches(UNSIZED).count() != 1 {
		return Err(
			format!("{source_path} has no unique storage-buffer descriptor declaration").into(),
		);
	}
	let strict: Value = serde_json::from_slice(&fs::read(strict_reflection)?)?;
	let strict_params = array_at(&strict, "parameters")?;
	let push = named(strict_params, "push")?;
	let fields = push
		.pointer("/type/elementType/fields")
		.and_then(Value::as_array)
		.ok_or("strict push fields missing")?;
	let buffer_count = fields
		.iter()
		.filter(|field| {
			field["name"]
				.as_str()
				.is_some_and(|name| name.ends_with("_index") || name.ends_with("_idx"))
		})
		.count();
	if buffer_count == 0 {
		return Err(format!("{stem} has no reflected buffer indices").into());
	}
	let bounded = source.replace(
		UNSIZED,
		&format!("[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[{buffer_count}];"),
	);
	let bounded_source = build.output_directory.join(format!("{stem}_bounded.slang"));
	let spirv = build.output_directory.join(format!("{stem}_bounded.spv"));
	let reflection = build
		.output_directory
		.join(format!("{stem}_bounded.reflection.json"));
	fs::write(&bounded_source, bounded)?;
	let mut command = Command::new(build.slangc);
	command.arg(&bounded_source).args([
		"-entry",
		ENTRY_POINT,
		"-stage",
		"compute",
		"-target",
		"spirv",
		"-profile",
		"glsl_460",
		"-capability",
		"spirv_1_5",
		"-fvk-use-entrypoint-name",
		"-warnings-as-errors",
		"all",
		"-I",
		"src/slang/core",
		"-I",
		"src/slang/core/math",
		"-I",
		"src/slang/core/rng",
		"-I",
		"src/slang/matrix/rng",
		"-I",
		"src/slang/cryptography/hash",
		"-I",
		"src/slang/cryptography/pqc/common",
		"-I",
		"src/slang/cryptography/pqc/mldsa",
		"-I",
		"src/slang/cryptography/pqc/mlkem",
	]);
	if metadata["slang_optimization"].as_str() == Some("O0") {
		command.arg("-O0");
	}
	if let Some(capabilities) = metadata["slang_capabilities"].as_array() {
		for capability in capabilities {
			command
				.arg("-capability")
				.arg(capability.as_str().ok_or("non-string Slang capability")?);
		}
	}
	let output = command
		.arg("-reflection-json")
		.arg(&reflection)
		.arg("-o")
		.arg(&spirv)
		.output()?;
	if !output.status.success() {
		return Err(
			format!(
				"bounded {stem} Slang compilation failed: {}",
				String::from_utf8_lossy(&output.stderr)
			)
			.into(),
		);
	}
	let bounded: Value = serde_json::from_slice(&fs::read(&reflection)?)?;
	let bounded_params = array_at(&bounded, "parameters")?;
	require_equal(
		named(bounded_params, "push")?,
		named(strict_params, "push")?,
		"bounded push ABI",
	)?;
	let storage = named(bounded_params, "storage_buffers")?;
	require_equal(
		&storage["binding"],
		&json!({"kind":"descriptorTableSlot","index":0}),
		"bounded storage binding",
	)?;
	require_equal(
		&storage["type"]["elementCount"],
		&json!(buffer_count),
		"bounded storage count",
	)?;
	let strict_entry = array_at(&strict, "entryPoints")?
		.iter()
		.find(|entry| entry["name"] == ENTRY_POINT)
		.ok_or("strict entry point missing")?;
	let bounded_entry = array_at(&bounded, "entryPoints")?
		.iter()
		.find(|entry| entry["name"] == ENTRY_POINT)
		.ok_or("bounded entry point missing")?;
	require_equal(
		&bounded_entry["threadGroupSize"],
		&strict_entry["threadGroupSize"],
		"bounded workgroup",
	)?;
	let output = Command::new(build.spirv_val)
		.args(["--target-env", "vulkan1.2"])
		.arg(&spirv)
		.output()?;
	if !output.status.success() {
		return Err(
			format!(
				"bounded {stem} Vulkan 1.2 validation failed: {}",
				String::from_utf8_lossy(&output.stderr)
			)
			.into(),
		);
	}
	let bytes = fs::read(&spirv)?;
	let (words, remainder) = bytes.as_chunks::<4>();
	if !remainder.is_empty() || words.len() < 5 {
		return Err(format!("bounded {stem} has malformed SPIR-V words").into());
	}
	let words = words
		.iter()
		.map(|&word| u32::from_le_bytes(word))
		.collect::<Vec<_>>();
	if words.get(1).copied().unwrap_or(u32::MAX) > 0x0001_0500 {
		return Err(format!("bounded {stem} exceeds SPIR-V 1.5").into());
	}
	let mut cursor = 5;
	while cursor < words.len() {
		let instruction = words[cursor];
		let count = (instruction >> 16) as usize;
		if count == 0 || cursor + count > words.len() {
			return Err(format!("bounded {stem} has malformed SPIR-V").into());
		}
		if instruction as u16 == 17 && words[cursor + 1] == 5302 {
			return Err(format!("bounded {stem} still requires RuntimeDescriptorArray").into());
		}
		cursor += count;
	}
	Ok(())
}

fn validate_reflection(
	path: &Path,
	operation: &Value,
	kernel_name: &str,
	dtype: &str,
	workgroup_size: &Value,
	storage_count: u32,
) -> Result<(), Box<dyn std::error::Error>> {
	let name = string_at(operation, "name")?;
	let kind = string_at(operation, "kind")?;
	let reflection: Value = serde_json::from_slice(&fs::read(path)?)?;
	let entry_points = array_at(&reflection, "entryPoints")?;
	let entry = entry_points
		.iter()
		.find(|entry| entry["name"] == ENTRY_POINT)
		.ok_or_else(|| format!("reflection does not contain {ENTRY_POINT}"))?;
	require_equal(&entry["stage"], &json!("compute"), "entry-point stage")?;
	require_equal(
		&entry["threadGroupSize"],
		workgroup_size,
		"thread-group size",
	)?;
	let attributes = entry["userAttribs"]
		.as_array()
		.ok_or("reflection does not contain OA kernel attributes")?;
	let variant = operation["variant"].as_str().unwrap_or("generic");
	for (attribute_name, argument) in [
		("kernel_name", kernel_name),
		("domain", "matrix"),
		("variant", variant),
		("dtype", dtype),
		("status", "experimental"),
	] {
		let attribute = named(attributes, attribute_name)?;
		require_equal(
			&attribute["arguments"],
			&json!([argument]),
			"kernel attribute",
		)?;
	}

	let parameters = array_at(&reflection, "parameters")?;
	let push = named(parameters, "push")?;
	require_equal(
		&push["binding"],
		&json!({"kind": "pushConstantBuffer", "index": 0}),
		"push-constant binding",
	)?;
	let fields = push
		.pointer("/type/elementType/fields")
		.and_then(Value::as_array)
		.ok_or("reflection does not describe push-constant fields")?;
	let expected_fields = expected_push_fields(kind)?;
	if fields.len() != expected_fields.len() {
		return Err(
			format!(
				"matrix.{name} push constants contain {} fields; expected {}",
				fields.len(),
				expected_fields.len()
			)
			.into(),
		);
	}
	for (field, (field_name, scalar_type, offset)) in fields.iter().zip(expected_fields) {
		require_equal(
			&field["name"],
			&json!(field_name),
			"push-constant field name",
		)?;
		require_equal(
			&field["type"]["scalarType"],
			&json!(scalar_type),
			"push-constant field type",
		)?;
		require_equal(
			&field["binding"]["offset"],
			&json!(offset),
			"push-constant offset",
		)?;
		require_equal(&field["binding"]["size"], &json!(4), "push-constant size")?;
	}

	let storage = named(parameters, "storage_buffers")?;
	require_equal(
		&storage["binding"],
		&json!({"kind": "descriptorTableSlot", "index": 0}),
		"storage-buffer binding",
	)?;
	require_equal(
		&storage["type"]["kind"],
		&json!("array"),
		"storage-buffer array",
	)?;
	require_equal(
		&storage["type"]["elementCount"],
		&json!(storage_count),
		"storage-buffer count",
	)?;
	require_equal(
		&storage["type"]["elementType"]["baseShape"],
		&json!("byteAddressBuffer"),
		"storage-buffer shape",
	)?;
	require_equal(
		&storage["type"]["elementType"]["access"],
		&json!("readWrite"),
		"storage-buffer access",
	)?;
	Ok(())
}

fn build_schema_shader(
	operation: &Value,
	domain: &str,
	dtype: &str,
	workgroup_size: &Value,
	build: &ShaderBuild<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
	let name = string_at(operation, "name")?;
	let source = string_at(operation, "source")?;
	println!("cargo:rerun-if-changed={source}");
	let spirv = build
		.output_directory
		.join(format!("{domain}_{name}_{dtype}.spv"));
	let reflection = build
		.output_directory
		.join(format!("{domain}_{name}_{dtype}.reflection.json"));

	let mut command = Command::new(build.slangc);
	command.arg(source).args([
		"-entry",
		ENTRY_POINT,
		"-stage",
		"compute",
		"-target",
		"spirv",
		"-profile",
		"glsl_460",
		"-capability",
		build.spirv_capability,
		"-fvk-use-entrypoint-name",
		"-warnings-as-errors",
		"all",
		"-I",
		"src/slang/core",
		"-I",
		"src/slang/core/math",
		"-I",
		"src/slang/core/rng",
		"-I",
		"src/slang/matrix/rng",
		"-I",
		"src/slang/cryptography/hash",
		"-I",
		"src/slang/cryptography/pqc/common",
		"-I",
		"src/slang/cryptography/pqc/mldsa",
		"-I",
		"src/slang/cryptography/pqc/mlkem",
	]);
	if let Some(optimization) = operation["slang_optimization"].as_str() {
		match optimization {
			"O0" => {
				command.arg("-O0");
			}
			_ => return Err(format!("unsupported Slang optimization {optimization}").into()),
		}
	}
	if let Some(capabilities) = operation["slang_capabilities"].as_array() {
		for capability in capabilities {
			command.arg("-capability").arg(
				capability
					.as_str()
					.ok_or_else(|| format!("{domain}.{name} has a non-string Slang capability"))?,
			);
		}
	}
	let slang_output = command
		.arg("-reflection-json")
		.arg(&reflection)
		.arg("-o")
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.slangc))?;
	if !slang_output.status.success() {
		return Err(
			format!(
				"{:?} failed for {domain}.{name} with {}\n{}",
				build.slangc,
				slang_output.status,
				String::from_utf8_lossy(&slang_output.stderr)
			)
			.into(),
		);
	}

	validate_schema_reflection(&reflection, operation, domain, dtype, workgroup_size)?;
	let validation_output = Command::new(build.spirv_val)
		.args(["--target-env", build.vulkan_target])
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.spirv_val))?;
	if !validation_output.status.success() {
		return Err(
			format!(
				"{:?} failed for {domain}.{name} with {}\n{}",
				build.spirv_val,
				validation_output.status,
				String::from_utf8_lossy(&validation_output.stderr)
			)
			.into(),
		);
	}
	build_bounded_shader(
		source,
		&format!("{domain}_{name}_{dtype}"),
		&reflection,
		operation,
		build,
	)?;
	Ok(())
}

fn build_schema_pqc_shader(
	operation: &Value,
	domain: &str,
	dtype: &str,
	workgroup_size: &Value,
	build: &ShaderBuild<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
	let name = string_at(operation, "name")?;
	let source = string_at(operation, "source")?;
	println!("cargo:rerun-if-changed={source}");
	let spirv = build
		.output_directory
		.join(format!("{domain}_{name}_{dtype}.spv"));
	let reflection = build
		.output_directory
		.join(format!("{domain}_{name}_{dtype}.reflection.json"));

	let mut command = Command::new(build.slangc);
	command.arg(source).args([
		"-entry",
		ENTRY_POINT,
		"-stage",
		"compute",
		"-target",
		"spirv",
		"-profile",
		"glsl_460",
		"-capability",
		build.spirv_capability,
		"-fvk-use-entrypoint-name",
		"-warnings-as-errors",
		"all",
		"-I",
		"src/slang/core",
		"-I",
		"src/slang/core/math",
		"-I",
		"src/slang/cryptography/hash",
		"-I",
		"src/slang/cryptography/pqc/common",
		"-I",
		"src/slang/cryptography/pqc/mldsa",
		"-I",
		"src/slang/cryptography/pqc/mlkem",
	]);
	if let Some(capabilities) = operation["slang_capabilities"].as_array() {
		for capability in capabilities {
			command.arg("-capability").arg(
				capability
					.as_str()
					.ok_or_else(|| format!("{domain}.{name} has a non-string Slang capability"))?,
			);
		}
	}
	let slang_output = command
		.arg("-reflection-json")
		.arg(&reflection)
		.arg("-o")
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.slangc))?;
	if !slang_output.status.success() {
		return Err(
			format!(
				"{:?} failed for {domain}.{name} with {}\n{}",
				build.slangc,
				slang_output.status,
				String::from_utf8_lossy(&slang_output.stderr)
			)
			.into(),
		);
	}

	validate_schema_reflection(&reflection, operation, domain, dtype, workgroup_size)?;
	let validation_output = Command::new(build.spirv_val)
		.args(["--target-env", build.vulkan_target])
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.spirv_val))?;
	if !validation_output.status.success() {
		return Err(
			format!(
				"{:?} failed for {domain}.{name} with {}\n{}",
				build.spirv_val,
				validation_output.status,
				String::from_utf8_lossy(&validation_output.stderr)
			)
			.into(),
		);
	}
	build_bounded_shader(
		source,
		&format!("{domain}_{name}_{dtype}"),
		&reflection,
		operation,
		build,
	)?;
	Ok(())
}

fn validate_schema_reflection(
	path: &Path,
	operation: &Value,
	domain: &str,
	dtype: &str,
	workgroup_size: &Value,
) -> Result<(), Box<dyn std::error::Error>> {
	let name = string_at(operation, "name")?;
	let reflection: Value = serde_json::from_slice(&fs::read(path)?)?;
	let entry = array_at(&reflection, "entryPoints")?
		.iter()
		.find(|entry| entry["name"] == ENTRY_POINT)
		.ok_or_else(|| format!("reflection does not contain {ENTRY_POINT}"))?;
	require_equal(&entry["stage"], &json!("compute"), "entry-point stage")?;
	require_equal(
		&entry["threadGroupSize"],
		workgroup_size,
		"thread-group size",
	)?;
	let attributes = entry["userAttribs"]
		.as_array()
		.ok_or("reflection does not contain OA kernel attributes")?;
	let variant = operation["variant"].as_str().unwrap_or("generic");
	let reflection_name = operation["reflection_name"].as_str().unwrap_or(name);
	for (attribute_name, argument) in [
		("kernel_name", reflection_name),
		("domain", domain),
		("variant", variant),
		("dtype", dtype),
		("status", "experimental"),
	] {
		let attribute = named(attributes, attribute_name)?;
		require_equal(
			&attribute["arguments"],
			&json!([argument]),
			"kernel attribute",
		)?;
	}

	let parameters = array_at(&reflection, "parameters")?;
	let push = named(parameters, "push")?;
	require_equal(
		&push["binding"],
		&json!({"kind": "pushConstantBuffer", "index": 0}),
		"push-constant binding",
	)?;
	let fields = push
		.pointer("/type/elementType/fields")
		.and_then(Value::as_array)
		.ok_or("reflection does not describe push-constant fields")?;
	let expected_fields = array_at(operation, "push_fields")?;
	if fields.len() != expected_fields.len() {
		return Err(
			format!(
				"{domain}.{name} push constants contain {} fields; expected {}",
				fields.len(),
				expected_fields.len()
			)
			.into(),
		);
	}
	for (index, (field, expected)) in fields.iter().zip(expected_fields).enumerate() {
		let pair = expected
			.as_array()
			.ok_or("ML push-field schema entry is not an array")?;
		let field_name = pair[0]
			.as_str()
			.ok_or("ML push-field name is not a string")?;
		let scalar_type = pair[1]
			.as_str()
			.ok_or("ML push-field type is not a string")?;
		require_equal(
			&field["name"],
			&json!(field_name),
			"push-constant field name",
		)?;
		require_equal(
			&field["type"]["scalarType"],
			&json!(scalar_type),
			"push-constant field type",
		)?;
		require_equal(
			&field["binding"]["offset"],
			&json!(u32::try_from(index)? * 4),
			"push-constant offset",
		)?;
		require_equal(&field["binding"]["size"], &json!(4), "push-constant size")?;
	}

	let storage = named(parameters, "storage_buffers")?;
	require_equal(
		&storage["binding"],
		&json!({"kind": "descriptorTableSlot", "index": 0}),
		"storage-buffer binding",
	)?;
	require_equal(
		&storage["type"]["kind"],
		&json!("array"),
		"storage-buffer array",
	)?;
	require_equal(
		&storage["type"]["elementCount"],
		&json!(0),
		"runtime storage-buffer count",
	)?;
	require_equal(
		&storage["type"]["elementType"]["baseShape"],
		&json!("byteAddressBuffer"),
		"storage-buffer shape",
	)?;
	require_equal(
		&storage["type"]["elementType"]["access"],
		&json!("readWrite"),
		"storage-buffer access",
	)?;
	Ok(())
}

fn expected_push_fields(kind: &str) -> Result<Vec<(&'static str, &'static str, u32)>, String> {
	match kind {
		"binary" => Ok(vec![
			("left_index", "uint32", 0),
			("right_index", "uint32", 4),
			("output_index", "uint32", 8),
			("element_count", "uint32", 12),
		]),
		"unary" => Ok(vec![
			("input_index", "uint32", 0),
			("output_index", "uint32", 4),
			("element_count", "uint32", 8),
		]),
		"unary_scalar" => Ok(vec![
			("input_index", "uint32", 0),
			("output_index", "uint32", 4),
			("element_count", "uint32", 8),
			("scalar", "float32", 12),
		]),
		"mat_mul_nt" => Ok(vec![
			("left_index", "uint32", 0),
			("right_index", "uint32", 4),
			("output_index", "uint32", 8),
			("m", "uint32", 12),
			("n", "uint32", 16),
			("k", "uint32", 20),
		]),
		_ => Err(format!("unsupported matrix operation kind {kind}")),
	}
}

fn array_at<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
	value[key]
		.as_array()
		.ok_or_else(|| format!("field {key} is not an array"))
}

fn string_at<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
	value[key]
		.as_str()
		.ok_or_else(|| format!("field {key} is not a string"))
}

fn named<'a>(values: &'a [Value], name: &str) -> Result<&'a Value, String> {
	values
		.iter()
		.find(|value| value["name"] == name)
		.ok_or_else(|| format!("reflection does not contain parameter {name}"))
}

fn require_equal(actual: &Value, expected: &Value, label: &str) -> Result<(), String> {
	if actual == expected {
		Ok(())
	} else {
		Err(format!(
			"unexpected {label}: expected {expected}, found {actual}"
		))
	}
}
