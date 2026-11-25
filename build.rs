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
const CRYPTOGRAPHY_MERKLE_SCHEMA: &str = "tool/gen/fn/schema/crypto/crypto_merkle.json";
const CRYPTOGRAPHY_NTT_SCHEMA: &str = "tool/gen/fn/schema/crypto/crypto_ntt.json";
const CRYPTOGRAPHY_HASH_SCHEMA: &str = "tool/gen/fn/schema/crypto/crypto_hash.json";
const CRYPTOGRAPHY_BIGINT_SCHEMA: &str = "tool/gen/fn/schema/crypto/crypto_bigint.json";
const CRYPTOGRAPHY_PQC_SCHEMA: &str = "tool/gen/fn/schema/crypto/crypto_pqc.json";
const IMAGE_SCHEMA: &str = "tool/gen/fn/schema/image/image.json";
const VISION_SCHEMA: &str = "tool/gen/fn/schema/vision/vision_detection.json";
const VIDEO_SCHEMA: &str = "tool/gen/fn/schema/video/video.json";
const GENERATOR: &str = "tool/gen/fn/generate.py";
const STORAGE: &str = "src/slang/core/math/storage.slang";
const ATTRIBUTES: &str = "src/slang/core/attributes.slang";
const LINEAR_FEATURES: &str = "src/slang/ml/nn/attention/linear_features.slang";
const TRADING_COMMON: &str = "src/slang/ml/environment/trading/common.slang";
const ACTIVATIONS: &str = "src/slang/core/math/activations.slang";
const ELEMENTWISE: &str = "src/slang/core/math/elementwise.slang";
const SSM_MATH: &str = "src/slang/core/math/ssm_math.slang";
const PHILOX: &str = "src/slang/core/rng/philox.slang";
const DROPOUT_RNG: &str = "src/slang/matrix/rng/dropout_rng.slang";
const CRYPTOGRAPHY_KECCAK: &str = "src/slang/crypto/hash/keccak.slang";
const CRYPTOGRAPHY_PQC_MLDSA_FIELD: &str = "src/slang/crypto/pqc/mldsa/mldsa_field.slang";
const CRYPTOGRAPHY_PQC_MLDSA_NTT: &str = "src/slang/crypto/pqc/mldsa/mldsa_ntt.slang";
const CRYPTOGRAPHY_PQC_MLDSA_PACKING: &str = "src/slang/crypto/pqc/mldsa/mldsa_packing.slang";
const CRYPTOGRAPHY_PQC_MLDSA_SAMPLING: &str = "src/slang/crypto/pqc/mldsa/mldsa_sampling.slang";
const CRYPTOGRAPHY_PQC_SHAKE: &str = "src/slang/crypto/pqc/common/shake.slang";
const CRYPTOGRAPHY_PQC_VERIFICATION: &str = "src/slang/crypto/pqc/mldsa/mldsa_verification.slang";
const CRYPTOGRAPHY_PQC_VERIFICATION_PAR: &str =
	"src/slang/crypto/pqc/mldsa/mldsa_verification_par.slang";
const CRYPTOGRAPHY_PQC_MLDSA_KEYGEN_PAR: &str = "src/slang/crypto/pqc/mldsa/mldsa_keygen_par.slang";
const CRYPTOGRAPHY_PQC_MLDSA_SIGN_PAR: &str = "src/slang/crypto/pqc/mldsa/mldsa_sign_par.slang";
const CRYPTOGRAPHY_PQC_SHA3: &str = "src/slang/crypto/pqc/common/sha3.slang";
const ENTRY_POINT: &str = "main";

struct ShaderBuild<'a> {
	output_directory: &'a Path,
	slangc: &'a std::ffi::OsStr,
	spirv_val: &'a std::ffi::OsStr,
	profile: BuildProfile,
	spirv_capability: &'static str,
	vulkan_target: &'static str,
	/// Documentation-only builds (docs.rs sets `DOCS_RS`) have no Slang
	/// toolchain: emit placeholder artifacts so `include_bytes!` sites
	/// compile without executing shader compilation or validation.
	stub: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuildProfile {
	Core,
	Compatibility,
}

impl BuildProfile {
	fn parse(value: Option<&str>) -> Result<Self, String> {
		match value {
			None | Some("core") => Ok(Self::Core),
			Some("compatibility") => Ok(Self::Compatibility),
			Some(value) => Err(format!(
				"unsupported OA_BUILD_PROFILE={value:?}; expected core or compatibility"
			)),
		}
	}

	fn name(self) -> &'static str {
		match self {
			Self::Core => "core",
			Self::Compatibility => "compatibility",
		}
	}

	fn shader_target(self) -> Result<(&'static str, &'static str), Box<dyn std::error::Error>> {
		// Retain the Core SPIR-V 1.5 audit override without changing its binding ABI.
		match (self, env::var("OA_SPIRV_TARGET").as_deref()) {
			(Self::Core, Err(env::VarError::NotPresent) | Ok("1.6")) => Ok(("spirv_1_6", "vulkan1.3")),
			(_, Err(env::VarError::NotPresent) | Ok("1.5")) => Ok(("spirv_1_5", "vulkan1.2")),
			(_, Ok(value)) => Err(
				format!(
					"unsupported OA_SPIRV_TARGET={value:?} for {}; expected {}",
					self.name(),
					if self == Self::Core {
						"1.5 or 1.6"
					} else {
						"1.5"
					}
				)
				.into(),
			),
			(_, Err(error)) => Err(format!("invalid OA_SPIRV_TARGET: {error}").into()),
		}
	}

	fn storage_count(self, buffer_count: u32) -> u32 {
		match self {
			Self::Core => 0,
			Self::Compatibility => buffer_count,
		}
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
const RENDER_SURFACE_VERT: &str = "src/slang/render/surface.vert.slang";
const UI_CANVAS_GRID_VERT: &str = "src/slang/ui/canvas_grid.vert.slang";
const UI_CANVAS_GRID_FRAG: &str = "src/slang/ui/canvas_grid.frag.slang";
const RENDER_SURFACE_FRAG: &str = "src/slang/render/surface.frag.slang";

const UI_RASTER_FULLSCREEN_VERT: &str = "src/slang/ui/fullscreen.vert.slang";
const UI_RASTER_CLEAR_FRAG: &str = "src/slang/ui/clear.frag.slang";
const UI_RASTER_RECT_FRAG: &str = "src/slang/ui/rect.frag.slang";
const UI_RASTER_RECT_OUTLINE_FRAG: &str = "src/slang/ui/rect_outline.frag.slang";
const UI_RASTER_LINE_FRAG: &str = "src/slang/ui/line.frag.slang";
const UI_RASTER_TRIANGLE_FRAG: &str = "src/slang/ui/triangle.frag.slang";
const UI_RASTER_MASK_FRAG: &str = "src/slang/ui/mask.frag.slang";
const UI_RASTER_BLIT_TEXTURE_FRAG: &str = "src/slang/ui/blit_texture.frag.slang";
const UI_RASTER_BLIT_PLANAR_FRAG: &str = "src/slang/ui/blit_planar.frag.slang";
const UI_RASTER_BLIT_YUV_FRAG: &str = "src/slang/ui/blit_yuv.frag.slang";
const UI_RASTER_HEATMAP_FRAG: &str = "src/slang/ui/heatmap.frag.slang";
const UI_RASTER_WAVEFORM_FRAG: &str = "src/slang/ui/waveform.frag.slang";
const UI_RASTER_PLOT_LINE_FRAG: &str = "src/slang/ui/plot_line.frag.slang";
const UI_RASTER_GLYPHS_VERT: &str = "src/slang/ui/glyphs.vert.slang";
const UI_RASTER_GLYPHS_FRAG: &str = "src/slang/ui/glyphs.frag.slang";
const UI_RASTER_RECT_OUTLINES_VERT: &str = "src/slang/ui/rect_outlines.vert.slang";
const UI_RASTER_RECT_OUTLINES_FRAG: &str = "src/slang/ui/rect_outlines.frag.slang";
const UI_COMPOSITOR_COMMON: &str = "src/slang/ui/compositor_common.slang";
const VIDEO_CVT_RGBA_TO_NV12: &str = "src/slang/video/cvt_rgba_to_nv12.slang";
const VIDEO_CVT_RGBA64_TO_P010: &str = "src/slang/video/cvt_rgba64_to_p010.slang";
const VIDEO_CVT_YUV420_TO_RGBA8: &str = "src/slang/video/cvt_yuv420_to_rgba8.slang";

fn build_shaders() -> Result<(), Box<dyn std::error::Error>> {
	println!("cargo:rerun-if-env-changed=OA_BUILD_PROFILE");
	println!("cargo:rerun-if-env-changed=DOCS_RS");
	println!("cargo:rustc-check-cfg=cfg(oa_profile, values(\"core\", \"compatibility\"))");
	let selected = env::var("OA_BUILD_PROFILE");
	let profile = match &selected {
		Ok(value) => BuildProfile::parse(Some(value))?,
		Err(env::VarError::NotPresent) => BuildProfile::parse(None)?,
		Err(error) => return Err(format!("invalid OA_BUILD_PROFILE: {error}").into()),
	};
	let (spirv_capability, vulkan_target) = profile.shader_target()?;
	println!("cargo:rustc-cfg=oa_profile=\"{}\"", profile.name());
	for source in [
		SCHEMA,
		BLAS_SCHEMA,
		REDUCE_SCHEMA,
		RNG_SCHEMA,
		INDEX_SCHEMA,
		ML_SCHEMA,
		AUDIO_SCHEMA,
		CRYPTOGRAPHY_HASH_SCHEMA,
		CRYPTOGRAPHY_BIGINT_SCHEMA,
		CRYPTOGRAPHY_PQC_SCHEMA,
		IMAGE_SCHEMA,
		VISION_SCHEMA,
		GENERATOR,
		"tool/gen/fn/template/matrix",
		"tool/gen/fn/template/audio",
		"tool/gen/fn/template/image",
		"tool/gen/fn/template/vision",
		"tool/gen/fn/template/crypto",
		"tool/gen/fn/template/python",
		STORAGE,
		ATTRIBUTES,
		ACTIVATIONS,
		ELEMENTWISE,
		LINEAR_FEATURES,
		TRADING_COMMON,
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
		CRYPTOGRAPHY_PQC_VERIFICATION_PAR,
		CRYPTOGRAPHY_PQC_MLDSA_KEYGEN_PAR,
		CRYPTOGRAPHY_PQC_MLDSA_SIGN_PAR,
		RENDER_VERT,
		RENDER_FRAG,
		RENDER_MATERIAL_COMMON,
		RENDER_BRDF,
		RENDER_LIGHTS,
		RENDER_FLAT_COLOR_VERT,
		RENDER_FLAT_COLOR_FRAG,
		RENDER_UNLIT_VERT,
		RENDER_UNLIT_FRAG,
		RENDER_SURFACE_VERT,
		RENDER_SURFACE_FRAG,
		UI_CANVAS_GRID_VERT,
		UI_CANVAS_GRID_FRAG,
		UI_COMPOSITOR_COMMON,
		UI_RASTER_FULLSCREEN_VERT,
		UI_RASTER_CLEAR_FRAG,
		UI_RASTER_RECT_FRAG,
		UI_RASTER_RECT_OUTLINE_FRAG,
		UI_RASTER_LINE_FRAG,
		UI_RASTER_TRIANGLE_FRAG,
		UI_RASTER_MASK_FRAG,
		UI_RASTER_BLIT_TEXTURE_FRAG,
		UI_RASTER_BLIT_PLANAR_FRAG,
		UI_RASTER_BLIT_YUV_FRAG,
		UI_RASTER_HEATMAP_FRAG,
		UI_RASTER_WAVEFORM_FRAG,
		UI_RASTER_PLOT_LINE_FRAG,
		UI_RASTER_GLYPHS_VERT,
		UI_RASTER_GLYPHS_FRAG,
		UI_RASTER_RECT_OUTLINES_VERT,
		UI_RASTER_RECT_OUTLINES_FRAG,
	] {
		println!("cargo:rerun-if-changed={source}");
	}
	println!("cargo:rerun-if-env-changed=SLANGC");
	println!("cargo:rerun-if-env-changed=SPIRV_VAL");
	println!("cargo:rerun-if-env-changed=PYTHON");
	println!("cargo:rerun-if-env-changed=OA_SPIRV_TARGET");
	// The published crate ships canonical generated output but excludes src/py;
	// the freshness gate only applies to a complete development checkout.
	if Path::new("src/py").is_dir() {
		verify_generated_sources()?;
	}

	let schema: Value = serde_json::from_slice(&fs::read(SCHEMA)?)?;
	let operations = array_at(&schema, "operations")?;
	let default_dtype = string_at(&schema, "dtype")?;
	let workgroup_size = &schema["workgroup_size"];
	if operations.is_empty() {
		return Err("matrix elementwise schema contains no operations".into());
	}
	let output_directory = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is unavailable")?)
		.join("shaders")
		.join(profile.name());
	fs::create_dir_all(&output_directory)?;
	// Only build-owned shader files in the selected directory are retired.
	// A profile switch cannot import an old or opposite-profile content ID.
	for entry in fs::read_dir(&output_directory)? {
		let path = entry?.path();
		if matches!(
			path.extension().and_then(|ext| ext.to_str()),
			Some("spv" | "json" | "slang")
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
	let build = ShaderBuild {
		output_directory: &output_directory,
		slangc: &slangc,
		spirv_val: &spirv_val,
		profile,
		spirv_capability,
		vulkan_target,
		stub: env::var_os("DOCS_RS").is_some(),
	};
	for operation in operations {
		// Lowering-only kernels own a handwritten Slang source and a fixed push
		// layout; contract variants generate their sources instead.
		if operation["lowering_only"].as_bool() == Some(true) {
			build_schema_shader(operation, "matrix", default_dtype, workgroup_size, &build)?;
			continue;
		}
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
	if let Some(candidates) = blas_schema["candidate_kernels"].as_array() {
		for candidate in candidates {
			build_shader(
				candidate,
				candidate,
				blas_dtype,
				blas_workgroup_size,
				"src/slang/matrix/blas",
				&build,
			)?;
		}
	}

	let reduce_schema: Value = serde_json::from_slice(&fs::read(REDUCE_SCHEMA)?)?;
	let reduce_operations = array_at(&reduce_schema, "operations")?;
	let reduce_dtype = string_at(&reduce_schema, "dtype")?;
	let reduce_workgroup_size = &reduce_schema["workgroup_size"];
	if reduce_operations.is_empty() {
		return Err("matrix Reduce schema contains no operations".into());
	}
	for operation in reduce_operations {
		// Composite records such as `mean` lower to existing kernels; there is no shader to build.
		if operation["source"].as_str().is_none() {
			continue;
		}
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
		// Composite records such as `split` lower to existing kernels; there is no shader to build.
		if operation["source"].as_str().is_none() {
			continue;
		}
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

	let merkle_schema: Value = serde_json::from_slice(&fs::read(CRYPTOGRAPHY_MERKLE_SCHEMA)?)?;
	println!("cargo:rerun-if-changed={CRYPTOGRAPHY_MERKLE_SCHEMA}");
	for kernel in array_at(&merkle_schema, "kernels")? {
		build_schema_shader(
			kernel,
			"cryptography",
			"u32",
			&kernel["workgroup_size"],
			&build,
		)?;
	}
	let ntt_schema: Value = serde_json::from_slice(&fs::read(CRYPTOGRAPHY_NTT_SCHEMA)?)?;
	println!("cargo:rerun-if-changed={CRYPTOGRAPHY_NTT_SCHEMA}");
	for kernel in array_at(&ntt_schema, "kernels")? {
		build_schema_shader(
			kernel,
			"cryptography",
			"u32",
			&kernel["workgroup_size"],
			&build,
		)?;
	}
	let bigint_schema: Value = serde_json::from_slice(&fs::read(CRYPTOGRAPHY_BIGINT_SCHEMA)?)?;
	for kernel in array_at(&bigint_schema, "kernels")? {
		build_schema_shader(
			kernel,
			"cryptography",
			"u32",
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
			kernel["dtype"].as_str().unwrap_or(cryptography_dtype),
			&kernel["workgroup_size"],
			&build,
		)?;
	}

	// Imported private formulas are build inputs too. Track each family and
	// directory membership so additions/removals cannot leave stale artifacts.
	for directory in [
		"src/slang/crypto/bigint",
		"src/slang/crypto/hash",
		"src/slang/crypto/pqc/mlkem",
		"src/slang/crypto/pqc/mldsa",
		"src/slang/video",
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
	for helper in [
		"src/slang/core/math/wide_u32.slang",
		"src/slang/core/math/bigint_words.slang",
		"src/slang/core/math/bigint_modular.slang",
		"src/slang/core/math/bigint_montgomery.slang",
		"src/slang/core/math/bigint_barrett.slang",
		"src/slang/core/math/bigint_division.slang",
		"src/slang/core/math/prime31.slang",
		"src/slang/core/math/prime62.slang",
		"src/slang/core/math/poseidon2_field.slang",
		"src/slang/core/math/poseidon2.slang",
		"src/slang/core/math/merkle_general.slang",
		"src/slang/core/math/merkle_poseidon2.slang",
	] {
		println!("cargo:rerun-if-changed={helper}");
	}
	let cryptography_pqc_schema: Value = serde_json::from_slice(&fs::read(CRYPTOGRAPHY_PQC_SCHEMA)?)?;
	let cryptography_pqc_kernels = array_at(&cryptography_pqc_schema, "kernels")?;
	let cryptography_pqc_dtype = string_at(&cryptography_pqc_schema, "dtype")?;
	if cryptography_pqc_schema["private_kernels"]
		.as_array()
		.is_none_or(Vec::is_empty)
	{
		return Err("Cryptography PQC schema contains no private kernels".into());
	}
	for kernel in cryptography_pqc_kernels.iter().chain(
		cryptography_pqc_schema["private_kernels"]
			.as_array()
			.into_iter()
			.flatten(),
	) {
		build_schema_shader(
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
		let kernel_dtype = kernel["dtype"].as_str().unwrap_or(image_dtype);
		build_schema_shader(
			kernel,
			"image",
			kernel_dtype,
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

	println!("cargo:rerun-if-changed={VIDEO_SCHEMA}");
	let video_schema: Value = serde_json::from_slice(&fs::read(VIDEO_SCHEMA)?)?;
	for kernel in array_at(&video_schema, "kernels")? {
		if kernel["registered"].as_bool() == Some(false) {
			continue;
		}
		build_schema_shader(
			kernel,
			"video",
			kernel["dtype"].as_str().unwrap_or("u8"),
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
	// Surface — opaque base variant (no texture defines).
	build_graphics_shader("surface_vert", RENDER_SURFACE_VERT, "vertex", &build)?;
	build_graphics_shader("surface_frag", RENDER_SURFACE_FRAG, "fragment", &build)?;
	build_graphics_shader("ui_canvas_grid_vert", UI_CANVAS_GRID_VERT, "vertex", &build)?;
	build_graphics_shader(
		"ui_canvas_grid_frag",
		UI_CANVAS_GRID_FRAG,
		"fragment",
		&build,
	)?;
	// UI raster shaders — one graphics pair per compositor family. The
	// fullscreen vertex stage is shared by every per-pixel family; the two
	// instanced families (glyphs, detection outlines) bring their own.
	build_graphics_shader(
		"ui_raster_fullscreen_vert",
		UI_RASTER_FULLSCREEN_VERT,
		"vertex",
		&build,
	)?;
	for (stem, stage, source) in [
		("ui_raster_clear_frag", "fragment", UI_RASTER_CLEAR_FRAG),
		("ui_raster_rect_frag", "fragment", UI_RASTER_RECT_FRAG),
		(
			"ui_raster_rect_outline_frag",
			"fragment",
			UI_RASTER_RECT_OUTLINE_FRAG,
		),
		("ui_raster_line_frag", "fragment", UI_RASTER_LINE_FRAG),
		(
			"ui_raster_triangle_frag",
			"fragment",
			UI_RASTER_TRIANGLE_FRAG,
		),
		("ui_raster_mask_frag", "fragment", UI_RASTER_MASK_FRAG),
		(
			"ui_raster_blit_texture_frag",
			"fragment",
			UI_RASTER_BLIT_TEXTURE_FRAG,
		),
		(
			"ui_raster_blit_planar_frag",
			"fragment",
			UI_RASTER_BLIT_PLANAR_FRAG,
		),
		(
			"ui_raster_blit_yuv_frag",
			"fragment",
			UI_RASTER_BLIT_YUV_FRAG,
		),
		("ui_raster_heatmap_frag", "fragment", UI_RASTER_HEATMAP_FRAG),
		(
			"ui_raster_waveform_frag",
			"fragment",
			UI_RASTER_WAVEFORM_FRAG,
		),
		(
			"ui_raster_plot_line_frag",
			"fragment",
			UI_RASTER_PLOT_LINE_FRAG,
		),
		("ui_raster_glyphs_vert", "vertex", UI_RASTER_GLYPHS_VERT),
		("ui_raster_glyphs_frag", "fragment", UI_RASTER_GLYPHS_FRAG),
		(
			"ui_raster_rect_outlines_vert",
			"vertex",
			UI_RASTER_RECT_OUTLINES_VERT,
		),
		(
			"ui_raster_rect_outlines_frag",
			"fragment",
			UI_RASTER_RECT_OUTLINES_FRAG,
		),
	] {
		build_graphics_shader(stem, source, stage, &build)?;
	}
	// Video conversion requires the Core storage-image heap.
	if profile == BuildProfile::Core {
		build_video_shader("cvt_rgba_to_nv12", VIDEO_CVT_RGBA_TO_NV12, &build)?;
		build_video_shader("cvt_rgba64_to_p010", VIDEO_CVT_RGBA64_TO_P010, &build)?;
		build_video_shader("cvt_yuv420_to_rgba8", VIDEO_CVT_YUV420_TO_RGBA8, &build)?;
		build_video_shader(
			"cvt_ycbcr_to_rgba8",
			"src/slang/video/cvt_ycbcr_to_rgba8.slang",
			&build,
		)?;
	}
	emit_shader_content_ids(&output_directory)?;
	Ok(())
}

/// Write a minimal well-formed SPIR-V module header. Documentation-only
/// builds run without a Slang toolchain; `include_bytes!` sites only
/// require the artifact to exist.
fn write_stub_spirv(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
	const HEADER: [u32; 5] = [0x0723_0203, 0x0001_0600, 0, 1, 0];
	let mut bytes = Vec::with_capacity(HEADER.len() * 4);
	for word in HEADER {
		bytes.extend_from_slice(&word.to_le_bytes());
	}
	fs::write(path, bytes)?;
	Ok(())
}

/// Emit `OUT_DIR/shader_content_ids.rs` with an FNV-1a content hash per
/// produced `.spv`, keyed by the artifact stem in upper snake case.
fn emit_shader_content_ids(output_directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
	let mut entries = Vec::new();
	for entry in fs::read_dir(output_directory)? {
		let path = entry?.path();
		if path.extension().and_then(|ext| ext.to_str()) != Some("spv") {
			continue;
		}
		let bytes = fs::read(&path)?;
		let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
			(hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
		});
		let stem: String = path
			.file_stem()
			.and_then(|stem| stem.to_str())
			.ok_or("SPIR-V artifact name is not UTF-8")?
			.chars()
			.map(|ch| {
				if ch.is_ascii_alphanumeric() {
					ch.to_ascii_uppercase()
				} else {
					'_'
				}
			})
			.collect();
		entries.push((stem, hash));
	}
	entries.sort();
	let mut source = String::from("// @generated by build.rs; DO NOT EDIT.\n");
	for (name, hash) in entries {
		source.push_str(&format!("pub(crate) const {name}: u64 = 0x{hash:016x};\n"));
	}
	fs::write(output_directory.join("shader_content_ids.rs"), source)?;
	Ok(())
}

fn build_video_shader(
	stem: &str,
	source: &str,
	build: &ShaderBuild<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
	println!("cargo:rerun-if-changed={source}");
	let artifact = format!("video_{stem}");
	let spirv = build.output_directory.join(format!("{artifact}.spv"));
	let reflection_path = build
		.output_directory
		.join(format!("{artifact}.reflection.json"));
	if build.stub {
		return write_stub_spirv(&spirv);
	}
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
			"src/slang/video",
			"-I",
			"src/slang/core/math",
		])
		.arg("-reflection-json")
		.arg(&reflection_path)
		.arg("-o")
		.arg(&spirv)
		.output()?;
	if !output.status.success() {
		return Err(
			format!(
				"{artifact} Slang compilation failed: {}",
				String::from_utf8_lossy(&output.stderr)
			)
			.into(),
		);
	}
	let reflection: Value = serde_json::from_slice(&fs::read(&reflection_path)?)?;
	let parameters = array_at(&reflection, "parameters")?;
	require_equal(
		&named(parameters, "pc")?["type"]["elementType"]["kind"],
		&json!("struct"),
		"video push-constant layout",
	)?;
	require_equal(
		&named(parameters, "heap")?["binding"],
		&json!({"kind":"descriptorTableSlot","index":0}),
		"video storage-buffer binding",
	)?;
	// Kernels that sample or store images declare `storage_images` at binding 1;
	// buffer-only kernels omit it rather than bind an unused heap.
	if let Some(images) = find_named(parameters, "storage_images") {
		require_equal(
			&images["binding"],
			&json!({"kind":"descriptorTableSlot","index":1}),
			"video storage-image binding",
		)?;
	}
	if let Some(image) = find_named(parameters, "ycbcr_image") {
		require_equal(
			&image["binding"],
			&json!({"kind":"descriptorTableSlot", "space":1, "index":0}),
			"immutable YCbCr source binding",
		)?;
	}
	let output = Command::new(build.spirv_val)
		.args(["--target-env", build.vulkan_target])
		.arg(&spirv)
		.output()?;
	if !output.status.success() {
		return Err(
			format!(
				"{artifact} SPIR-V validation failed: {}",
				String::from_utf8_lossy(&output.stderr)
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
	if build.stub {
		return write_stub_spirv(&spirv);
	}
	// Slang lowers fragment discard to OpDemoteToHelperInvocation for SPIR-V
	// 1.6. The UI uses ordinary pixel rejection without the optional Vulkan
	// demote feature. SPIR-V 1.5 lowers the same discard to OpKill.
	let spirv_capability = if stage == "fragment" && build.spirv_capability == "spirv_1_6" {
		"spirv_1_5"
	} else {
		build.spirv_capability
	};
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
			spirv_capability,
			"-fvk-use-entrypoint-name",
			"-warnings-as-errors",
			"all",
			"-I",
			"src/slang/core",
			"-I",
			"src/slang/core/math",
			"-I",
			"src/slang/ui",
			"-I",
			"src/slang/video",
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
	let source = variant["source"].as_str().map_or_else(
		|| format!("{source_directory}/{source_stem}.gen.slang"),
		str::to_owned,
	);
	println!("cargo:rerun-if-changed={source}");
	let spirv = build
		.output_directory
		.join(format!("matrix_{name}_{dtype}.spv"));
	let reflection = build
		.output_directory
		.join(format!("matrix_{name}_{dtype}.reflection.json"));
	if build.stub {
		return write_stub_spirv(&spirv);
	}

	let fields = expected_push_fields(
		string_at(operation, "kind")?,
		variant["variant"].as_str().unwrap_or("generic"),
	)?;
	let buffer_count = u32::try_from(
		fields
			.iter()
			.filter(|(name, _, _)| name.ends_with("_index") || name.ends_with("_idx"))
			.count(),
	)?;
	let selected_source = compute_source(
		&source,
		&format!("matrix_{name}_{dtype}"),
		buffer_count,
		build,
	)?;
	let slang_output = Command::new(build.slangc)
		.arg(&selected_source)
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
			"src/slang/crypto/hash",
			"-I",
			source_directory,
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
		build.profile.storage_count(buffer_count),
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
	if build.profile == BuildProfile::Compatibility {
		validate_bounded_spirv(&spirv, name)?;
	}
	Ok(())
}

// Mechanical binding adaptation only; shader math and push ABI stay unchanged.
fn compute_source(
	source_path: &str,
	stem: &str,
	buffer_count: u32,
	build: &ShaderBuild<'_>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
	if build.profile == BuildProfile::Core {
		return Ok(PathBuf::from(source_path));
	}
	const UNSIZED: &str = "[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];";
	let source = fs::read_to_string(source_path)?;
	if buffer_count == 0 || source.matches(UNSIZED).count() != 1 {
		return Err(
			format!("{source_path} has no unique storage-buffer declaration or buffer metadata").into(),
		);
	}
	let bounded = source.replace(
		UNSIZED,
		&format!("[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[{buffer_count}];"),
	);
	let path = build.output_directory.join(format!("{stem}.slang"));
	fs::write(&path, bounded)?;
	Ok(path)
}

fn schema_buffer_count(operation: &Value) -> Result<u32, Box<dyn std::error::Error>> {
	let count = array_at(operation, "push_fields")?
		.iter()
		.filter(|field| {
			field[0]
				.as_str()
				.is_some_and(|name| name.ends_with("_index") || name.ends_with("_idx"))
		})
		.count();
	Ok(u32::try_from(count)?)
}

fn validate_bounded_spirv(spirv: &Path, stem: &str) -> Result<(), Box<dyn std::error::Error>> {
	let bytes = fs::read(spirv)?;
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
	let expected_fields = expected_push_fields(kind, variant)?;
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
	if build.stub {
		return write_stub_spirv(&spirv);
	}

	let buffer_count = schema_buffer_count(operation)?;
	let selected_source = compute_source(
		source,
		&format!("{domain}_{name}_{dtype}"),
		buffer_count,
		build,
	)?;
	let mut command = Command::new(build.slangc);
	if let Some(parent) = Path::new(source).parent() {
		command.arg("-I").arg(parent);
	}
	command.arg(&selected_source).args([
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
		"src/slang/crypto/hash",
		"-I",
		"src/slang/crypto/pqc/common",
		"-I",
		"src/slang/crypto/pqc/mldsa",
		"-I",
		"src/slang/crypto/pqc/mlkem",
		"-I",
		"src/slang/video",
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

	validate_schema_reflection(
		&reflection,
		operation,
		domain,
		dtype,
		workgroup_size,
		build.profile.storage_count(buffer_count),
	)?;
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
	if build.profile == BuildProfile::Compatibility {
		validate_bounded_spirv(&spirv, name)?;
	}
	Ok(())
}

fn validate_schema_reflection(
	path: &Path,
	operation: &Value,
	domain: &str,
	dtype: &str,
	workgroup_size: &Value,
	storage_count: u32,
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

fn expected_push_fields(
	kind: &str,
	variant: &str,
) -> Result<Vec<(&'static str, &'static str, u32)>, String> {
	if let ("matmul_nt", "strided" | "strided_tiled") = (kind, variant) {
		// Arbitrary-layout candidates carry the MatmulLayout fields of
		// all three operands: row/column/batch strides plus element offset.
		return Ok(vec![
			("left_index", "uint32", 0),
			("right_index", "uint32", 4),
			("output_index", "uint32", 8),
			("m", "uint32", 12),
			("n", "uint32", 16),
			("k", "uint32", 20),
			("a_row_stride", "uint32", 24),
			("a_col_stride", "uint32", 28),
			("a_batch_stride", "uint32", 32),
			("a_offset", "uint32", 36),
			("b_row_stride", "uint32", 40),
			("b_col_stride", "uint32", 44),
			("b_batch_stride", "uint32", 48),
			("b_offset", "uint32", 52),
			("c_row_stride", "uint32", 56),
			("c_col_stride", "uint32", 60),
			("c_batch_stride", "uint32", 64),
			("c_offset", "uint32", 68),
		]);
	}
	if let ("matmul_nt", "split_k") = (kind, variant) {
		// The partial kernel carries the K-slice extent after the canonical
		// problem fields; `group_id.z * k_chunk` bounds each slice.
		return Ok(vec![
			("left_index", "uint32", 0),
			("right_index", "uint32", 4),
			("partial_index", "uint32", 8),
			("m", "uint32", 12),
			("n", "uint32", 16),
			("k", "uint32", 20),
			("k_chunk", "uint32", 24),
		]);
	}
	if let ("matmul_nt", "split_k_reduce") = (kind, variant) {
		return Ok(vec![
			("partial_index", "uint32", 0),
			("output_index", "uint32", 4),
			("element_count", "uint32", 8),
			("split_k", "uint32", 12),
		]);
	}
	if let ("matmul_nt", "bias" | "bias_relu" | "bias_gelu" | "bias_silu") = (kind, variant) {
		return Ok(vec![
			("left_index", "uint32", 0),
			("right_index", "uint32", 4),
			("bias_index", "uint32", 8),
			("output_index", "uint32", 12),
			("m", "uint32", 16),
			("n", "uint32", 20),
			("k", "uint32", 24),
		]);
	}
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
		"matmul_nt" => Ok(vec![
			("left_index", "uint32", 0),
			("right_index", "uint32", 4),
			("output_index", "uint32", 8),
			("m", "uint32", 12),
			("n", "uint32", 16),
			("k", "uint32", 20),
		]),
		// Storage casts and the F16-storage reduction share the unary
		// [input, output, count] push contract.
		"cast" | "sum" => Ok(vec![
			("input_index", "uint32", 0),
			("output_index", "uint32", 4),
			("element_count", "uint32", 8),
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
	find_named(values, name).ok_or_else(|| format!("reflection does not contain parameter {name}"))
}

fn find_named<'a>(values: &'a [Value], name: &str) -> Option<&'a Value> {
	values.iter().find(|value| value["name"] == name)
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

#[cfg(test)]
#[path = "test/rs/runtime/build_profile_unit.rs"]
mod build_profile_tests;
