use std::{
	env, fs,
	path::{Path, PathBuf},
	process::Command,
};

use serde_json::{Value, json};

const SCHEMA: &str = "tools/gen/fn/schema/matrix_elemwise.json";
const BLAS_SCHEMA: &str = "tools/gen/fn/schema/matrix_blas.json";
const REDUCE_SCHEMA: &str = "tools/gen/fn/schema/matrix_reduce.json";
const RNG_SCHEMA: &str = "tools/gen/fn/schema/matrix_rng.json";
const ML_SCHEMA: &str = "tools/gen/fn/schema/ml_training.json";
const AUDIO_SCHEMA: &str = "tools/gen/fn/schema/audio.json";
const CRYPTOGRAPHY_HASH_SCHEMA: &str = "tools/gen/fn/schema/cryptography_hash.json";
const IMAGE_SCHEMA: &str = "tools/gen/fn/schema/image.json";
const VISION_SCHEMA: &str = "tools/gen/fn/schema/vision_detection.json";
const GENERATOR: &str = "tools/gen/fn/generate.py";
const STORAGE: &str = "src/slang/core/math/storage.slang";
const ATTRIBUTES: &str = "src/slang/core/attributes.slang";
const ACTIVATIONS: &str = "src/slang/core/math/activations.slang";
const PHILOX: &str = "src/slang/core/rng/philox.slang";
const DROPOUT_RNG: &str = "src/slang/matrix/rng/dropout_rng.slang";
const CRYPTOGRAPHY_KECCAK: &str = "src/slang/cryptography/hash/keccak.slang";
const ENTRY_POINT: &str = "main";

struct ShaderBuild<'a> {
	output_directory: &'a Path,
	slangc: &'a std::ffi::OsStr,
	spirv_val: &'a std::ffi::OsStr,
}

fn main() {
	if let Err(error) = build_shaders() {
		panic!("OA shader build failed: {error}");
	}
}

fn build_shaders() -> Result<(), Box<dyn std::error::Error>> {
	for source in [
		SCHEMA,
		BLAS_SCHEMA,
		REDUCE_SCHEMA,
		RNG_SCHEMA,
		ML_SCHEMA,
		AUDIO_SCHEMA,
		CRYPTOGRAPHY_HASH_SCHEMA,
		IMAGE_SCHEMA,
		VISION_SCHEMA,
		GENERATOR,
		STORAGE,
		ATTRIBUTES,
		ACTIVATIONS,
		PHILOX,
		DROPOUT_RNG,
		CRYPTOGRAPHY_KECCAK,
	] {
		println!("cargo:rerun-if-changed={source}");
	}
	println!("cargo:rerun-if-env-changed=SLANGC");
	println!("cargo:rerun-if-env-changed=SPIRV_VAL");
	println!("cargo:rerun-if-env-changed=PYTHON");
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
	let build = ShaderBuild {
		output_directory: &output_directory,
		slangc: &slangc,
		spirv_val: &spirv_val,
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
	Ok(())
}

fn verify_generated_sources() -> Result<(), Box<dyn std::error::Error>> {
	let python = env::var_os("PYTHON").unwrap_or_else(|| "python3".into());
	let output = Command::new(&python)
		.args([GENERATOR, "--check"])
		.output()
		.map_err(|source| format!("could not execute {python:?}: {source}"))?;
	if !output.status.success() {
		return Err(format!(
			"generated operation sources are stale; run `python3 {GENERATOR}`\n{}",
			String::from_utf8_lossy(&output.stderr)
		)
		.into());
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
			"spirv_1_6",
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
		return Err(format!(
			"{:?} failed for matrix.{name} with {}\n{}",
			build.slangc,
			slang_output.status,
			String::from_utf8_lossy(&slang_output.stderr)
		)
		.into());
	}

	validate_reflection(&reflection, operation, kernel_name, dtype, workgroup_size)?;

	let validation_output = Command::new(build.spirv_val)
		.args(["--target-env", "vulkan1.3"])
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.spirv_val))?;
	if !validation_output.status.success() {
		return Err(format!(
			"{:?} failed for matrix.{name} with {}\n{}",
			build.spirv_val,
			validation_output.status,
			String::from_utf8_lossy(&validation_output.stderr)
		)
		.into());
	}
	Ok(())
}

fn validate_reflection(
	path: &Path,
	operation: &Value,
	kernel_name: &str,
	dtype: &str,
	workgroup_size: &Value,
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
		return Err(format!(
			"matrix.{name} push constants contain {} fields; expected {}",
			fields.len(),
			expected_fields.len()
		)
		.into());
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

	let slang_output = Command::new(build.slangc)
		.arg(source)
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
			"spirv_1_6",
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
		return Err(format!(
			"{:?} failed for {domain}.{name} with {}\n{}",
			build.slangc,
			slang_output.status,
			String::from_utf8_lossy(&slang_output.stderr)
		)
		.into());
	}

	validate_schema_reflection(&reflection, operation, domain, dtype, workgroup_size)?;
	let validation_output = Command::new(build.spirv_val)
		.args(["--target-env", "vulkan1.3"])
		.arg(&spirv)
		.output()
		.map_err(|source| format!("could not execute {:?}: {source}", build.spirv_val))?;
	if !validation_output.status.success() {
		return Err(format!(
			"{:?} failed for {domain}.{name} with {}\n{}",
			build.spirv_val,
			validation_output.status,
			String::from_utf8_lossy(&validation_output.stderr)
		)
		.into());
	}
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
		return Err(format!(
			"{domain}.{name} push constants contain {} fields; expected {}",
			fields.len(),
			expected_fields.len()
		)
		.into());
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
