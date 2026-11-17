use super::*;

#[test]
fn default_and_explicit_profiles_are_exclusive() {
	assert_eq!(BuildProfile::parse(None).unwrap(), BuildProfile::Core);
	assert_eq!(
		BuildProfile::parse(Some("core")).unwrap(),
		BuildProfile::Core
	);
	assert_eq!(
		BuildProfile::parse(Some("compatibility")).unwrap(),
		BuildProfile::Compatibility
	);
	for invalid in [
		"",
		"automatic",
		"both",
		"strict",
		"CORE",
		"core,compatibility",
	] {
		assert!(BuildProfile::parse(Some(invalid)).is_err(), "{invalid}");
	}
}

#[test]
fn compatibility_adapts_source_without_a_core_compilation() {
	let output = TemporaryDirectory::new();
	let source = output.path().join("donor.slang");
	let body = "[[vk::binding(0, 0)]] RWByteAddressBuffer storage_buffers[];\n// unchanged math\n";
	fs::write(&source, body).unwrap();
	let mut build = ShaderBuild {
		output_directory: output.path(),
		slangc: std::ffi::OsStr::new("nonexistent-compiler"),
		spirv_val: std::ffi::OsStr::new("nonexistent-validator"),
		profile: BuildProfile::Compatibility,
		spirv_capability: "spirv_1_5",
		vulkan_target: "vulkan1.2",
	};
	let adapted = compute_source(source.to_str().unwrap(), "selected", 3, &build).unwrap();
	assert_eq!(
		fs::read_to_string(adapted).unwrap(),
		body.replace("storage_buffers[]", "storage_buffers[3]")
	);
	assert_eq!(fs::read_to_string(&source).unwrap(), body);
	assert!(compute_source(source.to_str().unwrap(), "zero", 0, &build).is_err());
	build.profile = BuildProfile::Core;
	assert_eq!(
		compute_source(source.to_str().unwrap(), "selected", 3, &build).unwrap(),
		source
	);
}

#[test]
fn schema_and_family_metadata_define_descriptor_capacity() {
	let operation = json!({"push_fields": [["input_index", "uint32"], ["output_idx", "uint32"], ["count", "uint32"]]});
	assert_eq!(schema_buffer_count(&operation).unwrap(), 2);
	let fields = expected_push_fields("mat_mul_nt", "bias_gelu").unwrap();
	let count = fields
		.iter()
		.filter(|(name, _, _)| name.ends_with("_index"))
		.count();
	assert_eq!(count, 4);
	assert_eq!(BuildProfile::Core.storage_count(4), 0);
	assert_eq!(BuildProfile::Compatibility.storage_count(4), 4);
}

#[test]
fn content_ids_include_only_selected_directory_artifacts() {
	let output = TemporaryDirectory::new();
	let core = output.path().join("core");
	let compatibility = output.path().join("compatibility");
	fs::create_dir_all(&core).unwrap();
	fs::create_dir_all(&compatibility).unwrap();
	fs::write(core.join("core_only.spv"), b"core").unwrap();
	fs::write(compatibility.join("selected.spv"), b"compatibility").unwrap();
	emit_shader_content_ids(&compatibility).unwrap();
	let ids = fs::read_to_string(compatibility.join("shader_content_ids.rs")).unwrap();
	assert!(ids.contains("SELECTED"));
	assert!(!ids.contains("CORE_ONLY"));
}

struct TemporaryDirectory(std::path::PathBuf);

impl TemporaryDirectory {
	fn new() -> Self {
		static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
		let nonce = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
		let path = env::temp_dir().join(format!("oa-shader-build-{}-{nonce}", std::process::id()));
		fs::create_dir(&path).unwrap();
		Self(path)
	}

	fn path(&self) -> &Path {
		&self.0
	}
}

impl Drop for TemporaryDirectory {
	fn drop(&mut self) {
		let _ = fs::remove_dir_all(&self.0);
	}
}
