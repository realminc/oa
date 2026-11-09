//! `.vkcache` envelope contract tests — bounded parse, integrity, scope
//! isolation, merge preservation, atomic persistence and corruption fallback.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::jit::{JitIndexRecord, JitSpirvRecord};
use super::{
	SECTION_DRIVER_PIPELINE_CACHE, SECTION_FLAG_REQUIRED, SECTION_GEMM_MEASURED_ROUTES,
	SECTION_SCOPE, SectionUpdate, VkCacheFile, VkCacheScope, VkCacheSection, atomic_commit,
	driver_blob_matches_scope, jit, serialize, update,
};

static NEXT_DIR: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> crate::Path {
	let unique = NEXT_DIR.fetch_add(1, Ordering::Relaxed);
	let dir = crate::Path::new_from_path_buf(
		std::env::temp_dir().join(format!("oa-vkcache-test-{}-{unique}", std::process::id())),
	);
	crate::Filesystem::create_directories(&dir).expect("temp dir");
	dir
}

fn scope() -> VkCacheScope {
	VkCacheScope {
		key: 0x5c0e_0000_0000_0001,
		vendor_id: 0x8086,
		device_id: 0x0a16,
		driver_id: 6,
		driver_version: 0x0680_2005,
		api_version: 0x0040_0162,
		profile: 1,
		capability_identity: 0xb0a2_19cf_9ef5_22e6,
		pipeline_cache_uuid: [0x5au8; 16],
	}
}

fn other_scope() -> VkCacheScope {
	VkCacheScope {
		key: 0x5c0e_0000_0000_0002,
		vendor_id: 0x10de,
		..scope()
	}
}

fn artifact_with(entries: &[(u32, u32, &[u8])], scope: &VkCacheScope) -> Vec<u8> {
	let mut sections = BTreeMap::new();
	sections.insert(
		SECTION_SCOPE,
		VkCacheSection {
			version: 1,
			bytes: scope.encode().to_vec(),
		},
	);
	for (section_type, version, bytes) in entries {
		sections.insert(
			*section_type,
			VkCacheSection {
				version: *version,
				bytes: bytes.to_vec(),
			},
		);
	}
	serialize(&sections).expect("test artifact must serialize")
}

fn write_artifact(path: &crate::Path, bytes: &[u8]) {
	crate::Filesystem::write_binary(path, bytes).expect("artifact write");
}

#[test]
fn envelope_round_trips_sections() {
	let dir = temp_dir();
	let path = dir.join("oa-test.vkcache");
	let scope = scope();
	update(
		&path,
		&scope,
		&[SectionUpdate::replace(
			SECTION_DRIVER_PIPELINE_CACHE,
			1,
			vec![1, 2, 3, 4, 5],
		)],
	)
	.expect("commit must succeed");

	let file = VkCacheFile::open(&path, &scope).expect("artifact must open");
	assert_eq!(
		file
			.section(SECTION_DRIVER_PIPELINE_CACHE)
			.map(|section| section.bytes.as_slice()),
		Some(&[1u8, 2, 3, 4, 5][..])
	);
	assert!(file.section(SECTION_SCOPE).is_some());
	let _ = std::fs::remove_dir_all(dir.into_path_buf());
}

#[test]
fn envelope_rejects_truncation_and_corruption() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	let bytes = artifact_with(&[(SECTION_DRIVER_PIPELINE_CACHE, 1, &[9; 64])], &scope);
	write_artifact(&path, &bytes);
	assert!(VkCacheFile::open(&path, &scope).is_ok());

	write_artifact(&path, &bytes[..bytes.len() - 8]);
	assert!(
		VkCacheFile::open(&path, &scope).is_err(),
		"truncated artifact must fail closed"
	);

	let mut flipped = bytes.clone();
	let last = flipped.len() - 1;
	flipped[last] ^= 0xff;
	write_artifact(&path, &flipped);
	assert!(
		VkCacheFile::open(&path, &scope).is_err(),
		"payload corruption must fail checksum"
	);

	let mut bad_dir = bytes.clone();
	bad_dir[48] ^= 0xff;
	write_artifact(&path, &bad_dir);
	assert!(
		VkCacheFile::open(&path, &scope).is_err(),
		"directory corruption must fail checksum"
	);

	let mut bad_magic = bytes.clone();
	bad_magic[0] ^= 0xff;
	write_artifact(&path, &bad_magic);
	assert!(VkCacheFile::open(&path, &scope).is_err());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn envelope_rejects_foreign_scope() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	write_artifact(&path, &artifact_with(&[], &scope));
	assert!(VkCacheFile::open(&path, &other_scope()).is_err());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn envelope_skips_unknown_optional_and_rejects_required() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	let unknown_payload = [0xabu8; 37];
	write_artifact(&path, &artifact_with(&[(99, 7, &unknown_payload)], &scope));
	// Unknown optional sections are skipped on read but preserved on merge.
	let file = VkCacheFile::open(&path, &scope).expect("optional unknown must open");
	update(
		&path,
		&scope,
		&[SectionUpdate::replace(
			SECTION_GEMM_MEASURED_ROUTES,
			1,
			vec![7],
		)],
	)
	.expect("merge must succeed");
	let merged = VkCacheFile::open(&path, &scope).expect("merged artifact must open");
	let _ = &file;
	assert_eq!(
		merged.section(99).map(|section| section.bytes.as_slice()),
		Some(&unknown_payload[..]),
		"merge must preserve unmodified optional sections verbatim"
	);
	assert!(merged.section(SECTION_GEMM_MEASURED_ROUTES).is_some());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn envelope_unknown_required_section_fails_closed() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	let mut sections = BTreeMap::new();
	sections.insert(
		SECTION_SCOPE,
		VkCacheSection {
			version: 1,
			bytes: scope.encode().to_vec(),
		},
	);
	// Craft a required unknown entry by serializing then setting the flag in
	// the directory entry for type 77 and fixing the directory checksum.
	let mut writer_sections = sections.clone();
	writer_sections.insert(
		77,
		VkCacheSection {
			version: 1,
			bytes: vec![1],
		},
	);
	let mut bytes = serialize(&writer_sections).expect("serialize");
	// Directory entry layout: 48-byte header, then entries sorted by type —
	// SCOPE (1) is first, unknown (77) second at offset 48+48.
	let entry_flags_offset = 48 + 48 + 40;
	bytes[entry_flags_offset..entry_flags_offset + 4]
		.copy_from_slice(&SECTION_FLAG_REQUIRED.to_le_bytes());
	let directory_end = 48 + 96;
	let checksum = super::fnv1a64(&bytes[48..directory_end]);
	bytes[0x20..0x28].copy_from_slice(&checksum.to_le_bytes());
	write_artifact(&path, &bytes);
	assert!(VkCacheFile::open(&path, &scope).is_err());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn merge_replaces_removes_and_preserves() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	update(
		&path,
		&scope,
		&[SectionUpdate::replace(
			SECTION_DRIVER_PIPELINE_CACHE,
			1,
			vec![1],
		)],
	)
	.expect("commit");
	update(
		&path,
		&scope,
		&[SectionUpdate::replace(
			SECTION_GEMM_MEASURED_ROUTES,
			1,
			vec![2, 3],
		)],
	)
	.expect("commit");
	let file = VkCacheFile::open(&path, &scope).expect("open");
	assert!(file.section(SECTION_DRIVER_PIPELINE_CACHE).is_some());
	assert_eq!(
		file
			.section(SECTION_GEMM_MEASURED_ROUTES)
			.map(|s| s.bytes.as_slice()),
		Some(&[2u8, 3][..])
	);

	update(
		&path,
		&scope,
		&[SectionUpdate::remove(SECTION_DRIVER_PIPELINE_CACHE)],
	)
	.expect("commit");
	let file = VkCacheFile::open(&path, &scope).expect("open");
	assert!(file.section(SECTION_DRIVER_PIPELINE_CACHE).is_none());
	assert!(file.section(SECTION_GEMM_MEASURED_ROUTES).is_some());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn live_lock_skips_persistence() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	// A live OS lock makes update a no-op even if its inode is old.
	let lock = super::CacheLock::acquire(&path)
		.expect("lock acquire")
		.expect("exclusive lock");
	lock
		._file
		.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(600))
		.expect("old inode");
	update(
		&path,
		&scope,
		&[SectionUpdate::replace(
			SECTION_DRIVER_PIPELINE_CACHE,
			1,
			vec![1],
		)],
	)
	.expect("skip must not error");
	assert!(
		!path.is_file(),
		"a held lock must leave the artifact untouched"
	);
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn abandoned_lock_inode_does_not_block_persistence() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	let lock = path.with_extension("lock");
	std::fs::write(&lock, b"crashed-writer").expect("lock write");
	// An unlocked inode left by a crashed or previous writer is reusable.
	let stale_time = std::time::SystemTime::now() - std::time::Duration::from_secs(600);
	std::fs::File::options()
		.write(true)
		.open(&lock)
		.expect("lock open")
		.set_modified(stale_time)
		.expect("lock mtime");
	update(
		&path,
		&scope,
		&[SectionUpdate::replace(
			SECTION_DRIVER_PIPELINE_CACHE,
			1,
			vec![1],
		)],
	)
	.expect("stale lock must not block persistence");
	assert!(
		path.is_file(),
		"artifact must commit after acquiring abandoned lock inode"
	);
	assert!(lock.exists(), "keep the lock inode stable across writers");
	assert!(
		super::CacheLock::acquire(&path).unwrap().is_some(),
		"OS lock must be released after commit"
	);
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn corrupt_prior_artifact_is_replaced_not_merged() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	crate::Filesystem::write_binary(&path, &[0xde; 128]).expect("seed");
	update(
		&path,
		&scope,
		&[SectionUpdate::replace(
			SECTION_GEMM_MEASURED_ROUTES,
			1,
			vec![4],
		)],
	)
	.expect("commit");
	let file = VkCacheFile::open(&path, &scope).expect("open");
	assert!(file.section(SECTION_GEMM_MEASURED_ROUTES).is_some());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn atomic_commit_leaves_no_temporary_file() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	atomic_commit(&path, &artifact_with(&[], &scope)).expect("commit");
	assert!(path.is_file());
	assert!(!path.with_extension("tmp").exists());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn driver_blob_header_validates_device_identity() {
	let scope = scope();
	let mut blob = Vec::new();
	blob.extend_from_slice(&32u32.to_le_bytes()); // header_size
	blob.extend_from_slice(&1u32.to_le_bytes()); // header_version = ONE
	blob.extend_from_slice(&scope.vendor_id.to_le_bytes());
	blob.extend_from_slice(&scope.device_id.to_le_bytes());
	blob.extend_from_slice(&scope.pipeline_cache_uuid);
	blob.extend_from_slice(&[0xcc; 64]);
	assert!(driver_blob_matches_scope(&blob, &scope));

	let mut bad_vendor = blob.clone();
	bad_vendor[8] ^= 0xff;
	assert!(!driver_blob_matches_scope(&bad_vendor, &scope));
	let mut bad_uuid = blob.clone();
	bad_uuid[20] ^= 0xff;
	assert!(!driver_blob_matches_scope(&bad_uuid, &scope));
	let mut bad_version = blob.clone();
	bad_version[4..8].copy_from_slice(&2u32.to_le_bytes());
	assert!(!driver_blob_matches_scope(&bad_version, &scope));
	assert!(!driver_blob_matches_scope(&blob[..20], &scope));
}

#[test]
fn jit_sections_round_trip_and_restart_reuse() {
	let scope = scope();
	let path = temp_dir().join("oa-test.vkcache");
	let key = vec![0x11u8; 64];
	let artifact_id = jit::artifact_id(&key, 1, 1);
	let index = jit::encode_index(&[JitIndexRecord {
		artifact_id,
		key: key.clone(),
		descriptor_profile: 1,
		spirv_environment: 0,
		workgroup_size: 256,
		binding_count: 5,
		push_bytes: 8,
		emitter_version: 1,
		compiler: b"slang-test".to_vec(),
		flags: 0,
	}])
	.expect("index encode");
	let spirv = jit::encode_spirv(&[JitSpirvRecord {
		artifact_id,
		words: vec![0x0723_0203, 1, 2, 3],
	}])
	.expect("spirv encode");
	update(
		&path,
		&scope,
		&[
			SectionUpdate::replace(super::SECTION_JIT_ARTIFACT_INDEX, 1, index),
			SectionUpdate::replace(super::SECTION_JIT_SPIRV_ARTIFACTS, 1, spirv),
		],
	)
	.expect("commit");

	// Simulated restart: reopen the artifact and decode both sections.
	let file = VkCacheFile::open(&path, &scope).expect("reopen");
	let index = jit::decode_index(
		&file
			.section(super::SECTION_JIT_ARTIFACT_INDEX)
			.expect("index section")
			.bytes,
	)
	.expect("index decode");
	let spirv = jit::decode_spirv(
		&file
			.section(super::SECTION_JIT_SPIRV_ARTIFACTS)
			.expect("spirv section")
			.bytes,
	)
	.expect("spirv decode");
	assert_eq!(index.len(), 1);
	assert_eq!(index[0].key, key);
	assert_eq!(spirv.len(), 1);
	assert_eq!(
		spirv[0].words,
		vec![0x0723_0203, 1, 2, 3],
		"persisted artifact must survive restart with identical words"
	);
	assert_eq!(index[0].artifact_id, spirv[0].artifact_id);
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn jit_codecs_reject_out_of_bounds_and_duplicates() {
	let oversized = JitIndexRecord {
		artifact_id: 1,
		key: vec![0u8; 4097],
		descriptor_profile: 0,
		spirv_environment: 0,
		workgroup_size: 256,
		binding_count: 1,
		push_bytes: 4,
		emitter_version: 1,
		compiler: Vec::new(),
		flags: 0,
	};
	assert!(jit::encode_index(&[oversized]).is_err());
	let duplicate = JitIndexRecord {
		artifact_id: 7,
		key: vec![1],
		descriptor_profile: 0,
		spirv_environment: 0,
		workgroup_size: 256,
		binding_count: 1,
		push_bytes: 4,
		emitter_version: 1,
		compiler: Vec::new(),
		flags: 0,
	};
	assert!(jit::encode_index(&[duplicate.clone(), duplicate]).is_err());

	let trailing = {
		let mut bytes = jit::encode_index(&[]).expect("empty index encodes");
		bytes.push(0);
		bytes
	};
	assert!(jit::decode_index(&trailing).is_err());
	let bad_spirv = {
		let mut bytes = jit::encode_spirv(&[]).expect("empty table encodes");
		bytes.extend_from_slice(&7u64.to_le_bytes());
		bytes
	};
	assert!(jit::decode_spirv(&bad_spirv).is_err());
}

#[test]
fn route_observations_round_trip_and_reject_malformed() {
	let records = vec![
		jit::RouteObservationRecord {
			signature_key: jit::signature_key(b"program-a"),
			selected_route: jit::ROUTE_SPECIALIZED,
			sample_count: 9,
			unfused_p50_ns: 0,
			tape_p50_ns: 4_200,
			specialized_p50_ns: 1_900,
			last_step: 41,
		},
		jit::RouteObservationRecord {
			signature_key: jit::signature_key(b"program-b"),
			selected_route: jit::ROUTE_TAPE,
			sample_count: 5,
			unfused_p50_ns: 0,
			tape_p50_ns: 800,
			specialized_p50_ns: 2_400,
			last_step: 12,
		},
	];
	let bytes = jit::encode_observations(&records).expect("encode");
	assert_eq!(jit::decode_observations(&bytes).expect("decode"), records);

	// Out-of-vocabulary route selection and duplicate signatures reject.
	let mut bad_route = bytes.clone();
	bad_route[8 + 8] = 7; // first record's selected_route low byte
	assert!(jit::decode_observations(&bad_route).is_err());
	let mut dup = records.clone();
	dup[1].signature_key = dup[0].signature_key;
	assert!(jit::encode_observations(&dup).is_ok());
	assert!(jit::decode_observations(&jit::encode_observations(&dup).unwrap()).is_err());
	for cut in [0, 7, 8, 9, 20, bytes.len() - 1] {
		assert!(
			jit::decode_observations(&bytes[..cut]).is_err(),
			"truncated observation payload at {cut} must fail closed"
		);
	}
	let mut trailing = bytes.clone();
	trailing.push(0);
	assert!(jit::decode_observations(&trailing).is_err());
}

#[test]
fn envelope_parse_never_panics_on_mutated_bytes() {
	let scope = scope();
	let artifact = artifact_with(
		&[
			(SECTION_DRIVER_PIPELINE_CACHE, 1, &[0xAA; 64]),
			(SECTION_GEMM_MEASURED_ROUTES, 1, &[0xBB; 32]),
		],
		&scope,
	);
	// Deterministic walk: flip one byte across the whole artifact plus a few
	// truncations. The parse must return an error, never panic or over-read.
	for position in 0..artifact.len() {
		let mut mutated = artifact.clone();
		mutated[position] ^= 0x5A;
		let _ = VkCacheFile::decode(&mutated);
	}
	for cut in [0, 1, 16, artifact.len() / 2, artifact.len() - 1] {
		let _ = VkCacheFile::decode(&artifact[..cut]);
	}
	let mut oversized = artifact.clone();
	oversized.extend_from_slice(&[0; 4096]);
	let _ = VkCacheFile::decode(&oversized);
}

/// `update` is a no-op under a live lock, so a writer that requires its
/// section to persist must retry until the commit is observable.
fn commit_until_visible(path: &crate::Path, section: u32, marker: u8) {
	for round in 0..10_000u32 {
		update(
			path,
			&scope(),
			&[SectionUpdate::replace(section, 1, vec![marker])],
		)
		.expect("writer commit");
		if VkCacheFile::open(path, &scope())
			.ok()
			.is_some_and(|file| file.section(section).is_some())
		{
			return;
		}
		if round.is_multiple_of(64) {
			std::thread::sleep(std::time::Duration::from_millis(1));
		}
	}
	panic!("writer starved by a permanently held lock");
}

#[test]
fn concurrent_writers_merge_without_corruption() {
	let path = temp_dir().join("oa-test.vkcache");
	atomic_commit(&path, &artifact_with(&[], &scope())).expect("seed");
	let writer = |section: u32, marker: u8| {
		let path = path.clone();
		std::thread::spawn(move || commit_until_visible(&path, section, marker))
	};
	let first = writer(super::SECTION_JIT_ARTIFACT_INDEX, 0xA0);
	let second = writer(SECTION_GEMM_MEASURED_ROUTES, 0xB0);
	let reader = {
		let path = path.clone();
		let reader_scope = scope();
		std::thread::spawn(move || {
			for _ in 0..32 {
				// A racing reader must see either a complete artifact or a
				// clean miss — never a panic or partial parse.
				if let Ok(file) = VkCacheFile::open(&path, &reader_scope) {
					let _ = file.section(SECTION_SCOPE);
				}
			}
		})
	};
	first.join().expect("writer");
	second.join().expect("writer");
	reader.join().expect("reader");
	// Merge semantics: once each writer committed, both sections persist —
	// a later writer preserves sections it does not own.
	let file = VkCacheFile::open(&path, &scope()).expect("final artifact must parse");
	assert!(file.section(super::SECTION_JIT_ARTIFACT_INDEX).is_some());
	assert!(file.section(SECTION_GEMM_MEASURED_ROUTES).is_some());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn cross_process_writers_preserve_unrelated_sections() {
	const CHILD_ENV: &str = "OA_VKCACHE_TEST_CHILD";
	const PATH_ENV: &str = "OA_VKCACHE_TEST_PATH";
	if std::env::var_os(CHILD_ENV).is_some() {
		// Child role: write its own section until it is observable, then exit.
		let path = std::path::PathBuf::from(std::env::var_os(PATH_ENV).expect("child path"));
		let path = crate::Path::new_from_path_buf(path);
		commit_until_visible(&path, super::SECTION_JIT_SPIRV_ARTIFACTS, 0xC0);
		return;
	}
	let path = temp_dir().join("oa-test.vkcache");
	atomic_commit(&path, &artifact_with(&[], &scope())).expect("seed");
	let child = std::process::Command::new(std::env::current_exe().expect("test binary"))
		.args([
			"--exact",
			concat!(
				"runtime::device::vkcache::tests::",
				"cross_process_writers_preserve_unrelated_sections"
			),
			"--test-threads=1",
		])
		.env(CHILD_ENV, "1")
		.env(PATH_ENV, path.as_os_str())
		.spawn()
		.expect("child process");
	commit_until_visible(&path, SECTION_GEMM_MEASURED_ROUTES, 0xD0);
	assert!(
		child
			.wait_with_output()
			.expect("join child")
			.status
			.success()
	);
	// Two OS processes interleaving lock-protected commits must converge on
	// an artifact carrying both writers' sections.
	let file = VkCacheFile::open(&path, &scope()).expect("final artifact must parse");
	assert!(file.section(super::SECTION_JIT_SPIRV_ARTIFACTS).is_some());
	assert!(file.section(SECTION_GEMM_MEASURED_ROUTES).is_some());
	let _ = std::fs::remove_dir_all(path.parent().unwrap().into_path_buf());
}

#[test]
fn oversized_file_is_rejected_before_payload_allocation() {
	let dir = temp_dir();
	let path = dir.join("oversized.vkcache");
	let file = std::fs::File::create(&path).unwrap();
	file.set_len(super::VKCACHE_MAX_FILE_BYTES + 1).unwrap();
	assert!(VkCacheFile::open(&path, &scope()).is_err());
	drop(file);
	std::fs::remove_dir_all(dir.into_path_buf()).unwrap();
}
