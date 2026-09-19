use std::path::Path;

use oa::{ErrorKind, Filesystem};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Create a fresh temporary directory scoped to one test.
fn tmp_dir(label: &str) -> std::path::PathBuf {
	let dir = std::env::temp_dir().join("oa_test_filesystem").join(label);
	let _ = std::fs::remove_dir_all(&dir);
	std::fs::create_dir_all(&dir).unwrap();
	dir
}

// ─── Existence & Info ─────────────────────────────────────────────────────────

#[test]
fn exists_file_and_directory() {
	let dir = tmp_dir("exists");
	let file = dir.join("hello.txt");
	std::fs::write(&file, b"hi").unwrap();

	assert!(Filesystem::exists(&dir));
	assert!(Filesystem::exists(&file));
	assert!(!Filesystem::exists(&dir.join("absent")));
}

#[test]
fn is_file_and_is_directory_are_exclusive() {
	let dir = tmp_dir("kind");
	let file = dir.join("f.txt");
	std::fs::write(&file, b"x").unwrap();

	assert!(Filesystem::is_file(&file));
	assert!(!Filesystem::is_directory(&file));
	assert!(Filesystem::is_directory(&dir));
	assert!(!Filesystem::is_file(&dir));
}

#[test]
fn get_file_size_matches_written_bytes() {
	let dir = tmp_dir("size");
	let file = dir.join("data.bin");
	std::fs::write(&file, b"hello world").unwrap();
	assert_eq!(Filesystem::get_file_size(&file).unwrap(), 11);
}

#[test]
fn get_last_modified_advances_after_write() {
	let dir = tmp_dir("mtime");
	let file = dir.join("t.txt");
	std::fs::write(&file, b"v1").unwrap();
	let t1 = Filesystem::get_last_modified(&file).unwrap();
	// Overwrite with a new mtime (sleep gives the OS time to tick).
	std::thread::sleep(std::time::Duration::from_millis(10));
	std::fs::write(&file, b"v2 longer").unwrap();
	let t2 = Filesystem::get_last_modified(&file).unwrap();
	// mtime is at least as large — on fast FSes it may be identical.
	assert!(t2 >= t1);
}

// ─── Directory operations ─────────────────────────────────────────────────────

#[test]
fn create_directory_is_idempotent() {
	let dir = tmp_dir("mkdir");
	let target = dir.join("new");
	Filesystem::create_directory(&target).unwrap();
	Filesystem::create_directory(&target).unwrap(); // second call: no error
	assert!(target.is_dir());
}

#[test]
fn create_directories_creates_nested_path() {
	let dir = tmp_dir("mkdirs");
	let deep = dir.join("a/b/c");
	Filesystem::create_directories(&deep).unwrap();
	assert!(deep.is_dir());
}

#[test]
fn remove_file_is_silent_on_missing() {
	let dir = tmp_dir("rm_absent");
	let absent = dir.join("nope.txt");
	Filesystem::remove_file(&absent).unwrap(); // must not error
}

#[test]
fn remove_file_deletes_existing_file() {
	let dir = tmp_dir("rm_file");
	let file = dir.join("bye.txt");
	std::fs::write(&file, b"gone").unwrap();
	Filesystem::remove_file(&file).unwrap();
	assert!(!file.exists());
}

#[test]
fn remove_directory_non_recursive_rejects_non_empty() {
	let dir = tmp_dir("rmdir_ne");
	let target = dir.join("populated");
	std::fs::create_dir_all(target.join("child")).unwrap();
	let err = Filesystem::remove_directory(&target, false).unwrap_err();
	assert_eq!(err.kind(), ErrorKind::Io);
}

#[test]
fn remove_directory_recursive_clears_tree() {
	let dir = tmp_dir("rmdir_rec");
	let target = dir.join("tree");
	std::fs::create_dir_all(target.join("a/b")).unwrap();
	std::fs::write(target.join("a/b/f.txt"), b"x").unwrap();
	Filesystem::remove_directory(&target, true).unwrap();
	assert!(!target.exists());
}

#[test]
fn remove_directory_silent_when_absent() {
	let dir = tmp_dir("rmdir_abs");
	Filesystem::remove_directory(&dir.join("ghost"), false).unwrap();
}

#[test]
fn copy_file_creates_parents_and_matches_content() {
	let dir = tmp_dir("copy");
	let src = dir.join("src.bin");
	std::fs::write(&src, b"copy me").unwrap();
	let dst = dir.join("sub/dst.bin");
	Filesystem::copy(&src, &dst).unwrap();
	assert_eq!(std::fs::read(&dst).unwrap(), b"copy me");
}

#[test]
fn move_path_renames_file() {
	let dir = tmp_dir("mv");
	let from = dir.join("from.txt");
	let to = dir.join("nested/to.txt");
	std::fs::write(&from, b"moved").unwrap();
	Filesystem::move_path(&from, &to).unwrap();
	assert!(!from.exists());
	assert_eq!(std::fs::read(&to).unwrap(), b"moved");
}

// ─── Listing ──────────────────────────────────────────────────────────────────

#[test]
fn list_files_returns_sorted_regular_files() {
	let dir = tmp_dir("list_files");
	for name in ["z.txt", "a.txt", "m.txt"] {
		std::fs::write(dir.join(name), b"").unwrap();
	}
	std::fs::create_dir(dir.join("subdir")).unwrap();

	let files = Filesystem::list_files(&dir, "").unwrap();
	let names: Vec<_> = files
		.iter()
		.map(|p| p.file_name().unwrap().to_str().unwrap())
		.collect();
	assert_eq!(names, ["a.txt", "m.txt", "z.txt"]);
}

#[test]
fn list_files_filters_by_extension() {
	let dir = tmp_dir("list_ext");
	for name in ["a.png", "b.jpg", "c.png"] {
		std::fs::write(dir.join(name), b"").unwrap();
	}
	let pngs = Filesystem::list_files(&dir, ".png").unwrap();
	assert_eq!(pngs.len(), 2);
	assert!(pngs.iter().all(|p| p.extension().unwrap() == "png"));
}

#[test]
fn list_files_errors_on_missing_directory() {
	let err = Filesystem::list_files(Path::new("/tmp/oa_no_such_dir_xyz"), "").unwrap_err();
	assert_eq!(err.kind(), ErrorKind::NotFound);
}

#[test]
fn list_directories_returns_only_dirs_sorted() {
	let dir = tmp_dir("list_dirs");
	for name in ["z", "a", "m"] {
		std::fs::create_dir(dir.join(name)).unwrap();
	}
	std::fs::write(dir.join("file.txt"), b"").unwrap();

	let dirs = Filesystem::list_directories(&dir).unwrap();
	let names: Vec<_> = dirs
		.iter()
		.map(|p| p.file_name().unwrap().to_str().unwrap())
		.collect();
	assert_eq!(names, ["a", "m", "z"]);
}

#[test]
fn list_all_non_recursive_matches_immediate_children() {
	let dir = tmp_dir("list_all_flat");
	std::fs::write(dir.join("f.txt"), b"").unwrap();
	std::fs::create_dir(dir.join("d")).unwrap();
	let all = Filesystem::list_all(&dir, false).unwrap();
	assert_eq!(all.len(), 2);
}

#[test]
fn list_all_recursive_descends_into_subdirs() {
	let dir = tmp_dir("list_all_rec");
	std::fs::create_dir_all(dir.join("a/b")).unwrap();
	std::fs::write(dir.join("a/b/deep.txt"), b"").unwrap();
	std::fs::write(dir.join("root.txt"), b"").unwrap();
	let all = Filesystem::list_all(&dir, true).unwrap();
	// root.txt, a/, a/b/, a/b/deep.txt → 4 entries
	assert_eq!(all.len(), 4);
}

// ─── Text operations ─────────────────────────────────────────────────────────

#[test]
fn write_text_creates_parents_and_round_trips() {
	let dir = tmp_dir("write_text");
	let path = dir.join("sub/file.txt");
	Filesystem::write_text(&path, "hello\nworld").unwrap();
	let back = Filesystem::read_text(&path).unwrap();
	assert_eq!(back, "hello\nworld");
}

#[test]
fn append_text_accumulates() {
	let dir = tmp_dir("append");
	let path = dir.join("log.txt");
	Filesystem::write_text(&path, "line1\n").unwrap();
	Filesystem::append_text(&path, "line2\n").unwrap();
	assert_eq!(Filesystem::read_text(&path).unwrap(), "line1\nline2\n");
}

#[test]
fn read_lines_splits_on_lf_and_crlf() {
	let dir = tmp_dir("read_lines");
	let path = dir.join("lines.txt");
	// Write CRLF + LF mixed, no trailing newline.
	std::fs::write(&path, b"first\r\nsecond\nthird").unwrap();
	let lines = Filesystem::read_lines(&path).unwrap();
	assert_eq!(lines, ["first", "second", "third"]);
}

#[test]
fn read_lines_empty_file_gives_empty_vec() {
	let dir = tmp_dir("lines_empty");
	let path = dir.join("empty.txt");
	std::fs::write(&path, b"").unwrap();
	assert!(Filesystem::read_lines(&path).unwrap().is_empty());
}

// ─── Binary operations ───────────────────────────────────────────────────────

#[test]
fn write_binary_and_read_binary_round_trip() {
	let dir = tmp_dir("binary");
	let path = dir.join("data.bin");
	let data: Vec<u8> = (0u8..=255).collect();
	Filesystem::write_binary(&path, &data).unwrap();
	assert_eq!(Filesystem::read_binary(&path).unwrap(), data);
}

// ─── Absolute resolution ─────────────────────────────────────────────────────

#[test]
fn absolute_returns_rooted_path() {
	let abs = Filesystem::absolute(Path::new(".")).unwrap();
	assert!(abs.is_absolute());
}

// ─── Glob ─────────────────────────────────────────────────────────────────────

#[test]
fn glob_star_matches_any_sequence() {
	let dir = tmp_dir("glob_star");
	for name in ["audio.wav", "audio.mp3", "video.mp4", "README.md"] {
		std::fs::write(dir.join(name), b"").unwrap();
	}
	let wav = Filesystem::glob(&dir, "*.wav").unwrap();
	assert_eq!(wav.len(), 1);
	assert_eq!(wav[0].file_name().unwrap(), "audio.wav");

	let audio = Filesystem::glob(&dir, "audio.*").unwrap();
	assert_eq!(audio.len(), 2);
}

#[test]
fn glob_question_mark_matches_one_char() {
	let dir = tmp_dir("glob_q");
	for name in ["a1.txt", "a2.txt", "ab.txt", "abc.txt"] {
		std::fs::write(dir.join(name), b"").unwrap();
	}
	let two_char_stem = Filesystem::glob(&dir, "??.txt").unwrap();
	let names: Vec<_> = two_char_stem
		.iter()
		.map(|p| p.file_name().unwrap().to_str().unwrap())
		.collect();
	assert_eq!(names, ["a1.txt", "a2.txt", "ab.txt"]);
}

#[test]
fn glob_star_matches_all_files() {
	let dir = tmp_dir("glob_all");
	for name in ["x", "y", "z"] {
		std::fs::write(dir.join(name), b"").unwrap();
	}
	let all = Filesystem::glob(&dir, "*").unwrap();
	assert_eq!(all.len(), 3);
}

#[test]
fn glob_errors_on_missing_directory() {
	let err = Filesystem::glob(Path::new("/tmp/oa_no_such_glob_dir_xyz"), "*").unwrap_err();
	assert_eq!(err.kind(), ErrorKind::NotFound);
}

// ─── Root re-export ───────────────────────────────────────────────────────────

#[test]
fn root_re_export_is_the_same_type_as_core() {
	use std::any::TypeId;
	assert_eq!(
		TypeId::of::<oa::Filesystem>(),
		TypeId::of::<oa::core::Filesystem>()
	);
}
