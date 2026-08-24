// OA CORE - Filesystem
//
// Filesystem operations for reading/writing data files.
// Supports text, binary, directory operations, and glob matching.
//
// Fallible value-producing and mutating calls return oa::Status/oa::Result.
// Predicates return false on absence or an inaccessible path.
// No exceptions. No raw new/delete. Uses oa::memcpy for POD reads.

#pragma once

#include <oa/core/memory.h>
#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa {

// Filesystem — stateless host-filesystem operations.
//
// oa::Path owns lexical path manipulation. oa::Paths owns OA location discovery.
class Filesystem {
public:
	// ─── Existence & Info ─────────────────────────────────────────────────
	[[nodiscard]] static bool exists(const oa::Path& inPath);
	[[nodiscard]] static bool isFile(const oa::Path& inPath);
	[[nodiscard]] static bool isDirectory(const oa::Path& inPath);
	[[nodiscard]] static oa::Result<oa::Usize> getFileSize(const oa::Path& inPath);
	[[nodiscard]] static oa::Result<oa::I64> getLastModified(const oa::Path& inPath);

	// ─── directory operations ─────────────────────────────────────────────
	[[nodiscard]] static oa::Status createDirectory(const oa::Path& inPath);
	[[nodiscard]] static oa::Status createDirectories(const oa::Path& inPath);
	[[nodiscard]] static oa::Status removeFile(const oa::Path& inPath);
	[[nodiscard]] static oa::Status removeDirectory(const oa::Path& inPath, bool inRecursive = false);
	[[nodiscard]] static oa::Status copy(const oa::Path& inFrom, const oa::Path& inTo);
	[[nodiscard]] static oa::Status move(const oa::Path& inFrom, const oa::Path& inTo);

	// ─── Listing ──────────────────────────────────────────────────────────
	[[nodiscard]] static oa::Result<oa::Vec<oa::Path>>
	listFiles(const oa::Path& inDir, oa::StringView inExtension = "");
	[[nodiscard]] static oa::Result<oa::Vec<oa::Path>>
	listDirectories(const oa::Path& inDir);
	[[nodiscard]] static oa::Result<oa::Vec<oa::Path>>
	listAll(const oa::Path& inDir, bool inRecursive = false);

	// ─── text file operations ─────────────────────────────────────────────
	[[nodiscard]] static oa::Result<oa::String> readText(const oa::Path& inPath);
	[[nodiscard]] static oa::Status writeText(const oa::Path& inPath, oa::StringView inContent);
	[[nodiscard]] static oa::Status appendText(const oa::Path& inPath, oa::StringView inContent);
	[[nodiscard]] static oa::Result<oa::Vec<oa::String>>
	readLines(const oa::Path& inPath);

	// ─── Binary file operations ───────────────────────────────────────────
	[[nodiscard]] static oa::Result<oa::Vec<oa::U8>> readBinary(const oa::Path& inPath);
	[[nodiscard]] static oa::Status writeBinary(const oa::Path& inPath, oa::Span<const oa::U8> inData);

	template <typename T>
	[[nodiscard]] static oa::Result<oa::Vec<T>> readPod(const oa::Path& inPath) {
		auto result = readBinary(inPath);
		if (!result.isOk()) return result.getStatus();
		const auto& data = result.getValue();
		if (data.size() % sizeof(T) != 0)
			return oa::Status::invalidArgument("file size not aligned to type size");
		oa::Vec<T> out(data.size() / sizeof(T));
		oa::memcpy(out.data(), data.data(), data.size());
		return out;
	}

	template <typename T>
	[[nodiscard]] static oa::Status writePod(const oa::Path& inPath, oa::Span<const T> inData) {
		return writeBinary(inPath,
			oa::Span<const oa::U8>(reinterpret_cast<const oa::U8*>(inData.data()),
			inData.size() * sizeof(T)));
	}

	// ─── Filesystem Resolution ────────────────────────────────────────────
	[[nodiscard]] static oa::Result<oa::Path> absolute(const oa::Path& inPath);

	// ─── Glob Pattern Matching ────────────────────────────────────────────
	[[nodiscard]] static oa::Result<oa::Vec<oa::Path>> glob(const oa::Path& inDir, oa::StringView inPattern);
};

// Legacy alias — remove once call sites are migrated.

} // namespace oa
