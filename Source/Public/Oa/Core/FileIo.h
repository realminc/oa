// OA CORE - File I/O
//
// File system operations for reading/writing data files.
// Supports text, binary, directory operations, and glob matching.
//
// All functions return OaStatus/OaResult for error handling.
// No exceptions. No raw new/delete. Uses OaMemcpy for POD reads.

#pragma once

#include <cstdlib>

#include <Oa/Core/Memory.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

// OaFileIo — Static utility class for file operations
class OaFileIo {
public:
	// ─── Existence & Info ─────────────────────────────────────────────────
	[[nodiscard]] static bool Exists(const OaPath &InPath);
	[[nodiscard]] static bool IsFile(const OaPath &InPath);
	[[nodiscard]] static bool IsDirectory(const OaPath &InPath);
	[[nodiscard]] static OaResult<OaUsize> GetFileSize(const OaPath &InPath);
	[[nodiscard]] static OaResult<OaI64> GetLastModified(const OaPath &InPath);

	// ─── Directory Operations ─────────────────────────────────────────────
	[[nodiscard]] static OaStatus CreateDirectory(const OaPath &InPath);
	[[nodiscard]] static OaStatus CreateDirectories(const OaPath &InPath);
	[[nodiscard]] static OaStatus RemoveFile(const OaPath &InPath);
	[[nodiscard]] static OaStatus RemoveDirectory(const OaPath &InPath, bool InRecursive = false);
	[[nodiscard]] static OaStatus Copy(const OaPath &InFrom, const OaPath &InTo);
	[[nodiscard]] static OaStatus Move(const OaPath &InFrom, const OaPath &InTo);

	// ─── Listing ──────────────────────────────────────────────────────────
	[[nodiscard]] static OaResult<OaVec<OaPath>>
	ListFiles(const OaPath &InDir, OaStringView InExtension = "");
	[[nodiscard]] static OaResult<OaVec<OaPath>>
	ListDirectories(const OaPath &InDir);
	[[nodiscard]] static OaResult<OaVec<OaPath>>
	ListAll(const OaPath &InDir, bool InRecursive = false);

	// ─── Text File Operations ─────────────────────────────────────────────
	[[nodiscard]] static OaResult<OaString> ReadText(const OaPath &InPath);
	[[nodiscard]] static OaStatus WriteText(const OaPath &InPath, OaStringView InContent);
	[[nodiscard]] static OaStatus AppendText(const OaPath &InPath, OaStringView InContent);
	[[nodiscard]] static OaResult<OaVec<OaString>>
	ReadLines(const OaPath &InPath);

	// ─── Binary File Operations ───────────────────────────────────────────
	[[nodiscard]] static OaResult<OaVec<OaU8>> ReadBinary(const OaPath &InPath);
	[[nodiscard]] static OaStatus WriteBinary(const OaPath &InPath, OaSpan<const OaU8> InData);

	/// Read a file as a vector of POD structs (size must be aligned to sizeof(T))
	template <typename T>
	[[nodiscard]] static OaResult<OaVec<T>> ReadPod(const OaPath &InPath) {
		auto result = ReadBinary(InPath);
		if (!result.IsOk()) {
			return result.GetStatus();
		}

		const auto &data = result.GetValue();
		if (data.Size() % sizeof(T) != 0) {
			return OaStatus::InvalidArgument("File size not aligned to type size");
		}

		OaVec<T> out(data.Size() / sizeof(T));
		OaMemcpy(out.Data(), data.Data(), data.Size());
		return out;
	}

	/// Write a vector of POD structs as binary
	template <typename T>
	[[nodiscard]] static OaStatus WritePod(
		const OaPath &InPath,
		OaSpan<const T> InData
	) {
		return WriteBinary(
			InPath,
			OaSpan<const OaU8>(reinterpret_cast<const OaU8 *>(InData.data()),
			InData.size() * sizeof(T))
		);
	}

	// ─── Var Directory (repo-root-relative) ──────────────────────────────
	// Inline so each consumer repo's OA_REPO_ROOT is used at the call site.
	// Resolution: OA_VAR_DIR env > OA_REPO_ROOT/var/ > ./var/
	[[nodiscard]] static inline OaPath GetVarDir() {
		const char* env = std::getenv("OA_VAR_DIR");
		if (env && env[0] != '\0') return OaPath(env);
#ifdef OA_REPO_ROOT
		return OaPath(OA_REPO_ROOT) / "var";
#else
		return OaPath("var");
#endif
	}
	[[nodiscard]] static inline OaPath GetVarDir(OaStringView InSubdir) {
		return GetVarDir() / OaPath(InSubdir);
	}

	// ─── Path Utilities ───────────────────────────────────────────────────
	[[nodiscard]] static OaPath GetCurrentDirectory();
	[[nodiscard]] static OaPath GetHomeDirectory();
	[[nodiscard]] static OaPath GetTempDirectory();
	[[nodiscard]] static OaPath GetAbsolutePath(const OaPath &InPath);
	[[nodiscard]] static OaPath Join(const OaPath &InBase, const OaPath &InRelative);
	[[nodiscard]] static OaString GetExtension(const OaPath &InPath);
	[[nodiscard]] static OaString GetStem(const OaPath &InPath);
	[[nodiscard]] static OaPath GetParent(const OaPath &InPath);

	// ─── Glob Pattern Matching ────────────────────────────────────────────
	/// Simple glob: * matches any chars, ? matches single char
	[[nodiscard]] static OaResult<OaVec<OaPath>> Glob(const OaPath &InDir, OaStringView InPattern);
};
