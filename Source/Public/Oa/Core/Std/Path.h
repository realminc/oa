#pragma once

// OaStdPath — native storage: **`OaStdString`** (native path string, same as `std::filesystem::path::string()`).
// Lexical queries and **`StdPath()`** build a temporary **`std::filesystem::path`** so behavior matches **`std`**;
// **`OaStdFilesystem`** / **`OaFileIo`** still use **`std::filesystem`** at the syscall boundary — see `docs/OaStd.md`.

#include <Oa/Core/Std/String.h>
#include <Oa/Core/Std/Utility.h>

#include <filesystem>
#include <string>

class OaStdPath {
public:
	OaStdPath() = default;

	explicit OaStdPath(const std::filesystem::path& InP) : Str_(InP.string()) {}

	explicit OaStdPath(std::string InS) : Str_(OaStdMove(InS)) {}

	OaStdPath(const char* InCStr) : Str_(InCStr != nullptr ? InCStr : "") {}

	OaStdPath(OaStdStringView InV) : Str_(InV) {}

	explicit OaStdPath(const OaStdString& InS) : Str_(InS) {}

	[[nodiscard]] std::filesystem::path StdPath() const { return std::filesystem::path(Str_.StdStr()); }

	[[nodiscard]] operator std::filesystem::path() const { return StdPath(); }

	[[nodiscard]] OaStdString String() const { return Str_; }

	[[nodiscard]] OaStdString GenericString() const {
		const std::filesystem::path native(Str_.StdStr());
		return OaStdString(native.generic_string());
	}

	[[nodiscard]] std::string string() const { return Str_.StdStr(); }

	[[nodiscard]] const char* CStr() const noexcept { return Str_.CStr(); }

	[[nodiscard]] bool Empty() const { return Str_.Empty(); }

	[[nodiscard]] bool empty() const { return Empty(); }

	void Clear() { Str_.Clear(); }

	[[nodiscard]] bool has_parent_path() const {
		const std::filesystem::path native(Str_.StdStr());
		return native.has_parent_path();
	}

	[[nodiscard]] OaStdPath parent_path() const {
		const std::filesystem::path native(Str_.StdStr());
		return OaStdPath(native.parent_path());
	}

	OaStdPath& Append(const OaStdPath& InO) {
		std::filesystem::path work(Str_.StdStr());
		work /= std::filesystem::path(InO.Str_.StdStr());
		Str_ = OaStdString(work.string());
		return *this;
	}

	OaStdPath& operator/=(const OaStdPath& InO) { return Append(InO); }

	[[nodiscard]] OaStdPath ParentPath() const { return parent_path(); }

	[[nodiscard]] OaStdPath Filename() const {
		const std::filesystem::path native(Str_.StdStr());
		return OaStdPath(native.filename());
	}

	[[nodiscard]] OaStdPath Stem() const {
		const std::filesystem::path native(Str_.StdStr());
		return OaStdPath(native.stem());
	}

	[[nodiscard]] OaStdPath Extension() const {
		const std::filesystem::path native(Str_.StdStr());
		return OaStdPath(native.extension());
	}

	[[nodiscard]] bool IsAbsolute() const {
		const std::filesystem::path native(Str_.StdStr());
		return native.is_absolute();
	}

	[[nodiscard]] bool IsRelative() const {
		const std::filesystem::path native(Str_.StdStr());
		return native.is_relative();
	}

	[[nodiscard]] OaStdPath LexicallyNormal() const {
		const std::filesystem::path native(Str_.StdStr());
		return OaStdPath(native.lexically_normal());
	}

	[[nodiscard]] bool operator==(const OaStdPath& InO) const {
		return std::filesystem::path(Str_.StdStr()) == std::filesystem::path(InO.Str_.StdStr());
	}

	[[nodiscard]] bool operator!=(const OaStdPath& InO) const { return not(*this == InO); }

	void Swap(OaStdPath& InO) noexcept {
		using std::swap;
		swap(Str_, InO.Str_);
	}

private:
	OaStdString Str_;
};

inline OaStdPath operator/(const OaStdPath& InA, const OaStdPath& InB) {
	OaStdPath result(InA);
	result /= InB;
	return result;
}

inline OaStdPath operator/(const OaStdPath& InA, const char* InB) {
	return InA / OaStdPath(InB != nullptr ? InB : "");
}

inline OaStdPath operator/(const OaStdPath& InA, OaStdStringView InB) {
	return InA / OaStdPath(InB);
}
