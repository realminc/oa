#pragma once

#include <oa/core/std/string.h>
#include <oa/core/std/utility.h>

#include <filesystem>
#include <string>

namespace oa {

// OA-owned path value. Native storage is oa::String; std::filesystem is used
// only at lexical and system-call boundaries.
class Path {
public:
	Path() = default;
	explicit Path(const std::filesystem::path& inPath) : string_(inPath.string()) {}
	explicit Path(std::string inString) : string_(oa::move(inString)) {}
	Path(const char* inString) : string_(inString != nullptr ? inString : "") {}
	Path(StringView inView) : string_(inView) {}
	Path(const String& inString) : string_(inString) {}

	[[nodiscard]] std::filesystem::path stdPath() const {
		return std::filesystem::path(string_.stdStr());
	}
	[[nodiscard]] operator std::filesystem::path() const { return stdPath(); }

	[[nodiscard]] String string() const { return string_; }
	[[nodiscard]] String genericString() const {
		return String(stdPath().generic_string());
	}
	[[nodiscard]] const char* cStr() const noexcept { return string_.cStr(); }
	[[nodiscard]] bool empty() const noexcept { return string_.empty(); }
	void clear() { string_.clear(); }

	[[nodiscard]] bool hasParentPath() const { return stdPath().has_parent_path(); }
	[[nodiscard]] Path parentPath() const { return Path(stdPath().parent_path()); }

	Path& append(const Path& inOther) {
		std::filesystem::path path = stdPath();
		path /= inOther.stdPath();
		string_ = String(path.string());
		return *this;
	}
	Path& operator/=(const Path& inOther) { return append(inOther); }

	[[nodiscard]] Path filename() const { return Path(stdPath().filename()); }
	[[nodiscard]] Path stem() const { return Path(stdPath().stem()); }
	[[nodiscard]] Path extension() const { return Path(stdPath().extension()); }
	[[nodiscard]] bool isAbsolute() const { return stdPath().is_absolute(); }
	[[nodiscard]] bool isRelative() const { return stdPath().is_relative(); }
	[[nodiscard]] Path lexicallyNormal() const { return Path(stdPath().lexically_normal()); }

	[[nodiscard]] bool operator==(const Path& inOther) const {
		return stdPath() == inOther.stdPath();
	}
	[[nodiscard]] bool operator!=(const Path& inOther) const { return !(*this == inOther); }

	void swap(Path& inOther) noexcept { oa::swapValues(string_, inOther.string_); }

private:
	String string_;
};

inline Path operator/(const Path& inLeft, const Path& inRight) {
	Path result(inLeft);
	result /= inRight;
	return result;
}

inline Path operator/(const Path& inLeft, const char* inRight) {
	return inLeft / Path(inRight != nullptr ? inRight : "");
}

inline Path operator/(const Path& inLeft, StringView inRight) {
	return inLeft / Path(inRight);
}

} // namespace oa
