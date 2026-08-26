#pragma once

// OA-owned lexical path value. Path manipulation is independent from the
// hosted filesystem library; operating-system access belongs to Filesystem's
// private backend.

#include <oa/core/std/string.h>
#include <oa/core/std/utility.h>
#include <oa/core/std/vec.h>

namespace oa {

class Path {
public:
	Path() = default;
	Path(const char* inString) : string_(inString != nullptr ? inString : "") {}
	Path(StringView inView) : string_(inView) {}
	Path(const String& inString) : string_(inString) {}
	Path(String&& inString) noexcept : string_(oa::move(inString)) {}

	[[nodiscard]] String string() const { return string_; }
	[[nodiscard]] String genericString() const { return string_; }
	[[nodiscard]] const char* cStr() const noexcept { return string_.cStr(); }
	[[nodiscard]] bool empty() const noexcept { return string_.empty(); }
	void clear() noexcept { string_.clear(); }

	[[nodiscard]] bool hasParentPath() const { return not parentPath().empty(); }

	[[nodiscard]] Path parentPath() const {
		const StringView text = string_.view();
		if (text.empty()) return {};

		Usize end = text.size();
		while (end > 1 and isSeparator_(text[end - 1])) --end;
		if (end == 1 and isSeparator_(text[0])) return Path("/");
		if (end < text.size()) return Path(text.subStr(0, end));

		Usize separator = end;
		while (separator > 0 and not isSeparator_(text[separator - 1])) --separator;
		if (separator == 0) return {};
		while (separator > 1 and isSeparator_(text[separator - 1])) --separator;
		return Path(text.subStr(0, separator));
	}

	Path& append(const Path& inOther) {
		if (inOther.empty()) return *this;
		if (inOther.isAbsolute() or empty()) {
			string_ = inOther.string_;
			return *this;
		}
		if (not isSeparator_(string_.back())) string_.pushBack('/');
		string_.append(inOther.string_.view());
		return *this;
	}

	Path& operator/=(const Path& inOther) { return append(inOther); }

	[[nodiscard]] Path filename() const {
		const StringView text = string_.view();
		if (text.empty() or isSeparator_(text.back())) return {};
		Usize begin = text.size();
		while (begin > 0 and not isSeparator_(text[begin - 1])) --begin;
		return Path(text.subStr(begin));
	}

	[[nodiscard]] Path stem() const {
		const Path namePath = filename();
		const StringView name = namePath.string_.view();
		if (name == "." or name == "..") return Path(name);
		const Usize dot = name.rfind('.');
		if (dot == StringView::Npos or dot == 0) return Path(name);
		return Path(name.subStr(0, dot));
	}

	[[nodiscard]] Path extension() const {
		const Path namePath = filename();
		const StringView name = namePath.string_.view();
		if (name == "." or name == "..") return {};
		const Usize dot = name.rfind('.');
		if (dot == StringView::Npos or dot == 0) return {};
		return Path(name.subStr(dot));
	}

	[[nodiscard]] bool isAbsolute() const noexcept {
		return not string_.empty() and isSeparator_(string_[0]);
	}

	[[nodiscard]] bool isRelative() const noexcept { return not isAbsolute(); }

	[[nodiscard]] Path lexicallyNormal() const {
		const StringView text = string_.view();
		if (text.empty()) return {};

		const bool absolute = isAbsolute();
		Vec<StringView> components;
		Usize cursor = 0;
		while (cursor < text.size()) {
			while (cursor < text.size() and isSeparator_(text[cursor])) ++cursor;
			const Usize begin = cursor;
			while (cursor < text.size() and not isSeparator_(text[cursor])) ++cursor;
			if (begin == cursor) continue;

			const StringView component = text.subStr(begin, cursor - begin);
			if (component == ".") continue;
			if (component == "..") {
				if (not components.empty() and components.back() != "..") {
					components.popBack();
				} else if (not absolute) {
					components.pushBack(component);
				}
				continue;
			}
			components.pushBack(component);
		}

		String normalized;
		if (absolute) normalized.pushBack('/');
		for (Usize index = 0; index < components.size(); ++index) {
			if (not normalized.empty() and normalized.back() != '/') normalized.pushBack('/');
			normalized.append(components[index]);
		}
		if (normalized.empty() and not absolute) normalized.pushBack('.');
		return Path(oa::move(normalized));
	}

	[[nodiscard]] bool operator==(const Path& inOther) const noexcept {
		return string_ == inOther.string_;
	}

	[[nodiscard]] bool operator!=(const Path& inOther) const noexcept {
		return not (*this == inOther);
	}

	void swap(Path& inOther) noexcept { oa::swapValues(string_, inOther.string_); }

private:
	[[nodiscard]] static constexpr bool isSeparator_(char inCharacter) noexcept {
		return inCharacter == '/';
	}

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
