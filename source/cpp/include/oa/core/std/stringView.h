#pragma once

// Native StringView — non-owning `const char*` + length.
//
// `Npos` for find failures; `at` / `subStr` throw `std::out_of_range`; `operator[]` unchecked.
// Interop: `stdView()` → `std::string_view`; explicit ctor from `std::string_view`.

#include <cassert>
#include <cstddef>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace oa {

class StringView {
public:
	using value_type = char;
	using size_type = std::size_t;
	using const_iterator = const char*;

	static constexpr size_type Npos = static_cast<size_type>(-1);

	StringView() noexcept = default;

	// Array binding (string literals AND fixed-size char buffers). Length is the
	// C-string length bounded by the array capacity — i.e. up to the first '\0',
	// else the full N. For an exact-sized literal this equals the historical N-1,
	// so the constexpr name tables are unchanged; for a short string in a larger
	// buffer (e.g. snprintf into char[16]) it stops at the real terminator instead
	// of capturing the embedded NUL + uninitialised tail (which silently corrupted
	// dotted module paths in checkpoint serialization).
	template<std::size_t N>
	constexpr StringView(const char (&ins)[N]) noexcept
		: ptr_(ins),
		  len_(boundedCStrLen(ins, N)) {}

	constexpr StringView(const char* ins, size_type inlen) noexcept : ptr_(ins), len_(inlen) {}

	StringView(const char* innullTerminated) noexcept
		: ptr_(innullTerminated),
		  len_(innullTerminated != nullptr ? std::strlen(innullTerminated) : 0) {}

	explicit StringView(std::string_view inv) noexcept : ptr_(inv.data()), len_(inv.size()) {}

	[[nodiscard]] std::string_view stdView() const noexcept { return std::string_view(ptr_, len_); }

	[[nodiscard]] size_type size() const noexcept { return len_; }
	[[nodiscard]] bool empty() const noexcept { return len_ == 0; }
	[[nodiscard]] const char* data() const noexcept { return ptr_; }

	[[nodiscard]] char operator[](size_type inidx) const noexcept { return ptr_[inidx]; }

	[[nodiscard]] char at(size_type inidx) const {
		if (inidx >= len_) {
			throw std::out_of_range("StringView::at");
		}
		return ptr_[inidx];
	}

	[[nodiscard]] char front() const noexcept { return *ptr_; }
	[[nodiscard]] char back() const noexcept { return ptr_[len_ - 1U]; }

	[[nodiscard]] StringView subStr(size_type inpos = 0, size_type incount = Npos) const {
		if (inpos > len_) {
			throw std::out_of_range("StringView::subStr");
		}
		size_type const avail = len_ - inpos;
		size_type n = avail;
		if (incount != Npos) {
			n = incount < avail ? incount : avail;
		}
		return StringView(ptr_ + inpos, n);
	}

	void removePrefix(size_type inn) noexcept {
		assert(inn <= len_);
		ptr_ += inn;
		len_ -= inn;
	}

	void removeSuffix(size_type inn) noexcept {
		assert(inn <= len_);
		len_ -= inn;
	}

	[[nodiscard]] const_iterator begin() const noexcept { return ptr_; }
	[[nodiscard]] const_iterator end() const noexcept { return ptr_ + len_; }

	[[nodiscard]] bool equals(StringView ino) const noexcept {
		if (len_ != ino.len_) {
			return false;
		}
		return len_ == 0 || std::memcmp(ptr_, ino.ptr_, len_) == 0;
	}

	[[nodiscard]] size_type find(char inch, size_type inpos = 0) const noexcept {
		if (inpos >= len_) {
			return Npos;
		}
		for (size_type i = inpos; i < len_; ++i) {
			if (ptr_[i] == inch) {
				return i;
			}
		}
		return Npos;
	}

	[[nodiscard]] size_type find(StringView inneedle, size_type inpos = 0) const noexcept {
		const size_type nl = inneedle.size();
		if (nl == 0) {
			return inpos <= len_ ? inpos : Npos;
		}
		if (inpos > len_ || len_ - inpos < nl) {
			return Npos;
		}
		for (size_type i = inpos; i + nl <= len_; ++i) {
			if (std::memcmp(ptr_ + i, inneedle.data(), nl) == 0) {
				return i;
			}
		}
		return Npos;
	}

	[[nodiscard]] size_type find(const char* ins, size_type inpos, size_type incount) const noexcept {
		if (ins == nullptr || incount == 0) {
			return inpos <= len_ ? inpos : Npos;
		}
		return find(StringView(ins, incount), inpos);
	}

	[[nodiscard]] size_type find(const char* ins, size_type inpos = 0) const noexcept {
		if (ins == nullptr) {
			return inpos <= len_ ? inpos : Npos;
		}
		return find(StringView(ins), inpos);
	}

	[[nodiscard]] int compare(StringView ino) const noexcept {
		size_type const lhs = len_;
		size_type const rhs = ino.len_;
		size_type const n = lhs < rhs ? lhs : rhs;
		int const c = n == 0 ? 0 : std::memcmp(ptr_, ino.ptr_, n);
		if (c != 0) {
			return c < 0 ? -1 : 1;
		}
		if (lhs < rhs) {
			return -1;
		}
		if (lhs > rhs) {
			return 1;
		}
		return 0;
	}

private:
	// C-string length within a fixed-capacity array: index of the first '\0', or
	// inn if none. constexpr so literal bindings stay compile-time.
	static constexpr size_type boundedCStrLen(const char* ins, size_type inn) noexcept {
		size_type i = 0;
		while (i < inn && ins[i] != '\0') {
			++i;
		}
		return i;
	}

	const char* ptr_ = nullptr;
	size_type len_ = 0;
};

inline bool operator==(StringView ina, StringView inb) noexcept {
	return ina.equals(inb);
}
inline bool operator!=(StringView ina, StringView inb) noexcept {
	return !ina.equals(inb);
}

inline bool operator==(StringView ina, const char* inb) noexcept {
	if (inb == nullptr) {
		return ina.empty();
	}
	return ina.equals(StringView(inb));
}
inline bool operator==(const char* ina, StringView inb) noexcept {
	return inb == ina;
}
inline bool operator!=(StringView ina, const char* inb) noexcept {
	return !(ina == inb);
}
inline bool operator!=(const char* ina, StringView inb) noexcept {
	return !(ina == inb);
}

inline StringView::const_iterator begin(StringView inv) noexcept { return inv.begin(); }
inline StringView::const_iterator end(StringView inv) noexcept { return inv.end(); }

inline std::ostream& operator<<(std::ostream& inos, StringView inv) {
	return inos.write(inv.data(), static_cast<std::streamsize>(inv.size()));
}

} // namespace oa
