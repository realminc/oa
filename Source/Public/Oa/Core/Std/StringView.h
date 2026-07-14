#pragma once

// Native OaStdStringView — non-owning `const char*` + length.
//
// `Npos` for find failures; `At` / `SubStr` throw `std::out_of_range`; `operator[]` unchecked.
// Interop: `StdView()` → `std::string_view`; explicit ctor from `std::string_view`.

#include <cassert>
#include <cstddef>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>

class OaStdStringView {
public:
	using value_type = char;
	using size_type = std::size_t;
	using const_iterator = const char*;

	static constexpr size_type Npos = static_cast<size_type>(-1);
	static constexpr size_type npos = Npos;

	OaStdStringView() noexcept = default;

	// Array binding (string literals AND fixed-size char buffers). Length is the
	// C-string length bounded by the array capacity — i.e. up to the first '\0',
	// else the full N. For an exact-sized literal this equals the historical N-1,
	// so the constexpr name tables are unchanged; for a short string in a larger
	// buffer (e.g. snprintf into char[16]) it stops at the real terminator instead
	// of capturing the embedded NUL + uninitialised tail (which silently corrupted
	// dotted module paths in checkpoint serialization).
	template<std::size_t N>
	constexpr OaStdStringView(const char (&InS)[N]) noexcept
		: Ptr_(InS),
		  Len_(BoundedCStrLen(InS, N)) {}

	constexpr OaStdStringView(const char* InS, size_type InLen) noexcept : Ptr_(InS), Len_(InLen) {}

	OaStdStringView(const char* InNullTerminated) noexcept
		: Ptr_(InNullTerminated),
		  Len_(InNullTerminated != nullptr ? std::strlen(InNullTerminated) : 0) {}

	explicit OaStdStringView(std::string_view InV) noexcept : Ptr_(InV.data()), Len_(InV.size()) {}

	[[nodiscard]] std::string_view StdView() const noexcept { return std::string_view(Ptr_, Len_); }

	[[nodiscard]] size_type Size() const noexcept { return Len_; }
	[[nodiscard]] size_type size() const noexcept { return Len_; }
	[[nodiscard]] bool Empty() const noexcept { return Len_ == 0; }
	[[nodiscard]] bool empty() const noexcept { return Empty(); }
	[[nodiscard]] const char* Data() const noexcept { return Ptr_; }
	[[nodiscard]] const char* data() const noexcept { return Ptr_; }

	[[nodiscard]] char operator[](size_type InIdx) const noexcept { return Ptr_[InIdx]; }

	[[nodiscard]] char At(size_type InIdx) const {
		if (InIdx >= Len_) {
			throw std::out_of_range("OaStdStringView::At");
		}
		return Ptr_[InIdx];
	}

	[[nodiscard]] char Front() const noexcept { return *Ptr_; }
	[[nodiscard]] char Back() const noexcept { return Ptr_[Len_ - 1U]; }

	[[nodiscard]] OaStdStringView SubStr(size_type InPos = 0, size_type InCount = Npos) const {
		if (InPos > Len_) {
			throw std::out_of_range("OaStdStringView::SubStr");
		}
		size_type const avail = Len_ - InPos;
		size_type n = avail;
		if (InCount != Npos) {
			n = InCount < avail ? InCount : avail;
		}
		return OaStdStringView(Ptr_ + InPos, n);
	}

	void RemovePrefix(size_type InN) noexcept {
		assert(InN <= Len_);
		Ptr_ += InN;
		Len_ -= InN;
	}

	void RemoveSuffix(size_type InN) noexcept {
		assert(InN <= Len_);
		Len_ -= InN;
	}

	[[nodiscard]] const_iterator Begin() const noexcept { return Ptr_; }
	[[nodiscard]] const_iterator End() const noexcept { return Ptr_ + Len_; }

	[[nodiscard]] bool Equals(OaStdStringView InO) const noexcept {
		if (Len_ != InO.Len_) {
			return false;
		}
		return Len_ == 0 || std::memcmp(Ptr_, InO.Ptr_, Len_) == 0;
	}

	[[nodiscard]] size_type find(char InCh, size_type InPos = 0) const noexcept {
		if (InPos >= Len_) {
			return Npos;
		}
		for (size_type i = InPos; i < Len_; ++i) {
			if (Ptr_[i] == InCh) {
				return i;
			}
		}
		return Npos;
	}

	[[nodiscard]] size_type find(OaStdStringView InNeedle, size_type InPos = 0) const noexcept {
		const size_type nl = InNeedle.Size();
		if (nl == 0) {
			return InPos <= Len_ ? InPos : Npos;
		}
		if (InPos > Len_ || Len_ - InPos < nl) {
			return Npos;
		}
		for (size_type i = InPos; i + nl <= Len_; ++i) {
			if (std::memcmp(Ptr_ + i, InNeedle.Data(), nl) == 0) {
				return i;
			}
		}
		return Npos;
	}

	[[nodiscard]] size_type find(const char* InS, size_type InPos, size_type InCount) const noexcept {
		if (InS == nullptr || InCount == 0) {
			return InPos <= Len_ ? InPos : Npos;
		}
		return find(OaStdStringView(InS, InCount), InPos);
	}

	[[nodiscard]] size_type find(const char* InS, size_type InPos = 0) const noexcept {
		if (InS == nullptr) {
			return InPos <= Len_ ? InPos : Npos;
		}
		return find(OaStdStringView(InS), InPos);
	}

	[[nodiscard]] int Compare(OaStdStringView InO) const noexcept {
		size_type const lhs = Len_;
		size_type const rhs = InO.Len_;
		size_type const n = lhs < rhs ? lhs : rhs;
		int const c = n == 0 ? 0 : std::memcmp(Ptr_, InO.Ptr_, n);
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
	// InN if none. constexpr so literal bindings stay compile-time.
	static constexpr size_type BoundedCStrLen(const char* InS, size_type InN) noexcept {
		size_type i = 0;
		while (i < InN && InS[i] != '\0') {
			++i;
		}
		return i;
	}

	const char* Ptr_ = nullptr;
	size_type Len_ = 0;
};

inline bool operator==(OaStdStringView InA, OaStdStringView InB) noexcept {
	return InA.Equals(InB);
}
inline bool operator!=(OaStdStringView InA, OaStdStringView InB) noexcept {
	return !InA.Equals(InB);
}

inline bool operator==(OaStdStringView InA, const char* InB) noexcept {
	if (InB == nullptr) {
		return InA.Empty();
	}
	return InA.Equals(OaStdStringView(InB));
}
inline bool operator==(const char* InA, OaStdStringView InB) noexcept {
	return InB == InA;
}
inline bool operator!=(OaStdStringView InA, const char* InB) noexcept {
	return !(InA == InB);
}
inline bool operator!=(const char* InA, OaStdStringView InB) noexcept {
	return !(InA == InB);
}

inline OaStdStringView::const_iterator begin(OaStdStringView InV) noexcept { return InV.Begin(); }
inline OaStdStringView::const_iterator end(OaStdStringView InV) noexcept { return InV.End(); }

inline std::ostream& operator<<(std::ostream& InOs, OaStdStringView InV) {
	return InOs.write(InV.Data(), static_cast<std::streamsize>(InV.Size()));
}
