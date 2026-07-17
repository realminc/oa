#pragma once

// Phase 2b OaStd — small-string optimization (`SsoCap` chars) + heap tail via OaStdAllocBytes.
//
// Copies use `OaMemcpy` where contiguous; growth releases SSO to heap when needed.
// Interop: `StdStr()` copies to `std::string`; includes `<string>` only for that boundary.

#include <Oa/Core/Memory.h>
#include <Oa/Core/Std/Allocator.h>
#include <Oa/Core/Std/StringView.h>

#include <atomic>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef OA_ASSERT
#include <cassert>
#define OA_ASSERT(expr) assert(expr)
#endif

class OaStdString {
public:
	using size_type = std::size_t;

	static constexpr size_type SsoCap = 22;
	static constexpr size_type npos = OaStdStringView::Npos;

	OaStdString() noexcept {
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
	}

	OaStdString(const OaStdString& InO) {
		InitEmpty();
		AssignRange(InO.Data(), InO.Size());
	}

	OaStdString(OaStdString&& InO) noexcept {
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
		if (!InO.IsHeap_) {
			const size_type n = InO.SsoLen();
			if (n > 0) {
				OaMemcpy(SsoData(), InO.SsoData(), static_cast<OaUsize>(n));
			}
			Rep_.Sso.Buf[n] = '\0';
			Rep_.Sso.Len = InO.Rep_.Sso.Len;
			InO.Rep_.Sso.Len = 0;
			InO.Rep_.Sso.Buf[0] = '\0';
		} else {
			Rep_.Heap = InO.Rep_.Heap;
			IsHeap_ = true;
			InO.IsHeap_ = false;
			InO.Rep_.Sso.Len = 0;
			InO.Rep_.Sso.Buf[0] = '\0';
		}
	}

	explicit OaStdString(std::string InS) {
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
		AssignRange(InS.data(), InS.size());
	}

	OaStdString(const char* InCStr) {
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
		if (InCStr == nullptr) {
			return;
		}
		const size_type n = std::strlen(InCStr);
		AssignRange(InCStr, n);
	}

	OaStdString(const char* InData, size_type InLen) {
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
		AssignRange(InData, InLen);
	}

	explicit OaStdString(OaStdStringView InV) {
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
		AssignRange(InV.Data(), InV.Size());
	}

	~OaStdString() { DestroyHeap(); }

	OaStdString& operator=(const OaStdString& InO) {
		if (this == &InO) {
			return *this;
		}
		AssignRange(InO.Data(), InO.Size());
		return *this;
	}

	OaStdString& operator=(OaStdString&& InO) noexcept {
		if (this == &InO) {
			return *this;
		}
		DestroyHeap();
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
		if (!InO.IsHeap_) {
			const size_type n = InO.SsoLen();
			if (n > 0) {
				OaMemcpy(SsoData(), InO.SsoData(), static_cast<OaUsize>(n));
			}
			Rep_.Sso.Buf[n] = '\0';
			Rep_.Sso.Len = InO.Rep_.Sso.Len;
			InO.Rep_.Sso.Len = 0;
			InO.Rep_.Sso.Buf[0] = '\0';
		} else {
			Rep_.Heap = InO.Rep_.Heap;
			IsHeap_ = true;
			InO.IsHeap_ = false;
			InO.Rep_.Sso.Len = 0;
			InO.Rep_.Sso.Buf[0] = '\0';
		}
		return *this;
	}

	[[nodiscard]] std::string StdStr() const { return std::string(Data(), Size()); }

	[[nodiscard]] size_type Size() const noexcept {
		return IsHeap_ ? Rep_.Heap.Len : static_cast<size_type>(Rep_.Sso.Len);
	}

	[[nodiscard]] size_type size() const noexcept { return Size(); }

	[[nodiscard]] bool Empty() const noexcept { return Size() == 0; }
	[[nodiscard]] bool empty() const noexcept { return Empty(); }

	[[nodiscard]] const char* Data() const noexcept {
		return IsHeap_ ? Rep_.Heap.Ptr : SsoData();
	}

	[[nodiscard]] const char* CStr() const noexcept { return Data(); }
	[[nodiscard]] const char* c_str() const noexcept { return Data(); }

	void Clear() noexcept {
		DestroyHeap();
		InitEmpty();
		Rep_.Sso.Buf[0] = '\0';
	}

	void clear() noexcept { Clear(); }

	// Zero live buffer bytes (best-effort against compiler reordering), then release storage.
	void SecureWipeSecrets() noexcept {
		const size_type n = Size();
		if (n == 0) {
			return;
		}
		volatile char* ptr = MutableData();
		for (size_type i = 0; i < n; ++i) {
			ptr[i] = 0;
		}
		std::atomic_thread_fence(std::memory_order_seq_cst);
		Clear();
	}

	// Mirrors std::string::erase(pos, count); returns index of character after erased range.
	size_type erase(size_type InPos = 0, size_type InCount = npos) {
		const size_type sz = Size();
		if (InPos > sz) {
			throw std::out_of_range("OaStdString::erase");
		}
		const size_type tail = sz - InPos;
		const size_type removeN = (InCount == npos || InCount > tail) ? tail : InCount;
		if (removeN == 0) {
			return InPos;
		}
		const size_type newSz = sz - removeN;
		char* d = MutableData();
		const size_type keep = tail - removeN;
		if (keep > 0) {
			std::memmove(d + InPos, d + InPos + removeN, keep);
		}
		SetLen(newSz);
		DowngradeToSsoIfFits();
		return InPos;
	}

	void Reserve(size_type InCap) {
		const size_type sz = Size();
		const size_type need = InCap < sz ? sz : InCap;
		if (need <= SsoCap) {
			return;
		}
		EnsureHeapCapacityAtLeast(need);
	}

	void reserve(size_type InCap) { Reserve(InCap); }

	[[nodiscard]] size_type Capacity() const noexcept {
		return IsHeap_ ? Rep_.Heap.Cap : SsoCap;
	}

	[[nodiscard]] size_type capacity() const noexcept { return Capacity(); }

	[[nodiscard]] OaStdString substr(size_type InPos = 0, size_type InCount = npos) const {
		return OaStdString(View().SubStr(InPos, InCount));
	}

	void Resize(size_type InN) { Resize(InN, '\0'); }

	// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — mirrors std::string::resize(n, ch)
	void Resize(size_type InN, char InCh) {
		const size_type old = Size();
		if (InN == old) {
			return;
		}
		if (InN < old) {
			SetLen(InN);
			DowngradeToSsoIfFits();
			return;
		}
		EnsureTotalCapacity(InN);
		char* d = MutableData();
		for (size_type i = old; i < InN; ++i) {
			d[i] = InCh;
		}
		SetLen(InN);
	}

	void PushBack(char InCh) {
		const size_type old = Size();
		if (old == std::numeric_limits<size_type>::max()) {
			throw std::bad_array_new_length();
		}
		const size_type n = old + 1;
		EnsureTotalCapacity(n);
		char* d = MutableData();
		d[old] = InCh;
		SetLen(n);
	}

	void PopBack() {
		OA_ASSERT(!Empty());
		const size_type n = Size() - 1;
		SetLen(n);
		DowngradeToSsoIfFits();
	}

	void pop_back() { PopBack(); }

	[[nodiscard]] size_type length() const noexcept { return Size(); }

	[[nodiscard]] char& Front() noexcept {
		OA_ASSERT(!Empty());
		return MutableData()[0];
	}
	[[nodiscard]] const char& Front() const noexcept {
		OA_ASSERT(!Empty());
		return Data()[0];
	}
	[[nodiscard]] char& Back() noexcept {
		OA_ASSERT(!Empty());
		return MutableData()[Size() - 1U];
	}
	[[nodiscard]] const char& Back() const noexcept {
		OA_ASSERT(!Empty());
		return Data()[Size() - 1U];
	}
	[[nodiscard]] char& front() noexcept { return Front(); }
	[[nodiscard]] const char& front() const noexcept { return Front(); }
	[[nodiscard]] char& back() noexcept { return Back(); }
	[[nodiscard]] const char& back() const noexcept { return Back(); }

	OaStdString& Append(OaStdStringView InV) {
		const char* p = InV.Data();
		size_type n = InV.Size();
		if (n == 0) {
			return *this;
		}
		if (!IsHeap_) {
			if (p >= SsoData() && p < SsoData() + Size()) {
				OaStdString tmp(InV);
				return Append(tmp.View());
			}
		} else {
			if (p >= Rep_.Heap.Ptr && p < Rep_.Heap.Ptr + Rep_.Heap.Len) {
				OaStdString tmp(InV);
				return Append(tmp.View());
			}
		}
		const size_type old = Size();
		if (n > std::numeric_limits<size_type>::max() - old) {
			throw std::bad_array_new_length();
		}
		const size_type newLen = old + n;
		EnsureTotalCapacity(newLen);
		OaMemcpy(MutableData() + old, p, static_cast<OaUsize>(n));
		SetLen(newLen);
		return *this;
	}

	OaStdString& Append(const char* InS) {
		if (InS == nullptr) {
			return *this;
		}
		return Append(OaStdStringView(InS));
	}

	// Operator overloading
	OaStdString& operator+=(OaStdStringView InV) { return Append(InV); }
	OaStdString& operator+=(const char* InS) { return Append(InS); }
	OaStdString& operator+=(char InCh) { PushBack(InCh); return *this; }
	OaStdString& operator+=(const OaStdString& InO) {
		return Append(OaStdStringView(InO.Data(), InO.Size()));
	}
	OaStdString& operator+=(std::string_view InV) {
		return Append(OaStdStringView(InV.data(), InV.size()));
	}
	OaStdString& operator+=(const std::string& InS) {
		return Append(OaStdStringView(InS.data(), InS.size()));
	}

	[[nodiscard]] char& operator[](size_type InIdx) { return MutableData()[InIdx]; }

	[[nodiscard]] const char& operator[](size_type InIdx) const { return Data()[InIdx]; }

	[[nodiscard]] char& At(size_type InIdx) {
		if (InIdx >= Size()) {
			throw std::out_of_range("OaStdString::At");
		}
		return MutableData()[InIdx];
	}

	[[nodiscard]] const char& At(size_type InIdx) const {
		if (InIdx >= Size()) {
			throw std::out_of_range("OaStdString::At");
		}
		return Data()[InIdx];
	}

	[[nodiscard]] OaStdStringView View() const noexcept {
		return OaStdStringView(Data(), Size());
	}

	// Member begin/end so range-for mutates this string (free begin(OaStdStringView) would not).
	[[nodiscard]] char* begin() noexcept { return MutableData(); }
	[[nodiscard]] const char* begin() const noexcept { return Data(); }
	[[nodiscard]] char* end() noexcept { return MutableData() + Size(); }
	[[nodiscard]] const char* end() const noexcept { return Data() + Size(); }

	// Converts to view into this string; do not store the view past the string's lifetime.
	[[nodiscard]] operator OaStdStringView() const noexcept { return View(); }

	[[nodiscard]] size_type find(OaStdStringView InNeedle, size_type InPos = 0) const noexcept {
		return View().find(InNeedle, InPos);
	}

	[[nodiscard]] size_type find(char InCh, size_type InPos = 0) const noexcept {
		return View().find(InCh, InPos);
	}

	[[nodiscard]] size_type find(const char* InS, size_type InPos = 0) const noexcept {
		return View().find(InS, InPos);
	}

	[[nodiscard]] bool Equals(OaStdStringView InV) const noexcept {
		const size_type n = Size();
		if (n != InV.Size()) {
			return false;
		}
		return n == 0 || std::memcmp(Data(), InV.Data(), n) == 0;
	}

	[[nodiscard]] bool Equals(const OaStdString& InO) const noexcept { return Equals(InO.View()); }

	// Lexicographic ordering (unsigned char semantics via memcmp).
	[[nodiscard]] int Compare(OaStdStringView InV) const noexcept {
		const size_type na = Size();
		const size_type nb = InV.Size();
		const size_type n = na < nb ? na : nb;
		if (n > 0) {
			const int cmp = std::memcmp(Data(), InV.Data(), static_cast<size_t>(n));
			if (cmp != 0) {
				return cmp < 0 ? -1 : 1;
			}
		}
		if (na < nb) return -1;
		if (na > nb) return 1;
		return 0;
	}

	[[nodiscard]] int Compare(const OaStdString& InO) const noexcept { return Compare(InO.View()); }

private:
	union Rep {
		struct {
			char Buf[SsoCap + 1];
			unsigned char Len;
		} Sso;
		struct {
			char* Ptr;
			size_type Len;
			size_type Cap;
		} Heap;
	} Rep_{};
	bool IsHeap_{false};

	void InitEmpty() noexcept {
		IsHeap_ = false;
		Rep_.Sso.Len = 0;
	}

	void DestroyHeap() noexcept {
		if (IsHeap_) {
			OaStdFreeBytes(Rep_.Heap.Ptr);
			IsHeap_ = false;
			Rep_.Sso.Len = 0;
		}
	}

	[[nodiscard]] size_type SsoLen() const noexcept {
		return static_cast<size_type>(Rep_.Sso.Len);
	}

	[[nodiscard]] char* SsoData() noexcept { return Rep_.Sso.Buf; }

	[[nodiscard]] const char* SsoData() const noexcept { return Rep_.Sso.Buf; }

	[[nodiscard]] char* MutableData() noexcept {
		return IsHeap_ ? Rep_.Heap.Ptr : SsoData();
	}

	void SetSso(const char* InP, size_type InLen) {
		OA_ASSERT(InLen <= SsoCap);
		if (InLen > 0) {
			OaMemcpy(SsoData(), InP, static_cast<OaUsize>(InLen));
		}
		Rep_.Sso.Buf[InLen] = '\0';
		Rep_.Sso.Len = static_cast<unsigned char>(InLen);
		IsHeap_ = false;
	}

	void SetLen(size_type InN) {
		if (IsHeap_) {
			Rep_.Heap.Len = InN;
			Rep_.Heap.Ptr[InN] = '\0';
		} else {
			Rep_.Sso.Len = static_cast<unsigned char>(InN);
			Rep_.Sso.Buf[InN] = '\0';
		}
	}

	void DowngradeToSsoIfFits() {
		if (!IsHeap_ || Rep_.Heap.Len > SsoCap) {
			return;
		}
		char tmp[SsoCap + 1];
		const size_type len = Rep_.Heap.Len;
		if (len > 0) {
			OaMemcpy(tmp, Rep_.Heap.Ptr, static_cast<OaUsize>(len));
		}
		OaStdFreeBytes(Rep_.Heap.Ptr);
		IsHeap_ = false;
		SetSso(tmp, len);
	}

	void EnsureHeapCapacityAtLeast(size_type InMinCap) {
		if (InMinCap <= SsoCap) {
			return;
		}
		if (!IsHeap_) {
			const size_type len = SsoLen();
			const size_type allocBytes = InMinCap + 1;
			if (allocBytes <= InMinCap) {
				throw std::bad_array_new_length();
			}
			void* raw = OaStdAllocBytes(allocBytes, 1);
			char* p = static_cast<char*>(raw);
			if (len > 0) {
				OaMemcpy(p, SsoData(), static_cast<OaUsize>(len));
			}
			p[len] = '\0';
			Rep_.Heap.Ptr = p;
			Rep_.Heap.Len = len;
			Rep_.Heap.Cap = InMinCap;
			IsHeap_ = true;
			return;
		}
		if (Rep_.Heap.Cap >= InMinCap) {
			return;
		}
		size_type newCap = Rep_.Heap.Cap;
		while (newCap < InMinCap) {
			if (newCap >= (std::numeric_limits<size_type>::max() / 2) - 16) {
				newCap = InMinCap;
				break;
			}
			const size_type next = (newCap * 2) + 16;
			newCap = next < InMinCap ? InMinCap : next;
		}
		const size_type allocBytes = newCap + 1;
		if (allocBytes <= newCap) {
			throw std::bad_array_new_length();
		}
		void* raw = OaStdAllocBytes(allocBytes, 1);
		char* p = static_cast<char*>(raw);
		OaMemcpy(p, Rep_.Heap.Ptr, static_cast<OaUsize>(Rep_.Heap.Len));
		OaStdFreeBytes(Rep_.Heap.Ptr);
		Rep_.Heap.Ptr = p;
		Rep_.Heap.Cap = newCap;
		Rep_.Heap.Ptr[Rep_.Heap.Len] = '\0';
	}

	void EnsureTotalCapacity(size_type InNeedLen) {
		if (InNeedLen <= SsoCap) {
			return;
		}
		EnsureHeapCapacityAtLeast(InNeedLen);
	}

	void AssignRange(const char* InP, size_type InN) {
		if (InN == 0) {
			Clear();
			return;
		}
		if (InN <= SsoCap) {
			if (IsHeap_) {
				OaStdFreeBytes(Rep_.Heap.Ptr);
				IsHeap_ = false;
			}
			SetSso(InP, InN);
			return;
		}
		EnsureHeapCapacityAtLeast(InN);
		OaMemcpy(Rep_.Heap.Ptr, InP, static_cast<OaUsize>(InN));
		Rep_.Heap.Ptr[InN] = '\0';
		Rep_.Heap.Len = InN;
		IsHeap_ = true;
	}
};

inline bool operator==(const OaStdString& InA, const OaStdString& InB) noexcept {
	return InA.Equals(InB);
}

inline bool operator!=(const OaStdString& InA, const OaStdString& InB) noexcept {
	return !InA.Equals(InB);
}

inline bool operator<(const OaStdString& InA, const OaStdString& InB) noexcept {
	return InA.Compare(InB.View()) < 0;
}

inline bool operator==(const OaStdString& InA, OaStdStringView InB) noexcept {
	return InA.Equals(InB);
}

inline bool operator==(OaStdStringView InA, const OaStdString& InB) noexcept {
	return InB.Equals(InA);
}

inline bool operator!=(const OaStdString& InA, OaStdStringView InB) noexcept {
	return !InA.Equals(InB);
}

inline bool operator!=(OaStdStringView InA, const OaStdString& InB) noexcept {
	return !InB.Equals(InA);
}

inline bool operator==(const OaStdString& InA, const char* InB) noexcept {
	return InA.Equals(OaStdStringView(InB));
}

inline bool operator==(const char* InA, const OaStdString& InB) noexcept {
	return InB.Equals(OaStdStringView(InA));
}

inline bool operator!=(const OaStdString& InA, const char* InB) noexcept {
	return !InA.Equals(OaStdStringView(InB));
}

inline bool operator!=(const char* InA, const OaStdString& InB) noexcept {
	return !InB.Equals(OaStdStringView(InA));
}

inline bool operator==(const OaStdString& InA, const std::string& InB) noexcept {
	return InA.Equals(OaStdStringView(InB.data(), InB.size()));
}

inline bool operator==(const std::string& InA, const OaStdString& InB) noexcept {
	return InB.Equals(OaStdStringView(InA.data(), InA.size()));
}

inline bool operator!=(const OaStdString& InA, const std::string& InB) noexcept {
	return !InA.Equals(OaStdStringView(InB.data(), InB.size()));
}

inline bool operator!=(const std::string& InA, const OaStdString& InB) noexcept {
	return !InB.Equals(OaStdStringView(InA.data(), InA.size()));
}

inline OaStdString operator+(const OaStdString& InA, const OaStdString& InB) {
	OaStdString r(InA);
	r.Append(InB.View());
	return r;
}

inline OaStdString operator+(const OaStdString& InA, OaStdStringView InB) {
	OaStdString r(InA);
	r.Append(InB);
	return r;
}

inline OaStdString operator+(OaStdStringView InA, const OaStdString& InB) {
	OaStdString r(InA);
	r.Append(InB.View());
	return r;
}

inline OaStdString operator+(const OaStdString& InA, const char* InB) {
	OaStdString r(InA);
	r.Append(InB);
	return r;
}

inline OaStdString operator+(const char* InA, const OaStdString& InB) {
	OaStdString r(InA);
	r.Append(InB.View());
	return r;
}

inline OaStdString operator+(const OaStdString& InA, const std::string& InB) {
	OaStdString r(InA);
	r.Append(OaStdStringView(InB.data(), InB.size()));
	return r;
}

inline OaStdString operator+(const std::string& InA, const OaStdString& InB) {
	OaStdString r(InA.data(), InA.size());
	r.Append(InB.View());
	return r;
}

namespace std {

template<>
struct hash<OaStdString> {
	std::size_t operator()(const OaStdString& InS) const noexcept {
		return hash<string_view>{}(string_view(InS.Data(), InS.Size()));
	}
};

template<>
struct hash<OaStdStringView> {
	std::size_t operator()(OaStdStringView InV) const noexcept {
		return hash<string_view>{}(InV.StdView());
	}
};

} // namespace std
