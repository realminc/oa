#pragma once

// OA-owned string with small-string optimization (`SsoCap` chars) and an
// oa::allocBytes-backed heap tail.
//
// Copies use `oa::memcpy` where contiguous; growth releases SSO to heap when needed.
// Hosted-library conversions belong at their owning filesystem, parser, CLI,
// binding, or serialization boundary rather than in this foundation type.

#include <oa/core/memory.h>
#include <oa/core/assert.h>
#include <oa/core/std/allocator.h>
#include <oa/core/std/atomic.h>
#include <oa/core/std/stringView.h>

namespace oa {

class String {
public:
	using size_type = oa::Usize;

	static constexpr size_type SsoCap = 22;
	static constexpr size_type Npos = oa::StringView::Npos;

	String() noexcept {
		initEmpty();
		rep_.sso.buf[0] = '\0';
	}

	String(const String& inO) {
		initEmpty();
		assignRange(inO.data(), inO.size());
	}

	String(String&& inO) noexcept {
		initEmpty();
		rep_.sso.buf[0] = '\0';
		if (!inO.isHeap_) {
			const size_type n = inO.ssoLen();
			if (n > 0) {
				oa::memcpy(ssoData(), inO.ssoData(), n);
			}
			rep_.sso.buf[n] = '\0';
			rep_.sso.len = inO.rep_.sso.len;
			inO.rep_.sso.len = 0;
			inO.rep_.sso.buf[0] = '\0';
		} else {
			rep_.heap = inO.rep_.heap;
			isHeap_ = true;
			inO.isHeap_ = false;
			inO.rep_.sso.len = 0;
			inO.rep_.sso.buf[0] = '\0';
		}
	}

	String(const char* inCStr) {
		initEmpty();
		rep_.sso.buf[0] = '\0';
		if (inCStr == nullptr) {
			return;
		}
		const size_type n = oa::strlen(inCStr);
		assignRange(inCStr, n);
	}

	String(const char* inData, size_type inLen) {
		initEmpty();
		rep_.sso.buf[0] = '\0';
		assignRange(inData, inLen);
	}

	explicit String(oa::StringView inV) {
		initEmpty();
		rep_.sso.buf[0] = '\0';
		assignRange(inV.data(), inV.size());
	}

	~String() { destroyHeap(); }

	String& operator=(const String& inO) {
		if (this == &inO) {
			return *this;
		}
		assignRange(inO.data(), inO.size());
		return *this;
	}

	String& operator=(String&& inO) noexcept {
		if (this == &inO) {
			return *this;
		}
		destroyHeap();
		initEmpty();
		rep_.sso.buf[0] = '\0';
		if (!inO.isHeap_) {
			const size_type n = inO.ssoLen();
			if (n > 0) {
				oa::memcpy(ssoData(), inO.ssoData(), n);
			}
			rep_.sso.buf[n] = '\0';
			rep_.sso.len = inO.rep_.sso.len;
			inO.rep_.sso.len = 0;
			inO.rep_.sso.buf[0] = '\0';
		} else {
			rep_.heap = inO.rep_.heap;
			isHeap_ = true;
			inO.isHeap_ = false;
			inO.rep_.sso.len = 0;
			inO.rep_.sso.buf[0] = '\0';
		}
		return *this;
	}

	[[nodiscard]] size_type size() const noexcept {
		return isHeap_ ? rep_.heap.len : static_cast<size_type>(rep_.sso.len);
	}

	[[nodiscard]] bool empty() const noexcept { return size() == 0; }

	[[nodiscard]] const char* data() const noexcept {
		return isHeap_ ? rep_.heap.ptr : ssoData();
	}
	[[nodiscard]] const char* cStr() const noexcept { return data(); }

	void clear() noexcept {
		destroyHeap();
		initEmpty();
		rep_.sso.buf[0] = '\0';
	}

	// Zero live buffer bytes (best-effort against compiler reordering), then release storage.
	void secureWipeSecrets() noexcept {
		const size_type n = size();
		if (n == 0) {
			return;
		}
		volatile char* ptr = mutableData();
		for (size_type i = 0; i < n; ++i) {
			ptr[i] = 0;
		}
		oa::atomicThreadFence(oa::MemoryOrder::Sequential);
		clear();
	}

	// Mirrors std::string::erase(pos, count); returns index of character after erased range.
	size_type erase(size_type inPos = 0, size_type inCount = Npos) {
		const size_type sz = size();
		OA_REQUIRE(inPos <= sz);
		const size_type tail = sz - inPos;
		const size_type removeN = (inCount == Npos || inCount > tail) ? tail : inCount;
		if (removeN == 0) {
			return inPos;
		}
		const size_type newSz = sz - removeN;
		char* d = mutableData();
		const size_type keep = tail - removeN;
		if (keep > 0) {
			__builtin_memmove(d + inPos, d + inPos + removeN, keep);
		}
		setLen(newSz);
		downgradeToSsoIfFits();
		return inPos;
	}

	void reserve(size_type inCap) {
		const size_type sz = size();
		const size_type need = inCap < sz ? sz : inCap;
		if (need <= SsoCap) {
			return;
		}
		ensureHeapCapacityAtLeast(need);
	}

	[[nodiscard]] size_type capacity() const noexcept {
		return isHeap_ ? rep_.heap.cap : SsoCap;
	}

	[[nodiscard]] String substr(size_type inPos = 0, size_type inCount = Npos) const {
		return String(view().subStr(inPos, inCount));
	}

	void resize(size_type inN) { resize(inN, '\0'); }

	// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) — mirrors std::string::resize(n, ch)
	void resize(size_type inN, char inCh) {
		const size_type old = size();
		if (inN == old) {
			return;
		}
		if (inN < old) {
			setLen(inN);
			downgradeToSsoIfFits();
			return;
		}
		ensureTotalCapacity(inN);
		char* d = mutableData();
		for (size_type i = old; i < inN; ++i) {
			d[i] = inCh;
		}
		setLen(inN);
	}

	void pushBack(char inCh) {
		if (isHeap_) {
			const size_type old = rep_.heap.len;
			if (old == rep_.heap.cap) {
				growForPushBack(old);
			}
			char* const data = rep_.heap.ptr;
			data[old] = inCh;
			data[old + 1] = '\0';
			rep_.heap.len = old + 1;
			return;
		}

		const size_type old = ssoLen();
		if (old < SsoCap) {
			rep_.sso.buf[old] = inCh;
			rep_.sso.buf[old + 1] = '\0';
			rep_.sso.len = static_cast<unsigned char>(old + 1);
			return;
		}

		ensureHeapCapacityAtLeast(SsoCap + 1);
		rep_.heap.ptr[old] = inCh;
		rep_.heap.ptr[old + 1] = '\0';
		rep_.heap.len = old + 1;
	}

	void popBack() {
		OA_REQUIRE(!empty());
		const size_type n = size() - 1;
		setLen(n);
		downgradeToSsoIfFits();
	}

	[[nodiscard]] size_type length() const noexcept { return size(); }

	[[nodiscard]] char& front() noexcept {
		OA_REQUIRE(!empty());
		return mutableData()[0];
	}
	[[nodiscard]] const char& front() const noexcept {
		OA_REQUIRE(!empty());
		return data()[0];
	}
	[[nodiscard]] char& back() noexcept {
		OA_REQUIRE(!empty());
		return mutableData()[size() - 1U];
	}
	[[nodiscard]] const char& back() const noexcept {
		OA_REQUIRE(!empty());
		return data()[size() - 1U];
	}
	String& append(oa::StringView inV) {
		const char* p = inV.data();
		size_type n = inV.size();
		if (n == 0) {
			return *this;
		}
		if (!isHeap_) {
			if (p >= ssoData() && p < ssoData() + size()) {
				String tmp(inV);
				return append(tmp.view());
			}
		} else {
			if (p >= rep_.heap.ptr && p < rep_.heap.ptr + rep_.heap.len) {
				String tmp(inV);
				return append(tmp.view());
			}
		}
		const size_type old = size();
		if (n > static_cast<size_type>(-1) - old) {
			oa::allocationFailed(oa::AllocationError::SizeOverflow, old, 1);
		}
		const size_type newLen = old + n;
		ensureTotalCapacity(newLen);
		oa::memcpy(mutableData() + old, p, static_cast<oa::Usize>(n));
		setLen(newLen);
		return *this;
	}

	String& append(const char* inS) {
		if (inS == nullptr) {
			return *this;
		}
		return append(oa::StringView(inS));
	}

	// Operator overloading
	String& operator+=(oa::StringView inV) { return append(inV); }
	String& operator+=(const char* inS) { return append(inS); }
	String& operator+=(char inCh) { pushBack(inCh); return *this; }
	String& operator+=(const String& inO) {
		return append(oa::StringView(inO.data(), inO.size()));
	}

	[[nodiscard]] char& operator[](size_type inIdx) { return mutableData()[inIdx]; }

	[[nodiscard]] const char& operator[](size_type inIdx) const { return data()[inIdx]; }

	[[nodiscard]] char& at(size_type inIdx) {
		OA_REQUIRE(inIdx < size());
		return mutableData()[inIdx];
	}

	[[nodiscard]] const char& at(size_type inIdx) const {
		OA_REQUIRE(inIdx < size());
		return data()[inIdx];
	}

	[[nodiscard]] oa::StringView view() const noexcept {
		return oa::StringView(data(), size());
	}

	// Member begin/end so range-for mutates this string (free begin(oa::StringView) would not).
	[[nodiscard]] char* begin() noexcept { return mutableData(); }
	[[nodiscard]] const char* begin() const noexcept { return data(); }
	[[nodiscard]] char* end() noexcept { return mutableData() + size(); }
	[[nodiscard]] const char* end() const noexcept { return data() + size(); }

	// Converts to view into this string; do not store the view past the string's lifetime.
	[[nodiscard]] operator oa::StringView() const noexcept { return view(); }

	[[nodiscard]] size_type find(oa::StringView inNeedle, size_type inPos = 0) const noexcept {
		return view().find(inNeedle, inPos);
	}

	[[nodiscard]] size_type find(char inCh, size_type inPos = 0) const noexcept {
		return view().find(inCh, inPos);
	}

	[[nodiscard]] size_type find(const char* inS, size_type inPos = 0) const noexcept {
		return view().find(inS, inPos);
	}

	[[nodiscard]] bool equals(oa::StringView inV) const noexcept {
		const size_type n = size();
		if (n != inV.size()) {
			return false;
		}
		return n == 0 || oa::memcmp(data(), inV.data(), n) == 0;
	}

	[[nodiscard]] bool equals(const String& inO) const noexcept { return equals(inO.view()); }

	// Lexicographic ordering (unsigned char semantics via memcmp).
	[[nodiscard]] int compare(oa::StringView inV) const noexcept {
		const size_type na = size();
		const size_type nb = inV.size();
		const size_type n = na < nb ? na : nb;
		if (n > 0) {
			const int cmp = oa::memcmp(data(), inV.data(), n);
			if (cmp != 0) {
				return cmp < 0 ? -1 : 1;
			}
		}
		if (na < nb) return -1;
		if (na > nb) return 1;
		return 0;
	}

	[[nodiscard]] int compare(const String& inO) const noexcept { return compare(inO.view()); }

private:
	union Rep {
		struct {
			char buf[SsoCap + 1];
			unsigned char len;
		} sso;
		struct {
			char* ptr;
			size_type len;
			size_type cap;
		} heap;
	} rep_{};
	bool isHeap_{false};

	void initEmpty() noexcept {
		isHeap_ = false;
		rep_.sso.len = 0;
	}

	void destroyHeap() noexcept {
		if (isHeap_) {
			oa::freeBytes(rep_.heap.ptr);
			isHeap_ = false;
			rep_.sso.len = 0;
		}
	}

	[[nodiscard]] size_type ssoLen() const noexcept {
		return static_cast<size_type>(rep_.sso.len);
	}

	[[nodiscard]] char* ssoData() noexcept { return rep_.sso.buf; }

	[[nodiscard]] const char* ssoData() const noexcept { return rep_.sso.buf; }

	[[nodiscard]] char* mutableData() noexcept {
		return isHeap_ ? rep_.heap.ptr : ssoData();
	}

	void setSso(const char* inP, size_type inLen) {
		OA_ASSERT(inLen <= SsoCap);
		if (inLen > 0) {
			oa::memcpy(ssoData(), inP, static_cast<oa::Usize>(inLen));
		}
		rep_.sso.buf[inLen] = '\0';
		rep_.sso.len = static_cast<unsigned char>(inLen);
		isHeap_ = false;
	}

	void setLen(size_type inN) {
		if (isHeap_) {
			rep_.heap.len = inN;
			rep_.heap.ptr[inN] = '\0';
		} else {
			rep_.sso.len = static_cast<unsigned char>(inN);
			rep_.sso.buf[inN] = '\0';
		}
	}

	void downgradeToSsoIfFits() {
		if (!isHeap_ || rep_.heap.len > SsoCap) {
			return;
		}
		char tmp[SsoCap + 1];
		const size_type len = rep_.heap.len;
		if (len > 0) {
			oa::memcpy(tmp, rep_.heap.ptr, static_cast<oa::Usize>(len));
		}
		oa::freeBytes(rep_.heap.ptr);
		isHeap_ = false;
		setSso(tmp, len);
	}

	void ensureHeapCapacityAtLeast(size_type inMinCap) {
		if (inMinCap <= SsoCap) {
			return;
		}
		if (!isHeap_) {
			const size_type len = ssoLen();
			const size_type allocBytes = inMinCap + 1;
			if (allocBytes <= inMinCap) {
				oa::allocationFailed(oa::AllocationError::SizeOverflow, inMinCap, 1);
			}
			void* raw = oa::allocBytes(allocBytes, 1);
			char* p = static_cast<char*>(raw);
			if (len > 0) {
				oa::memcpy(p, ssoData(), static_cast<oa::Usize>(len));
			}
			p[len] = '\0';
			rep_.heap.ptr = p;
			rep_.heap.len = len;
			rep_.heap.cap = inMinCap;
			isHeap_ = true;
			return;
		}
		if (rep_.heap.cap >= inMinCap) {
			return;
		}
		size_type newCap = rep_.heap.cap;
		while (newCap < inMinCap) {
			if (newCap >= (static_cast<size_type>(-1) / 2) - 16) {
				newCap = inMinCap;
				break;
			}
			const size_type next = (newCap * 2) + 16;
			newCap = next < inMinCap ? inMinCap : next;
		}
		const size_type allocBytes = newCap + 1;
		if (allocBytes <= newCap) {
			oa::allocationFailed(oa::AllocationError::SizeOverflow, newCap, 1);
		}
		void* raw = oa::allocBytes(allocBytes, 1);
		char* p = static_cast<char*>(raw);
		oa::memcpy(p, rep_.heap.ptr, static_cast<oa::Usize>(rep_.heap.len));
		oa::freeBytes(rep_.heap.ptr);
		rep_.heap.ptr = p;
		rep_.heap.cap = newCap;
		rep_.heap.ptr[rep_.heap.len] = '\0';
	}

	void ensureTotalCapacity(size_type inNeedLen) {
		if (inNeedLen <= SsoCap) {
			return;
		}
		ensureHeapCapacityAtLeast(inNeedLen);
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((noinline)) __attribute__((cold))
#endif
	void growForPushBack(size_type inOldLen) {
		if (inOldLen == static_cast<size_type>(-1)) {
			oa::allocationFailed(oa::AllocationError::SizeOverflow, inOldLen, 1);
		}
		ensureHeapCapacityAtLeast(inOldLen + 1);
	}

	void assignRange(const char* inP, size_type inN) {
		if (inN == 0) {
			clear();
			return;
		}
		if (inN <= SsoCap) {
			if (isHeap_) {
				oa::freeBytes(rep_.heap.ptr);
				isHeap_ = false;
			}
			setSso(inP, inN);
			return;
		}
		ensureHeapCapacityAtLeast(inN);
		oa::memcpy(rep_.heap.ptr, inP, static_cast<oa::Usize>(inN));
		rep_.heap.ptr[inN] = '\0';
		rep_.heap.len = inN;
		isHeap_ = true;
	}
};

inline bool operator==(const String& inA, const String& inB) noexcept {
	return inA.equals(inB);
}

inline bool operator!=(const String& inA, const String& inB) noexcept {
	return !inA.equals(inB);
}

inline bool operator<(const String& inA, const String& inB) noexcept {
	return inA.compare(inB.view()) < 0;
}

inline bool operator==(const String& inA, oa::StringView inB) noexcept {
	return inA.equals(inB);
}

inline bool operator==(oa::StringView inA, const String& inB) noexcept {
	return inB.equals(inA);
}

inline bool operator!=(const String& inA, oa::StringView inB) noexcept {
	return !inA.equals(inB);
}

inline bool operator!=(oa::StringView inA, const String& inB) noexcept {
	return !inB.equals(inA);
}

inline bool operator==(const String& inA, const char* inB) noexcept {
	return inA.equals(oa::StringView(inB));
}

inline bool operator==(const char* inA, const String& inB) noexcept {
	return inB.equals(oa::StringView(inA));
}

inline bool operator!=(const String& inA, const char* inB) noexcept {
	return !inA.equals(oa::StringView(inB));
}

inline bool operator!=(const char* inA, const String& inB) noexcept {
	return !inB.equals(oa::StringView(inA));
}

inline String operator+(const String& inA, const String& inB) {
	String r(inA);
	r.append(inB.view());
	return r;
}

inline String operator+(const String& inA, oa::StringView inB) {
	String r(inA);
	r.append(inB);
	return r;
}

inline String operator+(oa::StringView inA, const String& inB) {
	String r(inA);
	r.append(inB.view());
	return r;
}

inline String operator+(const String& inA, const char* inB) {
	String r(inA);
	r.append(inB);
	return r;
}

inline String operator+(const char* inA, const String& inB) {
	String r(inA);
	r.append(inB.view());
	return r;
}

} // namespace oa
