#pragma once

// Vec<T> — phase 2 (OA standard library roadmap)
//
// Header-only template (same as std::vector): full definition must stay in headers
// unless you add explicit instantiations in a .cpp for specific T.
//
// Growable array: 2× growth (same amortized pattern as libstdc++/libc++).
// Trivially copyable T with alignof(T) <= max_align_t: malloc/realloc — in-place
// expansion when libc allows (avoids memcpy+free on most growth steps).
// Larger / over-aligned T: oa::allocBytes / oa::freeBytes (operator new/delete,
// same as oa::Allocator) + oa::memcpy (or element-wise move). Trivial path stays
// malloc/realloc (no oa::allocBytes — realloc in-place growth).
// Non-trivial T: per-element move + destructor on grow/shrink.
//
// Hot paths: a three-pointer representation keeps begin/end/capacity in registers;
// pushBack and its uncommon growth edge inline so Clang can scalar-replace that
// state. Trivial T stores use a restrict-qualified slot (no construct_at).
// Trivial popBack skips destroy_at. clear() ends at destroyAll only.
// Trivial bulk: append(indata, incount) uses oa::memcpy; value-init resize uses oa::memzero
// for arithmetic/enum/pointer T; contiguous iterator ranges use oa::memcpy when allowed.
//
// Include as <oa/core/std/vec.h> or through <oa/core/std.h> or <oa/core/types.h>.

#include <oa/core/memory.h>
#include <oa/core/assert.h>
#include <oa/core/std/allocator.h>
#include <oa/core/std/iter.h>
#include <oa/core/std/lifetime.h>
#include <oa/core/std/span.h>
#include <oa/core/std/typeTraits.h>

#include <initializer_list>

#ifndef OA_LIKELY
#define OA_LIKELY(x) (x)
#define OA_UNLIKELY(x) (x)
#endif

#ifndef OA_RESTRICT
	#if defined(__clang__) || defined(__GNUC__)
		#define OA_RESTRICT __restrict__
	#else
		#define OA_RESTRICT
	#endif
#endif

namespace oa {

template<typename T>
class Vec {
public:
	using value_type = T;
	using size_type = oa::Usize;
	using difference_type = oa::Isize;
	using reference = T&;
	using const_reference = const T&;
	using pointer = T*;
	using const_pointer = const T*;
	using iterator = T*;
	using const_iterator = const T*;
	using reverse_iterator = oa::ReverseIterator<iterator>;
	using const_reverse_iterator = oa::ReverseIterator<const_iterator>;

	Vec() noexcept = default;

	explicit Vec(size_type incount) { resizeValueInit(incount); }

	Vec(size_type incount, const T& inval) {
		if (incount == 0) return;
		reserve(incount);
		if constexpr (oa::IsTriviallyCopyableV<T>) {
			for (size_type i = 0; i < incount; ++i) ptr_[i] = inval;
			end_ = ptr_ + incount;
		} else {
			for (size_type i = 0; i < incount; ++i) pushBack(inval);
		}
	}

	Vec(std::initializer_list<T> inInit) : Vec(inInit.begin(), inInit.end()) {}

	// Exclude integral It so Vec<int>(3, 4) resolves to (size_type, const T&), not iterator pair.
	template<typename It>
	requires (!oa::IsIntegralV<oa::RemoveCvrefT<It>>
		&& requires(It inIt) { ++inIt; *inIt; })
	Vec(It infirst, It inlast) {
#if __cplusplus >= 202002L
		if constexpr (oa::IsRandomAccessIteratorV<It>) {
			const auto dist = inlast - infirst;
			if (dist <= 0) return;
			const size_type n = static_cast<size_type>(dist);
			reserve(n);
			if constexpr (oa::IsContiguousIteratorV<It>) {
				if constexpr (oa::IsSameV<oa::RemoveConstT<oa::IterValueT<It>>, T> && oa::IsTriviallyCopyableV<T>) {
					oa::memcpy(ptr_, oa::toAddress(infirst), n * sizeof(T));
					end_ = ptr_ + n;
					return;
				}
			}
			It it = infirst;
			for (size_type i = 0; i < n; ++i, ++it) {
				if constexpr (oa::IsTriviallyCopyableV<T>) ptr_[i] = *it;
				else oa::constructAt(ptr_ + i, *it);
			}
			end_ = ptr_ + n;
			return;
		}
#endif
		for (; infirst != inlast; ++infirst) pushBack(*infirst);
	}

	Vec(const Vec& inother) requires oa::IsCopyConstructibleV<T>
		: Vec(inother.begin(), inother.end()) {}

	Vec(const Vec&) requires (!oa::IsCopyConstructibleV<T>) = delete;

	Vec(Vec&& inother) noexcept
		: ptr_(inother.ptr_), end_(inother.end_), capEnd_(inother.capEnd_) {
		inother.ptr_ = nullptr;
		inother.end_ = nullptr;
		inother.capEnd_ = nullptr;
	}

	Vec& operator=(const Vec& inother) requires oa::IsCopyConstructibleV<T> {
		if (this == &inother) return *this;
		Vec tmp(inother);
		swap(tmp);
		return *this;
	}

	Vec& operator=(const Vec&) requires (!oa::IsCopyConstructibleV<T>) = delete;

	Vec& operator=(Vec&& inother) noexcept(
		oa::IsNothrowMoveConstructibleV<T> && oa::IsNothrowMoveAssignableV<T>) {
		if (this == &inother) return *this;
		destroyAll();
		deallocate();
		ptr_ = inother.ptr_;
		end_ = inother.end_;
		capEnd_ = inother.capEnd_;
		inother.ptr_ = nullptr;
		inother.end_ = nullptr;
		inother.capEnd_ = nullptr;
		return *this;
	}

	Vec& operator=(std::initializer_list<T> inInit) {
		assign(inInit.begin(), inInit.end());
		return *this;
	}

	~Vec() {
		destroyAll();
		deallocate();
	}

	reference operator[](size_type inidx) {
		OA_ASSERT(inidx < size());
		return ptr_[inidx];
	}

	const_reference operator[](size_type inidx) const {
		OA_ASSERT(inidx < size());
		return ptr_[inidx];
	}

	reference at(size_type inidx) noexcept {
		OA_REQUIRE(inidx < size());
		return ptr_[inidx];
	}

	const_reference at(size_type inidx) const noexcept {
		OA_REQUIRE(inidx < size());
		return ptr_[inidx];
	}

	reference front() {
		OA_ASSERT(!empty());
		return ptr_[0];
	}

	const_reference front() const {
		OA_ASSERT(!empty());
		return ptr_[0];
	}

	reference back() {
		OA_ASSERT(!empty());
		return end_[-1];
	}

	const_reference back() const {
		OA_ASSERT(!empty());
		return end_[-1];
	}

	pointer data() noexcept { return ptr_; }

	const_pointer data() const noexcept { return ptr_; }

	[[nodiscard]] bool empty() const noexcept { return end_ == ptr_; }

	[[nodiscard]] size_type size() const noexcept {
		return ptr_ == nullptr ? 0 : static_cast<size_type>(end_ - ptr_);
	}

	[[nodiscard]] oa::Span<T> span() noexcept { return oa::Span<T>(data(), size()); }

	[[nodiscard]] oa::Span<const T> span() const noexcept {
		return oa::Span<const T>(data(), size());
	}

	[[nodiscard]] size_type capacity() const noexcept {
		return ptr_ == nullptr ? 0 : static_cast<size_type>(capEnd_ - ptr_);
	}

	void reserve(size_type incap) {
		if (incap > capacity()) reallocatePreserve(incap);
	}

	void shrinkToFit() {
		const size_type currentSize = size();
		if (currentSize == 0) {
			deallocate();
			return;
		}
		if (currentSize == capacity()) return;
		reallocateExact(currentSize);
	}

	void clear() noexcept {
		destroyAll();
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void pushBack(const T& inval) {
		if (OA_UNLIKELY(end_ == capEnd_)) growCapacity(addCapacity(size(), 1));
		T* OA_RESTRICT slot = end_;
		if constexpr (oa::IsTriviallyCopyableV<T>) *slot = inval;
		else oa::constructAt(slot, inval);
		end_ = slot + 1;
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void pushBack(T&& inval) {
		if (OA_UNLIKELY(end_ == capEnd_)) growCapacity(addCapacity(size(), 1));
		T* OA_RESTRICT slot = end_;
		if constexpr (oa::IsTriviallyCopyableV<T>) *slot = oa::move(inval);
		else oa::constructAt(slot, oa::move(inval));
		end_ = slot + 1;
	}

	template<typename... Args>
	reference emplaceBack(Args&&... inargs) {
		if (OA_UNLIKELY(end_ == capEnd_)) growCapacity(addCapacity(size(), 1));
		T* const slot = end_;
		oa::constructAt(slot, oa::forward<Args>(inargs)...);
		end_ = slot + 1;
		return *slot;
	}

	// Trivial elements only: one oa::memcpy (inline asm / NT path from memory.h) vs N× pushBack.
	template<typename U = T, typename = oa::EnableIfT<oa::IsTriviallyCopyableV<U>>>
	void append(const T* indata, size_type incount) {
		if (incount == 0) return;
		reserve(addCapacity(size(), incount));
		oa::memcpy(end_, indata, static_cast<oa::Usize>(incount * sizeof(T)));
		end_ += incount;
	}

	void popBack() {
		OA_ASSERT(!empty());
		if constexpr (!oa::IsTriviallyDestructibleV<T>) {
			oa::destroyAt(end_ - 1);
		}
		--end_;
	}

	void resize(size_type incount) {
		const size_type currentSize = size();
		if (incount < currentSize) {
			shrinkToSize(incount);
			return;
		}
		if (incount > currentSize) {
			reserve(incount);
			const size_type add = incount - currentSize;
			if constexpr ((oa::IsArithmeticV<T> || oa::IsEnumV<T> || oa::IsPointerV<T>) &&
				oa::IsTriviallyCopyableV<T>) {
				oa::memzero(end_, static_cast<oa::Usize>(add * sizeof(T)));
				end_ = ptr_ + incount;
			} else {
				while (size() < incount) {
					oa::constructAt(end_);
					++end_;
				}
			}
		}
	}

	void resize(size_type incount, const T& inval) {
		if (incount < size()) {
			shrinkToSize(incount);
			return;
		}
		reserve(incount);
		while (size() < incount) pushBack(inval);
	}

	void swap(Vec& inother) noexcept {
		T* tPtr = ptr_;
		T* tEnd = end_;
		T* tCapEnd = capEnd_;
		ptr_ = inother.ptr_;
		end_ = inother.end_;
		capEnd_ = inother.capEnd_;
		inother.ptr_ = tPtr;
		inother.end_ = tEnd;
		inother.capEnd_ = tCapEnd;
	}

	template<typename It>
	void assign(It infirst, It inlast) {
		clear();
		for (; infirst != inlast; ++infirst) pushBack(*infirst);
	}

	void assign(size_type incount, const T& inval) {
		clear();
		reserve(incount);
		if constexpr (oa::IsTriviallyCopyableV<T>) {
			for (size_type i = 0; i < incount; ++i) ptr_[i] = inval;
			end_ = ptr_ + incount;
		} else {
			for (size_type i = 0; i < incount; ++i) pushBack(inval);
		}
	}

	iterator insert(const_iterator inpos, const T& inval) {
		return insertRebuild(positionIndex(inpos), inval);
	}

	iterator insert(const_iterator inpos, T&& inval) {
		return insertRebuildMove(positionIndex(inpos), oa::move(inval));
	}

	template<typename It>
	iterator insert(const_iterator inpos, It infirst, It inlast) {
		const size_type i = positionIndex(inpos);
		const auto distance = oa::distance(infirst, inlast);
		OA_REQUIRE(distance >= 0);
		Vec tmp;
		const size_type currentSize = size();
		tmp.reserve(addCapacity(currentSize, static_cast<size_type>(distance)));
		for (size_type j = 0; j < i; ++j) tmp.pushBack(ptr_[j]);
		for (; infirst != inlast; ++infirst) tmp.pushBack(*infirst);
		for (size_type j = i; j < currentSize; ++j) tmp.pushBack(ptr_[j]);
		swap(tmp);
		return pointerAt(i);
	}

	iterator erase(const_iterator inpos) {
		const size_type index = positionIndex(inpos);
		OA_REQUIRE(index < size());
		return erase(inpos, inpos + 1);
	}

	iterator erase(const_iterator infirst, const_iterator inlast) {
		const size_type lo = positionIndex(infirst);
		const size_type hi = positionIndex(inlast);
		OA_REQUIRE(lo <= hi);
		if (lo == hi) return pointerAt(lo);
		Vec tmp;
		const size_type currentSize = size();
		size_type newLen = currentSize - (hi - lo);
		tmp.reserve(newLen);
		for (size_type j = 0; j < lo; ++j) tmp.pushBack(oa::move(ptr_[j]));
		for (size_type j = hi; j < currentSize; ++j) tmp.pushBack(oa::move(ptr_[j]));
		swap(tmp);
		return pointerAt(lo);
	}

	iterator begin() noexcept { return ptr_; }

	iterator end() noexcept { return end_; }

	const_iterator begin() const noexcept { return ptr_; }

	const_iterator end() const noexcept { return end_; }

	const_iterator cbegin() const noexcept { return ptr_; }

	const_iterator cend() const noexcept { return end_; }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

private:
	T* ptr_ = nullptr;
	T* end_ = nullptr;
	T* capEnd_ = nullptr;

	// Element storage alignment (not forced to 64B — that penalizes malloc vs std::vector).
	static constexpr size_type storageAlign() noexcept {
		constexpr size_type kMin = sizeof(void*);
		size_type a = alignof(T);
		if (a < kMin) a = kMin;
		return a;
	}

	// Hot path: libc malloc/realloc when C++ guarantees they are sufficient.
	static constexpr bool useMallocRealloc() noexcept {
		return oa::IsTriviallyCopyableV<T>
			&& storageAlign() <= oa::defaultAllocationAlignment();
	}

	static size_type byteCapacityFor(size_type inelemCap) {
		if (inelemCap == 0) return 0;
		if (inelemCap > static_cast<size_type>(-1) / sizeof(T)) {
			oa::allocationFailed(
				oa::AllocationError::SizeOverflow,
				0,
				storageAlign()
			);
		}
		return inelemCap * sizeof(T);
	}

	static size_type addCapacity(size_type inLeft, size_type inRight) {
		if (inRight > static_cast<size_type>(-1) - inLeft) {
			oa::allocationFailed(
				oa::AllocationError::SizeOverflow,
				0,
				storageAlign()
			);
		}
		return inLeft + inRight;
	}

	[[nodiscard]] size_type positionIndex(const_iterator inPosition) const noexcept {
		if (ptr_ == nullptr) {
			OA_REQUIRE(inPosition == nullptr);
			return 0;
		}
		OA_REQUIRE(inPosition >= ptr_ && inPosition <= end_);
		return static_cast<size_type>(inPosition - ptr_);
	}

	[[nodiscard]] iterator pointerAt(size_type inIndex) noexcept {
		OA_REQUIRE(inIndex <= size());
		return ptr_ == nullptr ? nullptr : ptr_ + inIndex;
	}

	void deallocate() noexcept {
		if (!ptr_) {
			end_ = nullptr;
			capEnd_ = nullptr;
			return;
		}
		oa::freeBytes(ptr_, storageAlign());
		ptr_ = nullptr;
		end_ = nullptr;
		capEnd_ = nullptr;
	}

	void destroyAll() noexcept {
		if constexpr (!oa::IsTriviallyDestructibleV<T>) {
			for (T* it = ptr_; it != end_; ++it) oa::destroyAt(it);
		}
		end_ = ptr_;
	}

	void shrinkToSize(size_type innewSize) {
		const size_type currentSize = size();
		OA_ASSERT(innewSize <= currentSize);
		if constexpr (!oa::IsTriviallyDestructibleV<T>) {
			for (size_type i = innewSize; i < currentSize; ++i) oa::destroyAt(ptr_ + i);
		}
		end_ = ptr_ + innewSize;
	}

	void resizeValueInit(size_type incount) {
		reserve(incount);
		const size_type currentSize = size();
		if (currentSize >= incount) return;
		const size_type add = incount - currentSize;
		if constexpr ((oa::IsArithmeticV<T> || oa::IsEnumV<T> || oa::IsPointerV<T>) &&
			oa::IsTriviallyCopyableV<T>) {
			oa::memzero(end_, static_cast<oa::Usize>(add * sizeof(T)));
			end_ = ptr_ + incount;
		} else {
			while (size() < incount) {
				oa::constructAt(end_);
				++end_;
			}
		}
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void growCapacity(size_type inminCap) {
		const size_type currentCapacity = capacity();
		if (inminCap <= currentCapacity) return;
		size_type n = currentCapacity ? currentCapacity : 16;
		while (n < inminCap) {
			if (n > static_cast<size_type>(-1) / 2) {
				n = inminCap;
				break;
			}
			const size_type next = n * 2;
			if (next < n) {
				n = inminCap;
				break;
			}
			n = next;
		}
		reallocatePreserve(n);
	}

	void reallocatePreserve(size_type innewCap) {
		if (innewCap <= capacity()) return;
		const size_type currentSize = size();
		const size_type bytes = byteCapacityFor(innewCap);
		if constexpr (useMallocRealloc()) {
			ptr_ = ptr_ == nullptr
				? static_cast<T*>(oa::detail::allocFreshBytes(bytes, storageAlign()))
				: static_cast<T*>(oa::reallocBytes(ptr_, bytes, storageAlign()));
			end_ = ptr_ + currentSize;
			capEnd_ = ptr_ + innewCap;
			return;
		}
		const size_type align = storageAlign();
		void* const raw = oa::allocBytes(bytes, align);
		T* newPtr = static_cast<T*>(raw);
		if (ptr_) {
			if constexpr (oa::IsTriviallyCopyableV<T>) {
				if (currentSize > 0) oa::memcpy(newPtr, ptr_, currentSize * sizeof(T));
			} else {
				for (size_type i = 0; i < currentSize; ++i) {
					oa::constructAt(newPtr + i, oa::move(ptr_[i]));
					oa::destroyAt(ptr_ + i);
				}
			}
			oa::freeBytes(ptr_, align);
		}
		ptr_ = newPtr;
		end_ = ptr_ + currentSize;
		capEnd_ = ptr_ + innewCap;
	}

	void reallocateExact(size_type innewCap) {
		const size_type currentSize = size();
		const size_type bytes = byteCapacityFor(innewCap);
		if constexpr (useMallocRealloc()) {
			ptr_ = ptr_ == nullptr
				? static_cast<T*>(oa::detail::allocFreshBytes(bytes, storageAlign()))
				: static_cast<T*>(oa::reallocBytes(ptr_, bytes, storageAlign()));
			end_ = ptr_ + currentSize;
			capEnd_ = ptr_ + innewCap;
			return;
		}
		const size_type align = storageAlign();
		void* const raw = oa::allocBytes(bytes, align);
		T* newPtr = static_cast<T*>(raw);
		if (ptr_) {
			if constexpr (oa::IsTriviallyCopyableV<T>) {
				if (currentSize > 0) oa::memcpy(newPtr, ptr_, currentSize * sizeof(T));
			} else {
				for (size_type i = 0; i < currentSize; ++i) {
					oa::constructAt(newPtr + i, oa::move(ptr_[i]));
					oa::destroyAt(ptr_ + i);
				}
			}
			oa::freeBytes(ptr_, align);
		}
		ptr_ = newPtr;
		end_ = ptr_ + currentSize;
		capEnd_ = ptr_ + innewCap;
	}

	iterator insertRebuild(size_type inidx, const T& inval) {
		const size_type currentSize = size();
		OA_REQUIRE(inidx <= currentSize);
		Vec tmp;
		tmp.reserve(addCapacity(currentSize, 1));
		for (size_type j = 0; j < inidx; ++j) tmp.pushBack(ptr_[j]);
		tmp.pushBack(inval);
		for (size_type j = inidx; j < currentSize; ++j) tmp.pushBack(ptr_[j]);
		swap(tmp);
		return pointerAt(inidx);
	}

	iterator insertRebuildMove(size_type inidx, T&& inval) {
		const size_type currentSize = size();
		OA_REQUIRE(inidx <= currentSize);
		Vec tmp;
		tmp.reserve(addCapacity(currentSize, 1));
		for (size_type j = 0; j < inidx; ++j) tmp.pushBack(oa::move(ptr_[j]));
		tmp.pushBack(oa::move(inval));
		for (size_type j = inidx; j < currentSize; ++j) tmp.pushBack(oa::move(ptr_[j]));
		swap(tmp);
		return pointerAt(inidx);
	}
};

// ADL hooks for range-for and std::begin/end (member API is PascalCase).
template<typename T>
typename Vec<T>::iterator begin(Vec<T>& invec) noexcept {
	return invec.begin();
}
template<typename T>
typename Vec<T>::const_iterator begin(const Vec<T>& invec) noexcept {
	return invec.begin();
}
template<typename T>
typename Vec<T>::iterator begin(Vec<T>&& invec) noexcept {
	return invec.begin();
}
template<typename T>
typename Vec<T>::iterator end(Vec<T>& invec) noexcept {
	return invec.end();
}
template<typename T>
typename Vec<T>::const_iterator end(const Vec<T>& invec) noexcept {
	return invec.end();
}
template<typename T>
typename Vec<T>::iterator end(Vec<T>&& invec) noexcept {
	return invec.end();
}

template<typename T>
bool operator==(const Vec<T>& ina, const Vec<T>& inb) {
	if (ina.size() != inb.size()) return false;
	if (ina.data() == inb.data()) return true;
	if constexpr (oa::IsTriviallyCopyableV<T>) {
		if (ina.size() == 0) return true;
		return oa::memEqual(ina.data(), inb.data(), ina.size() * sizeof(T));
	}
	for (oa::Usize i = 0; i < ina.size(); ++i) {
		if (!(ina.data()[i] == inb.data()[i])) return false;
	}
	return true;
}

template<typename T>
bool operator!=(const Vec<T>& ina, const Vec<T>& inb) {
	return !(ina == inb);
}

} // namespace oa
