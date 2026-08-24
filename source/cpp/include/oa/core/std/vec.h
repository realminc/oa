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
// Hot paths: pushBack always_inline (Clang/GCC), OA_UNLIKELY on capacity, trivial T
// stores via restrict-qualified slot pointer (no construct_at). growCapacity is
// noinline+cold. Trivial popBack skips destroy_at. clear() ends at destroyAll only.
// Trivial bulk: append(indata, incount) uses oa::memcpy; value-init resize uses oa::memzero
// for arithmetic/enum/pointer T; contiguous iterator ranges use oa::memcpy when allowed.
//
// Include as <oa/core/std/vec.h> or through <oa/core/std.h> or <oa/core/types.h>.

#include <oa/core/memory.h>
#include <oa/core/std/allocator.h>
#include <oa/core/std/iter.h>
#include <oa/core/std/span.h>
#include <oa/core/std/typeTraits.h>

#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef OA_ASSERT
#include <cassert>
#define OA_ASSERT(expr) assert(expr)
#endif

#ifndef OA_LIKELY
#define OA_LIKELY(x) (x)
#define OA_UNLIKELY(x) (x)
#endif

#ifndef OA_RESTRICT
#define OA_RESTRICT
#endif

namespace oa {

template<typename T>
class Vec {
public:
	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
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
			size_ = incount;
		} else {
			for (size_type i = 0; i < incount; ++i) pushBack(inval);
		}
	}

	Vec(std::initializer_list<T> ininit) : Vec(ininit.begin(), ininit.end()) {}

	// Exclude integral It so Vec<int>(3, 4) resolves to (size_type, const T&), not iterator pair.
	template<typename It>
	requires (!oa::IsIntegralV<oa::RemoveCvrefT<It>>)
	Vec(It infirst, It inlast) {
#if __cplusplus >= 202002L
		if constexpr (oa::IsRandomAccessIteratorV<It>) {
			const auto dist = inlast - infirst;
			if (dist <= 0) return;
			const size_type n = static_cast<size_type>(dist);
			reserve(n);
			if constexpr (oa::IsContiguousIteratorV<It>) {
				if constexpr (oa::IsSameV<oa::RemoveConstT<oa::IterValueT<It>>, T> && oa::IsTriviallyCopyableV<T>) {
					oa::memcpy(ptr_, std::to_address(infirst), static_cast<oa::Usize>(n * sizeof(T)));
					size_ = n;
					return;
				}
			}
			It it = infirst;
			for (size_type i = 0; i < n; ++i, ++it) {
				if constexpr (oa::IsTriviallyCopyableV<T>) ptr_[i] = *it;
				else std::construct_at(ptr_ + i, *it);
			}
			size_ = n;
			return;
		}
#endif
		for (; infirst != inlast; ++infirst) pushBack(*infirst);
	}

	Vec(const std::vector<T>& invec) : Vec(invec.begin(), invec.end()) {}

	Vec(std::vector<T>&& invec) noexcept(oa::IsNothrowMoveConstructibleV<T>) {
		reserve(invec.size());
		for (auto& x : invec) pushBack(oa::move(x));
		invec.clear();
	}

	Vec(const Vec& inother) requires oa::IsCopyConstructibleV<T>
		: Vec(inother.begin(), inother.end()) {}

	Vec(const Vec&) requires (!oa::IsCopyConstructibleV<T>) = delete;

	Vec(Vec&& inother) noexcept : ptr_(inother.ptr_), size_(inother.size_), cap_(inother.cap_) {
		inother.ptr_ = nullptr;
		inother.size_ = 0;
		inother.cap_ = 0;
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
		size_ = inother.size_;
		cap_ = inother.cap_;
		inother.ptr_ = nullptr;
		inother.size_ = 0;
		inother.cap_ = 0;
		return *this;
	}

	Vec& operator=(std::initializer_list<T> ininit) {
		assign(ininit);
		return *this;
	}

	~Vec() {
		destroyAll();
		deallocate();
	}

	reference operator[](size_type inidx) {
		OA_ASSERT(inidx < size_);
		return ptr_[inidx];
	}

	const_reference operator[](size_type inidx) const {
		OA_ASSERT(inidx < size_);
		return ptr_[inidx];
	}

	reference at(size_type inidx) {
		if (inidx >= size_) throw std::out_of_range("Vec::at");
		return ptr_[inidx];
	}

	const_reference at(size_type inidx) const {
		if (inidx >= size_) throw std::out_of_range("Vec::at");
		return ptr_[inidx];
	}

	reference front() {
		OA_ASSERT(size_ > 0);
		return ptr_[0];
	}

	const_reference front() const {
		OA_ASSERT(size_ > 0);
		return ptr_[0];
	}

	reference back() {
		OA_ASSERT(size_ > 0);
		return ptr_[size_ - 1];
	}

	const_reference back() const {
		OA_ASSERT(size_ > 0);
		return ptr_[size_ - 1];
	}

	pointer data() noexcept { return ptr_; }

	const_pointer data() const noexcept { return ptr_; }

	[[nodiscard]] bool empty() const noexcept { return size_ == 0; }

	[[nodiscard]] size_type size() const noexcept { return size_; }

	[[nodiscard]] oa::Span<T> span() noexcept { return oa::Span<T>(data(), size()); }

	[[nodiscard]] oa::Span<const T> span() const noexcept {
		return oa::Span<const T>(data(), size());
	}

	[[nodiscard]] size_type capacity() const noexcept { return cap_; }

	void reserve(size_type incap) {
		if (incap > cap_) reallocatePreserve(incap);
	}

	void shrinkToFit() {
		if (size_ == 0) {
			deallocate();
			ptr_ = nullptr;
			cap_ = 0;
			return;
		}
		if (size_ == cap_) return;
		reallocateExact(size_);
	}

	void clear() noexcept {
		destroyAll();
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void pushBack(const T& inval) {
		if (OA_UNLIKELY(size_ == cap_)) growCapacity(size_ + 1);
		T* OA_RESTRICT p = ptr_;
		const size_type s = size_;
		if constexpr (oa::IsTriviallyCopyableV<T>) p[s] = inval;
		else std::construct_at(p + s, inval);
		size_ = s + 1;
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void pushBack(T&& inval) {
		if (OA_UNLIKELY(size_ == cap_)) growCapacity(size_ + 1);
		T* OA_RESTRICT p = ptr_;
		const size_type s = size_;
		if constexpr (oa::IsTriviallyCopyableV<T>) p[s] = oa::move(inval);
		else std::construct_at(p + s, oa::move(inval));
		size_ = s + 1;
	}

	template<typename... Args>
	reference emplaceBack(Args&&... inargs) {
		if (OA_UNLIKELY(size_ == cap_)) growCapacity(size_ + 1);
		std::construct_at(ptr_ + size_, oa::forward<Args>(inargs)...);
		++size_;
		return ptr_[size_ - 1];
	}

	// Trivial elements only: one oa::memcpy (inline asm / NT path from memory.h) vs N× pushBack.
	template<typename U = T, typename = oa::EnableIfT<oa::IsTriviallyCopyableV<U>>>
	void append(const T* indata, size_type incount) {
		if (incount == 0) return;
		reserve(size_ + incount);
		oa::memcpy(ptr_ + size_, indata, static_cast<oa::Usize>(incount * sizeof(T)));
		size_ += incount;
	}

	void popBack() {
		OA_ASSERT(size_ > 0);
		if constexpr (!oa::IsTriviallyDestructibleV<T>) {
			std::destroy_at(ptr_ + size_ - 1);
		}
		--size_;
	}

	void resize(size_type incount) {
		if (incount < size_) {
			shrinkToSize(incount);
			return;
		}
		if (incount > size_) {
			reserve(incount);
			const size_type add = incount - size_;
			if constexpr ((oa::IsArithmeticV<T> || oa::IsEnumV<T> || oa::IsPointerV<T>) &&
				oa::IsTriviallyCopyableV<T>) {
				oa::memzero(ptr_ + size_, static_cast<oa::Usize>(add * sizeof(T)));
				size_ = incount;
			} else {
				while (size_ < incount) {
					std::construct_at(ptr_ + size_);
					++size_;
				}
			}
		}
	}

	void resize(size_type incount, const T& inval) {
		if (incount < size_) {
			shrinkToSize(incount);
			return;
		}
		reserve(incount);
		while (size_ < incount) pushBack(inval);
	}

	void swap(Vec& inother) noexcept {
		T* tPtr = ptr_;
		size_type tSize = size_;
		size_type tCap = cap_;
		ptr_ = inother.ptr_;
		size_ = inother.size_;
		cap_ = inother.cap_;
		inother.ptr_ = tPtr;
		inother.size_ = tSize;
		inother.cap_ = tCap;
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
			size_ = incount;
		} else {
			for (size_type i = 0; i < incount; ++i) pushBack(inval);
		}
	}

	void assign(std::initializer_list<T> ininit) { assign(ininit.begin(), ininit.end()); }

	iterator insert(const_iterator inpos, const T& inval) {
		return insertRebuild(static_cast<size_type>(inpos - cbegin()), inval);
	}

	iterator insert(const_iterator inpos, T&& inval) {
		return insertRebuildMove(static_cast<size_type>(inpos - cbegin()), oa::move(inval));
	}

	template<typename It>
	iterator insert(const_iterator inpos, It infirst, It inlast) {
		size_type i = static_cast<size_type>(inpos - cbegin());
		Vec tmp;
		tmp.reserve(size_ + static_cast<size_type>(oa::distance(infirst, inlast)));
		for (size_type j = 0; j < i; ++j) tmp.pushBack(ptr_[j]);
		for (; infirst != inlast; ++infirst) tmp.pushBack(*infirst);
		for (size_type j = i; j < size_; ++j) tmp.pushBack(ptr_[j]);
		swap(tmp);
		return begin() + static_cast<difference_type>(i);
	}

	iterator insert(const_iterator inpos, std::initializer_list<T> ininit) {
		return insert(inpos, ininit.begin(), ininit.end());
	}

	iterator erase(const_iterator inpos) {
		return erase(inpos, inpos + 1);
	}

	iterator erase(const_iterator infirst, const_iterator inlast) {
		size_type lo = static_cast<size_type>(infirst - cbegin());
		size_type hi = static_cast<size_type>(inlast - cbegin());
		OA_ASSERT(hi <= size_ && lo <= hi);
		if (lo == hi) return begin() + static_cast<difference_type>(lo);
		Vec tmp;
		size_type newLen = size_ - (hi - lo);
		tmp.reserve(newLen);
		for (size_type j = 0; j < lo; ++j) tmp.pushBack(oa::move(ptr_[j]));
		for (size_type j = hi; j < size_; ++j) tmp.pushBack(oa::move(ptr_[j]));
		swap(tmp);
		return begin() + static_cast<difference_type>(lo);
	}

	iterator begin() noexcept { return ptr_; }

	iterator end() noexcept { return ptr_ + size_; }

	const_iterator begin() const noexcept { return ptr_; }

	const_iterator end() const noexcept { return ptr_ + size_; }

	const_iterator cbegin() const noexcept { return ptr_; }

	const_iterator cend() const noexcept { return ptr_ + size_; }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

private:
	T* ptr_ = nullptr;
	size_type size_ = 0;
	size_type cap_ = 0;

	// Element storage alignment (not forced to 64B — that penalizes malloc vs std::vector).
	static constexpr size_type storageAlign() noexcept {
		constexpr size_type kMin = sizeof(void*);
		size_type a = alignof(T);
		if (a < kMin) a = kMin;
		return a;
	}

	// Hot path: libc malloc/realloc when C++ guarantees they are sufficient.
	static constexpr bool useMallocRealloc() noexcept {
		return oa::IsTriviallyCopyableV<T> && storageAlign() <= alignof(std::max_align_t);
	}

	static size_type byteCapacityFor(size_type inelemCap) {
		if (inelemCap == 0) return 0;
		if (inelemCap > std::numeric_limits<size_type>::max() / sizeof(T)) {
			throw std::length_error("Vec capacity overflow");
		}
		return inelemCap * sizeof(T);
	}

	void deallocate() noexcept {
		if (!ptr_) {
			cap_ = 0;
			return;
		}
		if constexpr (useMallocRealloc()) std::free(ptr_);
		else oa::freeBytes(ptr_, storageAlign());
		ptr_ = nullptr;
		cap_ = 0;
	}

	void destroyAll() noexcept {
		if constexpr (!oa::IsTriviallyDestructibleV<T>) {
			for (size_type i = 0; i < size_; ++i) std::destroy_at(ptr_ + i);
		}
		size_ = 0;
	}

	void shrinkToSize(size_type innewSize) {
		OA_ASSERT(innewSize <= size_);
		if constexpr (!oa::IsTriviallyDestructibleV<T>) {
			for (size_type i = innewSize; i < size_; ++i) std::destroy_at(ptr_ + i);
		}
		size_ = innewSize;
	}

	void resizeValueInit(size_type incount) {
		reserve(incount);
		if (size_ >= incount) return;
		const size_type add = incount - size_;
		if constexpr ((oa::IsArithmeticV<T> || oa::IsEnumV<T> || oa::IsPointerV<T>) &&
			oa::IsTriviallyCopyableV<T>) {
			oa::memzero(ptr_ + size_, static_cast<oa::Usize>(add * sizeof(T)));
			size_ = incount;
		} else {
			while (size_ < incount) {
				std::construct_at(ptr_ + size_);
				++size_;
			}
		}
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((noinline)) __attribute__((cold))
#endif
	void growCapacity(size_type inminCap) {
		if (inminCap <= cap_) return;
		size_type n = cap_ ? cap_ : 16;
		while (n < inminCap) {
			if (n > std::numeric_limits<size_type>::max() / 2) {
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
		if (innewCap <= cap_) return;
		const size_type bytes = byteCapacityFor(innewCap);
		if constexpr (useMallocRealloc()) {
			void* p = std::realloc(static_cast<void*>(ptr_), bytes);
			if (p || bytes == 0) {
				ptr_ = static_cast<T*>(p);
				cap_ = innewCap;
				return;
			}
			// realloc failed — new buffer + copy + free (rare: OOM pressure).
			void* raw = std::malloc(bytes);
			OA_ASSERT(raw != nullptr);
			T* newPtr = static_cast<T*>(raw);
			if (ptr_ && size_ > 0) oa::memcpy(newPtr, ptr_, size_ * sizeof(T));
			std::free(ptr_);
			ptr_ = newPtr;
			cap_ = innewCap;
			return;
		}
		const size_type align = storageAlign();
		void* const raw = oa::allocBytes(bytes, align);
		T* newPtr = static_cast<T*>(raw);
		if (ptr_) {
			if constexpr (oa::IsTriviallyCopyableV<T>) {
				if (size_ > 0) oa::memcpy(newPtr, ptr_, size_ * sizeof(T));
			} else {
				for (size_type i = 0; i < size_; ++i) {
					new (&newPtr[i]) T(oa::move(ptr_[i]));
					ptr_[i].~T();
				}
			}
			oa::freeBytes(ptr_, align);
		}
		ptr_ = newPtr;
		cap_ = innewCap;
	}

	void reallocateExact(size_type innewCap) {
		const size_type bytes = byteCapacityFor(innewCap);
		if constexpr (useMallocRealloc()) {
			void* p = std::realloc(static_cast<void*>(ptr_), bytes);
			if (p || bytes == 0) {
				ptr_ = static_cast<T*>(p);
				cap_ = innewCap;
				return;
			}
			void* raw = std::malloc(bytes);
			OA_ASSERT(raw != nullptr);
			T* newPtr = static_cast<T*>(raw);
			if (ptr_ && size_ > 0) oa::memcpy(newPtr, ptr_, size_ * sizeof(T));
			std::free(ptr_);
			ptr_ = newPtr;
			cap_ = innewCap;
			return;
		}
		const size_type align = storageAlign();
		void* const raw = oa::allocBytes(bytes, align);
		T* newPtr = static_cast<T*>(raw);
		if (ptr_) {
			if constexpr (oa::IsTriviallyCopyableV<T>) {
				if (size_ > 0) oa::memcpy(newPtr, ptr_, size_ * sizeof(T));
			} else {
				for (size_type i = 0; i < size_; ++i) {
					new (&newPtr[i]) T(oa::move(ptr_[i]));
					ptr_[i].~T();
				}
			}
			oa::freeBytes(ptr_, align);
		}
		ptr_ = newPtr;
		cap_ = innewCap;
	}

	iterator insertRebuild(size_type inidx, const T& inval) {
		OA_ASSERT(inidx <= size_);
		Vec tmp;
		tmp.reserve(size_ + 1);
		for (size_type j = 0; j < inidx; ++j) tmp.pushBack(ptr_[j]);
		tmp.pushBack(inval);
		for (size_type j = inidx; j < size_; ++j) tmp.pushBack(ptr_[j]);
		swap(tmp);
		return begin() + static_cast<difference_type>(inidx);
	}

	iterator insertRebuildMove(size_type inidx, T&& inval) {
		OA_ASSERT(inidx <= size_);
		Vec tmp;
		tmp.reserve(size_ + 1);
		for (size_type j = 0; j < inidx; ++j) tmp.pushBack(oa::move(ptr_[j]));
		tmp.pushBack(oa::move(inval));
		for (size_type j = inidx; j < size_; ++j) tmp.pushBack(oa::move(ptr_[j]));
		swap(tmp);
		return begin() + static_cast<difference_type>(inidx);
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
