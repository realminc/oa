#pragma once

// OaVec<T> — Phase 2 (OaStd roadmap)
//
// Header-only template (same as std::vector): full definition must stay in headers
// unless you add explicit instantiations in a .cpp for specific T.
//
// Growable array: 2× growth (same amortized pattern as libstdc++/libc++).
// Trivially copyable T with alignof(T) <= max_align_t: malloc/realloc — in-place
// expansion when libc allows (avoids memcpy+free on most growth steps).
// Larger / over-aligned T: OaStdAllocBytes / OaStdFreeBytes (operator new/delete,
// same as OaStdAllocator) + OaMemcpy (or element-wise move). Trivial path stays
// malloc/realloc (no OaStdAllocBytes — realloc in-place growth).
// Non-trivial T: per-element move + destructor on grow/shrink.
//
// Hot paths: PushBack always_inline (Clang/GCC), OA_UNLIKELY on capacity, trivial T
// stores via restrict-qualified slot pointer (no construct_at). GrowCapacity is
// noinline+cold. Trivial PopBack skips destroy_at. Clear() ends at DestroyAll only.
// Trivial bulk: Append(InData, InCount) uses OaMemcpy; value-init Resize uses OaMemzero
// for arithmetic/enum/pointer T; contiguous iterator ranges use OaMemcpy when allowed.
//
// Source/Public/Oa/Core/Std/Vec.h — include as <Oa/Core/Std/Vec.h> or via <Oa/Core/Std.h> / <Oa/Core/Types.h>.

#include <Oa/Core/Memory.h>
#include <Oa/Core/Std/Allocator.h>
#include <Oa/Core/Std/Iter.h>
#include <Oa/Core/Std/Span.h>
#include <Oa/Core/Std/TypeTraits.h>

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

template<typename T>
class OaVec {
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
	using reverse_iterator = OaStdReverseIterator<iterator>;
	using const_reverse_iterator = OaStdReverseIterator<const_iterator>;

	OaVec() noexcept = default;

	explicit OaVec(size_type InCount) { ResizeValueInit(InCount); }

	OaVec(size_type InCount, const T& InVal) {
		if (InCount == 0) return;
		Reserve(InCount);
		if constexpr (OaStdIsTriviallyCopyableV<T>) {
			for (size_type i = 0; i < InCount; ++i) Ptr_[i] = InVal;
			Size_ = InCount;
		} else {
			for (size_type i = 0; i < InCount; ++i) PushBack(InVal);
		}
	}

	OaVec(std::initializer_list<T> InInit) : OaVec(InInit.begin(), InInit.end()) {}

	// Exclude integral It so OaVec<int>(3, 4) resolves to (size_type, const T&), not iterator pair.
	template<typename It>
	requires (!OaStdIsIntegralV<OaStdRemoveCvrefT<It>>)
	OaVec(It InFirst, It InLast) {
#if __cplusplus >= 202002L
		if constexpr (OaStdIsRandomAccessIteratorV<It>) {
			const auto dist = InLast - InFirst;
			if (dist <= 0) return;
			const size_type n = static_cast<size_type>(dist);
			Reserve(n);
			if constexpr (OaStdIsContiguousIteratorV<It>) {
				if constexpr (OaStdIsSameV<OaStdRemoveConstT<OaStdIterValueT<It>>, T> && OaStdIsTriviallyCopyableV<T>) {
					OaMemcpy(Ptr_, std::to_address(InFirst), static_cast<OaUsize>(n * sizeof(T)));
					Size_ = n;
					return;
				}
			}
			It it = InFirst;
			for (size_type i = 0; i < n; ++i, ++it) {
				if constexpr (OaStdIsTriviallyCopyableV<T>) Ptr_[i] = *it;
				else std::construct_at(Ptr_ + i, *it);
			}
			Size_ = n;
			return;
		}
#endif
		for (; InFirst != InLast; ++InFirst) PushBack(*InFirst);
	}

	OaVec(const std::vector<T>& InVec) : OaVec(InVec.begin(), InVec.end()) {}

	OaVec(std::vector<T>&& InVec) noexcept(OaStdIsNothrowMoveConstructibleV<T>) {
		Reserve(InVec.size());
		for (auto& x : InVec) PushBack(OaStdMove(x));
		InVec.clear();
	}

	OaVec(const OaVec& InOther) requires OaStdIsCopyConstructibleV<T>
		: OaVec(InOther.Begin(), InOther.End()) {}

	OaVec(const OaVec&) requires (!OaStdIsCopyConstructibleV<T>) = delete;

	OaVec(OaVec&& InOther) noexcept : Ptr_(InOther.Ptr_), Size_(InOther.Size_), Cap_(InOther.Cap_) {
		InOther.Ptr_ = nullptr;
		InOther.Size_ = 0;
		InOther.Cap_ = 0;
	}

	OaVec& operator=(const OaVec& InOther) requires OaStdIsCopyConstructibleV<T> {
		if (this == &InOther) return *this;
		OaVec Tmp(InOther);
		Swap(Tmp);
		return *this;
	}

	OaVec& operator=(const OaVec&) requires (!OaStdIsCopyConstructibleV<T>) = delete;

	OaVec& operator=(OaVec&& InOther) noexcept(
		OaStdIsNothrowMoveConstructibleV<T> && OaStdIsNothrowMoveAssignableV<T>) {
		if (this == &InOther) return *this;
		DestroyAll();
		Deallocate();
		Ptr_ = InOther.Ptr_;
		Size_ = InOther.Size_;
		Cap_ = InOther.Cap_;
		InOther.Ptr_ = nullptr;
		InOther.Size_ = 0;
		InOther.Cap_ = 0;
		return *this;
	}

	OaVec& operator=(std::initializer_list<T> InInit) {
		Assign(InInit);
		return *this;
	}

	~OaVec() {
		DestroyAll();
		Deallocate();
	}

	reference operator[](size_type InIdx) {
		OA_ASSERT(InIdx < Size_);
		return Ptr_[InIdx];
	}

	const_reference operator[](size_type InIdx) const {
		OA_ASSERT(InIdx < Size_);
		return Ptr_[InIdx];
	}

	reference At(size_type InIdx) {
		if (InIdx >= Size_) throw std::out_of_range("OaVec::At");
		return Ptr_[InIdx];
	}

	const_reference At(size_type InIdx) const {
		if (InIdx >= Size_) throw std::out_of_range("OaVec::At");
		return Ptr_[InIdx];
	}

	reference Front() {
		OA_ASSERT(Size_ > 0);
		return Ptr_[0];
	}

	const_reference Front() const {
		OA_ASSERT(Size_ > 0);
		return Ptr_[0];
	}

	reference Back() {
		OA_ASSERT(Size_ > 0);
		return Ptr_[Size_ - 1];
	}

	const_reference Back() const {
		OA_ASSERT(Size_ > 0);
		return Ptr_[Size_ - 1];
	}

	pointer Data() noexcept { return Ptr_; }

	const_pointer Data() const noexcept { return Ptr_; }

	[[nodiscard]] bool Empty() const noexcept { return Size_ == 0; }

	[[nodiscard]] size_type Size() const noexcept { return Size_; }

	[[nodiscard]] OaStdSpan<T> Span() noexcept { return OaStdSpan<T>(Data(), Size()); }

	[[nodiscard]] OaStdSpan<const T> Span() const noexcept {
		return OaStdSpan<const T>(Data(), Size());
	}

	[[nodiscard]] size_type Capacity() const noexcept { return Cap_; }

	void Reserve(size_type InCap) {
		if (InCap > Cap_) ReallocatePreserve(InCap);
	}

	void ShrinkToFit() {
		if (Size_ == 0) {
			Deallocate();
			Ptr_ = nullptr;
			Cap_ = 0;
			return;
		}
		if (Size_ == Cap_) return;
		ReallocateExact(Size_);
	}

	void Clear() noexcept {
		DestroyAll();
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void PushBack(const T& InVal) {
		if (OA_UNLIKELY(Size_ == Cap_)) GrowCapacity(Size_ + 1);
		T* OA_RESTRICT p = Ptr_;
		const size_type s = Size_;
		if constexpr (OaStdIsTriviallyCopyableV<T>) p[s] = InVal;
		else std::construct_at(p + s, InVal);
		Size_ = s + 1;
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void PushBack(T&& InVal) {
		if (OA_UNLIKELY(Size_ == Cap_)) GrowCapacity(Size_ + 1);
		T* OA_RESTRICT p = Ptr_;
		const size_type s = Size_;
		if constexpr (OaStdIsTriviallyCopyableV<T>) p[s] = OaStdMove(InVal);
		else std::construct_at(p + s, OaStdMove(InVal));
		Size_ = s + 1;
	}

	template<typename... Args>
	reference EmplaceBack(Args&&... InArgs) {
		if (OA_UNLIKELY(Size_ == Cap_)) GrowCapacity(Size_ + 1);
		std::construct_at(Ptr_ + Size_, OaStdForward<Args>(InArgs)...);
		++Size_;
		return Ptr_[Size_ - 1];
	}

	// Trivial elements only: one OaMemcpy (inline asm / NT path from memory.h) vs N× PushBack.
	template<typename U = T, typename = OaStdEnableIfT<OaStdIsTriviallyCopyableV<U>>>
	void Append(const T* InData, size_type InCount) {
		if (InCount == 0) return;
		Reserve(Size_ + InCount);
		OaMemcpy(Ptr_ + Size_, InData, static_cast<OaUsize>(InCount * sizeof(T)));
		Size_ += InCount;
	}

	void PopBack() {
		OA_ASSERT(Size_ > 0);
		if constexpr (!OaStdIsTriviallyDestructibleV<T>) {
			std::destroy_at(Ptr_ + Size_ - 1);
		}
		--Size_;
	}

	void Resize(size_type InCount) {
		if (InCount < Size_) {
			ShrinkToSize(InCount);
			return;
		}
		if (InCount > Size_) {
			Reserve(InCount);
			const size_type add = InCount - Size_;
			if constexpr ((OaStdIsArithmeticV<T> || OaStdIsEnumV<T> || OaStdIsPointerV<T>) &&
				OaStdIsTriviallyCopyableV<T>) {
				OaMemzero(Ptr_ + Size_, static_cast<OaUsize>(add * sizeof(T)));
				Size_ = InCount;
			} else {
				while (Size_ < InCount) {
					std::construct_at(Ptr_ + Size_);
					++Size_;
				}
			}
		}
	}

	void Resize(size_type InCount, const T& InVal) {
		if (InCount < Size_) {
			ShrinkToSize(InCount);
			return;
		}
		Reserve(InCount);
		while (Size_ < InCount) PushBack(InVal);
	}

	void Swap(OaVec& InOther) noexcept {
		T* tPtr = Ptr_;
		size_type tSize = Size_;
		size_type tCap = Cap_;
		Ptr_ = InOther.Ptr_;
		Size_ = InOther.Size_;
		Cap_ = InOther.Cap_;
		InOther.Ptr_ = tPtr;
		InOther.Size_ = tSize;
		InOther.Cap_ = tCap;
	}

	template<typename It>
	void Assign(It InFirst, It InLast) {
		Clear();
		for (; InFirst != InLast; ++InFirst) PushBack(*InFirst);
	}

	void Assign(size_type InCount, const T& InVal) {
		Clear();
		Reserve(InCount);
		if constexpr (OaStdIsTriviallyCopyableV<T>) {
			for (size_type i = 0; i < InCount; ++i) Ptr_[i] = InVal;
			Size_ = InCount;
		} else {
			for (size_type i = 0; i < InCount; ++i) PushBack(InVal);
		}
	}

	void Assign(std::initializer_list<T> InInit) { Assign(InInit.begin(), InInit.end()); }

	iterator Insert(const_iterator InPos, const T& InVal) {
		return InsertRebuild(static_cast<size_type>(InPos - CBegin()), InVal);
	}

	iterator Insert(const_iterator InPos, T&& InVal) {
		return InsertRebuildMove(static_cast<size_type>(InPos - CBegin()), OaStdMove(InVal));
	}

	template<typename It>
	iterator Insert(const_iterator InPos, It InFirst, It InLast) {
		size_type i = static_cast<size_type>(InPos - CBegin());
		OaVec Tmp;
		Tmp.Reserve(Size_ + static_cast<size_type>(OaStdDistance(InFirst, InLast)));
		for (size_type j = 0; j < i; ++j) Tmp.PushBack(Ptr_[j]);
		for (; InFirst != InLast; ++InFirst) Tmp.PushBack(*InFirst);
		for (size_type j = i; j < Size_; ++j) Tmp.PushBack(Ptr_[j]);
		Swap(Tmp);
		return Begin() + static_cast<difference_type>(i);
	}

	iterator Insert(const_iterator InPos, std::initializer_list<T> InInit) {
		return Insert(InPos, InInit.begin(), InInit.end());
	}

	iterator Erase(const_iterator InPos) {
		return Erase(InPos, InPos + 1);
	}

	iterator Erase(const_iterator InFirst, const_iterator InLast) {
		size_type lo = static_cast<size_type>(InFirst - CBegin());
		size_type hi = static_cast<size_type>(InLast - CBegin());
		OA_ASSERT(hi <= Size_ && lo <= hi);
		if (lo == hi) return Begin() + static_cast<difference_type>(lo);
		OaVec Tmp;
		size_type newLen = Size_ - (hi - lo);
		Tmp.Reserve(newLen);
		for (size_type j = 0; j < lo; ++j) Tmp.PushBack(OaStdMove(Ptr_[j]));
		for (size_type j = hi; j < Size_; ++j) Tmp.PushBack(OaStdMove(Ptr_[j]));
		Swap(Tmp);
		return Begin() + static_cast<difference_type>(lo);
	}

	iterator Begin() noexcept { return Ptr_; }

	iterator End() noexcept { return Ptr_ + Size_; }

	const_iterator Begin() const noexcept { return Ptr_; }

	const_iterator End() const noexcept { return Ptr_ + Size_; }

	const_iterator CBegin() const noexcept { return Ptr_; }

	const_iterator CEnd() const noexcept { return Ptr_ + Size_; }

	reverse_iterator RBegin() noexcept { return reverse_iterator(End()); }

	reverse_iterator REnd() noexcept { return reverse_iterator(Begin()); }

	const_reverse_iterator RBegin() const noexcept { return const_reverse_iterator(End()); }

	const_reverse_iterator REnd() const noexcept { return const_reverse_iterator(Begin()); }

	// std::vector-compatible aliases (legacy call sites, std algorithm interop).
	[[nodiscard]] size_type size() const noexcept { return Size(); }
	[[nodiscard]] pointer data() noexcept { return Data(); }
	[[nodiscard]] const_pointer data() const noexcept { return Data(); }
	[[nodiscard]] bool empty() const noexcept { return Empty(); }
	[[nodiscard]] size_type capacity() const noexcept { return Capacity(); }

	void reserve(size_type InCap) { Reserve(InCap); }
	void clear() { Clear(); }
	void push_back(const T& InVal) { PushBack(InVal); }
	void push_back(T&& InVal) { PushBack(OaStdMove(InVal)); }
	void pop_back() { PopBack(); }
	void resize(size_type InCount) { Resize(InCount); }
	void resize(size_type InCount, const T& InVal) { Resize(InCount, InVal); }

	template<typename It>
	void assign(It InFirst, It InLast) {
		Assign(InFirst, InLast);
	}
	void assign(size_type InCount, const T& InVal) { Assign(InCount, InVal); }
	void assign(std::initializer_list<T> InInit) { Assign(InInit); }

	iterator begin() noexcept { return Begin(); }
	iterator end() noexcept { return End(); }
	const_iterator begin() const noexcept { return Begin(); }
	const_iterator end() const noexcept { return End(); }
	const_iterator cbegin() const noexcept { return CBegin(); }
	const_iterator cend() const noexcept { return CEnd(); }
	reverse_iterator rbegin() noexcept { return RBegin(); }
	reverse_iterator rend() noexcept { return REnd(); }
	const_reverse_iterator rbegin() const noexcept { return RBegin(); }
	const_reverse_iterator rend() const noexcept { return REnd(); }
	const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(CEnd()); }
	const_reverse_iterator crend() const noexcept { return const_reverse_iterator(CBegin()); }

	reference at(size_type InIdx) { return At(InIdx); }
	const_reference at(size_type InIdx) const { return At(InIdx); }

	reference front() noexcept { return Front(); }
	const_reference front() const noexcept { return Front(); }
	reference back() noexcept { return Back(); }
	const_reference back() const noexcept { return Back(); }

	iterator insert(const_iterator InPos, const T& InVal) { return Insert(InPos, InVal); }
	iterator insert(const_iterator InPos, T&& InVal) { return Insert(InPos, OaStdMove(InVal)); }

	template<typename It>
	iterator insert(const_iterator InPos, It InFirst, It InLast) {
		return Insert(InPos, InFirst, InLast);
	}

	iterator insert(const_iterator InPos, std::initializer_list<T> InInit) {
		return Insert(InPos, InInit);
	}

	iterator erase(const_iterator InPos) { return Erase(InPos); }
	iterator erase(const_iterator InFirst, const_iterator InLast) { return Erase(InFirst, InLast); }

private:
	T* Ptr_ = nullptr;
	size_type Size_ = 0;
	size_type Cap_ = 0;

	// Element storage alignment (not forced to 64B — that penalizes malloc vs std::vector).
	static constexpr size_type StorageAlign() noexcept {
		constexpr size_type kMin = sizeof(void*);
		size_type a = alignof(T);
		if (a < kMin) a = kMin;
		return a;
	}

	// Hot path: libc malloc/realloc when C++ guarantees they are sufficient.
	static constexpr bool UseMallocRealloc() noexcept {
		return OaStdIsTriviallyCopyableV<T> && StorageAlign() <= alignof(std::max_align_t);
	}

	static size_type ByteCapacityFor(size_type InElemCap) {
		if (InElemCap == 0) return 0;
		if (InElemCap > std::numeric_limits<size_type>::max() / sizeof(T)) {
			throw std::length_error("OaVec capacity overflow");
		}
		return InElemCap * sizeof(T);
	}

	void Deallocate() noexcept {
		if (!Ptr_) {
			Cap_ = 0;
			return;
		}
		if constexpr (UseMallocRealloc()) std::free(Ptr_);
		else OaStdFreeBytes(Ptr_, StorageAlign());
		Ptr_ = nullptr;
		Cap_ = 0;
	}

	void DestroyAll() noexcept {
		if constexpr (!OaStdIsTriviallyDestructibleV<T>) {
			for (size_type i = 0; i < Size_; ++i) std::destroy_at(Ptr_ + i);
		}
		Size_ = 0;
	}

	void ShrinkToSize(size_type InNewSize) {
		OA_ASSERT(InNewSize <= Size_);
		if constexpr (!OaStdIsTriviallyDestructibleV<T>) {
			for (size_type i = InNewSize; i < Size_; ++i) std::destroy_at(Ptr_ + i);
		}
		Size_ = InNewSize;
	}

	void ResizeValueInit(size_type InCount) {
		Reserve(InCount);
		if (Size_ >= InCount) return;
		const size_type add = InCount - Size_;
		if constexpr ((OaStdIsArithmeticV<T> || OaStdIsEnumV<T> || OaStdIsPointerV<T>) &&
			OaStdIsTriviallyCopyableV<T>) {
			OaMemzero(Ptr_ + Size_, static_cast<OaUsize>(add * sizeof(T)));
			Size_ = InCount;
		} else {
			while (Size_ < InCount) {
				std::construct_at(Ptr_ + Size_);
				++Size_;
			}
		}
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((noinline)) __attribute__((cold))
#endif
	void GrowCapacity(size_type InMinCap) {
		if (InMinCap <= Cap_) return;
		size_type n = Cap_ ? Cap_ : 16;
		while (n < InMinCap) {
			if (n > std::numeric_limits<size_type>::max() / 2) {
				n = InMinCap;
				break;
			}
			const size_type next = n * 2;
			if (next < n) {
				n = InMinCap;
				break;
			}
			n = next;
		}
		ReallocatePreserve(n);
	}

	void ReallocatePreserve(size_type InNewCap) {
		if (InNewCap <= Cap_) return;
		const size_type bytes = ByteCapacityFor(InNewCap);
		if constexpr (UseMallocRealloc()) {
			void* p = std::realloc(static_cast<void*>(Ptr_), bytes);
			if (p || bytes == 0) {
				Ptr_ = static_cast<T*>(p);
				Cap_ = InNewCap;
				return;
			}
			// realloc failed — new buffer + copy + free (rare: OOM pressure).
			void* raw = std::malloc(bytes);
			OA_ASSERT(raw != nullptr);
			T* newPtr = static_cast<T*>(raw);
			if (Ptr_ && Size_ > 0) OaMemcpy(newPtr, Ptr_, Size_ * sizeof(T));
			std::free(Ptr_);
			Ptr_ = newPtr;
			Cap_ = InNewCap;
			return;
		}
		const size_type align = StorageAlign();
		void* const raw = OaStdAllocBytes(bytes, align);
		T* newPtr = static_cast<T*>(raw);
		if (Ptr_) {
			if constexpr (OaStdIsTriviallyCopyableV<T>) {
				if (Size_ > 0) OaMemcpy(newPtr, Ptr_, Size_ * sizeof(T));
			} else {
				for (size_type i = 0; i < Size_; ++i) {
					new (&newPtr[i]) T(OaStdMove(Ptr_[i]));
					Ptr_[i].~T();
				}
			}
			OaStdFreeBytes(Ptr_, align);
		}
		Ptr_ = newPtr;
		Cap_ = InNewCap;
	}

	void ReallocateExact(size_type InNewCap) {
		const size_type bytes = ByteCapacityFor(InNewCap);
		if constexpr (UseMallocRealloc()) {
			void* p = std::realloc(static_cast<void*>(Ptr_), bytes);
			if (p || bytes == 0) {
				Ptr_ = static_cast<T*>(p);
				Cap_ = InNewCap;
				return;
			}
			void* raw = std::malloc(bytes);
			OA_ASSERT(raw != nullptr);
			T* newPtr = static_cast<T*>(raw);
			if (Ptr_ && Size_ > 0) OaMemcpy(newPtr, Ptr_, Size_ * sizeof(T));
			std::free(Ptr_);
			Ptr_ = newPtr;
			Cap_ = InNewCap;
			return;
		}
		const size_type align = StorageAlign();
		void* const raw = OaStdAllocBytes(bytes, align);
		T* newPtr = static_cast<T*>(raw);
		if (Ptr_) {
			if constexpr (OaStdIsTriviallyCopyableV<T>) {
				if (Size_ > 0) OaMemcpy(newPtr, Ptr_, Size_ * sizeof(T));
			} else {
				for (size_type i = 0; i < Size_; ++i) {
					new (&newPtr[i]) T(OaStdMove(Ptr_[i]));
					Ptr_[i].~T();
				}
			}
			OaStdFreeBytes(Ptr_, align);
		}
		Ptr_ = newPtr;
		Cap_ = InNewCap;
	}

	iterator InsertRebuild(size_type InIdx, const T& InVal) {
		OA_ASSERT(InIdx <= Size_);
		OaVec Tmp;
		Tmp.Reserve(Size_ + 1);
		for (size_type j = 0; j < InIdx; ++j) Tmp.PushBack(Ptr_[j]);
		Tmp.PushBack(InVal);
		for (size_type j = InIdx; j < Size_; ++j) Tmp.PushBack(Ptr_[j]);
		Swap(Tmp);
		return Begin() + static_cast<difference_type>(InIdx);
	}

	iterator InsertRebuildMove(size_type InIdx, T&& InVal) {
		OA_ASSERT(InIdx <= Size_);
		OaVec Tmp;
		Tmp.Reserve(Size_ + 1);
		for (size_type j = 0; j < InIdx; ++j) Tmp.PushBack(OaStdMove(Ptr_[j]));
		Tmp.PushBack(OaStdMove(InVal));
		for (size_type j = InIdx; j < Size_; ++j) Tmp.PushBack(OaStdMove(Ptr_[j]));
		Swap(Tmp);
		return Begin() + static_cast<difference_type>(InIdx);
	}
};

// ADL hooks for range-for and std::begin/end (member API is PascalCase).
template<typename T>
typename OaVec<T>::iterator begin(OaVec<T>& InVec) noexcept {
	return InVec.Begin();
}
template<typename T>
typename OaVec<T>::const_iterator begin(const OaVec<T>& InVec) noexcept {
	return InVec.Begin();
}
template<typename T>
typename OaVec<T>::iterator begin(OaVec<T>&& InVec) noexcept {
	return InVec.Begin();
}
template<typename T>
typename OaVec<T>::iterator end(OaVec<T>& InVec) noexcept {
	return InVec.End();
}
template<typename T>
typename OaVec<T>::const_iterator end(const OaVec<T>& InVec) noexcept {
	return InVec.End();
}
template<typename T>
typename OaVec<T>::iterator end(OaVec<T>&& InVec) noexcept {
	return InVec.End();
}

template<typename T>
bool operator==(const OaVec<T>& InA, const OaVec<T>& InB) {
	if (InA.Size() != InB.Size()) return false;
	if (InA.Data() == InB.Data()) return true;
	if constexpr (OaStdIsTriviallyCopyableV<T>) {
		if (InA.Size() == 0) return true;
		return OaMemEqual(InA.Data(), InB.Data(), InA.Size() * sizeof(T));
	}
	for (OaUsize i = 0; i < InA.Size(); ++i) {
		if (!(InA.Data()[i] == InB.Data()[i])) return false;
	}
	return true;
}

template<typename T>
bool operator!=(const OaVec<T>& InA, const OaVec<T>& InB) {
	return !(InA == InB);
}
