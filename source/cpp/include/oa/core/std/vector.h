#pragma once

// Vector<T> — phase 2 (OA standard library roadmap)
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
// for arithmetic/enum T; contiguous iterator ranges use oa::memcpy when allowed.
//
// Include as <oa/core/std/vector.h> or through <oa/core/std.h> or <oa/core/types.h>.

#include <oa/core/std/memory.h>
#include <oa/core/std/assert.h>
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
class Vector {
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

	Vector() noexcept = default;

	explicit Vector(size_type incount) {
		ConstructionGuard guard(this);
		resizeValueInit(incount);
		guard.release();
	}

	Vector(size_type incount, const T& inval) {
		ConstructionGuard guard(this);
		if (incount == 0) {
			guard.release();
			return;
		}
		reserve(incount);
		if constexpr (oa::isTriviallyCopyableV<T>) {
			for (size_type i = 0; i < incount; ++i) ptr_[i] = inval;
			end_ = ptr_ + incount;
		} else {
			for (size_type i = 0; i < incount; ++i) pushBack(inval);
		}
		guard.release();
	}

	Vector(std::initializer_list<T> inInit) : Vector(inInit.begin(), inInit.end()) {}

	// Exclude integral It so Vector<int>(3, 4) resolves to (size_type, const T&), not iterator pair.
	template<typename It>
	requires (!oa::isIntegralV<oa::RemoveCvrefT<It>>
		&& requires(It inIt) { ++inIt; *inIt; })
	Vector(It infirst, It inlast) {
		ConstructionGuard guard(this);
#if __cplusplus >= 202002L
		if constexpr (oa::isRandomAccessIteratorV<It>) {
			const auto dist = inlast - infirst;
			if (dist <= 0) {
				guard.release();
				return;
			}
			const size_type n = static_cast<size_type>(dist);
			reserve(n);
			if constexpr (oa::isContiguousIteratorV<It>) {
				if constexpr (oa::isSameV<oa::RemoveConstT<oa::IterValueT<It>>, T> && oa::isTriviallyCopyableV<T>) {
					oa::memcpy(ptr_, oa::toAddress(infirst), n * sizeof(T));
					end_ = ptr_ + n;
					guard.release();
					return;
				}
			}
			It it = infirst;
			for (size_type i = 0; i < n; ++i, ++it) {
				if constexpr (oa::isTriviallyCopyableV<T>) {
					ptr_[i] = *it;
				} else {
					oa::constructAt(end_, *it);
					++end_;
				}
			}
			if constexpr (oa::isTriviallyCopyableV<T>) {
				end_ = ptr_ + n;
			}
			guard.release();
			return;
		}
#endif
		for (; infirst != inlast; ++infirst) pushBack(*infirst);
		guard.release();
	}

	Vector(const Vector& inother) requires oa::isCopyConstructibleV<T>
		: Vector(inother.begin(), inother.end()) {}

	Vector(const Vector&) requires (!oa::isCopyConstructibleV<T>) = delete;

	Vector(Vector&& inother) noexcept
		: ptr_(inother.ptr_), end_(inother.end_), capEnd_(inother.capEnd_) {
		inother.ptr_ = nullptr;
		inother.end_ = nullptr;
		inother.capEnd_ = nullptr;
	}

	Vector& operator=(const Vector& inother) requires oa::isCopyConstructibleV<T> {
		if (this == &inother) return *this;
		Vector tmp(inother);
		swap(tmp);
		return *this;
	}

	Vector& operator=(const Vector&) requires (!oa::isCopyConstructibleV<T>) = delete;

	Vector& operator=(Vector&& inother) noexcept(
		oa::isNothrowMoveConstructibleV<T> && oa::isNothrowMoveAssignableV<T>) {
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

	Vector& operator=(std::initializer_list<T> inInit) {
		assign(inInit.begin(), inInit.end());
		return *this;
	}

	~Vector() {
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
		OA_REQUIRE(!empty());
		return ptr_[0];
	}

	const_reference front() const {
		OA_REQUIRE(!empty());
		return ptr_[0];
	}

	reference back() {
		OA_REQUIRE(!empty());
		return end_[-1];
	}

	const_reference back() const {
		OA_REQUIRE(!empty());
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
		if (OA_UNLIKELY(end_ == capEnd_)) {
			// The input may name an element in this Vector. Stabilize it before a
			// realloc can invalidate the reference; keeping growth inline also
			// lets the optimizer scalar-replace short-lived trivial Vectors.
			T stable(inval);
			growCapacity(addCapacity(size(), 1));
			T* OA_RESTRICT slot = end_;
			if constexpr (oa::isTriviallyCopyableV<T>) *slot = stable;
			else oa::constructAt(slot, stable);
			end_ = slot + 1;
			return;
		}
		T* OA_RESTRICT slot = end_;
		if constexpr (oa::isTriviallyCopyableV<T>) *slot = inval;
		else oa::constructAt(slot, inval);
		end_ = slot + 1;
	}

#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	void pushBack(T&& inval) {
		if (OA_UNLIKELY(end_ == capEnd_)) {
			T stable(oa::move(inval));
			growCapacity(addCapacity(size(), 1));
			T* OA_RESTRICT slot = end_;
			if constexpr (oa::isTriviallyCopyableV<T>) *slot = oa::move(stable);
			else oa::constructAt(slot, oa::move(stable));
			end_ = slot + 1;
			return;
		}
		T* OA_RESTRICT slot = end_;
		if constexpr (oa::isTriviallyCopyableV<T>) *slot = oa::move(inval);
		else oa::constructAt(slot, oa::move(inval));
		end_ = slot + 1;
	}

	template<typename... Args>
	reference emplaceBack(Args&&... inargs) {
		if (OA_UNLIKELY(end_ == capEnd_)) {
			return growAndEmplaceBack(oa::forward<Args>(inargs)...);
		}
		T* const slot = end_;
		oa::constructAt(slot, oa::forward<Args>(inargs)...);
		end_ = slot + 1;
		return *slot;
	}

	// Trivial elements only: one oa::memcpy versus N pushBack calls.
	template<typename U = T, typename = oa::EnableIfT<oa::isTriviallyCopyableV<U>>>
	void append(const T* indata, size_type incount) {
		if (incount == 0) return;
		OA_REQUIRE(indata != nullptr);
		const size_type sourceBytes = byteCapacityFor(incount);
		const size_type sourceAddress = reinterpret_cast<size_type>(indata);
		OA_REQUIRE(sourceBytes <= static_cast<size_type>(-1) - sourceAddress);
		const size_type sourceEndAddress = sourceAddress + sourceBytes;

		size_type sourceIndex = InvalidElementIndex;
		if (ptr_ != nullptr) {
			const size_type beginAddress = reinterpret_cast<size_type>(ptr_);
			const size_type liveEndAddress = reinterpret_cast<size_type>(end_);
			const size_type storageEndAddress = reinterpret_cast<size_type>(capEnd_);
			const bool intersectsStorage = sourceAddress < storageEndAddress
				&& sourceEndAddress > beginAddress;
			if (intersectsStorage) {
				const bool isWhollyLive = sourceAddress >= beginAddress
					&& sourceAddress < liveEndAddress
					&& sourceEndAddress <= liveEndAddress
					&& ((sourceAddress - beginAddress) % sizeof(T)) == 0;
				OA_REQUIRE(isWhollyLive);
				sourceIndex = (sourceAddress - beginAddress) / sizeof(T);
			}
		}
		reserve(addCapacity(size(), incount));
		if (sourceIndex != InvalidElementIndex) {
			indata = ptr_ + sourceIndex;
		}
		oa::memcpy(end_, indata, sourceBytes);
		end_ += incount;
	}

	void popBack() {
		OA_REQUIRE(!empty());
		if constexpr (!oa::isTriviallyDestructibleV<T>) {
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
			if constexpr ((oa::isArithmeticV<T> || oa::isEnumV<T>) &&
				oa::isTriviallyCopyableV<T>) {
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
		if (incount > capacity() && elementIndex(&inval) != InvalidElementIndex) {
			T stable(inval);
			reserve(incount);
			while (size() < incount) pushBack(stable);
			return;
		}
		reserve(incount);
		while (size() < incount) pushBack(inval);
	}

	void swap(Vector& inother) noexcept {
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
	requires (!oa::isIntegralV<oa::RemoveCvrefT<It>>
		&& requires(It inIt) { ++inIt; *inIt; })
	void assign(It infirst, It inlast) {
		Vector tmp(infirst, inlast);
		swap(tmp);
	}

	void assign(size_type incount, const T& inval) {
		Vector tmp(incount, inval);
		swap(tmp);
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
		Vector tmp;
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
		Vector tmp;
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
		return oa::isTriviallyCopyableV<T>
			&& storageAlign() <= oa::defaultAllocationAlignment();
	}

	static constexpr size_type InvalidElementIndex = static_cast<size_type>(-1);

	[[nodiscard]] size_type elementIndex(const T* inValue) const noexcept {
		if (ptr_ == nullptr || inValue == nullptr) {
			return InvalidElementIndex;
		}
		const size_type address = reinterpret_cast<size_type>(inValue);
		const size_type beginAddress = reinterpret_cast<size_type>(ptr_);
		const size_type endAddress = reinterpret_cast<size_type>(end_);
		if (address < beginAddress || address >= endAddress
			|| ((address - beginAddress) % sizeof(T)) != 0) {
			return InvalidElementIndex;
		}
		return (address - beginAddress) / sizeof(T);
	}

	class ReallocationGuard final {
	public:
		ReallocationGuard(T* inData, size_type inAlignment) noexcept
			: data_(inData), alignment_(inAlignment) {}

		~ReallocationGuard() {
			if (data_ == nullptr) {
				return;
			}
			if constexpr (!oa::isTriviallyDestructibleV<T>) {
				for (size_type index = 0; index < constructed_; ++index) {
					oa::destroyAt(data_ + index);
				}
			}
			oa::freeBytes(data_, alignment_);
		}

		ReallocationGuard(const ReallocationGuard&) = delete;
		ReallocationGuard& operator=(const ReallocationGuard&) = delete;

		void markConstructed() noexcept { ++constructed_; }
		void release() noexcept { data_ = nullptr; }

	private:
		T* data_ = nullptr;
		size_type alignment_ = 0;
		size_type constructed_ = 0;
	};

	class ConstructionGuard final {
	public:
		explicit ConstructionGuard(Vector* inOwner) noexcept : owner_(inOwner) {}

		~ConstructionGuard() {
			if (owner_ == nullptr) {
				return;
			}
			owner_->destroyAll();
			owner_->deallocate();
		}

		ConstructionGuard(const ConstructionGuard&) = delete;
		ConstructionGuard& operator=(const ConstructionGuard&) = delete;

		void release() noexcept { owner_ = nullptr; }

	private:
		Vector* owner_;
	};

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
		const size_type address = reinterpret_cast<size_type>(inPosition);
		const size_type beginAddress = reinterpret_cast<size_type>(ptr_);
		const size_type endAddress = reinterpret_cast<size_type>(end_);
		OA_REQUIRE(address >= beginAddress && address <= endAddress);
		OA_REQUIRE(((address - beginAddress) % sizeof(T)) == 0);
		return (address - beginAddress) / sizeof(T);
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
		if constexpr (!oa::isTriviallyDestructibleV<T>) {
			for (T* it = ptr_; it != end_; ++it) oa::destroyAt(it);
		}
		end_ = ptr_;
	}

	void shrinkToSize(size_type innewSize) {
		const size_type currentSize = size();
		OA_ASSERT(innewSize <= currentSize);
		if constexpr (!oa::isTriviallyDestructibleV<T>) {
			for (size_type i = innewSize; i < currentSize; ++i) oa::destroyAt(ptr_ + i);
		}
		end_ = ptr_ + innewSize;
	}

	void resizeValueInit(size_type incount) {
		reserve(incount);
		const size_type currentSize = size();
		if (currentSize >= incount) return;
		const size_type add = incount - currentSize;
		if constexpr ((oa::isArithmeticV<T> || oa::isEnumV<T>) &&
			oa::isTriviallyCopyableV<T>) {
			oa::memzero(end_, static_cast<oa::Usize>(add * sizeof(T)));
			end_ = ptr_ + incount;
		} else {
			while (size() < incount) {
				oa::constructAt(end_);
				++end_;
			}
		}
	}

	template<typename... Args>
#if defined(__clang__) || defined(__GNUC__)
	__attribute__((noinline)) __attribute__((cold))
#endif
	reference growAndEmplaceBack(Args&&... inArgs) {
		// Constructor arguments may borrow an existing element. Materialize
		// before reallocation so a moved allocation cannot invalidate them.
		T stable(oa::forward<Args>(inArgs)...);
		growCapacity(addCapacity(size(), 1));
		T* const slot = end_;
		if constexpr (oa::isCopyConstructibleV<T>
			&& !oa::isNothrowMoveConstructibleV<T>) {
			oa::constructAt(slot, stable);
		} else {
			oa::constructAt(slot, oa::move(stable));
		}
		end_ = slot + 1;
		return *slot;
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
		ReallocationGuard guard(newPtr, align);
		if (ptr_) {
			if constexpr (oa::isTriviallyCopyableV<T>) {
				if (currentSize > 0) oa::memcpy(newPtr, ptr_, currentSize * sizeof(T));
			} else {
				for (size_type i = 0; i < currentSize; ++i) {
					if constexpr (oa::isCopyConstructibleV<T>
						&& !oa::isNothrowMoveConstructibleV<T>) {
						oa::constructAt(newPtr + i, ptr_[i]);
					} else {
						oa::constructAt(newPtr + i, oa::move(ptr_[i]));
					}
					guard.markConstructed();
				}
				for (size_type i = 0; i < currentSize; ++i) oa::destroyAt(ptr_ + i);
			}
			oa::freeBytes(ptr_, align);
		}
		guard.release();
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
		ReallocationGuard guard(newPtr, align);
		if (ptr_) {
			if constexpr (oa::isTriviallyCopyableV<T>) {
				if (currentSize > 0) oa::memcpy(newPtr, ptr_, currentSize * sizeof(T));
			} else {
				for (size_type i = 0; i < currentSize; ++i) {
					if constexpr (oa::isCopyConstructibleV<T>
						&& !oa::isNothrowMoveConstructibleV<T>) {
						oa::constructAt(newPtr + i, ptr_[i]);
					} else {
						oa::constructAt(newPtr + i, oa::move(ptr_[i]));
					}
					guard.markConstructed();
				}
				for (size_type i = 0; i < currentSize; ++i) oa::destroyAt(ptr_ + i);
			}
			oa::freeBytes(ptr_, align);
		}
		guard.release();
		ptr_ = newPtr;
		end_ = ptr_ + currentSize;
		capEnd_ = ptr_ + innewCap;
	}

	iterator insertRebuild(size_type inidx, const T& inval) {
		const size_type currentSize = size();
		OA_REQUIRE(inidx <= currentSize);
		Vector tmp;
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
		Vector tmp;
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
typename Vector<T>::iterator begin(Vector<T>& invec) noexcept {
	return invec.begin();
}
template<typename T>
typename Vector<T>::const_iterator begin(const Vector<T>& invec) noexcept {
	return invec.begin();
}
template<typename T>
typename Vector<T>::iterator begin(Vector<T>&& invec) noexcept {
	return invec.begin();
}
template<typename T>
typename Vector<T>::iterator end(Vector<T>& invec) noexcept {
	return invec.end();
}
template<typename T>
typename Vector<T>::const_iterator end(const Vector<T>& invec) noexcept {
	return invec.end();
}
template<typename T>
typename Vector<T>::iterator end(Vector<T>&& invec) noexcept {
	return invec.end();
}

template<typename T>
bool operator==(const Vector<T>& ina, const Vector<T>& inb) {
	if (ina.size() != inb.size()) return false;
	for (oa::Usize i = 0; i < ina.size(); ++i) {
		if (!(ina.data()[i] == inb.data()[i])) return false;
	}
	return true;
}

template<typename T>
bool operator!=(const Vector<T>& ina, const Vector<T>& inb) {
	return !(ina == inb);
}

} // namespace oa
