#pragma once

// OaStd — raw bytes + stateless allocator (libc++/libstdc++ parity for operator new/delete).
//
// OOM / errors
// - OaStdAllocBytes: uses ::operator new (and std::align_val_t when InAlignment exceeds
//   OaStdDefaultNewAlignment()). On failure throws std::bad_alloc (never returns nullptr).
// - OaStdAllocator::Allocate(InCount): overflow throws std::bad_array_new_length(); else
//   delegates to OaStdAllocBytes(InCount * sizeof(T), alignof(T)).
// OaVec (vec.h) keeps a separate fast path: malloc/realloc when trivial T and alignment
// ≤ max_align_t (realloc in-place growth). Over-aligned / non-trivial vec storage uses
// OaStdAllocBytes / OaStdFreeBytes so control blocks and vec agree on operator-new freeing.
//
// Why std::numeric_limits / std::type_traits: C++17 allocator requirements (max_size,
// rebind, propagate_on_* typedefs) and overflow checks — intentional parity with
// std::allocator_traits consumers.

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

static constexpr std::size_t OaStdDefaultNewAlignment() noexcept {
#if defined(__STDCPP_DEFAULT_NEW_ALIGNMENT__)
	return static_cast<std::size_t>(__STDCPP_DEFAULT_NEW_ALIGNMENT__);
#else
	return alignof(std::max_align_t);
#endif
}

// Raw bytes (containers, tools). InAlignment must be a power of two, ≥ 1.
// Pair with OaStdFreeBytes(ptr, InAlignment); the one-arg OaStdFreeBytes is only valid
// when the block was allocated with InAlignment ≤ OaStdDefaultNewAlignment().
[[nodiscard]] inline void* OaStdAllocBytes(std::size_t InBytes, std::size_t InAlignment) {
	std::size_t const al = InAlignment == 0 ? 1U : InAlignment;
	if (al <= OaStdDefaultNewAlignment()) {
		return ::operator new(InBytes);
	}
	return ::operator new(InBytes, std::align_val_t{al});
}

inline void OaStdFreeBytes(void* InPtr, std::size_t InAlignment) noexcept {
	if (!InPtr) {
		return;
	}
	std::size_t const al = InAlignment == 0 ? 1U : InAlignment;
	if (al <= OaStdDefaultNewAlignment()) {
		::operator delete(InPtr);
	} else {
		::operator delete(InPtr, std::align_val_t{al});
	}
}

inline void OaStdFreeBytes(void* InPtr) noexcept {
	OaStdFreeBytes(InPtr, OaStdDefaultNewAlignment());
}

template<typename T>
class OaStdAllocator {
public:
	// C++17 Allocator requirements: typedef names and `std::allocator_traits` hook names
	// (`allocate`, `deallocate`, `max_size`, …) are fixed by the standard. Primary API is
	// PascalCase (`Allocate`, `Deallocate`, `MaxSize`, …); snake_case members forward so
	// `std::vector<T, OaStdAllocator<T>>` and `allocator_traits` keep working.
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using void_pointer = void*;
	using const_void_pointer = const void*;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using propagate_on_container_copy_assignment = std::false_type;
	using propagate_on_container_move_assignment = std::true_type;
	using propagate_on_container_swap = std::false_type;
	using is_always_equal = std::true_type;

	constexpr OaStdAllocator() noexcept = default;

	template<typename U>
	constexpr OaStdAllocator(const OaStdAllocator<U>& InOther) noexcept {
		(void)InOther;
	}

	[[nodiscard]] OaStdAllocator SelectOnContainerCopyConstruction() const noexcept {
		return *this;
	}
	OaStdAllocator select_on_container_copy_construction() const noexcept {
		return SelectOnContainerCopyConstruction();
	}

	[[nodiscard]] pointer Allocate(size_type InCount) {
		if (InCount > std::numeric_limits<size_type>::max() / sizeof(T)) {
			throw std::bad_array_new_length();
		}
		const size_type bytes = InCount * sizeof(T);
		void* const raw = OaStdAllocBytes(bytes, alignof(T));
		return static_cast<pointer>(raw);
	}
	[[nodiscard]] pointer allocate(size_type InCount) { return Allocate(InCount); }

	// Deprecated in the standard; retained for allocator_traits / legacy containers.
	[[nodiscard]] pointer Allocate(size_type InCount, const_void_pointer /*InHint*/) {
		return Allocate(InCount);
	}
	[[nodiscard]] pointer allocate(size_type InCount, const_void_pointer InHint) {
		return Allocate(InCount, InHint);
	}

	void Deallocate(pointer InPtr, size_type InCount) noexcept {
		(void)InCount;
		OaStdFreeBytes(InPtr, alignof(T));
	}
	void deallocate(pointer InPtr, size_type InCount) noexcept {
		Deallocate(InPtr, InCount);
	}

	[[nodiscard]] constexpr size_type MaxSize() const noexcept {
		return std::numeric_limits<size_type>::max() / sizeof(T);
	}
	[[nodiscard]] constexpr size_type max_size() const noexcept { return MaxSize(); }

	template<typename U>
	struct rebind {
		using other = OaStdAllocator<U>;
	};

	friend constexpr bool operator==(
		const OaStdAllocator& InLhs, const OaStdAllocator& InRhs) noexcept {
		(void)InLhs;
		(void)InRhs;
		return true;
	}
	friend constexpr bool operator!=(
		const OaStdAllocator& InLhs, const OaStdAllocator& InRhs) noexcept {
		(void)InLhs;
		(void)InRhs;
		return false;
	}
};

template<typename T, typename U>
constexpr bool operator==(
	const OaStdAllocator<T>& InLhs, const OaStdAllocator<U>& InRhs) noexcept {
	(void)InLhs;
	(void)InRhs;
	return true;
}

template<typename T, typename U>
constexpr bool operator!=(
	const OaStdAllocator<T>& InLhs, const OaStdAllocator<U>& InRhs) noexcept {
	(void)InLhs;
	(void)InRhs;
	return false;
}
