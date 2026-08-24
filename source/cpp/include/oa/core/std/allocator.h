#pragma once

// OA standard library — raw bytes + stateless allocator (libc++/libstdc++ parity for operator new/delete).
//
// OOM / errors
// - allocBytes: uses ::operator new (and std::align_val_t when inAlignment exceeds
//   defaultNewAlignment()). On failure throws std::bad_alloc (never returns nullptr).
// - Allocator::allocate(inCount): overflow throws std::bad_array_new_length(); else
//   delegates to allocBytes(inCount * sizeof(T), alignof(T)).
// oa::Vec (vec.h) keeps a separate fast path: malloc/realloc when trivial T and alignment
// ≤ max_align_t (realloc in-place growth). Over-aligned / non-trivial vec storage uses
// allocBytes / freeBytes so control blocks and vec agree on operator-new freeing.
//
// Why std::numeric_limits / std::type_traits: C++17 allocator requirements (max_size,
// rebind, propagate_on_* typedefs) and overflow checks — intentional parity with
// std::allocator_traits consumers.

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>

namespace oa {

static constexpr std::size_t defaultNewAlignment() noexcept {
#if defined(__STDCPP_DEFAULT_NEW_ALIGNMENT__)
	return static_cast<std::size_t>(__STDCPP_DEFAULT_NEW_ALIGNMENT__);
#else
	return alignof(std::max_align_t);
#endif
}

// Raw bytes (containers, tools). inAlignment must be a power of two, ≥ 1.
// Pair with freeBytes(ptr, inAlignment); the one-arg freeBytes is only valid
// when the block was allocated with inAlignment ≤ defaultNewAlignment().
[[nodiscard]] inline void* allocBytes(std::size_t inBytes, std::size_t inAlignment) {
	std::size_t const al = inAlignment == 0 ? 1U : inAlignment;
	if (al <= defaultNewAlignment()) {
		return ::operator new(inBytes);
	}
	return ::operator new(inBytes, std::align_val_t{al});
}

inline void freeBytes(void* inPtr, std::size_t inAlignment) noexcept {
	if (!inPtr) {
		return;
	}
	std::size_t const al = inAlignment == 0 ? 1U : inAlignment;
	if (al <= defaultNewAlignment()) {
		::operator delete(inPtr);
	} else {
		::operator delete(inPtr, std::align_val_t{al});
	}
}

inline void freeBytes(void* inPtr) noexcept {
	freeBytes(inPtr, defaultNewAlignment());
}

template<typename T>
class Allocator {
public:
	// C++17 Allocator requirements: typedef names and `std::allocator_traits` hook names
	// (`allocate`, `deallocate`, `max_size`, …) are fixed by the standard. Primary API is
	// PascalCase (`allocate`, `deallocate`, `maxSize`, …); snake_case members forward so
	// `std::vector<T, Allocator<T>>` and `allocator_traits` keep working.
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

	constexpr Allocator() noexcept = default;

	template<typename U>
	constexpr Allocator(const Allocator<U>& inOther) noexcept {
		(void)inOther;
	}

	[[nodiscard]] Allocator selectOnContainerCopyConstruction() const noexcept {
		return *this;
	}
	Allocator select_on_container_copy_construction() const noexcept {
		return selectOnContainerCopyConstruction();
	}

	[[nodiscard]] pointer allocate(size_type inCount) {
		if (inCount > std::numeric_limits<size_type>::max() / sizeof(T)) {
			throw std::bad_array_new_length();
		}
		const size_type bytes = inCount * sizeof(T);
		void* const raw = allocBytes(bytes, alignof(T));
		return static_cast<pointer>(raw);
	}
	// Deprecated in the standard; retained for allocator_traits / legacy containers.
	[[nodiscard]] pointer allocate(size_type inCount, const_void_pointer /*inHint*/) {
		return allocate(inCount);
	}
	void deallocate(pointer inPtr, size_type inCount) noexcept {
		(void)inCount;
		freeBytes(inPtr, alignof(T));
	}
	[[nodiscard]] constexpr size_type maxSize() const noexcept {
		return std::numeric_limits<size_type>::max() / sizeof(T);
	}
	[[nodiscard]] constexpr size_type max_size() const noexcept { return maxSize(); }

	template<typename U>
	struct rebind {
		using other = Allocator<U>;
	};

	friend constexpr bool operator==(
		const Allocator& inLhs, const Allocator& inRhs) noexcept {
		(void)inLhs;
		(void)inRhs;
		return true;
	}
	friend constexpr bool operator!=(
		const Allocator& inLhs, const Allocator& inRhs) noexcept {
		(void)inLhs;
		(void)inRhs;
		return false;
	}
};

template<typename T, typename U>
constexpr bool operator==(
	const Allocator<T>& inLhs, const Allocator<U>& inRhs) noexcept {
	(void)inLhs;
	(void)inRhs;
	return true;
}

template<typename T, typename U>
constexpr bool operator!=(
	const Allocator<T>& inLhs, const Allocator<U>& inRhs) noexcept {
	(void)inLhs;
	(void)inRhs;
	return false;
}

} // namespace oa
