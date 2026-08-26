#pragma once

// OA host allocation boundary. Fallible operations report an explicit error;
// convenience allocation terminates through one documented fatal path. No
// exception or hosted C++ standard-library contract crosses this header.

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

#include <oa/core/std/typeTraits.h>

namespace oa {

enum class AllocationError : oa::U8 {
	None = 0,
	InvalidAlignment,
	SizeOverflow,
	OutOfMemory,
};

[[nodiscard]] constexpr const char* allocationErrorName(
	AllocationError inError
) noexcept {
	switch (inError) {
		case AllocationError::None:             return "none";
		case AllocationError::InvalidAlignment: return "invalid alignment";
		case AllocationError::SizeOverflow:     return "size overflow";
		case AllocationError::OutOfMemory:      return "out of memory";
		default:                                return "unknown allocation error";
	}
}

struct AllocationResult {
	void* data = nullptr;
	AllocationError error = AllocationError::None;

	[[nodiscard]] constexpr bool isOk() const noexcept {
		return error == AllocationError::None;
	}
	[[nodiscard]] constexpr bool isError() const noexcept { return !isOk(); }
	explicit constexpr operator bool() const noexcept { return isOk(); }
};

[[nodiscard]] constexpr oa::Usize defaultAllocationAlignment() noexcept {
#if defined(__STDCPP_DEFAULT_NEW_ALIGNMENT__)
	return static_cast<oa::Usize>(__STDCPP_DEFAULT_NEW_ALIGNMENT__);
#elif defined(__BIGGEST_ALIGNMENT__)
	return static_cast<oa::Usize>(__BIGGEST_ALIGNMENT__);
#else
	return alignof(void*);
#endif
}

[[nodiscard]] AllocationResult tryAllocBytes(
	oa::Usize inBytes,
	oa::Usize inAlignment = defaultAllocationAlignment()
) noexcept;

[[nodiscard]] AllocationResult tryAllocArray(
	oa::Usize inCount,
	oa::Usize inElementSize,
	oa::Usize inAlignment
) noexcept;

// Resize default-aligned storage while preserving its bytes. This is the
// explicit reallocation seam used by trivially copyable containers; callers
// must not pass over-aligned storage.
[[nodiscard]] AllocationResult tryReallocBytes(
	void* inPtr,
	oa::Usize inBytes,
	oa::Usize inAlignment = defaultAllocationAlignment()
) noexcept;

[[noreturn]] void allocationFailed(
	AllocationError inError,
	oa::Usize inBytes,
	oa::Usize inAlignment
) noexcept;

[[nodiscard]] inline void* allocBytes(
	oa::Usize inBytes,
	oa::Usize inAlignment = defaultAllocationAlignment()
) {
	const AllocationResult result = tryAllocBytes(inBytes, inAlignment);
	if (result.isError()) {
		allocationFailed(result.error, inBytes, inAlignment);
	}
	return result.data;
}

namespace detail {

// Keep this fresh-allocation call visible to optimizers that must prove a
// container's element storage disjoint from the container object. The ordinary
// allocation API remains inline; only this private optimizer seam is outlined.
[[nodiscard]]
#if defined(__clang__) || defined(__GNUC__)
__attribute__((malloc, alloc_size(1), alloc_align(2), noinline))
#endif
inline void* allocFreshBytes(
	oa::Usize inBytes,
	oa::Usize inAlignment = defaultAllocationAlignment()
) {
	return oa::allocBytes(inBytes, inAlignment);
}

} // namespace detail

[[nodiscard]] inline void* reallocBytes(
	void* inPtr,
	oa::Usize inBytes,
	oa::Usize inAlignment = defaultAllocationAlignment()
) {
	const AllocationResult result = tryReallocBytes(inPtr, inBytes, inAlignment);
	if (result.isError()) {
		allocationFailed(result.error, inBytes, inAlignment);
	}
	return result.data;
}

void freeBytes(
	void* inPtr,
	oa::Usize inAlignment = defaultAllocationAlignment()
) noexcept;

template<typename T>
class Allocator {
public:
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using void_pointer = void*;
	using const_void_pointer = const void*;
	using size_type = oa::Usize;
	using difference_type = oa::Isize;
	using propagate_on_container_copy_assignment = oa::FalseType;
	using propagate_on_container_move_assignment = oa::TrueType;
	using propagate_on_container_swap = oa::FalseType;
	using is_always_equal = oa::TrueType;

	constexpr Allocator() noexcept = default;

	template<typename U>
	constexpr Allocator(const Allocator<U>& inOther) noexcept {
		(void)inOther;
	}

	[[nodiscard]] Allocator selectOnContainerCopyConstruction() const noexcept {
		return *this;
	}
	[[nodiscard]] Allocator select_on_container_copy_construction() const noexcept {
		return selectOnContainerCopyConstruction();
	}

	[[nodiscard]] pointer allocate(size_type inCount) {
		const AllocationResult result = tryAllocArray(inCount, sizeof(T), alignof(T));
		if (result.isError()) {
			const size_type bytes = inCount <= maxSize() ? inCount * sizeof(T) : 0;
			allocationFailed(result.error, bytes, alignof(T));
		}
		return static_cast<pointer>(result.data);
	}

	[[nodiscard]] pointer allocate(
		size_type inCount,
		const_void_pointer /*inHint*/
	) {
		return allocate(inCount);
	}

	void deallocate(pointer inPtr, size_type inCount) noexcept {
		(void)inCount;
		freeBytes(inPtr, alignof(T));
	}

	[[nodiscard]] static constexpr size_type maxSize() noexcept {
		return static_cast<size_type>(-1) / sizeof(T);
	}
	[[nodiscard]] static constexpr size_type max_size() noexcept {
		return maxSize();
	}

	template<typename U>
	struct rebind {
		using other = Allocator<U>;
	};

	friend constexpr bool operator==(
		const Allocator& inLhs,
		const Allocator& inRhs
	) noexcept {
		(void)inLhs;
		(void)inRhs;
		return true;
	}

	friend constexpr bool operator!=(
		const Allocator& inLhs,
		const Allocator& inRhs
	) noexcept {
		(void)inLhs;
		(void)inRhs;
		return false;
	}
};

template<typename T, typename U>
constexpr bool operator==(
	const Allocator<T>& inLhs,
	const Allocator<U>& inRhs
) noexcept {
	(void)inLhs;
	(void)inRhs;
	return true;
}

template<typename T, typename U>
constexpr bool operator!=(
	const Allocator<T>& inLhs,
	const Allocator<U>& inRhs
) noexcept {
	(void)inLhs;
	(void)inRhs;
	return false;
}

} // namespace oa
