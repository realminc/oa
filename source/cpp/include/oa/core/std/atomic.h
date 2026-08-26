#pragma once

// OA atomic scalar operations. Compiler intrinsics provide the memory-model
// primitive directly; no hosted C++ standard-library type or ABI crosses this
// boundary.

#include <oa/core/std/typeTraits.h>

namespace oa {

enum class MemoryOrder {
	Relaxed,
	Consume,
	Acquire,
	Release,
	AcquireRelease,
	Sequential,
};

namespace atomicDetail {

[[nodiscard]] constexpr int builtinOrder(MemoryOrder inOrder) noexcept {
	switch (inOrder) {
		case MemoryOrder::Relaxed:        return __ATOMIC_RELAXED;
		case MemoryOrder::Consume:        return __ATOMIC_CONSUME;
		case MemoryOrder::Acquire:        return __ATOMIC_ACQUIRE;
		case MemoryOrder::Release:        return __ATOMIC_RELEASE;
		case MemoryOrder::AcquireRelease: return __ATOMIC_ACQ_REL;
		case MemoryOrder::Sequential:     return __ATOMIC_SEQ_CST;
		default:                          return __ATOMIC_SEQ_CST;
	}
}

[[nodiscard]] constexpr int failureOrder(MemoryOrder inOrder) noexcept {
	switch (inOrder) {
		case MemoryOrder::Release:        return __ATOMIC_RELAXED;
		case MemoryOrder::AcquireRelease: return __ATOMIC_ACQUIRE;
		default:                          return builtinOrder(inOrder);
	}
}

} // namespace atomicDetail

inline void atomicThreadFence(
	MemoryOrder inOrder = MemoryOrder::Sequential
) noexcept {
	__atomic_thread_fence(atomicDetail::builtinOrder(inOrder));
}

template<typename T>
class Atomic {
public:
	using ValueType = T;

	static_assert(oa::IsTriviallyCopyableV<T>,
		"oa::Atomic requires a trivially copyable scalar type");

	Atomic() noexcept = default;
	constexpr Atomic(T inDesired) noexcept : value_(inDesired) {}
	Atomic(const Atomic&) = delete;
	Atomic& operator=(const Atomic&) = delete;

	[[nodiscard]] T load(
		MemoryOrder inOrder = MemoryOrder::Sequential
	) const noexcept {
		return __atomic_load_n(&value_, atomicDetail::builtinOrder(inOrder));
	}

	void store(
		T inDesired,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept {
		__atomic_store_n(&value_, inDesired, atomicDetail::builtinOrder(inOrder));
	}

	T exchange(
		T inDesired,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept {
		return __atomic_exchange_n(
			&value_, inDesired, atomicDetail::builtinOrder(inOrder));
	}

	bool compareExchangeStrong(
		T& inExpected,
		T inDesired,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept {
		return __atomic_compare_exchange_n(
			&value_, &inExpected, inDesired, false,
			atomicDetail::builtinOrder(inOrder),
			atomicDetail::failureOrder(inOrder));
	}

	bool compareExchangeWeak(
		T& inExpected,
		T inDesired,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept {
		return __atomic_compare_exchange_n(
			&value_, &inExpected, inDesired, true,
			atomicDetail::builtinOrder(inOrder),
			atomicDetail::failureOrder(inOrder));
	}

	T fetchAdd(
		T inArg,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept requires(oa::IsIntegralV<T>) {
		return __atomic_fetch_add(
			&value_, inArg, atomicDetail::builtinOrder(inOrder));
	}

	T fetchSub(
		T inArg,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept requires(oa::IsIntegralV<T>) {
		return __atomic_fetch_sub(
			&value_, inArg, atomicDetail::builtinOrder(inOrder));
	}

	T fetchOr(
		T inArg,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept requires(oa::IsIntegralV<T>) {
		return __atomic_fetch_or(
			&value_, inArg, atomicDetail::builtinOrder(inOrder));
	}

	T fetchAnd(
		T inArg,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept requires(oa::IsIntegralV<T>) {
		return __atomic_fetch_and(
			&value_, inArg, atomicDetail::builtinOrder(inOrder));
	}

	T fetchXor(
		T inArg,
		MemoryOrder inOrder = MemoryOrder::Sequential
	) noexcept requires(oa::IsIntegralV<T>) {
		return __atomic_fetch_xor(
			&value_, inArg, atomicDetail::builtinOrder(inOrder));
	}

	operator T() const noexcept { return load(); }
	T operator=(T inDesired) noexcept {
		store(inDesired);
		return inDesired;
	}
	T operator++() noexcept requires(oa::IsIntegralV<T>) {
		return fetchAdd(static_cast<T>(1)) + static_cast<T>(1);
	}
	T operator++(int) noexcept requires(oa::IsIntegralV<T>) {
		return fetchAdd(static_cast<T>(1));
	}
	T operator--() noexcept requires(oa::IsIntegralV<T>) {
		return fetchSub(static_cast<T>(1)) - static_cast<T>(1);
	}
	T operator--(int) noexcept requires(oa::IsIntegralV<T>) {
		return fetchSub(static_cast<T>(1));
	}
	T operator+=(T inArg) noexcept requires(oa::IsIntegralV<T>) {
		return fetchAdd(inArg) + inArg;
	}
	T operator-=(T inArg) noexcept requires(oa::IsIntegralV<T>) {
		return fetchSub(inArg) - inArg;
	}

private:
	T value_{};
};

} // namespace oa
