#pragma once

// Native Fn<R(Args...)> — small-buffer type erasure with a heap fallback.

#include <oa/core/std/assert.h>
#include <oa/core/std/lifetime.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/std/utility.h>

namespace oa {

template<typename Sig>
class Fn;

namespace fnDetail {

template<typename R, typename F, typename... Args>
inline constexpr bool isInvocableRV = requires(F& inFunction, Args... inArgs) {
	static_cast<R>(inFunction(oa::forward<Args>(inArgs)...));
};

template<typename R, typename... Args>
struct VTable {
	using CallFn = R (*)(const void* storage, Args...);
	using DestroyFn = void (*)(void* storage);
	using CopyFn = void (*)(void* dst, const void* src);
	using MoveFn = void (*)(void* dst, void* src);

	CallFn call;
	DestroyFn destroy;
	CopyFn copy;
	MoveFn move;
};

template<typename R, typename F, typename... Args>
static R callSbo(const void* storage, Args... args) {
	using FD = oa::DecayT<F>;
	FD* p = const_cast<FD*>(reinterpret_cast<const FD*>(storage));
	return (*p)(oa::forward<Args>(args)...);
}

template<typename R, typename F, typename... Args>
static void destroySbo(void* storage) {
	using FD = oa::DecayT<F>;
	reinterpret_cast<FD*>(storage)->~FD();
}

template<typename R, typename F, typename... Args>
static void copySbo(void* dst, const void* src) {
	using FD = oa::DecayT<F>;
	oa::constructAt(reinterpret_cast<FD*>(dst), *reinterpret_cast<const FD*>(src));
}

template<typename R, typename F, typename... Args>
static void moveSbo(void* dst, void* src) {
	using FD = oa::DecayT<F>;
	oa::constructAt(reinterpret_cast<FD*>(dst), oa::move(*reinterpret_cast<FD*>(src)));
	oa::destroyAt(reinterpret_cast<FD*>(src));
}

template<typename R, typename F, typename... Args>
static const VTable<R, Args...>* vtableSbo() {
	static const VTable<R, Args...> vt = {&callSbo<R, F, Args...>, &destroySbo<R, F, Args...>,
		&copySbo<R, F, Args...>, &moveSbo<R, F, Args...>};
	return &vt;
}

template<typename R, typename... Args>
struct HeapBase {
	virtual ~HeapBase() = default;
	virtual HeapBase* clone() const = 0;
	virtual R invoke(Args...) const = 0;
};

template<typename R, typename F, typename... Args>
struct HeapImpl final : HeapBase<R, Args...> {
	mutable oa::DecayT<F> function_;

	template<typename G>
	explicit HeapImpl(G&& g) : function_(oa::forward<G>(g)) {}

	HeapBase<R, Args...>* clone() const override {
		return new HeapImpl<R, F, Args...>(function_);
	}

	R invoke(Args... args) const override {
		return const_cast<oa::DecayT<F>&>(function_)(oa::forward<Args>(args)...);
	}
};

template<typename R, typename... Args>
static R callHeap(const void* storage, Args... args) {
	auto* p = *static_cast<HeapBase<R, Args...>* const*>(storage);
	return p->invoke(oa::forward<Args>(args)...);
}

template<typename R, typename... Args>
static void destroyHeap(void* storage) {
	auto** pp = reinterpret_cast<HeapBase<R, Args...>**>(storage);
	delete *pp;
	*pp = nullptr;
}

template<typename R, typename... Args>
static void copyHeap(void* dst, const void* src) {
	auto* p = *static_cast<HeapBase<R, Args...>* const*>(src);
	*reinterpret_cast<HeapBase<R, Args...>**>(dst) = p->clone();
}

template<typename R, typename... Args>
static void moveHeap(void* dst, void* src) {
	auto** pd = reinterpret_cast<HeapBase<R, Args...>**>(dst);
	auto** ps = reinterpret_cast<HeapBase<R, Args...>**>(src);
	*pd = *ps;
	*ps = nullptr;
}

template<typename R, typename... Args>
static const VTable<R, Args...>* vtableHeap() {
	static const VTable<R, Args...> vt = {&callHeap<R, Args...>, &destroyHeap<R, Args...>,
		&copyHeap<R, Args...>, &moveHeap<R, Args...>};
	return &vt;
}

template<typename F, typename R, typename... Args>
static constexpr bool useSbo() {
	using FD = oa::DecayT<F>;
	constexpr decltype(sizeof(0)) Cap = 40;
	constexpr decltype(sizeof(0)) Alignment =
		alignof(long double) > alignof(void*) ? alignof(long double) : alignof(void*);
	return isInvocableRV<R, FD, Args...> && sizeof(FD) <= Cap &&
		alignof(FD) <= Alignment && oa::isNothrowMoveConstructibleV<FD>;
}

} // namespace fnDetail

template<typename R, typename... Args>
class Fn<R(Args...)> {
	static constexpr decltype(sizeof(0)) BufferSize = 40;
	static constexpr decltype(sizeof(0)) BufferAlignment =
		alignof(long double) > alignof(void*) ? alignof(long double) : alignof(void*);

	using VTable = fnDetail::VTable<R, Args...>;
	const VTable* vtable_{nullptr};
	alignas(BufferAlignment) unsigned char buffer_[BufferSize]{};

	void clear() noexcept {
		if (vtable_) {
			const VTable* retired = vtable_;
			vtable_ = nullptr;
			retired->destroy(buffer_);
		}
	}

	void initFrom(const Fn& inO) {
		if (!inO.vtable_) {
			return;
		}
		inO.vtable_->copy(buffer_, inO.buffer_);
		vtable_ = inO.vtable_;
	}

	void initFrom(Fn&& inO) noexcept {
		if (!inO.vtable_) {
			return;
		}
		const VTable* incoming = inO.vtable_;
		inO.vtable_ = nullptr;
		incoming->move(buffer_, inO.buffer_);
		vtable_ = incoming;
	}

public:
	Fn() noexcept = default;

	Fn(decltype(nullptr)) noexcept {}

	~Fn() { clear(); }

	Fn(const Fn& inO) { initFrom(inO); }

	Fn(Fn&& inO) noexcept { initFrom(oa::move(inO)); }

	Fn& operator=(const Fn& inO) {
		if (this == &inO) {
			return *this;
		}
		Fn tmp(inO);
		swap(tmp);
		return *this;
	}

	Fn& operator=(Fn&& inO) noexcept {
		if (this == &inO) {
			return *this;
		}
		clear();
		initFrom(oa::move(inO));
		return *this;
	}

	template<typename F>
	requires (!oa::isSameV<oa::DecayT<F>, Fn>
		&& fnDetail::isInvocableRV<R, oa::DecayT<F>, Args...>
		&& oa::isCopyConstructibleV<oa::DecayT<F>>)
	Fn(F&& inF) {
		using FD = oa::DecayT<F>;
		if constexpr (oa::isPointerV<FD>) {
			// Match std::function's typed-null function-pointer behavior. Treating
			// it as an engaged callable merely defers the fault until invocation.
			if (inF == nullptr) return;
		}

		if constexpr (fnDetail::useSbo<F, R, Args...>()) {
			oa::constructAt(reinterpret_cast<FD*>(buffer_), oa::forward<F>(inF));
			vtable_ = fnDetail::vtableSbo<R, FD, Args...>();
		} else {
			static_assert(sizeof(void*) <= BufferSize, "Fn: buffer too small for heap pointer");
			using Base = fnDetail::HeapBase<R, Args...>;
			Base* impl = new fnDetail::HeapImpl<R, FD, Args...>(oa::forward<F>(inF));
			*reinterpret_cast<Base**>(buffer_) = impl;
			vtable_ = fnDetail::vtableHeap<R, Args...>();
		}
	}

	R operator()(Args... inArgs) const {
		OA_REQUIRE(vtable_ != nullptr);
		return vtable_->call(buffer_, oa::forward<Args>(inArgs)...);
	}

	explicit operator bool() const noexcept { return vtable_ != nullptr; }

	[[nodiscard]] bool empty() const noexcept { return vtable_ == nullptr; }

	void swap(Fn& inO) noexcept {
		Fn tmp(oa::move(*this));
		*this = oa::move(inO);
		inO = oa::move(tmp);
	}

};

} // namespace oa
