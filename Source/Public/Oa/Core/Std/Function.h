#pragma once

// Native OaStdFn<R(Args...)> — small-buffer type erasure + heap fallback; `StdFn()` wraps for std boundaries.

#include <Oa/Core/Std/TypeTraits.h>
#include <Oa/Core/Std/Utility.h>

#include <functional>
#include <new>
#include <stdexcept>
#include <type_traits>

template<typename Sig>
class OaStdFn;

namespace OaStdFnDetail {

template<typename R, typename... Args>
struct VTable {
	using CallFn = R (*)(const void* storage, Args...);
	using DestroyFn = void (*)(void* storage);
	using CopyFn = void (*)(void* dst, const void* src);
	using MoveFn = void (*)(void* dst, void* src);

	CallFn Call;
	DestroyFn Destroy;
	CopyFn Copy;
	MoveFn Move;
};

template<typename R, typename F, typename... Args>
static R CallSbo(const void* storage, Args... args) {
	using FD = OaStdDecayT<F>;
	FD* p = const_cast<FD*>(reinterpret_cast<const FD*>(storage));
	return (*p)(OaStdForward<Args>(args)...);
}

template<typename R, typename F, typename... Args>
static void DestroySbo(void* storage) {
	using FD = OaStdDecayT<F>;
	reinterpret_cast<FD*>(storage)->~FD();
}

template<typename R, typename F, typename... Args>
static void CopySbo(void* dst, const void* src) {
	using FD = OaStdDecayT<F>;
	new (dst) FD(*reinterpret_cast<const FD*>(src));
}

template<typename R, typename F, typename... Args>
static void MoveSbo(void* dst, void* src) {
	using FD = OaStdDecayT<F>;
	new (dst) FD(OaStdMove(*reinterpret_cast<FD*>(src)));
	reinterpret_cast<FD*>(src)->~FD();
}

template<typename R, typename F, typename... Args>
static const VTable<R, Args...>* VtableSbo() {
	static const VTable<R, Args...> vt = {&CallSbo<R, F, Args...>, &DestroySbo<R, F, Args...>,
		&CopySbo<R, F, Args...>, &MoveSbo<R, F, Args...>};
	return &vt;
}

template<typename R, typename... Args>
struct HeapBase {
	virtual ~HeapBase() = default;
	virtual HeapBase* Clone() const = 0;
	virtual R Invoke(Args...) const = 0;
};

template<typename R, typename F, typename... Args>
struct HeapImpl final : HeapBase<R, Args...> {
	mutable OaStdDecayT<F> Fn_;

	template<typename G>
	explicit HeapImpl(G&& g) : Fn_(OaStdForward<G>(g)) {}

	HeapBase<R, Args...>* Clone() const override {
		return new HeapImpl<R, F, Args...>(Fn_);
	}

	R Invoke(Args... args) const override {
		return const_cast<OaStdDecayT<F>&>(Fn_)(OaStdForward<Args>(args)...);
	}
};

template<typename R, typename... Args>
static R CallHeap(const void* storage, Args... args) {
	auto* p = *static_cast<HeapBase<R, Args...>* const*>(storage);
	return p->Invoke(OaStdForward<Args>(args)...);
}

template<typename R, typename... Args>
static void DestroyHeap(void* storage) {
	auto** pp = reinterpret_cast<HeapBase<R, Args...>**>(storage);
	delete *pp;
	*pp = nullptr;
}

template<typename R, typename... Args>
static void CopyHeap(void* dst, const void* src) {
	auto* p = *static_cast<HeapBase<R, Args...>* const*>(src);
	*reinterpret_cast<HeapBase<R, Args...>**>(dst) = p->Clone();
}

template<typename R, typename... Args>
static void MoveHeap(void* dst, void* src) {
	auto** pd = reinterpret_cast<HeapBase<R, Args...>**>(dst);
	auto** ps = reinterpret_cast<HeapBase<R, Args...>**>(src);
	*pd = *ps;
	*ps = nullptr;
}

template<typename R, typename... Args>
static const VTable<R, Args...>* VtableHeap() {
	static const VTable<R, Args...> vt = {&CallHeap<R, Args...>, &DestroyHeap<R, Args...>,
		&CopyHeap<R, Args...>, &MoveHeap<R, Args...>};
	return &vt;
}

template<typename F, typename R, typename... Args>
static constexpr bool UseSbo() {
	using FD = OaStdDecayT<F>;
	constexpr std::size_t Cap = 40;
	return std::is_invocable_r_v<R, FD&, Args...> && sizeof(FD) <= Cap &&
		alignof(FD) <= alignof(std::max_align_t) && OaStdIsNothrowMoveConstructibleV<FD>;
}

} // namespace OaStdFnDetail

template<typename R, typename... Args>
class OaStdFn<R(Args...)> {
	static constexpr std::size_t BufSize = 40;

	using VTable = OaStdFnDetail::VTable<R, Args...>;
	const VTable* Vt_{nullptr};
	alignas(std::max_align_t) unsigned char Buf_[BufSize]{};

	void Clear() noexcept {
		if (Vt_) {
			Vt_->Destroy(Buf_);
			Vt_ = nullptr;
		}
	}

	void InitFrom(const OaStdFn& InO) {
		if (!InO.Vt_) {
			return;
		}
		InO.Vt_->Copy(Buf_, InO.Buf_);
		Vt_ = InO.Vt_;
	}

	void InitFrom(OaStdFn&& InO) noexcept {
		if (!InO.Vt_) {
			return;
		}
		InO.Vt_->Move(Buf_, InO.Buf_);
		Vt_ = InO.Vt_;
		InO.Vt_ = nullptr;
	}

public:
	OaStdFn() noexcept = default;

	OaStdFn(std::nullptr_t) noexcept {}

	~OaStdFn() { Clear(); }

	OaStdFn(const OaStdFn& InO) { InitFrom(InO); }

	OaStdFn(OaStdFn&& InO) noexcept { InitFrom(OaStdMove(InO)); }

	OaStdFn& operator=(const OaStdFn& InO) {
		if (this == &InO) {
			return *this;
		}
		OaStdFn tmp(InO);
		Swap(tmp);
		return *this;
	}

	OaStdFn& operator=(OaStdFn&& InO) noexcept {
		if (this == &InO) {
			return *this;
		}
		Clear();
		InitFrom(OaStdMove(InO));
		return *this;
	}

	template<typename F, typename = std::enable_if_t<!std::is_same_v<OaStdDecayT<F>, OaStdFn>>>
	OaStdFn(F&& InF) {
		using FD = OaStdDecayT<F>;
		static_assert(std::is_invocable_r_v<R, FD&, Args...>, "OaStdFn: F must be invocable with Args...");

		if constexpr (OaStdFnDetail::UseSbo<F, R, Args...>()) {
			new (Buf_) FD(OaStdForward<F>(InF));
			Vt_ = OaStdFnDetail::VtableSbo<R, FD, Args...>();
		} else {
			static_assert(sizeof(void*) <= BufSize, "OaStdFn: buffer too small for heap pointer");
			using Base = OaStdFnDetail::HeapBase<R, Args...>;
			Base* impl = new OaStdFnDetail::HeapImpl<R, FD, Args...>(OaStdForward<F>(InF));
			*reinterpret_cast<Base**>(Buf_) = impl;
			Vt_ = OaStdFnDetail::VtableHeap<R, Args...>();
		}
	}

	R operator()(Args... InArgs) const {
		if (!Vt_) {
			throw std::bad_function_call();
		}
		return Vt_->Call(Buf_, OaStdForward<Args>(InArgs)...);
	}

	explicit operator bool() const noexcept { return Vt_ != nullptr; }

	[[nodiscard]] bool Empty() const noexcept { return Vt_ == nullptr; }

	void Swap(OaStdFn& InO) noexcept {
		OaStdFn tmp(OaStdMove(*this));
		*this = OaStdMove(InO);
		InO = OaStdMove(tmp);
	}

	[[nodiscard]] std::function<R(Args...)> StdFn() const& {
		if (!Vt_) {
			return {};
		}
		OaStdFn self = *this;
		return [self](Args... a) mutable { return self(OaStdForward<Args>(a)...); };
	}

	[[nodiscard]] std::function<R(Args...)> StdFn() && {
		if (!Vt_) {
			return {};
		}
		OaStdFn self = OaStdMove(*this);
		return [self = OaStdMove(self)](Args... a) mutable {
			return self(OaStdForward<Args>(a)...);
		};
	}
};
