#pragma once

// Native Variant<Ts...> — bounded tagged storage for OA value alternatives.

#include <oa/core/assert.h>
#include <oa/core/std/lifetime.h>
#include <oa/core/std/typeTraits.h>
#include <oa/core/std/utility.h>

namespace oa {

namespace variantDetail {

using Size = decltype(sizeof(0));
inline constexpr Size Npos = static_cast<Size>(-1);

template<Size Index, typename Head, typename... Tail>
struct TypeAt : TypeAt<Index - 1, Tail...> {};

template<typename Head, typename... Tail>
struct TypeAt<0, Head, Tail...> {
	using Type = Head;
};

template<typename... Ts>
struct MaxSize;

template<typename T>
struct MaxSize<T> {
	static constexpr Size value = sizeof(T);
};

template<typename Head, typename... Tail>
struct MaxSize<Head, Tail...> {
	static constexpr Size tail = MaxSize<Tail...>::value;
	static constexpr Size value = sizeof(Head) > tail ? sizeof(Head) : tail;
};

template<typename... Ts>
struct MaxAlign;

template<typename T>
struct MaxAlign<T> {
	static constexpr Size value = alignof(T);
};

template<typename Head, typename... Tail>
struct MaxAlign<Head, Tail...> {
	static constexpr Size tail = MaxAlign<Tail...>::value;
	static constexpr Size value = alignof(Head) > tail ? alignof(Head) : tail;
};

template<typename U, Size Index, typename... Ts>
struct FirstIndex;

template<typename U, Size Index>
struct FirstIndex<U, Index> {
	static constexpr Size value = Npos;
};

template<typename U, Size Index, typename Head, typename... Tail>
struct FirstIndex<U, Index, Head, Tail...> {
	static constexpr Size value = oa::IsSameV<U, Head>
		? Index
		: FirstIndex<U, Index + 1, Tail...>::value;
};

template<typename U, Size Index, typename... Ts>
struct FirstDecayIndex;

template<typename U, Size Index>
struct FirstDecayIndex<U, Index> {
	static constexpr Size value = Npos;
};

template<typename U, Size Index, typename Head, typename... Tail>
struct FirstDecayIndex<U, Index, Head, Tail...> {
	static constexpr Size value = oa::IsSameV<U, oa::DecayT<Head>>
		? Index
		: FirstDecayIndex<U, Index + 1, Tail...>::value;
};

} // namespace variantDetail

template<typename... Ts>
class Variant {
public:
	static_assert(sizeof...(Ts) > 0, "Variant requires at least one alternative");

	using Size = variantDetail::Size;
	static constexpr Size Npos = variantDetail::Npos;
	static constexpr Size Count = sizeof...(Ts);
	static constexpr Size StorageSize = variantDetail::MaxSize<Ts...>::value;
	static constexpr Size StorageAlign = variantDetail::MaxAlign<Ts...>::value;

	template<Size Index>
	using Alt = typename variantDetail::TypeAt<Index, Ts...>::Type;

private:
	void* raw_() noexcept { return storage_; }
	const void* raw_() const noexcept { return storage_; }

	template<Size Index = 0>
	static void destroyAt_(Size inIndex, void* inStorage) noexcept {
		if constexpr (Index < Count) {
			if (inIndex == Index) {
				oa::destroyAt(reinterpret_cast<Alt<Index>*>(inStorage));
				return;
			}
			destroyAt_<Index + 1>(inIndex, inStorage);
		} else {
			OA_REQUIRE(false);
		}
	}

	void destroy_() noexcept {
		if (index_ != Npos) {
			destroyAt_(index_, raw_());
			index_ = Npos;
		}
	}

	template<Size Index = 0>
	static void copyConstructAt_(Size inIndex, void* inDst, const void* inSrc) {
		if constexpr (Index < Count) {
			if (inIndex == Index) {
				oa::constructAt(
					reinterpret_cast<Alt<Index>*>(inDst),
					*reinterpret_cast<const Alt<Index>*>(inSrc));
				return;
			}
			copyConstructAt_<Index + 1>(inIndex, inDst, inSrc);
		} else {
			OA_REQUIRE(false);
		}
	}

	template<Size Index = 0>
	static void moveConstructAt_(Size inIndex, void* inDst, void* inSrc) {
		if constexpr (Index < Count) {
			if (inIndex == Index) {
				oa::constructAt(
					reinterpret_cast<Alt<Index>*>(inDst),
					oa::move(*reinterpret_cast<Alt<Index>*>(inSrc)));
				return;
			}
			moveConstructAt_<Index + 1>(inIndex, inDst, inSrc);
		} else {
			OA_REQUIRE(false);
		}
	}

	template<Size Index = 0, typename F>
	static void visitMut_(Size inIndex, F&& inFunction, void* inStorage) {
		if constexpr (Index < Count) {
			if (inIndex == Index) {
				oa::forward<F>(inFunction)(*reinterpret_cast<Alt<Index>*>(inStorage));
				return;
			}
			visitMut_<Index + 1>(inIndex, oa::forward<F>(inFunction), inStorage);
		} else {
			OA_REQUIRE(false);
		}
	}

	template<Size Index = 0, typename F>
	static void visitConst_(Size inIndex, F&& inFunction, const void* inStorage) {
		if constexpr (Index < Count) {
			if (inIndex == Index) {
				oa::forward<F>(inFunction)(
					*reinterpret_cast<const Alt<Index>*>(inStorage));
				return;
			}
			visitConst_<Index + 1>(inIndex, oa::forward<F>(inFunction), inStorage);
		} else {
			OA_REQUIRE(false);
		}
	}

	template<Size Index = 0, typename F>
	static void visitMove_(Size inIndex, F&& inFunction, void* inStorage) {
		if constexpr (Index < Count) {
			if (inIndex == Index) {
				oa::forward<F>(inFunction)(
					oa::move(*reinterpret_cast<Alt<Index>*>(inStorage)));
				return;
			}
			visitMove_<Index + 1>(inIndex, oa::forward<F>(inFunction), inStorage);
		} else {
			OA_REQUIRE(false);
		}
	}

	template<typename T>
	static constexpr Size decayIndex_() {
		return variantDetail::FirstDecayIndex<oa::DecayT<T>, 0, Ts...>::value;
	}

	alignas(StorageAlign) unsigned char storage_[StorageSize]{};
	Size index_{Npos};

public:
	Variant() noexcept(oa::IsNothrowConstructibleV<Alt<0>>) : index_(0) {
		oa::constructAt(reinterpret_cast<Alt<0>*>(raw_()));
	}

	Variant(const Variant& inOther) : index_(inOther.index_) {
		if (index_ != Npos) {
			copyConstructAt_(index_, raw_(), inOther.raw_());
		}
	}

	Variant(Variant&& inOther) noexcept((oa::IsNothrowMoveConstructibleV<Ts> && ...))
		: index_(inOther.index_) {
		if (index_ != Npos) {
			moveConstructAt_(index_, raw_(), inOther.raw_());
			destroyAt_(inOther.index_, inOther.raw_());
			inOther.index_ = Npos;
		}
	}

	template<typename T>
	requires (!oa::IsSameV<oa::DecayT<T>, Variant> && decayIndex_<T>() != Npos)
	Variant(T&& inValue) : index_(decayIndex_<T>()) {
		constexpr Size Index = decayIndex_<T>();
		oa::constructAt(
			reinterpret_cast<Alt<Index>*>(raw_()), oa::forward<T>(inValue));
	}

	~Variant() { destroy_(); }

	Variant& operator=(const Variant& inOther) {
		if (this != &inOther) {
			Variant temporary(inOther);
			*this = oa::move(temporary);
		}
		return *this;
	}

	Variant& operator=(Variant&& inOther) noexcept(
		(oa::IsNothrowMoveConstructibleV<Ts> && ...) &&
		(oa::IsNothrowMoveAssignableV<Ts> && ...)) {
		if (this != &inOther) {
			destroy_();
			index_ = inOther.index_;
			if (index_ != Npos) {
				moveConstructAt_(index_, raw_(), inOther.raw_());
				destroyAt_(inOther.index_, inOther.raw_());
				inOther.index_ = Npos;
			}
		}
		return *this;
	}

	[[nodiscard]] Size index() const noexcept { return index_; }
	[[nodiscard]] bool empty() const noexcept { return index_ == Npos; }

	template<Size Index>
	[[nodiscard]] Alt<Index>& get() & {
		OA_REQUIRE(index_ == Index);
		return *reinterpret_cast<Alt<Index>*>(raw_());
	}

	template<Size Index>
	[[nodiscard]] const Alt<Index>& get() const& {
		OA_REQUIRE(index_ == Index);
		return *reinterpret_cast<const Alt<Index>*>(raw_());
	}

	template<Size Index>
	[[nodiscard]] Alt<Index>&& get() && {
		OA_REQUIRE(index_ == Index);
		return oa::move(*reinterpret_cast<Alt<Index>*>(raw_()));
	}

	template<typename U>
	[[nodiscard]] U& get() & {
		constexpr Size Index = variantDetail::FirstIndex<U, 0, Ts...>::value;
		static_assert(Index != Npos, "Variant::get type is not an alternative");
		return get<Index>();
	}

	template<typename U>
	[[nodiscard]] const U& get() const& {
		constexpr Size Index = variantDetail::FirstIndex<U, 0, Ts...>::value;
		static_assert(Index != Npos, "Variant::get type is not an alternative");
		return get<Index>();
	}

	template<typename U>
	[[nodiscard]] U&& get() && {
		constexpr Size Index = variantDetail::FirstIndex<U, 0, Ts...>::value;
		static_assert(Index != Npos, "Variant::get type is not an alternative");
		return oa::move(get<Index>());
	}

	template<typename U>
	[[nodiscard]] bool holdsAlternative() const noexcept {
		constexpr Size Index = variantDetail::FirstIndex<U, 0, Ts...>::value;
		static_assert(Index != Npos,
			"Variant::holdsAlternative type is not an alternative");
		return index_ == Index;
	}

	template<typename F>
	void visit(F&& inFunction) & {
		OA_REQUIRE(index_ != Npos);
		visitMut_(index_, oa::forward<F>(inFunction), raw_());
	}

	template<typename F>
	void visit(F&& inFunction) const& {
		OA_REQUIRE(index_ != Npos);
		visitConst_(index_, oa::forward<F>(inFunction), raw_());
	}

	template<typename F>
	void visit(F&& inFunction) && {
		OA_REQUIRE(index_ != Npos);
		visitMove_(index_, oa::forward<F>(inFunction), raw_());
		destroy_();
	}

	template<typename U, typename... Args>
	U& emplace(Args&&... inArgs) {
		constexpr Size Index = variantDetail::FirstIndex<U, 0, Ts...>::value;
		static_assert(Index != Npos, "Variant::emplace type is not an alternative");
		destroy_();
		oa::constructAt(
			reinterpret_cast<U*>(raw_()), oa::forward<Args>(inArgs)...);
		index_ = Index;
		return *reinterpret_cast<U*>(raw_());
	}

	void swap(Variant& inOther) noexcept(
		(oa::IsNothrowMoveConstructibleV<Ts> && ...) &&
		(oa::IsNothrowMoveAssignableV<Ts> && ...)) {
		Variant temporary(oa::move(*this));
		*this = oa::move(inOther);
		inOther = oa::move(temporary);
	}
};

} // namespace oa
