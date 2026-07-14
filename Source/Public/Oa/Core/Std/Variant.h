#pragma once

// Native OaStdVariant<Ts...> — tagged union; PascalCase API (`Get`/`Index`/`Visit`/`Emplace`/`Swap`/`HoldsAlternative`).
// `StdVariant()` builds `std::variant<Ts...>` for std boundaries.

#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace oa_std_var_detail {

template<typename Tuple, std::size_t... Is>
constexpr std::size_t MaxSize(std::index_sequence<Is...>) {
	return (std::max)({sizeof(std::tuple_element_t<Is, Tuple>)...});
}

template<typename Tuple, std::size_t... Is>
constexpr std::size_t MaxAlign(std::index_sequence<Is...>) {
	return (std::max)({alignof(std::tuple_element_t<Is, Tuple>)...});
}

template<typename U, typename Tuple, std::size_t... Is>
constexpr std::size_t FirstIndexOf(std::index_sequence<Is...>) {
	std::size_t r = std::numeric_limits<std::size_t>::max();
	((void)(std::is_same_v<U, std::tuple_element_t<Is, Tuple>> && r == std::numeric_limits<std::size_t>::max()
			 ? (r = Is, 0)
			 : 0),
		...);
	return r;
}

template<typename D, typename Tuple, std::size_t... Is>
constexpr std::size_t IndexOfDecay(std::index_sequence<Is...>) {
	std::size_t r = std::numeric_limits<std::size_t>::max();
	((void)(std::is_same_v<D, std::decay_t<std::tuple_element_t<Is, Tuple>>> &&
					r == std::numeric_limits<std::size_t>::max()
			 ? (r = Is, 0)
			 : 0),
		...);
	return r;
}

} // namespace oa_std_var_detail

template<typename... Ts>
struct OaStdVariant {
	static_assert(sizeof...(Ts) > 0, "OaStdVariant requires at least one alternative");

	static constexpr std::size_t Npos = std::numeric_limits<std::size_t>::max();
	using Tuple = std::tuple<Ts...>;
	static constexpr std::size_t Count = sizeof...(Ts);
	using Indices = std::index_sequence_for<Ts...>;

	static constexpr std::size_t StorageSize = oa_std_var_detail::MaxSize<Tuple>(Indices{});
	static constexpr std::size_t StorageAlign = oa_std_var_detail::MaxAlign<Tuple>(Indices{});

	template<std::size_t I>
	using Alt = std::tuple_element_t<I, Tuple>;

	void* Raw() noexcept { return Storage_; }
	const void* Raw() const noexcept { return Storage_; }

	template<std::size_t... Is>
	static void DestroyAt(std::size_t InIdx, void* InP, std::index_sequence<Is...>) {
		bool matched = false;
		((void)(matched || (InIdx == Is ? (reinterpret_cast<Alt<Is>*>(InP)->~Alt<Is>(), matched = true, true) : false)),
			...);
		(void)matched;
	}

	static void DestroyAt(std::size_t InIdx, void* InP) { DestroyAt(InIdx, InP, Indices{}); }

	void Destroy() noexcept {
		if (Index_ == Npos) {
			return;
		}
		DestroyAt(Index_, Raw());
		Index_ = Npos;
	}

	template<std::size_t... Is>
	static void CopyConstructAt(std::size_t InIdx, void* InDst, const void* InSrc, std::index_sequence<Is...>) {
		bool matched = false;
		((void)(matched ||
				 (InIdx == Is ? (new (InDst) Alt<Is>(*reinterpret_cast<const Alt<Is>*>(InSrc)), matched = true, true)
							  : false)),
			...);
		(void)matched;
	}

	static void CopyConstructAt(std::size_t InIdx, void* InDst, const void* InSrc) {
		CopyConstructAt(InIdx, InDst, InSrc, Indices{});
	}

	template<std::size_t... Is>
	static void MoveConstructAt(std::size_t InIdx, void* InDst, void* InSrc, std::index_sequence<Is...>) {
		bool matched = false;
		((void)(matched ||
				 (InIdx == Is ? (new (InDst) Alt<Is>(std::move(*reinterpret_cast<Alt<Is>*>(InSrc))), matched = true, true)
							  : false)),
			...);
		(void)matched;
	}

	static void MoveConstructAt(std::size_t InIdx, void* InDst, void* InSrc) {
		MoveConstructAt(InIdx, InDst, InSrc, Indices{});
	}

	template<typename F, std::size_t... Is>
	static void DispatchVoidMut(std::size_t InIdx, F&& InF, void* InP, std::index_sequence<Is...>) {
		bool done = false;
		((void)(done || (InIdx == Is ? (std::forward<F>(InF)(*reinterpret_cast<Alt<Is>*>(InP)), done = true, false) : false)),
			...);
		if (!done) {
			throw std::bad_variant_access();
		}
	}

	template<typename F, std::size_t... Is>
	static void DispatchVoidConst(std::size_t InIdx, F&& InF, const void* InP, std::index_sequence<Is...>) {
		bool done = false;
		((void)(done ||
				 (InIdx == Is ? (std::forward<F>(InF)(*reinterpret_cast<const Alt<Is>*>(InP)), done = true, false) : false)),
			...);
		if (!done) {
			throw std::bad_variant_access();
		}
	}

	template<std::size_t... Is>
	static std::variant<Ts...> ToStdVariant(std::size_t InIdx, const void* InP, std::index_sequence<Is...>) {
		std::variant<Ts...> out{};
		bool has = false;
		((void)(has || (InIdx == Is
							? (out.template emplace<Is>(*reinterpret_cast<const Alt<Is>*>(InP)), has = true, false)
							: false)),
			...);
		if (!has) {
			throw std::bad_variant_access();
		}
		return out;
	}

	template<typename F, std::size_t... Is>
	static void DispatchVoidRValue(std::size_t InIdx, F&& InF, void* InP, std::index_sequence<Is...>) {
		bool done = false;
		((void)(done || (InIdx == Is
							? (std::forward<F>(InF)(std::move(*reinterpret_cast<Alt<Is>*>(InP))), done = true, false)
							: false)),
			...);
		if (!done) {
			throw std::bad_variant_access();
		}
	}

	alignas(StorageAlign) unsigned char Storage_[StorageSize]{};
	std::size_t Index_{Npos};

	template<typename T, typename D = std::decay_t<T>>
	static constexpr std::size_t DecayIndex() {
		return oa_std_var_detail::IndexOfDecay<D, Tuple>(Indices{});
	}

public:
	OaStdVariant() noexcept(std::is_nothrow_default_constructible_v<Alt<0>>) : Index_(0) {
		new (Raw()) Alt<0>{};
	}

	OaStdVariant(const OaStdVariant& InO) : Index_(InO.Index_) {
		if (Index_ != Npos) {
			CopyConstructAt(Index_, Raw(), InO.Raw());
		}
	}

	OaStdVariant(OaStdVariant&& InO) noexcept((std::is_nothrow_move_constructible_v<Ts> && ...)) : Index_(InO.Index_) {
		if (Index_ != Npos) {
			MoveConstructAt(Index_, Raw(), InO.Raw());
			DestroyAt(InO.Index_, InO.Raw());
			InO.Index_ = Npos;
		}
	}

	template<typename T, typename D = std::decay_t<T>,
		typename = std::enable_if_t<!std::is_same_v<D, OaStdVariant>>,
		typename = std::enable_if_t<DecayIndex<T>() != Npos>>
	OaStdVariant(T&& InT) : Index_(DecayIndex<T>()) {
		constexpr std::size_t I = DecayIndex<T>();
		new (Raw()) Alt<I>(std::forward<T>(InT));
	}

	~OaStdVariant() { Destroy(); }

	OaStdVariant& operator=(const OaStdVariant& InO) {
		if (this == &InO) {
			return *this;
		}
		OaStdVariant tmp(InO);
		*this = std::move(tmp);
		return *this;
	}

	OaStdVariant& operator=(OaStdVariant&& InO) noexcept(
		(std::is_nothrow_move_constructible_v<Ts> && ...) && (std::is_nothrow_destructible_v<Ts> && ...)) {
		if (this == &InO) {
			return *this;
		}
		Destroy();
		Index_ = InO.Index_;
		if (Index_ != Npos) {
			MoveConstructAt(Index_, Raw(), InO.Raw());
			DestroyAt(InO.Index_, InO.Raw());
			InO.Index_ = Npos;
		}
		return *this;
	}

	[[nodiscard]] std::size_t Index() const noexcept { return Index_; }

	[[nodiscard]] bool ValuelessByException() const noexcept { return Index_ == Npos; }

	template<std::size_t I>
	[[nodiscard]] Alt<I>& Get() & {
		if (Index_ != I) {
			throw std::bad_variant_access();
		}
		return *reinterpret_cast<Alt<I>*>(Raw());
	}
	template<std::size_t I>
	[[nodiscard]] const Alt<I>& Get() const& {
		if (Index_ != I) {
			throw std::bad_variant_access();
		}
		return *reinterpret_cast<const Alt<I>*>(Raw());
	}
	template<std::size_t I>
	[[nodiscard]] Alt<I>&& Get() && {
		if (Index_ != I) {
			throw std::bad_variant_access();
		}
		return std::move(*reinterpret_cast<Alt<I>*>(Raw()));
	}

	template<typename U>
	[[nodiscard]] U& Get() & {
		constexpr std::size_t I = oa_std_var_detail::FirstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return Get<I>();
	}
	template<typename U>
	[[nodiscard]] const U& Get() const& {
		constexpr std::size_t I = oa_std_var_detail::FirstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return Get<I>();
	}
	template<typename U>
	[[nodiscard]] U&& Get() && {
		constexpr std::size_t I = oa_std_var_detail::FirstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return std::move(Get<I>());
	}

	template<typename U>
	[[nodiscard]] bool HoldsAlternative() const noexcept {
		constexpr std::size_t I = oa_std_var_detail::FirstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return Index_ == I;
	}

	template<typename F>
	void Visit(F&& InF) & {
		if (Index_ == Npos) {
			throw std::bad_variant_access();
		}
		DispatchVoidMut(Index_, std::forward<F>(InF), Raw(), Indices{});
	}

	template<typename F>
	void Visit(F&& InF) const& {
		if (Index_ == Npos) {
			throw std::bad_variant_access();
		}
		DispatchVoidConst(Index_, std::forward<F>(InF), Raw(), Indices{});
	}

	template<typename F>
	void Visit(F&& InF) && {
		if (Index_ == Npos) {
			throw std::bad_variant_access();
		}
		DispatchVoidRValue(Index_, std::forward<F>(InF), Raw(), Indices{});
		Destroy();
	}

	template<typename U, typename... Args>
	U& Emplace(Args&&... InArgs) {
		Destroy();
		constexpr std::size_t I = oa_std_var_detail::FirstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		new (Raw()) U(std::forward<Args>(InArgs)...);
		Index_ = I;
		return *reinterpret_cast<U*>(Raw());
	}

	void Swap(OaStdVariant& InO) noexcept(
		(std::is_nothrow_move_constructible_v<Ts> && ...) && (std::is_nothrow_swappable_v<Ts> && ...)) {
		OaStdVariant tmp(std::move(*this));
		*this = std::move(InO);
		InO = std::move(tmp);
	}

	[[nodiscard]] std::variant<Ts...> StdVariant() const& {
		if (Index_ == Npos) {
			return std::variant<Ts...>{};
		}
		return ToStdVariant(Index_, Raw(), Indices{});
	}

	[[nodiscard]] std::variant<Ts...> StdVariant() && {
		OaStdVariant tmp(std::move(*this));
		return tmp.StdVariant();
	}
};
