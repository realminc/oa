#pragma once

// Native Variant<Ts...> — tagged union with an OA-named value API.
// `stdVariant()` builds `std::variant<Ts...>` for std boundaries.

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace oa {

namespace variantDetail {

template<typename Tuple, std::size_t... Is>
constexpr std::size_t maxSize(std::index_sequence<Is...>) {
	return (std::max)({sizeof(std::tuple_element_t<Is, Tuple>)...});
}

template<typename Tuple, std::size_t... Is>
constexpr std::size_t maxAlign(std::index_sequence<Is...>) {
	return (std::max)({alignof(std::tuple_element_t<Is, Tuple>)...});
}

template<typename U, typename Tuple, std::size_t... Is>
constexpr std::size_t firstIndexOf(std::index_sequence<Is...>) {
	std::size_t r = std::numeric_limits<std::size_t>::max();
	((void)(std::is_same_v<U, std::tuple_element_t<Is, Tuple>> && r == std::numeric_limits<std::size_t>::max()
			 ? (r = Is, 0)
			 : 0),
		...);
	return r;
}

template<typename D, typename Tuple, std::size_t... Is>
constexpr std::size_t indexOfDecay(std::index_sequence<Is...>) {
	std::size_t r = std::numeric_limits<std::size_t>::max();
	((void)(std::is_same_v<D, std::decay_t<std::tuple_element_t<Is, Tuple>>> &&
					r == std::numeric_limits<std::size_t>::max()
			 ? (r = Is, 0)
			 : 0),
		...);
	return r;
}

} // namespace variantDetail

template<typename... Ts>
class Variant {
public:
	static_assert(sizeof...(Ts) > 0, "Variant requires at least one alternative");

	static constexpr std::size_t Npos = std::numeric_limits<std::size_t>::max();
	using Tuple = std::tuple<Ts...>;
	static constexpr std::size_t Count = sizeof...(Ts);
	using Indices = std::index_sequence_for<Ts...>;

	static constexpr std::size_t StorageSize = variantDetail::maxSize<Tuple>(Indices{});
	static constexpr std::size_t StorageAlign = variantDetail::maxAlign<Tuple>(Indices{});

	template<std::size_t I>
	using Alt = std::tuple_element_t<I, Tuple>;

private:

	void* raw_() noexcept { return storage_; }
	const void* raw_() const noexcept { return storage_; }

	template<std::size_t... Is>
	static void destroyAt_(std::size_t inIdx, void* inP, std::index_sequence<Is...>) {
		bool matched = false;
		((void)(matched || (inIdx == Is ? (reinterpret_cast<Alt<Is>*>(inP)->~Alt<Is>(), matched = true, true) : false)),
			...);
		(void)matched;
	}

	static void destroyAt_(std::size_t inIdx, void* inP) { destroyAt_(inIdx, inP, Indices{}); }

	void destroy_() noexcept {
		if (index_ == Npos) {
			return;
		}
		destroyAt_(index_, raw_());
		index_ = Npos;
	}

	template<std::size_t... Is>
	static void copyConstructAt_(std::size_t inIdx, void* inDst, const void* inSrc, std::index_sequence<Is...>) {
		bool matched = false;
		((void)(matched ||
				 (inIdx == Is ? (new (inDst) Alt<Is>(*reinterpret_cast<const Alt<Is>*>(inSrc)), matched = true, true)
							  : false)),
			...);
		(void)matched;
	}

	static void copyConstructAt_(std::size_t inIdx, void* inDst, const void* inSrc) {
		copyConstructAt_(inIdx, inDst, inSrc, Indices{});
	}

	template<std::size_t... Is>
	static void moveConstructAt_(std::size_t inIdx, void* inDst, void* inSrc, std::index_sequence<Is...>) {
		bool matched = false;
		((void)(matched ||
				 (inIdx == Is ? (new (inDst) Alt<Is>(std::move(*reinterpret_cast<Alt<Is>*>(inSrc))), matched = true, true)
							  : false)),
			...);
		(void)matched;
	}

	static void moveConstructAt_(std::size_t inIdx, void* inDst, void* inSrc) {
		moveConstructAt_(inIdx, inDst, inSrc, Indices{});
	}

	template<typename F, std::size_t... Is>
	static void dispatchVoidMut_(std::size_t inIdx, F&& inF, void* inP, std::index_sequence<Is...>) {
		bool done = false;
		((void)(done || (inIdx == Is ? (std::forward<F>(inF)(*reinterpret_cast<Alt<Is>*>(inP)), done = true, false) : false)),
			...);
		if (!done) {
			throw std::bad_variant_access();
		}
	}

	template<typename F, std::size_t... Is>
	static void dispatchVoidConst_(std::size_t inIdx, F&& inF, const void* inP, std::index_sequence<Is...>) {
		bool done = false;
		((void)(done ||
				 (inIdx == Is ? (std::forward<F>(inF)(*reinterpret_cast<const Alt<Is>*>(inP)), done = true, false) : false)),
			...);
		if (!done) {
			throw std::bad_variant_access();
		}
	}

	template<std::size_t... Is>
	static std::variant<Ts...> toStdVariant_(std::size_t inIdx, const void* inP, std::index_sequence<Is...>) {
		std::variant<Ts...> out{};
		bool has = false;
		((void)(has || (inIdx == Is
							? (out.template emplace<Is>(*reinterpret_cast<const Alt<Is>*>(inP)), has = true, false)
							: false)),
			...);
		if (!has) {
			throw std::bad_variant_access();
		}
		return out;
	}

	template<typename F, std::size_t... Is>
	static void dispatchVoidRValue_(std::size_t inIdx, F&& inF, void* inP, std::index_sequence<Is...>) {
		bool done = false;
		((void)(done || (inIdx == Is
							? (std::forward<F>(inF)(std::move(*reinterpret_cast<Alt<Is>*>(inP))), done = true, false)
							: false)),
			...);
		if (!done) {
			throw std::bad_variant_access();
		}
	}

	alignas(StorageAlign) unsigned char storage_[StorageSize]{};
	std::size_t index_{Npos};

	template<typename T, typename D = std::decay_t<T>>
	static constexpr std::size_t decayIndex_() {
		return variantDetail::indexOfDecay<D, Tuple>(Indices{});
	}

public:
	Variant() noexcept(std::is_nothrow_default_constructible_v<Alt<0>>) : index_(0) {
		new (raw_()) Alt<0>{};
	}

	Variant(const Variant& inO) : index_(inO.index_) {
		if (index_ != Npos) {
			copyConstructAt_(index_, raw_(), inO.raw_());
		}
	}

	Variant(Variant&& inO) noexcept((std::is_nothrow_move_constructible_v<Ts> && ...)) : index_(inO.index_) {
		if (index_ != Npos) {
			moveConstructAt_(index_, raw_(), inO.raw_());
			destroyAt_(inO.index_, inO.raw_());
			inO.index_ = Npos;
		}
	}

	template<typename T, typename D = std::decay_t<T>,
		typename = std::enable_if_t<!std::is_same_v<D, Variant>>,
		typename = std::enable_if_t<decayIndex_<T>() != Npos>>
	Variant(T&& inT) : index_(decayIndex_<T>()) {
		constexpr std::size_t I = decayIndex_<T>();
		new (raw_()) Alt<I>(std::forward<T>(inT));
	}

	~Variant() { destroy_(); }

	Variant& operator=(const Variant& inO) {
		if (this == &inO) {
			return *this;
		}
		Variant tmp(inO);
		*this = std::move(tmp);
		return *this;
	}

	Variant& operator=(Variant&& inO) noexcept(
		(std::is_nothrow_move_constructible_v<Ts> && ...) && (std::is_nothrow_destructible_v<Ts> && ...)) {
		if (this == &inO) {
			return *this;
		}
		destroy_();
		index_ = inO.index_;
		if (index_ != Npos) {
			moveConstructAt_(index_, raw_(), inO.raw_());
			destroyAt_(inO.index_, inO.raw_());
			inO.index_ = Npos;
		}
		return *this;
	}

	[[nodiscard]] std::size_t index() const noexcept { return index_; }

	[[nodiscard]] bool valuelessByException() const noexcept { return index_ == Npos; }

	template<std::size_t I>
	[[nodiscard]] Alt<I>& get() & {
		if (index_ != I) {
			throw std::bad_variant_access();
		}
		return *reinterpret_cast<Alt<I>*>(raw_());
	}
	template<std::size_t I>
	[[nodiscard]] const Alt<I>& get() const& {
		if (index_ != I) {
			throw std::bad_variant_access();
		}
		return *reinterpret_cast<const Alt<I>*>(raw_());
	}
	template<std::size_t I>
	[[nodiscard]] Alt<I>&& get() && {
		if (index_ != I) {
			throw std::bad_variant_access();
		}
		return std::move(*reinterpret_cast<Alt<I>*>(raw_()));
	}

	template<typename U>
	[[nodiscard]] U& get() & {
		constexpr std::size_t I = variantDetail::firstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return get<I>();
	}
	template<typename U>
	[[nodiscard]] const U& get() const& {
		constexpr std::size_t I = variantDetail::firstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return get<I>();
	}
	template<typename U>
	[[nodiscard]] U&& get() && {
		constexpr std::size_t I = variantDetail::firstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return std::move(get<I>());
	}

	template<typename U>
	[[nodiscard]] bool holdsAlternative() const noexcept {
		constexpr std::size_t I = variantDetail::firstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		return index_ == I;
	}

	template<typename F>
	void visit(F&& inF) & {
		if (index_ == Npos) {
			throw std::bad_variant_access();
		}
		dispatchVoidMut_(index_, std::forward<F>(inF), raw_(), Indices{});
	}

	template<typename F>
	void visit(F&& inF) const& {
		if (index_ == Npos) {
			throw std::bad_variant_access();
		}
		dispatchVoidConst_(index_, std::forward<F>(inF), raw_(), Indices{});
	}

	template<typename F>
	void visit(F&& inF) && {
		if (index_ == Npos) {
			throw std::bad_variant_access();
		}
		dispatchVoidRValue_(index_, std::forward<F>(inF), raw_(), Indices{});
		destroy_();
	}

	template<typename U, typename... Args>
	U& emplace(Args&&... inArgs) {
		destroy_();
		constexpr std::size_t I = variantDetail::firstIndexOf<U, Tuple>(Indices{});
		static_assert(I != Npos, "U not in variant");
		new (raw_()) U(std::forward<Args>(inArgs)...);
		index_ = I;
		return *reinterpret_cast<U*>(raw_());
	}

	void swap(Variant& inO) noexcept(
		(std::is_nothrow_move_constructible_v<Ts> && ...) && (std::is_nothrow_swappable_v<Ts> && ...)) {
		Variant tmp(std::move(*this));
		*this = std::move(inO);
		inO = std::move(tmp);
	}

	[[nodiscard]] std::variant<Ts...> stdVariant() const& {
		if (index_ == Npos) {
			return std::variant<Ts...>{};
		}
		return toStdVariant_(index_, raw_(), Indices{});
	}

	[[nodiscard]] std::variant<Ts...> stdVariant() && {
		Variant tmp(std::move(*this));
		return tmp.stdVariant();
	}
};

} // namespace oa
