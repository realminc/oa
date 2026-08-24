#pragma once

// Native OA pair. This intentionally mirrors the small aggregate semantics of
// std::pair while keeping OA-owned containers independent from the STL ABI.

namespace oa {

template<typename A, typename B>
struct Pair {
	A first{};
	B second{};

	Pair() = default;
	Pair(const A& inFirst, const B& inSecond) : first(inFirst), second(inSecond) {}

	friend bool operator==(const Pair& inLeft, const Pair& inRight) {
		return inLeft.first == inRight.first && inLeft.second == inRight.second;
	}
	friend bool operator!=(const Pair& inLeft, const Pair& inRight) {
		return !(inLeft == inRight);
	}
};

template<typename A, typename B>
[[nodiscard]] inline Pair<A, B> makePair(const A& inFirst, const B& inSecond) {
	return Pair<A, B>(inFirst, inSecond);
}

} // namespace oa
