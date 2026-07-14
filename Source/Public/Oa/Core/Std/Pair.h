#pragma once

// OaStdPair<A,B> — the std::pair replacement. PascalCase members (First/Second),
// value-initialized like std::pair. Native aggregate, no <utility> dependency.

template<typename A, typename B>
struct OaStdPair {
	A First{};
	B Second{};

	OaStdPair() = default;
	OaStdPair(const A& InFirst, const B& InSecond) : First(InFirst), Second(InSecond) {}

	friend bool operator==(const OaStdPair& InL, const OaStdPair& InR) {
		return InL.First == InR.First && InL.Second == InR.Second;
	}
	friend bool operator!=(const OaStdPair& InL, const OaStdPair& InR) {
		return !(InL == InR);
	}
};

template<typename A, typename B>
[[nodiscard]] inline OaStdPair<A, B> OaStdMakePair(const A& InA, const B& InB) {
	return OaStdPair<A, B>(InA, InB);
}

// Short public alias (canonical name stays OaStdPair; see OaStd.md naming).
template<typename A, typename B>
using OaPair = OaStdPair<A, B>;
