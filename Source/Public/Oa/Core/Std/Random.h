#pragma once

// OaStdRandom — deterministic, cross-platform pseudo-random generator.
//
// Why this exists instead of <random>: the C++ *engines* (mt19937, …) are
// portable, but std::uniform_int_distribution / uniform_real_distribution /
// normal_distribution are NOT specified bit-for-bit — the same engine + seed
// produces DIFFERENT draws on libstdc++ vs libc++ vs MSVC. For an ML stack that
// is a silent reproducibility footgun (shuffles, init, dropout differ per
// toolchain). OaStdRandom is a fixed PCG32 core with OA-owned distribution math,
// so one seed reproduces the exact same stream on every platform. Small, fast,
// header-only, no exceptions.
//
// NOT cryptographically secure — never use for keys/nonces (use the Crypto layer).
//
// Reference: M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
// Statistically Good Algorithms for Random Number Generation" (2014).

#define OA_TYPES_H_SKIP_REST
#include <Oa/Core/Types.h>
#undef OA_TYPES_H_SKIP_REST

#include <cmath>   // sqrt / log / sin / cos for the Gaussian transform

class OaStdRandom {
public:
	// Fixed default seed → reproducible by default. Provide a seed for a
	// specific stream; InSeq selects an independent stream for the same seed
	// (two generators with the same seed but different InSeq never correlate).
	explicit OaStdRandom(OaU64 InSeed = 0x853C49E6748FEA9BULL, OaU64 InSeq = 1u) {
		Seed(InSeed, InSeq);
	}

	void Seed(OaU64 InSeed, OaU64 InSeq = 1u) {
		State_ = 0u;
		Inc_   = (InSeq << 1u) | 1u;   // must be odd
		NextU32();
		State_ += InSeed;
		NextU32();
		HasCachedGaussian_ = false;
	}

	// ── Core engine: PCG32 (32 random bits) ─────────────────────────────────
	OaU32 NextU32() {
		const OaU64 old = State_;
		State_ = old * 6364136223846793005ULL + Inc_;
		const OaU32 xorshifted = static_cast<OaU32>(((old >> 18u) ^ old) >> 27u);
		const OaU32 rot        = static_cast<OaU32>(old >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
	}

	OaU64 NextU64() {
		const OaU64 hi = static_cast<OaU64>(NextU32());
		const OaU64 lo = static_cast<OaU64>(NextU32());
		return (hi << 32u) | lo;
	}

	// ── Uniform reals in [0, 1) ─────────────────────────────────────────────
	OaF32 NextFloat() {
		// 24 mantissa bits → [0, 1).
		return static_cast<OaF32>(NextU32() >> 8) * (1.0F / 16777216.0F);       // 2^24
	}
	OaF64 NextDouble() {
		// 53 mantissa bits → [0, 1).
		return static_cast<OaF64>(NextU64() >> 11) * (1.0 / 9007199254740992.0); // 2^53
	}

	// ── Uniform integer in [InMin, InMax] inclusive (unbiased via rejection) ─
	OaI64 NextRange(OaI64 InMin, OaI64 InMax) {
		if (InMax <= InMin) {
			return InMin;
		}
		const OaU64 range = static_cast<OaU64>(InMax - InMin) + 1u;
		if (range == 0u) {
			// Full 64-bit span (min=INT64_MIN, max=INT64_MAX): every value valid.
			return static_cast<OaI64>(NextU64());
		}
		// Reject the low `2^64 mod range` values so the modulo is unbiased.
		const OaU64 reject = (0u - range) % range;
		OaU64 r;
		do {
			r = NextU64();
		} while (r < reject);
		return InMin + static_cast<OaI64>(r % range);
	}

	// Uniform real in [InMin, InMax).
	OaF64 NextRangeF(OaF64 InMin, OaF64 InMax) {
		return InMin + (InMax - InMin) * NextDouble();
	}

	// Bernoulli — true with probability InP.
	bool NextBool(OaF64 InP = 0.5) {
		return NextDouble() < InP;
	}

	// ── Gaussian (Box–Muller, caches the paired variate) ────────────────────
	OaF64 NextGaussian(OaF64 InMean = 0.0, OaF64 InStdDev = 1.0) {
		if (HasCachedGaussian_) {
			HasCachedGaussian_ = false;
			return InMean + InStdDev * CachedGaussian_;
		}
		OaF64 u1 = NextDouble();
		if (u1 <= 1e-300) {
			u1 = 1e-300;  // guard log(0)
		}
		const OaF64 u2    = NextDouble();
		const OaF64 twoPi = 6.283185307179586476925286766559;
		const OaF64 mag   = std::sqrt(-2.0 * std::log(u1));
		CachedGaussian_    = mag * std::sin(twoPi * u2);
		HasCachedGaussian_ = true;
		return InMean + InStdDev * (mag * std::cos(twoPi * u2));
	}

	// ── Fisher–Yates shuffle over a contiguous range ────────────────────────
	template<typename T>
	void Shuffle(T* InData, OaU64 InCount) {
		for (OaU64 i = InCount; i > 1u; --i) {
			const OaU64 j = static_cast<OaU64>(NextRange(0, static_cast<OaI64>(i) - 1));
			T tmp                 = static_cast<T&&>(InData[i - 1u]);
			InData[i - 1u]        = static_cast<T&&>(InData[j]);
			InData[j]             = static_cast<T&&>(tmp);
		}
	}

	// Raw state accessors — for checkpointing a training run's RNG exactly.
	[[nodiscard]] OaU64 RawState() const { return State_; }
	[[nodiscard]] OaU64 RawInc()   const { return Inc_; }
	void SetRawState(OaU64 InState, OaU64 InInc) {
		State_ = InState;
		Inc_   = InInc | 1u;   // keep increment odd
		HasCachedGaussian_ = false;
	}

private:
	OaU64 State_ = 0u;
	OaU64 Inc_   = 0u;
	OaF64 CachedGaussian_    = 0.0;
	bool  HasCachedGaussian_ = false;
};

// Short public alias (canonical name stays OaStdRandom; see OaStd.md naming).
using OaRandom = OaStdRandom;
