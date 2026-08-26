#pragma once

// Random — deterministic, cross-platform pseudo-random generator.
//
// Why this exists instead of <random>: the C++ *engines* (mt19937, …) are
// portable, but std::uniform_int_distribution / uniform_real_distribution /
// normal_distribution are NOT specified bit-for-bit — the same engine + seed
// produces DIFFERENT draws on libstdc++ vs libc++ vs MSVC. For an ML stack that
// is a silent reproducibility footgun (shuffles, init, dropout differ per
// toolchain). Random is a fixed PCG32 core with OA-owned distribution math,
// so one seed reproduces the exact same stream on every platform. Small, fast,
// header-only, no exceptions.
//
// NOT cryptographically secure — never use for keys/nonces (use the Crypto layer).
//
// Reference: M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
// Statistically Good Algorithms for Random Number Generation" (2014).

#define OA_TYPES_H_SKIP_REST
#include <oa/core/types.h>
#undef OA_TYPES_H_SKIP_REST

#include <oa/core/std/typeTraits.h>
#include <oa/core/std/scalarMath.h>

namespace oa {

class Random {
public:
	// Fixed default seed → reproducible by default. Provide a seed for a
	// specific stream; inSeq selects an independent stream for the same seed
	// (two generators with the same seed but different inSeq never correlate).
	explicit Random(oa::U64 inSeed = 0x853C49E6748FEA9BULL, oa::U64 inSeq = 1u) {
		seed(inSeed, inSeq);
	}

	void seed(oa::U64 inSeed, oa::U64 inSeq = 1u) {
		state_ = 0u;
		increment_   = (inSeq << 1u) | 1u;   // must be odd
		nextU32();
		state_ += inSeed;
		nextU32();
		hasCachedGaussian_ = false;
	}

	// ── Core engine: PCG32 (32 random bits) ─────────────────────────────────
	oa::U32 nextU32() {
		const oa::U64 old = state_;
		state_ = old * 6364136223846793005ULL + increment_;
		const oa::U32 xorshifted = static_cast<oa::U32>(((old >> 18u) ^ old) >> 27u);
		const oa::U32 rot        = static_cast<oa::U32>(old >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
	}

	oa::U64 nextU64() {
		const oa::U64 hi = static_cast<oa::U64>(nextU32());
		const oa::U64 lo = static_cast<oa::U64>(nextU32());
		return (hi << 32u) | lo;
	}

	// ── Uniform reals in [0, 1) ─────────────────────────────────────────────
	oa::F32 nextFloat() {
		// 24 mantissa bits → [0, 1).
		return static_cast<oa::F32>(nextU32() >> 8) * (1.0F / 16777216.0F);       // 2^24
	}
	oa::F64 nextDouble() {
		// 53 mantissa bits → [0, 1).
		return static_cast<oa::F64>(nextU64() >> 11) * (1.0 / 9007199254740992.0); // 2^53
	}

	// ── Uniform integer in [inMin, inMax] inclusive (unbiased via rejection) ─
	template<
		typename T,
		typename = oa::EnableIfT<oa::IsIntegralV<T> && (sizeof(T) < sizeof(oa::I64))>
	>
	T nextRange(T inMin, T inMax) {
		return static_cast<T>(nextRange(
			static_cast<oa::I64>(inMin),
			static_cast<oa::I64>(inMax)
		));
	}

	oa::I64 nextRange(oa::I64 inMin, oa::I64 inMax) {
		if (inMax <= inMin) {
			return inMin;
		}
		const oa::U64 range = static_cast<oa::U64>(inMax - inMin) + 1u;
		if (range == 0u) {
			// Full 64-bit span (min=INT64_MIN, max=INT64_MAX): every value valid.
			return static_cast<oa::I64>(nextU64());
		}
		// Reject the low `2^64 mod range` values so the modulo is unbiased.
		const oa::U64 reject = (0u - range) % range;
		oa::U64 r;
		do {
			r = nextU64();
		} while (r < reject);
		return inMin + static_cast<oa::I64>(r % range);
	}

	// Uniform real in [inMin, inMax).
	oa::F64 nextRange(oa::F64 inMin, oa::F64 inMax) {
		return inMin + (inMax - inMin) * nextDouble();
	}

	// Bernoulli — true with probability inP.
	bool nextBool(oa::F64 inP = 0.5) {
		return nextDouble() < inP;
	}

	// ── Gaussian (Box–Muller, caches the paired variate) ────────────────────
	oa::F64 nextGaussian(oa::F64 inMean = 0.0, oa::F64 inStdDev = 1.0) {
		if (hasCachedGaussian_) {
			hasCachedGaussian_ = false;
			return inMean + inStdDev * cachedGaussian_;
		}
		oa::F64 u1 = nextDouble();
		if (u1 <= 1e-300) {
			u1 = 1e-300;  // guard log(0)
		}
		const oa::F64 u2    = nextDouble();
		const oa::F64 twoPi = 6.283185307179586476925286766559;
		const oa::F64 mag   = oa::sqrt(-2.0 * oa::log(u1));
		cachedGaussian_    = mag * oa::sin(twoPi * u2);
		hasCachedGaussian_ = true;
		return inMean + inStdDev * (mag * oa::cos(twoPi * u2));
	}

	// ── Fisher–Yates shuffle over a contiguous range ────────────────────────
	template<typename T>
	void shuffle(T* inData, oa::U64 inCount) {
		for (oa::U64 i = inCount; i > 1u; --i) {
			const oa::U64 j = static_cast<oa::U64>(nextRange(0, static_cast<oa::I64>(i) - 1));
			T tmp                 = static_cast<T&&>(inData[i - 1u]);
			inData[i - 1u]        = static_cast<T&&>(inData[j]);
			inData[j]             = static_cast<T&&>(tmp);
		}
	}

	// Raw state accessors — for checkpointing a training run's RNG exactly.
	[[nodiscard]] oa::U64 rawState() const { return state_; }
	[[nodiscard]] oa::U64 rawInc()   const { return increment_; }
	void setRawState(oa::U64 inState, oa::U64 inInc) {
		state_ = inState;
		increment_   = inInc | 1u;   // keep increment odd
		hasCachedGaussian_ = false;
	}

private:
	oa::U64 state_ = 0u;
	oa::U64 increment_   = 0u;
	oa::F64 cachedGaussian_    = 0.0;
	bool  hasCachedGaussian_ = false;
};

} // namespace oa
