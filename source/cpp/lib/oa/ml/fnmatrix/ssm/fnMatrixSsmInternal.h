#pragma once

#include <oa/core/fnMatrix.h>
#include <oa/core/types.h>

#include <bit>

namespace oa {

namespace FnMatrixPrivate {

[[nodiscard]] inline oa::U64 ssmFingerprintAppend(
	oa::U64 inHash, oa::U32 inValue) noexcept
{
	constexpr oa::U64 FnvPrime = 1099511628211ULL;
	for (oa::U32 shift = 0U; shift < 32U; shift += 8U) {
		inHash ^= static_cast<oa::U8>(inValue >> shift);
		inHash *= FnvPrime;
	}
	return inHash;
}

[[nodiscard]] inline oa::U64 ssmConfigIdentity(
	const oa::SsmConfig& inConfig) noexcept
{
	oa::U64 hash = 14695981039346656037ULL;
	hash = ssmFingerprintAppend(hash, inConfig.batch);
	hash = ssmFingerprintAppend(hash, inConfig.seqLen);
	hash = ssmFingerprintAppend(hash, inConfig.nHeads);
	hash = ssmFingerprintAppend(hash, inConfig.nGroups);
	hash = ssmFingerprintAppend(hash, inConfig.headDim);
	hash = ssmFingerprintAppend(hash, inConfig.stateSize);
	hash = ssmFingerprintAppend(hash, inConfig.numRopeAngles);
	hash = ssmFingerprintAppend(hash, inConfig.mimoRank);
	hash = ssmFingerprintAppend(hash, inConfig.hasZ);
	hash = ssmFingerprintAppend(hash, inConfig.hasD);
	hash = ssmFingerprintAppend(hash, inConfig.hasOutNorm);
	return hash;
}

[[nodiscard]] inline oa::U64 mamba3PreprocessConfigIdentity(
	const oa::Mamba3PreprocessConfig& inConfig) noexcept
{
	oa::U64 hash = 14695981039346656037ULL;
	hash = ssmFingerprintAppend(hash, static_cast<oa::U32>(inConfig.dInner));
	hash = ssmFingerprintAppend(hash, static_cast<oa::U32>(inConfig.dState));
	hash = ssmFingerprintAppend(hash, static_cast<oa::U32>(inConfig.nHeads));
	hash = ssmFingerprintAppend(
		hash, static_cast<oa::U32>(inConfig.numRopeAngles));
	hash = ssmFingerprintAppend(hash, static_cast<oa::U32>(inConfig.nGroups));
	hash = ssmFingerprintAppend(hash, static_cast<oa::U32>(inConfig.mimoRank));
	hash = ssmFingerprintAppend(hash, std::bit_cast<oa::U32>(inConfig.eps));
	hash = ssmFingerprintAppend(hash, std::bit_cast<oa::U32>(inConfig.dtMin));
	hash = ssmFingerprintAppend(hash, std::bit_cast<oa::U32>(inConfig.dtMax));
	hash = ssmFingerprintAppend(hash, std::bit_cast<oa::U32>(inConfig.aFloor));
	return hash;
}

} // namespace FnMatrixPrivate

} // namespace oa
