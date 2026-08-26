#pragma once

#include <oa/runtime/gemmTypes.h>
#include <oa/core/filesystem.h>
#include <oa/core/envFlag.h>
#include <oa/core/std/hashMap.h>
#include <oa/core/std/sync.h>
#include <oa/core/std/typeTraits.h>

namespace oa {

// Private route cache for per-device GEMM variant selection policy.
// Stores measured winners for exact operation contracts.
struct GemmRouteCache {
	static constexpr oa::U64 FileMagic = 0x4F4147454D4D5243ULL; // "OAGEMMRC"
	static constexpr oa::U32 FileVersion = 6;
	static constexpr const char* DefaultPath = "var/gemm_route_cache.bin";

	oa::HashMap<oa::RouteCacheKey, oa::RouteCacheValue, oa::RouteCacheKeyHash> map;
	mutable oa::Mutex mutex;
	oa::U64 PublicationStep = 0;

	// publish a newly measured route using a sequence owned by this engine's
	// cache. The sequence is diagnostic metadata, not process-global state.
	void publish(
		const oa::RouteCacheKey& inKey,
		oa::U64 inWinner,
		float inMedianGpuTimeMs,
		float inP95GpuTimeMs,
		oa::U32 inSampleCount)
	{
		oa::ScopedLock lock(mutex);
		updateLocked(inKey, inWinner, inMedianGpuTimeMs, inP95GpuTimeMs,
			inSampleCount, ++PublicationStep);
	}

	// Update cache with measured GPU time for a variant
	void update(
		const oa::RouteCacheKey& inKey,
		oa::U64          inWinner,
		float                     inGpuTimeMs,
		oa::U64                     inStep)
	{
		update(inKey, inWinner, inGpuTimeMs, inGpuTimeMs, 1U, inStep);
	}

	// publish an aggregated measurement. Keeping the distribution metadata in
	// the cache lets policy reject a fast-but-unstable route instead of
	// pretending one median is the complete measurement.
	void update(
		const oa::RouteCacheKey& inKey,
		oa::U64          inWinner,
		float                     inMedianGpuTimeMs,
		float                     inP95GpuTimeMs,
		oa::U32                     inSampleCount,
		oa::U64                     inStep)
	{
		oa::ScopedLock lock(mutex);
		updateLocked(inKey, inWinner, inMedianGpuTimeMs, inP95GpuTimeMs,
			inSampleCount, inStep);
		PublicationStep = oa::max(PublicationStep, inStep);
	}

	// query cache for a winning variant
	[[nodiscard]] bool query(
		const oa::RouteCacheKey& inKey,
		oa::U64&      outWinner) const
	{
		oa::ScopedLock lock(mutex);
		auto it = map.find(inKey);
		if (it != map.end() && it->second.sampleCount > 0) {
			outWinner = it->second.winnerVariant;
			return true;
		}
		return false;
	}

	// Versioned, field-wise format. Never dump C++ structs directly: padding,
	// bool size, and enum layout are not a persistent file contract.
	[[nodiscard]] bool save(const char* inPath) const {
		oa::ScopedLock lock(mutex);
		oa::Vec<oa::U8> bytes;

		auto write = [&](const auto& value) {
			using T = oa::RemoveCvrefT<decltype(value)>;
			static_assert(oa::IsTriviallyCopyableV<T>);
			bytes.append(reinterpret_cast<const oa::U8*>(&value), sizeof(T));
		};
		write(FileMagic);
		write(FileVersion);
		oa::U64 count = static_cast<oa::U64>(map.size());
		write(count);

		for (const auto& [key, value] : map) {
			write(key.vendorId); write(key.deviceId); write(key.driverId);
			write(key.driverVersionHash); write(key.shaderBuildId);
			write(key.m); write(key.n); write(key.k);
			write(key.batchCount);
			write(key.aOffset); write(key.aRowStride); write(key.aColStride); write(key.aBatchStride);
			write(key.bOffset); write(key.bRowStride); write(key.bColStride); write(key.bBatchStride);
			write(key.cOffset); write(key.cRowStride); write(key.cColStride); write(key.cBatchStride);
			write(static_cast<oa::U8>(key.aPrecision));
			write(static_cast<oa::U8>(key.bPrecision));
			write(static_cast<oa::U8>(key.outputPrecision));
			write(static_cast<oa::U8>(key.requestedPrecision));
			write(static_cast<oa::U8>(key.epilogue));
			write(static_cast<oa::U8>(key.aContiguous ? 1U : 0U));
			write(static_cast<oa::U8>(key.bContiguous ? 1U : 0U));
			write(static_cast<oa::U8>(key.bTransposed ? 1U : 0U));
			write(static_cast<oa::U8>(key.requiresPreActivation ? 1U : 0U));
			write(static_cast<oa::U8>(key.training ? 1U : 0U));
			write(value.winnerVariant);
			write(value.medianGpuTimeMs); write(value.p95GpuTimeMs);
			write(value.sampleCount); write(value.lastUpdatedStep);
		}

		return oa::Filesystem::writeBinary(
			oa::Path(inPath), oa::Span<const oa::U8>(bytes.data(), bytes.size())).isOk();
	}

	// load cache from disk
	[[nodiscard]] bool load(const char* inPath) {
		oa::ScopedLock lock(mutex);
		auto loaded = oa::Filesystem::readBinary(oa::Path(inPath));
		if (loaded.isError()) return false;
		const auto& bytes = *loaded;
		oa::Usize offset = 0;

		auto read = [&](auto& value) {
			using T = oa::RemoveCvrefT<decltype(value)>;
			static_assert(oa::IsTriviallyCopyableV<T>);
			if (sizeof(T) > bytes.size() - oa::min(offset, bytes.size())) return false;
			oa::memcpy(&value, bytes.data() + offset, sizeof(T));
			offset += sizeof(T);
			return true;
		};
		oa::U64 magic = 0;
		oa::U32 version = 0;
		if (!read(magic) || !read(version) || magic != FileMagic || version != FileVersion) {
			return false;
		}
		oa::U64 count = 0;
		if (!read(count) || count > 1000000ULL) return false;

		map.clear();
		PublicationStep = 0;
		for (oa::U64 i = 0; i < count; ++i) {
			oa::RouteCacheKey key{};
			oa::RouteCacheValue value{};
			oa::U8 aPrec = 0, bPrec = 0, outputPrec = 0, requestedPrec = 0, epilogue = 0;
			oa::U8 aContiguous = 0, bContiguous = 0, bTransposed = 0;
			oa::U8 requiresPreActivation = 0, training = 0;
			if (!read(key.vendorId) || !read(key.deviceId) || !read(key.driverId)
				|| !read(key.driverVersionHash) || !read(key.shaderBuildId)
				|| !read(key.m) || !read(key.n) || !read(key.k)
				|| !read(key.batchCount)
				|| !read(key.aOffset) || !read(key.aRowStride) || !read(key.aColStride) || !read(key.aBatchStride)
				|| !read(key.bOffset) || !read(key.bRowStride) || !read(key.bColStride) || !read(key.bBatchStride)
				|| !read(key.cOffset) || !read(key.cRowStride) || !read(key.cColStride) || !read(key.cBatchStride)
				|| !read(aPrec) || !read(bPrec) || !read(outputPrec)
				|| !read(requestedPrec) || !read(epilogue)
				|| !read(aContiguous) || !read(bContiguous) || !read(bTransposed)
				|| !read(requiresPreActivation) || !read(training)
				|| !read(value.winnerVariant)
				|| !read(value.medianGpuTimeMs) || !read(value.p95GpuTimeMs)
				|| !read(value.sampleCount) || !read(value.lastUpdatedStep)) {
				map.clear();
				PublicationStep = 0;
				return false;
			}
			const bool invalidEnum = aPrec > static_cast<oa::U8>(oa::GemmPrecision::Bf16)
				|| bPrec > static_cast<oa::U8>(oa::GemmPrecision::Bf16)
				|| outputPrec > static_cast<oa::U8>(oa::GemmPrecision::Bf16)
				|| requestedPrec > static_cast<oa::U8>(oa::GemmPrecision::Bf16)
				|| epilogue > static_cast<oa::U8>(oa::GemmEpilogue::SiluDual);
			const bool invalidBool = aContiguous > 1U || bContiguous > 1U
				|| bTransposed > 1U || requiresPreActivation > 1U || training > 1U;
			const bool invalidValue = value.winnerVariant == oa::invalidMatmulVariantId
				|| !oa::isFinite(value.medianGpuTimeMs)
				|| !oa::isFinite(value.p95GpuTimeMs)
				|| value.medianGpuTimeMs < 0.0F || value.p95GpuTimeMs < 0.0F
				|| value.sampleCount == 0U;
			if (invalidEnum || invalidBool || invalidValue
				|| key.m == 0U || key.n == 0U || key.k == 0U || key.batchCount == 0U
				|| key.aRowStride == 0U || key.aColStride == 0U
				|| key.bRowStride == 0U || key.bColStride == 0U
				|| key.cRowStride == 0U || key.cColStride == 0U) {
				map.clear();
				PublicationStep = 0;
				return false;
			}
			key.aPrecision = static_cast<oa::GemmPrecision>(aPrec);
			key.bPrecision = static_cast<oa::GemmPrecision>(bPrec);
			key.outputPrecision = static_cast<oa::GemmPrecision>(outputPrec);
			key.requestedPrecision = static_cast<oa::GemmPrecision>(requestedPrec);
			key.epilogue = static_cast<oa::GemmEpilogue>(epilogue);
			key.aContiguous = aContiguous != 0U;
			key.bContiguous = bContiguous != 0U;
			key.bTransposed = bTransposed != 0U;
			key.requiresPreActivation = requiresPreActivation != 0U;
			key.training = training != 0U;
			if (!map.emplace(key, value).second) {
				map.clear();
				PublicationStep = 0;
				return false;
			}
			PublicationStep = oa::max(PublicationStep, value.lastUpdatedStep);
		}

		// Reject trailing bytes as a corrupt or incompatible cache rather than
		// accepting records written with a different contract. Never leave a
		// partially accepted map live when load reports failure.
		if (offset != bytes.size()) {
			map.clear();
			PublicationStep = 0;
			return false;
		}
		return true;
	}

	// Check if autotune mode is enabled (per-run benchmarking vs cache)
	// environment variable: OA_GEMM_AUTOTUNE=1 enables, 0/default disables
	[[nodiscard]] static bool isAutotuneEnabled() {
		return oa::EnvFlag::isSet("OA_GEMM_AUTOTUNE");
	}

private:
	void updateLocked(
		const oa::RouteCacheKey& inKey,
		oa::U64 inWinner,
		float inMedianGpuTimeMs,
		float inP95GpuTimeMs,
		oa::U32 inSampleCount,
		oa::U64 inStep)
	{
		auto found = map.find(inKey);
		if (found == map.end()) {
			found = map.emplace(inKey, oa::RouteCacheValue{}).first;
		}
		auto& entry = found->second;
		entry.winnerVariant = inWinner;
		entry.medianGpuTimeMs = inMedianGpuTimeMs;
		entry.p95GpuTimeMs = inP95GpuTimeMs;
		entry.sampleCount = inSampleCount;
		entry.lastUpdatedStep = inStep;
	}
};

} // namespace oa
