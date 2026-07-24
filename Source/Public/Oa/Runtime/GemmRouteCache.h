#pragma once

#include <Oa/Runtime/GemmTypes.h>
#include <Oa/Core/Filesystem.h>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <type_traits>
#include <cmath>

// Route cache for per-device GEMM variant selection policy.
// Stores measured winners for exact operation contracts.
struct OaGemmRouteCache {
	static constexpr OaU64 FileMagic = 0x4F4147454D4D5243ULL; // "OAGEMMRC"
	static constexpr OaU32 FileVersion = 6;
	static constexpr const char* DefaultPath = "var/gemm_route_cache.bin";

	std::unordered_map<OaRouteCacheKey, OaRouteCacheValue, OaRouteCacheKeyHash> Map;
	mutable std::mutex Mutex;

	// Update cache with measured GPU time for a variant
	void Update(
		const OaRouteCacheKey& InKey,
		OaMatmulVariantId          InWinner,
		float                     InGpuTimeMs,
		OaU64                     InStep)
	{
		Update(InKey, InWinner, InGpuTimeMs, InGpuTimeMs, 1U, InStep);
	}

	// Publish an aggregated measurement. Keeping the distribution metadata in
	// the cache lets policy reject a fast-but-unstable route instead of
	// pretending one median is the complete measurement.
	void Update(
		const OaRouteCacheKey& InKey,
		OaMatmulVariantId          InWinner,
		float                     InMedianGpuTimeMs,
		float                     InP95GpuTimeMs,
		OaU32                     InSampleCount,
		OaU64                     InStep)
	{
		std::lock_guard<std::mutex> lock(Mutex);
		auto& entry = Map[InKey];
		entry.WinnerVariant = InWinner;
		entry.MedianGpuTimeMs = InMedianGpuTimeMs;
		entry.P95GpuTimeMs = InP95GpuTimeMs;
		entry.SampleCount = InSampleCount;
		entry.LastUpdatedStep = InStep;
	}

	// Query cache for a winning variant
	[[nodiscard]] bool Query(
		const OaRouteCacheKey& InKey,
		OaMatmulVariantId&      OutWinner) const
	{
		std::lock_guard<std::mutex> lock(Mutex);
		auto it = Map.find(InKey);
		if (it != Map.end() && it->second.SampleCount > 0) {
			OutWinner = it->second.WinnerVariant;
			return true;
		}
		return false;
	}

	// Versioned, field-wise format. Never dump C++ structs directly: padding,
	// bool size, and enum layout are not a persistent file contract.
	[[nodiscard]] bool Save(const char* InPath) const {
		std::lock_guard<std::mutex> lock(Mutex);
		std::ofstream file(InPath, std::ios::binary);
		if (!file) return false;

		auto write = [&](const auto& value) {
			using T = std::decay_t<decltype(value)>;
			static_assert(std::is_trivially_copyable_v<T>);
			file.write(reinterpret_cast<const char*>(&value), sizeof(T));
		};
		write(FileMagic);
		write(FileVersion);
		OaU64 count = static_cast<OaU64>(Map.size());
		write(count);

		for (const auto& [key, value] : Map) {
			write(key.VendorId); write(key.DeviceId); write(key.DriverId);
			write(key.DriverVersionHash); write(key.ShaderBuildId);
			write(key.M); write(key.N); write(key.K);
			write(key.BatchCount);
			write(key.AOffset); write(key.ARowStride); write(key.AColStride); write(key.ABatchStride);
			write(key.BOffset); write(key.BRowStride); write(key.BColStride); write(key.BBatchStride);
			write(key.COffset); write(key.CRowStride); write(key.CColStride); write(key.CBatchStride);
			write(static_cast<OaU8>(key.APrecision));
			write(static_cast<OaU8>(key.BPrecision));
			write(static_cast<OaU8>(key.OutputPrecision));
			write(static_cast<OaU8>(key.RequestedPrecision));
			write(static_cast<OaU8>(key.Epilogue));
			write(static_cast<OaU8>(key.AContiguous ? 1U : 0U));
			write(static_cast<OaU8>(key.BContiguous ? 1U : 0U));
			write(static_cast<OaU8>(key.BTransposed ? 1U : 0U));
			write(static_cast<OaU8>(key.RequiresPreActivation ? 1U : 0U));
			write(static_cast<OaU8>(key.Training ? 1U : 0U));
			write(value.WinnerVariant);
			write(value.MedianGpuTimeMs); write(value.P95GpuTimeMs);
			write(value.SampleCount); write(value.LastUpdatedStep);
		}

		return file.good();
	}

	// Load cache from disk
	[[nodiscard]] bool Load(const char* InPath) {
		std::lock_guard<std::mutex> lock(Mutex);
		std::ifstream file(InPath, std::ios::binary);
		if (!file) return false;

		auto read = [&](auto& value) {
			using T = std::decay_t<decltype(value)>;
			static_assert(std::is_trivially_copyable_v<T>);
			file.read(reinterpret_cast<char*>(&value), sizeof(T));
			return file.good();
		};
		OaU64 magic = 0;
		OaU32 version = 0;
		if (!read(magic) || !read(version) || magic != FileMagic || version != FileVersion) {
			return false;
		}
		OaU64 count = 0;
		if (!read(count) || count > 1000000ULL) return false;

		Map.clear();
		for (OaU64 i = 0; i < count; ++i) {
			OaRouteCacheKey key{};
			OaRouteCacheValue value{};
			OaU8 aPrec = 0, bPrec = 0, outputPrec = 0, requestedPrec = 0, epilogue = 0;
			OaU8 aContiguous = 0, bContiguous = 0, bTransposed = 0;
			OaU8 requiresPreActivation = 0, training = 0;
			if (!read(key.VendorId) || !read(key.DeviceId) || !read(key.DriverId)
				|| !read(key.DriverVersionHash) || !read(key.ShaderBuildId)
				|| !read(key.M) || !read(key.N) || !read(key.K)
				|| !read(key.BatchCount)
				|| !read(key.AOffset) || !read(key.ARowStride) || !read(key.AColStride) || !read(key.ABatchStride)
				|| !read(key.BOffset) || !read(key.BRowStride) || !read(key.BColStride) || !read(key.BBatchStride)
				|| !read(key.COffset) || !read(key.CRowStride) || !read(key.CColStride) || !read(key.CBatchStride)
				|| !read(aPrec) || !read(bPrec) || !read(outputPrec)
				|| !read(requestedPrec) || !read(epilogue)
				|| !read(aContiguous) || !read(bContiguous) || !read(bTransposed)
				|| !read(requiresPreActivation) || !read(training)
				|| !read(value.WinnerVariant)
				|| !read(value.MedianGpuTimeMs) || !read(value.P95GpuTimeMs)
				|| !read(value.SampleCount) || !read(value.LastUpdatedStep)) {
				Map.clear();
				return false;
			}
			const bool invalidEnum = aPrec > static_cast<OaU8>(OaGemmPrecision::Bf16)
				|| bPrec > static_cast<OaU8>(OaGemmPrecision::Bf16)
				|| outputPrec > static_cast<OaU8>(OaGemmPrecision::Bf16)
				|| requestedPrec > static_cast<OaU8>(OaGemmPrecision::Bf16)
				|| epilogue > static_cast<OaU8>(OaGemmEpilogue::SiluDual);
			const bool invalidBool = aContiguous > 1U || bContiguous > 1U
				|| bTransposed > 1U || requiresPreActivation > 1U || training > 1U;
			const bool invalidValue = value.WinnerVariant == OaInvalidMatmulVariantId
				|| !std::isfinite(value.MedianGpuTimeMs)
				|| !std::isfinite(value.P95GpuTimeMs)
				|| value.MedianGpuTimeMs < 0.0F || value.P95GpuTimeMs < 0.0F
				|| value.SampleCount == 0U;
			if (invalidEnum || invalidBool || invalidValue
				|| key.M == 0U || key.N == 0U || key.K == 0U || key.BatchCount == 0U
				|| key.ARowStride == 0U || key.AColStride == 0U
				|| key.BRowStride == 0U || key.BColStride == 0U
				|| key.CRowStride == 0U || key.CColStride == 0U) {
				Map.clear();
				return false;
			}
			key.APrecision = static_cast<OaGemmPrecision>(aPrec);
			key.BPrecision = static_cast<OaGemmPrecision>(bPrec);
			key.OutputPrecision = static_cast<OaGemmPrecision>(outputPrec);
			key.RequestedPrecision = static_cast<OaGemmPrecision>(requestedPrec);
			key.Epilogue = static_cast<OaGemmEpilogue>(epilogue);
			key.AContiguous = aContiguous != 0U;
			key.BContiguous = bContiguous != 0U;
			key.BTransposed = bTransposed != 0U;
			key.RequiresPreActivation = requiresPreActivation != 0U;
			key.Training = training != 0U;
			if (!Map.emplace(key, value).second) {
				Map.clear();
				return false;
			}
		}

		// Reject trailing bytes as a corrupt or incompatible cache rather than
		// accepting records written with a different contract. Never leave a
		// partially accepted map live when Load reports failure.
		if (file.peek() != std::ifstream::traits_type::eof()) {
			Map.clear();
			return false;
		}
		return true;
	}

	// Check if autotune mode is enabled (per-run benchmarking vs cache)
	// Environment variable: OA_GEMM_AUTOTUNE=1 enables, 0/default disables
	[[nodiscard]] static bool IsAutotuneEnabled() {
		const char* env = std::getenv("OA_GEMM_AUTOTUNE");
		if (!env) return false;
		std::string s(env);
		return s == "1" || s == "true" || s == "yes" || s == "on";
	}
};
