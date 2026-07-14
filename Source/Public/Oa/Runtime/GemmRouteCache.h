#pragma once

#include <Oa/Runtime/GemmTypes.h>
#include <Oa/Core/FileIo.h>
#include <unordered_map>
#include <mutex>
#include <fstream>
#include <type_traits>

// Route cache for per-device GEMM variant selection policy.
// Stores measured winners for exact operation contracts.
struct OaGemmRouteCache {
	static constexpr OaU64 FileMagic = 0x4F4147454D4D5243ULL; // "OAGEMMRC"
	static constexpr OaU32 FileVersion = 2;
	static constexpr const char* DefaultPath = "var/gemm_route_cache.bin";

	std::unordered_map<OaRouteCacheKey, OaRouteCacheValue, OaRouteCacheKeyHash> Map;
	mutable std::mutex Mutex;

	// Update cache with measured GPU time for a variant
	void Update(
		const OaRouteCacheKey& InKey,
		OaGemmKernel              InWinner,
		float                     InGpuTimeMs,
		OaU64                     InStep)
	{
		std::lock_guard<std::mutex> lock(Mutex);
		auto& entry = Map[InKey];
		entry.Winner = InWinner;
		entry.MedianGpuTimeMs = InGpuTimeMs;
		entry.P95GpuTimeMs = InGpuTimeMs;  // Simplified: single sample
		entry.SampleCount = 1;
		entry.LastUpdatedStep = InStep;
	}

	// Query cache for a winning variant
	[[nodiscard]] bool Query(
		const OaRouteCacheKey& InKey,
		OaGemmKernel&          OutWinner) const
	{
		std::lock_guard<std::mutex> lock(Mutex);
		auto it = Map.find(InKey);
		if (it != Map.end() && it->second.SampleCount > 0) {
			OutWinner = it->second.Winner;
			return true;
		}
		return false;
	}

	// Versioned, field-wise format. Never dump C++ structs directly: padding,
	// bool size, and enum layout are not a persistent file contract.
	[[nodiscard]] bool Save(const char* InPath) const {
		std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(Mutex));
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
			write(static_cast<OaU8>(key.Variant));
			write(key.M); write(key.N); write(key.K);
			write(static_cast<OaU8>(key.APrecision));
			write(static_cast<OaU8>(key.BPrecision));
			write(static_cast<OaU8>(key.Epilogue));
			write(static_cast<OaU8>(key.Training ? 1U : 0U));
			write(static_cast<OaU8>(key.UseTMA ? 1U : 0U));
			write(static_cast<OaU8>(value.Winner));
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
			OaU8 variant = 0, aPrec = 0, bPrec = 0, epilogue = 0;
			OaU8 training = 0, useTma = 0, winner = 0;
			if (!read(key.VendorId) || !read(key.DeviceId) || !read(key.DriverId)
				|| !read(key.DriverVersionHash) || !read(key.ShaderBuildId)
				|| !read(variant) || !read(key.M) || !read(key.N) || !read(key.K)
				|| !read(aPrec) || !read(bPrec) || !read(epilogue)
				|| !read(training) || !read(useTma) || !read(winner)
				|| !read(value.MedianGpuTimeMs) || !read(value.P95GpuTimeMs)
				|| !read(value.SampleCount) || !read(value.LastUpdatedStep)) {
				Map.clear();
				return false;
			}
			key.Variant = static_cast<OaGemmKernel>(variant);
			key.APrecision = static_cast<OaGemmPrecision>(aPrec);
			key.BPrecision = static_cast<OaGemmPrecision>(bPrec);
			key.Epilogue = static_cast<OaGemmEpilogue>(epilogue);
			key.Training = training != 0U;
			key.UseTMA = useTma != 0U;
			value.Winner = static_cast<OaGemmKernel>(winner);
			Map[key] = value;
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
