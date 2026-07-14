// Crypto benchmarks — CPU vs GPU batch performance measurement.
// Every GPU iteration records, submits, and synchronizes one complete batch;
// reported wall time therefore includes OA/Vulkan dispatch and submit overhead.
// Historical targets retained for context:
//   - SHAKE-256 batch: 20x speedup (10ms -> 500μs for 1000 hashes)
//   - Keccak-f[1600] batch: 25x speedup (5ms -> 200μs for 1000 states)
//   - Merkle root: 20x speedup (20ms -> 1ms for 1000 leaves)
//   - ML-DSA verify batch: 20x speedup (600ms -> 30ms for 1000 sigs)
//
// Run with: BenchHash --gtest_filter=OaVkEngineTestFixture.*GPU

#include "../../OaTest.h"
#include <Oa/Crypto/FnHash.h>
#include <Oa/Crypto/Hash.h>
#include <Oa/Crypto/Keccak.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>
#include <chrono>

#ifdef OA_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_RANGE_PUSH(name)
#define NVTX_RANGE_POP()
#endif

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

// ─── Helper: Format throughput ────────────────────────────────────────────

static OaString FormatThroughput(OaI64 InCount, double InMs) {
	double perSec = (InCount * 1000.0) / InMs;
	char buf[64];
	if (perSec >= 1e6) {
		snprintf(buf, sizeof(buf), "%.2f M/s", perSec / 1e6);
	} else if (perSec >= 1e3) {
		snprintf(buf, sizeof(buf), "%.2f K/s", perSec / 1e3);
	} else {
		snprintf(buf, sizeof(buf), "%.2f /s", perSec);
	}
	return OaString(buf);
}

// ─── Benchmark: SHAKE-256 Single (CPU Baseline) ──────────────────────────

TEST_F(OaVkEngineTestFixture, Shake256SingleCPU) {
	NVTX_RANGE_PUSH("BenchCrypto::Shake256SingleCPU");

	const OaI32 iterations = 1000;
	OaByte input[100];
	for (OaI32 i = 0; i < 100; ++i) input[i] = static_cast<OaByte>(i);

	auto start = Clock::now();
	for (OaI32 i = 0; i < iterations; ++i) {
		OaHash hash;
		OaShake256(input, 100, hash.Bytes.data(), 32);
		(void)hash;  // Prevent optimization
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgUs = (ms * 1000.0) / iterations;

	printf("  SHAKE-256 single (CPU): %.2f μs/hash, %s\n",
		avgUs, FormatThroughput(iterations, ms).CStr());

	NVTX_RANGE_POP();
}

// ─── Benchmark: SHAKE-256 Batch (GPU) ────────────────────────────────────

TEST_F(OaVkEngineTestFixture, Shake256BatchGPU) {

	NVTX_RANGE_PUSH("BenchCrypto::Shake256BatchGPU");

	const OaI32 batch_size = 1000;
	const OaI32 warmup = 10;
	const OaI32 iterations = 100;

	// Create batch input (1000 x 100 bytes)
	OaMatrix data = OaFnMatrix::Rand({batch_size, 100}, OaScalarType::UInt8);
	auto& ctx = OaContext::GetDefault();

	// Warmup
	for (OaI32 i = 0; i < warmup; ++i) {
		OaMatrix hash = OaFnHash::Shake256(data, 32);
		(void)hash;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		ctx.Clear();
	}

	// Benchmark
	auto start = Clock::now();
	for (OaI32 i = 0; i < iterations; ++i) {
		OaMatrix hash = OaFnHash::Shake256(data, 32);
		(void)hash;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		ctx.Clear();
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;
	double avgUsPerHash = (ms * 1000.0) / (iterations * batch_size);

	printf("  SHAKE-256 batch (GPU): %.2f ms/batch, %.2f μs/hash, %s\n",
		avgMs, avgUsPerHash, FormatThroughput(batch_size * iterations, ms).CStr());

	// Check speedup vs target (500μs for 1000 hashes = 0.5μs/hash)
	double target_us_per_hash = 0.5;
	double attainment = target_us_per_hash / avgUsPerHash;
	printf("  Target: 0.5 μs/hash, attainment: %.0f%% %s\n",
		attainment * 100.0, attainment >= 1.0 ? "✓" : "✗");

	NVTX_RANGE_POP();
}

// ─── Benchmark: Keccak-f[1600] Batch (GPU) ───────────────────────────────

TEST_F(OaVkEngineTestFixture, KeccakF1600BatchGPU) {

	NVTX_RANGE_PUSH("BenchCrypto::KeccakF1600BatchGPU");

	const OaI32 batch_size = 1000;
	const OaI32 warmup = 10;
	const OaI32 iterations = 100;

	// Create batch of Keccak states (1000 x 25 x uint64)
	OaMatrix states = OaFnMatrix::Zeros({batch_size, 25}, OaScalarType::UInt64);
	auto& ctx = OaContext::GetDefault();

	// Warmup
	for (OaI32 i = 0; i < warmup; ++i) {
		OaMatrix permuted = OaFnHash::KeccakF1600(states);
		(void)permuted;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		ctx.Clear();
	}

	// Benchmark
	auto start = Clock::now();
	for (OaI32 i = 0; i < iterations; ++i) {
		OaMatrix permuted = OaFnHash::KeccakF1600(states);
		(void)permuted;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		ctx.Clear();
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;
	double avgUsPerState = (ms * 1000.0) / (iterations * batch_size);

	printf("  Keccak-f[1600] batch (GPU): %.2f ms/batch, %.2f μs/state, %s\n",
		avgMs, avgUsPerState, FormatThroughput(batch_size * iterations, ms).CStr());

	// Check speedup vs target (200μs for 1000 states = 0.2μs/state)
	double target_us_per_state = 0.2;
	double attainment = target_us_per_state / avgUsPerState;
	printf("  Target: 0.2 μs/state, attainment: %.0f%% %s\n",
		attainment * 100.0, attainment >= 1.0 ? "✓" : "✗");

	NVTX_RANGE_POP();
}

// ─── Benchmark: Merkle Root (CPU Baseline) ───────────────────────────────

TEST_F(OaVkEngineTestFixture, MerkleRootCPU) {
	NVTX_RANGE_PUSH("BenchCrypto::MerkleRootCPU");

	const OaI32 num_leaves = 1024;  // Power of 2
	const OaI32 iterations = 10;

	// Create leaf hashes
	OaVec<OaHash> leaves;
	leaves.Reserve(num_leaves);
	for (OaI32 i = 0; i < num_leaves; ++i) {
		OaByte data[4] = {
			static_cast<OaByte>(i),
			static_cast<OaByte>(i >> 8),
			static_cast<OaByte>(i >> 16),
			static_cast<OaByte>(i >> 24)
		};
		OaHash hash;
		OaShake256(data, 4, hash.Bytes.data(), 32);
		leaves.PushBack(hash);
	}

	auto start = Clock::now();
	for (OaI32 iter = 0; iter < iterations; ++iter) {
		// Binary tree reduction
		OaVec<OaHash> current = leaves;
		while (current.Size() > 1) {
			OaVec<OaHash> next;
			next.Reserve(current.Size() / 2);
			for (OaUsize i = 0; i + 1 < current.Size(); i += 2) {
				OaByte combined[64];
				std::memcpy(combined, current[i].Bytes.data(), 32);
				std::memcpy(combined + 32, current[i + 1].Bytes.data(), 32);
				OaHash combinedHash;
				OaShake256(combined, 64, combinedHash.Bytes.data(), 32);
				next.PushBack(combinedHash);
			}
			current = std::move(next);
		}
		(void)current[0];  // Root hash
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;

	printf("  Merkle root (CPU, %d leaves): %.2f ms/tree\n", num_leaves, avgMs);

	NVTX_RANGE_POP();
}

// ─── Benchmark: Merkle Root (GPU) ─────────────────────────────────────────

TEST_F(OaVkEngineTestFixture, MerkleRootGPU) {

	NVTX_RANGE_PUSH("BenchCrypto::MerkleRootGPU");

	const OaI32 num_leaves = 1024;
	const OaI32 warmup = 10;
	const OaI32 iterations = 100;

	// Create leaf hashes (1024 x 32 bytes)
	OaMatrix leaves = OaFnMatrix::Rand({num_leaves, 32}, OaScalarType::UInt8);
	auto& ctx = OaContext::GetDefault();

	// Warmup
	for (OaI32 i = 0; i < warmup; ++i) {
		OaMatrix root = OaFnHash::MerkleRoot(leaves);
		(void)root;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		ctx.Clear();
	}

	// Benchmark
	auto start = Clock::now();
	for (OaI32 i = 0; i < iterations; ++i) {
		OaMatrix root = OaFnHash::MerkleRoot(leaves);
		(void)root;
		ASSERT_TRUE(ctx.Execute().IsOk());
		ASSERT_TRUE(ctx.Sync().IsOk());
		ctx.Clear();
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;

	printf("  Merkle root (GPU, %d leaves): %.2f ms/tree, %s\n",
		num_leaves, avgMs, FormatThroughput(iterations, ms).CStr());

	// Check speedup vs target (1ms for 1000 leaves)
	double target_ms = 1.0;
	double attainment = target_ms / avgMs;
	printf("  Target: 1.0 ms/tree, attainment: %.0f%% %s\n",
		attainment * 100.0, attainment >= 1.0 ? "✓" : "✗");

	NVTX_RANGE_POP();
}

// ─── Benchmark: CPU vs GPU Comparison ─────────────────────────────────────

TEST_F(OaVkEngineTestFixture, CPUvsGPUComparison) {

	printf("\n");
	printf("╔════════════════════════════════════════════════════════════════╗\n");
	printf("║          OA Crypto Performance Summary                         ║\n");
	printf("╠════════════════════════════════════════════════════════════════╣\n");
	printf("║ Operation              │ Target    │ Actual    │ Status       ║\n");
	printf("╠════════════════════════════════════════════════════════════════╣\n");
	printf("║ SHAKE-256 batch (1K)   │ 20x       │ TBD       │ ⏳ Pending  ║\n");
	printf("║ Keccak-f[1600] (1K)    │ 25x       │ TBD       │ ⏳ Pending  ║\n");
	printf("║ Merkle root (1K)       │ 20x       │ TBD       │ ⏳ Pending  ║\n");
	printf("║ ML-DSA verify (1K)     │ 20x       │ TBD       │ ⏳ Pending  ║\n");
	printf("╚════════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	printf("Run individual benchmarks to populate this table.\n");
	printf("Profile with: nsys profile --trace=vulkan,osrt,nvtx \\\n");
	printf("              --vulkan-gpu-workload=individual \\\n");
	printf("              --gpu-metrics-device=all \\\n");
	printf("              ./OaTest --gtest_filter=BenchCrypto.*\n");
}
