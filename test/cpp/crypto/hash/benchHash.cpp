// Crypto benchmarks — CPU vs GPU batch performance measurement.
// Every GPU iteration records, submits, and synchronizes one complete batch;
// reported wall time therefore includes OA/vulkan dispatch and submit overhead.
// Historical targets retained for context:
//   - SHAKE-256 batch: 20x speedup (10ms -> 500μs for 1000 hashes)
//   - Keccak-f[1600] batch: 25x speedup (5ms -> 200μs for 1000 states)
//   - Merkle root: 20x speedup (20ms -> 1ms for 1000 leaves)
//   - ML-DSA verify batch: 20x speedup (600ms -> 30ms for 1000 sigs)
//
// run with: BenchHash --gtest_filter=VkEngineTestFixture.*GPU

#include "../../oaTest.h"
#include <oa/crypto/fnHash.h>
#include <oa/crypto/hash.h>
#include <oa/crypto/keccak.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>
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

// ─── Helper: format throughput ────────────────────────────────────────────

static oa::String formatThroughput(oa::I64 inCount, double inMs) {
	double perSec = (inCount * 1000.0) / inMs;
	char buf[64];
	if (perSec >= 1e6) {
		snprintf(buf, sizeof(buf), "%.2f M/s", perSec / 1e6);
	} else if (perSec >= 1e3) {
		snprintf(buf, sizeof(buf), "%.2f K/s", perSec / 1e3);
	} else {
		snprintf(buf, sizeof(buf), "%.2f /s", perSec);
	}
	return oa::String(buf);
}

// ─── Benchmark: SHAKE-256 single (CPU baseline) ──────────────────────────

TEST_F(VkEngineTestFixture, Shake256SingleCPU) {
	NVTX_RANGE_PUSH("BenchCrypto::Shake256SingleCPU");

	const oa::I32 iterations = 1000;
	oa::Byte input[100];
	for (oa::I32 i = 0; i < 100; ++i) input[i] = static_cast<oa::Byte>(i);

	auto start = Clock::now();
	for (oa::I32 i = 0; i < iterations; ++i) {
		oa::Hash hash;
		oa::shake256(input, 100, hash.bytes.data(), 32);
		(void)hash;  // Prevent optimization
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgUs = (ms * 1000.0) / iterations;

	printf("  SHAKE-256 single (CPU): %.2f μs/hash, %s\n",
		avgUs, formatThroughput(iterations, ms).cStr());

	NVTX_RANGE_POP();
}

// ─── Benchmark: SHAKE-256 Batch (GPU) ────────────────────────────────────

TEST_F(VkEngineTestFixture, Shake256BatchGPU) {

	NVTX_RANGE_PUSH("BenchCrypto::Shake256BatchGPU");

	const oa::I32 batch_size = 1000;
	const oa::I32 warmup = 10;
	const oa::I32 iterations = 100;

	// Create batch input (1000 x 100 bytes)
	oa::Matrix data = oa::FnMatrix::rand({batch_size, 100}, oa::ScalarType::UInt8);
	auto& ctx = oa::ExecutionSession::getActive();

	// Warmup
	for (oa::I32 i = 0; i < warmup; ++i) {
		oa::Matrix hash = oa::FnHash::shake256(data, 32);
		(void)hash;
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		ctx.clear();
	}

	// Benchmark
	auto start = Clock::now();
	for (oa::I32 i = 0; i < iterations; ++i) {
		oa::Matrix hash = oa::FnHash::shake256(data, 32);
		(void)hash;
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		ctx.clear();
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;
	double avgUsPerHash = (ms * 1000.0) / (iterations * batch_size);

	printf("  SHAKE-256 batch (GPU): %.2f ms/batch, %.2f μs/hash, %s\n",
		avgMs, avgUsPerHash, formatThroughput(batch_size * iterations, ms).cStr());

	// Check speedup vs target (500μs for 1000 hashes = 0.5μs/hash)
	double target_us_per_hash = 0.5;
	double attainment = target_us_per_hash / avgUsPerHash;
	printf("  target: 0.5 μs/hash, attainment: %.0f%% %s\n",
		attainment * 100.0, attainment >= 1.0 ? "✓" : "✗");

	NVTX_RANGE_POP();
}

// ─── Benchmark: Keccak-f[1600] Batch (GPU) ───────────────────────────────

TEST_F(VkEngineTestFixture, KeccakF1600BatchGPU) {

	NVTX_RANGE_PUSH("BenchCrypto::KeccakF1600BatchGPU");

	const oa::I32 batch_size = 1000;
	const oa::I32 warmup = 10;
	const oa::I32 iterations = 100;

	// Create batch of Keccak states (1000 x 25 x uint64)
	oa::Matrix states = oa::FnMatrix::zeros({batch_size, 25}, oa::ScalarType::UInt64);
	auto& ctx = oa::ExecutionSession::getActive();

	// Warmup
	for (oa::I32 i = 0; i < warmup; ++i) {
		oa::Matrix permuted = oa::FnHash::keccakF1600(states);
		(void)permuted;
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		ctx.clear();
	}

	// Benchmark
	auto start = Clock::now();
	for (oa::I32 i = 0; i < iterations; ++i) {
		oa::Matrix permuted = oa::FnHash::keccakF1600(states);
		(void)permuted;
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		ctx.clear();
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;
	double avgUsPerState = (ms * 1000.0) / (iterations * batch_size);

	printf("  Keccak-f[1600] batch (GPU): %.2f ms/batch, %.2f μs/state, %s\n",
		avgMs, avgUsPerState, formatThroughput(batch_size * iterations, ms).cStr());

	// Check speedup vs target (200μs for 1000 states = 0.2μs/state)
	double target_us_per_state = 0.2;
	double attainment = target_us_per_state / avgUsPerState;
	printf("  target: 0.2 μs/state, attainment: %.0f%% %s\n",
		attainment * 100.0, attainment >= 1.0 ? "✓" : "✗");

	NVTX_RANGE_POP();
}

// ─── Benchmark: Merkle root (CPU baseline) ───────────────────────────────

TEST_F(VkEngineTestFixture, MerkleRootCPU) {
	NVTX_RANGE_PUSH("BenchCrypto::MerkleRootCPU");

	const oa::I32 num_leaves = 1024;  // Power of 2
	const oa::I32 iterations = 10;

	// Create leaf hashes
	oa::Vec<oa::Hash> leaves;
	leaves.reserve(num_leaves);
	for (oa::I32 i = 0; i < num_leaves; ++i) {
		oa::Byte data[4] = {
			static_cast<oa::Byte>(i),
			static_cast<oa::Byte>(i >> 8),
			static_cast<oa::Byte>(i >> 16),
			static_cast<oa::Byte>(i >> 24)
		};
		oa::Hash hash;
		oa::shake256(data, 4, hash.bytes.data(), 32);
		leaves.pushBack(hash);
	}

	auto start = Clock::now();
	for (oa::I32 iter = 0; iter < iterations; ++iter) {
		// Binary tree reduction
		oa::Vec<oa::Hash> current = leaves;
		while (current.size() > 1) {
			oa::Vec<oa::Hash> next;
			next.reserve(current.size() / 2);
			for (oa::Usize i = 0; i + 1 < current.size(); i += 2) {
				oa::Byte combined[64];
				std::memcpy(combined, current[i].bytes.data(), 32);
				std::memcpy(combined + 32, current[i + 1].bytes.data(), 32);
				oa::Hash combinedHash;
				oa::shake256(combined, 64, combinedHash.bytes.data(), 32);
				next.pushBack(combinedHash);
			}
			current = std::move(next);
		}
		(void)current[0];  // root hash
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;

	printf("  Merkle root (CPU, %d leaves): %.2f ms/tree\n", num_leaves, avgMs);

	NVTX_RANGE_POP();
}

// ─── Benchmark: Merkle root (GPU) ─────────────────────────────────────────

TEST_F(VkEngineTestFixture, MerkleRootGPU) {

	NVTX_RANGE_PUSH("BenchCrypto::MerkleRootGPU");

	const oa::I32 num_leaves = 1024;
	const oa::I32 warmup = 10;
	const oa::I32 iterations = 100;

	// Create leaf hashes (1024 x 32 bytes)
	oa::Matrix leaves = oa::FnMatrix::rand({num_leaves, 32}, oa::ScalarType::UInt8);
	auto& ctx = oa::ExecutionSession::getActive();

	// Warmup
	for (oa::I32 i = 0; i < warmup; ++i) {
		oa::Matrix root = oa::FnHash::merkleRoot(leaves);
		(void)root;
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		ctx.clear();
	}

	// Benchmark
	auto start = Clock::now();
	for (oa::I32 i = 0; i < iterations; ++i) {
		oa::Matrix root = oa::FnHash::merkleRoot(leaves);
		(void)root;
		ASSERT_TRUE(testSubmitAndWait(ctx).isOk());
		ctx.clear();
	}
	auto end = Clock::now();

	double ms = Duration(end - start).count();
	double avgMs = ms / iterations;

	printf("  Merkle root (GPU, %d leaves): %.2f ms/tree, %s\n",
		num_leaves, avgMs, formatThroughput(iterations, ms).cStr());

	// Check speedup vs target (1ms for 1000 leaves)
	double target_ms = 1.0;
	double attainment = target_ms / avgMs;
	printf("  target: 1.0 ms/tree, attainment: %.0f%% %s\n",
		attainment * 100.0, attainment >= 1.0 ? "✓" : "✗");

	NVTX_RANGE_POP();
}

// ─── Benchmark: CPU vs GPU Comparison ─────────────────────────────────────

TEST_F(VkEngineTestFixture, CPUvsGPUComparison) {

	printf("\n");
	printf("╔════════════════════════════════════════════════════════════════╗\n");
	printf("║          OA Crypto Performance summary                         ║\n");
	printf("╠════════════════════════════════════════════════════════════════╣\n");
	printf("║ Operation              │ target    │ actual    │ status       ║\n");
	printf("╠════════════════════════════════════════════════════════════════╣\n");
	printf("║ SHAKE-256 batch (1K)   │ 20x       │ TBD       │ ⏳ pending  ║\n");
	printf("║ Keccak-f[1600] (1K)    │ 25x       │ TBD       │ ⏳ pending  ║\n");
	printf("║ Merkle root (1K)       │ 20x       │ TBD       │ ⏳ pending  ║\n");
	printf("║ ML-DSA verify (1K)     │ 20x       │ TBD       │ ⏳ pending  ║\n");
	printf("╚════════════════════════════════════════════════════════════════╝\n");
	printf("\n");
	printf("run individual benchmarks to populate this table.\n");
	printf("profile with: nsys profile --trace=vulkan,osrt,nvtx \\\n");
	printf("              --vulkan-gpu-workload=individual \\\n");
	printf("              --gpu-metrics-device=all \\\n");
	printf("              ./OaTest --gtest_filter=BenchCrypto.*\n");
}
