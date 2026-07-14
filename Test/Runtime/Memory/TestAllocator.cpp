#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <cstring>

#include <Oa/Core/Memory.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Pool.h>

#include <Oa/Runtime/Vma/Slab.h>
#include <Oa/Core/Thread.h>
#include <Oa/Crypto/Buffer.h>

TEST(Preamble, SuiteInfo) {
	fprintf(stderr,
		"  [test_allocator] runtime: slab/spinlock (CPU) + OaVma/inference pool (Vulkan)\n");
	SUCCEED();
}

// Slab tests (pure CPU, no Vulkan)
TEST(Slab, AllocFree64Slots) {
	alignas(64) OaU8 backing[64 * 1024];
	OaVkSlab slab;
	slab.Init(backing, 1024, 64);

	EXPECT_EQ(slab.FreeCount(), 64u);
	EXPECT_FALSE(slab.IsFull());
	EXPECT_TRUE(slab.IsEmpty());

	OaVec<void*> ptrs;
	for (OaU32 i = 0; i < 64; ++i) {
		void* p = slab.Alloc();
		ASSERT_NE(p, nullptr) << "alloc failed at slot " << i;
		ptrs.PushBack(p);
	}

	EXPECT_EQ(slab.FreeCount(), 0u);
	EXPECT_TRUE(slab.IsFull());
	EXPECT_FALSE(slab.IsEmpty());

	EXPECT_EQ(slab.Alloc(), nullptr);

	for (void* p : ptrs) {
		slab.Free(p);
	}
	EXPECT_EQ(slab.FreeCount(), 64u);
	EXPECT_TRUE(slab.IsEmpty());
}

TEST(Slab, SlotPointersAreDistinct) {
	alignas(64) OaU8 backing[32 * 256];
	OaVkSlab slab;
	slab.Init(backing, 256, 32);

	OaVec<void*> ptrs;
	for (OaU32 i = 0; i < 32; ++i) {
		ptrs.PushBack(slab.Alloc());
	}

	for (OaU32 i = 0; i < 32; ++i) {
		for (OaU32 j = i + 1; j < 32; ++j) {
			EXPECT_NE(ptrs[i], ptrs[j]) << "slots " << i << " and " << j << " overlap";
		}
	}

	for (void* p : ptrs) slab.Free(p);
}

TEST(Slab, SmallCapacity) {
	alignas(64) OaU8 backing[8 * 64];
	OaVkSlab slab;
	slab.Init(backing, 64, 8);

	EXPECT_EQ(slab.FreeCount(), 8u);
	for (OaU32 i = 0; i < 8; ++i) {
		ASSERT_NE(slab.Alloc(), nullptr);
	}
	EXPECT_TRUE(slab.IsFull());
	EXPECT_EQ(slab.Alloc(), nullptr);
}

TEST(Slab, BenchmarkAllocFree) {
	alignas(64) OaU8 backing[64 * 1024];
	OaVkSlab slab;
	slab.Init(backing, 1024, 64);

	constexpr OaU32 kIters = 100000;
	void* ptrs[64];

	auto t0 = std::chrono::high_resolution_clock::now();
	for (OaU32 iter = 0; iter < kIters; ++iter) {
		for (OaU32 i = 0; i < 64; ++i) {
			ptrs[i] = slab.Alloc();
		}
		for (OaU32 i = 0; i < 64; ++i) {
			slab.Free(ptrs[i]);
		}
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	OaF64 nsPerOp = static_cast<OaF64>(us * 1000) / static_cast<OaF64>(kIters * 128);
	fprintf(stderr, "  Slab alloc+free: %.1f ns/op (%u iters x 128 ops)\n", nsPerOp, kIters);
	EXPECT_LT(nsPerOp, 50.0);
}

// Spinlock tests (pure CPU)

TEST(Spinlock, LockUnlock) {
	OaSpinlock lock;
	lock.Lock();
	lock.Unlock();
	SUCCEED();
}

TEST(Spinlock, TryLock) {
	OaSpinlock lock;
	EXPECT_TRUE(lock.TryLock());
	EXPECT_FALSE(lock.TryLock());
	lock.Unlock();
	EXPECT_TRUE(lock.TryLock());
	lock.Unlock();
}

TEST(Spinlock, Guard) {
	OaSpinlock lock;
	{
		OaSpinlockGuard guard(lock);
		EXPECT_FALSE(lock.TryLock());
	}
	EXPECT_TRUE(lock.TryLock());
	lock.Unlock();
}

// Vulkan allocator tests (require GPU)

static OaUniquePtr<OaComputeEngine> CreateTestEngine() {
	auto result = OaComputeEngine::Create({.AppName = "test_allocator", .MeshVulkanIndices = {}});
	return std::move(result.GetValue());   // move the owning pointer out (engine is pinned)
}

TEST(Allocator, AllocBarFallback) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;
	{
		const OaStringView name = rt.DeviceName();
		fprintf(stderr, "  [test_allocator] device: %.*s\n",
			static_cast<int>(name.size()), name.data());
	}

	auto result = rt.Allocator.AllocBar(4096);
	ASSERT_TRUE(result.IsOk()) << result.GetStatus().ToString();
	auto& buf = result.GetValue();

	EXPECT_NE(buf.Buffer, nullptr);
	EXPECT_NE(buf.MappedPtr, nullptr);
	EXPECT_EQ(buf.Size, 4096u);

	if (rt.HasSAM()) {
		EXPECT_TRUE(buf.IsBar());
		fprintf(stderr, "  AllocBar: SAM detected, BAR flag set\n");
	} else {
		fprintf(stderr, "  AllocBar: no SAM, fell back to host-visible\n");
	}

	rt.Allocator.Free(buf);
	rt.Destroy();
}

TEST(Allocator, UploadWeightsCorrectness) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	constexpr OaU32 N = 1024;
	constexpr OaU64 size = N * sizeof(OaF32);

	auto result = rt.Allocator.AllocBar(size);
	ASSERT_TRUE(result.IsOk());
	auto& buf = result.GetValue();

	OaVec<OaF32> src(N);
	for (OaU32 i = 0; i < N; ++i) {
		src[i] = static_cast<OaF32>(i) * 0.5f;
	}

	auto status = rt.Allocator.UploadWeights(buf, src.Data(), size);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	auto* dst = static_cast<OaF32*>(buf.MappedPtr);
	for (OaU32 i = 0; i < N; ++i) {
		EXPECT_FLOAT_EQ(dst[i], src[i]) << "mismatch at " << i;
	}

	rt.Allocator.Free(buf);
	rt.Destroy();
}

TEST(Allocator, BudgetQuery) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	auto stats = rt.Allocator.GetStats();
	fprintf(stderr, "  Budget: used=%llu MB, budget=%llu MB\n",
		(unsigned long long)(stats.UsedBytes / (1024 * 1024)),
		(unsigned long long)(stats.BudgetBytes / (1024 * 1024)));

	EXPECT_GT(stats.BudgetBytes, 0u);

	rt.Destroy();
}

TEST(Allocator, BenchmarkVmaVsSlab) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	constexpr OaU32 kVmaIters = 1000;
	constexpr OaU64 kBufSize = 1024;

	auto t0 = std::chrono::high_resolution_clock::now();
	for (OaU32 i = 0; i < kVmaIters; ++i) {
		auto r = rt.Allocator.AllocHostVisible(kBufSize);
		if (r.IsOk()) {
			auto& b = r.GetValue();
			rt.Allocator.Free(b);
		}
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	auto vmaUs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	OaF64 vmaNsPerOp = static_cast<OaF64>(vmaUs * 1000) / static_cast<OaF64>(kVmaIters);

	alignas(64) OaU8 backing[64 * kBufSize];
	OaVkSlab slab;
	slab.Init(backing, kBufSize, 64);

	constexpr OaU32 kSlabIters = 100000;
	auto t2 = std::chrono::high_resolution_clock::now();
	for (OaU32 i = 0; i < kSlabIters; ++i) {
		void* p = slab.Alloc();
		slab.Free(p);
	}
	auto t3 = std::chrono::high_resolution_clock::now();
	auto slabUs = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
	OaF64 slabNsPerOp = static_cast<OaF64>(slabUs * 1000) / static_cast<OaF64>(kSlabIters);

	fprintf(stderr, "  VMA alloc+free: %.0f ns/op (%u iters)\n", vmaNsPerOp, kVmaIters);
	fprintf(stderr, "  Slab alloc+free: %.1f ns/op (%u iters)\n", slabNsPerOp, kSlabIters);
	if (slabNsPerOp > 0) {
		fprintf(stderr, "  Speedup: %.0fx\n", vmaNsPerOp / slabNsPerOp);
	}

	rt.Destroy();
}

// Inference pool tests (require GPU)

TEST(InferencePool, PreAllocatedStablePointers) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 4;
	cfg.ActivationSize = 512;
	cfg.MaxBatchSize = 2;
	cfg.OutputRingBytes = 1024;

	OaVkInferencePool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	EXPECT_EQ(pool.NumLayers(), 4u);
	EXPECT_EQ(pool.MaxBatchSize(), 2u);
	EXPECT_EQ(pool.ActivationSize(), 512u);
	EXPECT_EQ(pool.TotalActivationBytes(), 4u * 2u * 512u);

	void* p00a = pool.GetActivation(0, 0);
	void* p00b = pool.GetActivation(0, 0);
	EXPECT_EQ(p00a, p00b);

	void* p31a = pool.GetActivation(3, 1);
	void* p31b = pool.GetActivation(3, 1);
	EXPECT_EQ(p31a, p31b);

	void* ptrs[8];
	for (OaU32 l = 0; l < 4; ++l) {
		for (OaU32 b = 0; b < 2; ++b) {
			ptrs[l * 2 + b] = pool.GetActivation(l, b);
		}
	}
	for (OaU32 i = 0; i < 8; ++i) {
		for (OaU32 j = i + 1; j < 8; ++j) {
			EXPECT_NE(ptrs[i], ptrs[j]) << "activation slots " << i << " and " << j << " overlap";
		}
	}

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(InferencePool, RingBufferWrapAround) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 1;
	cfg.ActivationSize = 64;
	cfg.MaxBatchSize = 1;
	cfg.OutputRingBytes = 64 * 4;

	OaVkInferencePool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	OaVec<void*> ringPtrs;
	for (OaU32 i = 0; i < 4; ++i) {
		void* p = pool.NextOutputSlot();
		ASSERT_NE(p, nullptr);
		ringPtrs.PushBack(p);
		pool.ConsumeOutputSlot();
	}

	void* wrap = pool.NextOutputSlot();
	EXPECT_EQ(wrap, ringPtrs[0]);

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(InferencePool, ResetSessionZeros) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 2;
	cfg.ActivationSize = 128;
	cfg.MaxBatchSize = 1;
	cfg.OutputRingBytes = 0;

	OaVkInferencePool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk());

	auto* p = static_cast<OaF32*>(pool.GetActivation(0, 0));
	for (OaU32 i = 0; i < 32; ++i) p[i] = 1.0f;

	pool.ResetSession();

	for (OaU32 i = 0; i < 32; ++i) {
		EXPECT_FLOAT_EQ(p[i], 0.0f) << "not zeroed at " << i;
	}

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(InferencePool, WeightPinning) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	auto bufResult = rt.Allocator.AllocHostVisible(4096);
	ASSERT_TRUE(bufResult.IsOk());
	auto& buf = bufResult.GetValue();

	OaVkInferencePoolConfig cfg{};
	cfg.NumLayers = 1;
	cfg.ActivationSize = 64;
	cfg.MaxBatchSize = 1;
	cfg.OutputRingBytes = 0;

	OaVkInferencePool pool;
	ASSERT_TRUE(pool.Init(rt.Allocator, cfg).IsOk());

	auto pinStatus = pool.PinWeights(buf);
	EXPECT_TRUE(pinStatus.IsOk());

	pool.Destroy(rt.Allocator);
	rt.Allocator.Free(buf);
	rt.Destroy();
}

// Secure buffer tests (pure CPU)

TEST(SecureBuffer, ZeroOnDestroy) {
	alignas(64) OaU8 backing[256];
	OaMemset(backing, 0xAA, 256);

	{
		OaSecureBuffer sec(backing, 256);
		EXPECT_TRUE(sec.IsValid());
	}
	// After OaSecureBuffer destructor, memory must be zeroed
	for (OaU32 i = 0; i < 256; ++i) {
		EXPECT_EQ(backing[i], 0u) << "byte " << i << " not zeroed";
	}
}

TEST(SecureBuffer, MoveSemantics) {
	alignas(64) OaU8 backing[128];
	OaMemset(backing, 0xBB, 128);

	OaSecureBuffer a(backing, 128);
	EXPECT_TRUE(a.IsValid());

	OaSecureBuffer b = std::move(a);
	EXPECT_FALSE(a.IsValid());
	EXPECT_TRUE(b.IsValid());
	EXPECT_EQ(b.Data(), backing);
	EXPECT_EQ(b.SizeBytes(), 128u);

	// Destroy b — backing should be zeroed
	b.Reset();
	EXPECT_FALSE(b.IsValid());
	for (OaU32 i = 0; i < 128; ++i) {
		EXPECT_EQ(backing[i], 0u);
	}
}

TEST(SecureBuffer, DefaultIsInvalid) {
	OaSecureBuffer sec;
	EXPECT_FALSE(sec.IsValid());
}

// PQC pool tests (require GPU)

TEST(PqcPool, NttSlabAllocFree) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	OaVkPqcPoolConfig cfg{};
	cfg.MaxConcurrentSigns = 8;
	cfg.MaxConcurrentHashes = 8;
	cfg.NttSlotSize = 1024;
	cfg.HashSlotSize = 200;

	OaVkPqcPool pool;
	auto status = pool.Init(rt.Allocator, cfg);
	ASSERT_TRUE(status.IsOk()) << status.ToString();

	EXPECT_EQ(pool.NttSlotsUsed(), 0u);

	OaVec<void*> slots;
	for (OaU32 i = 0; i < 8; ++i) {
		void* p = pool.AllocNttSlot();
		ASSERT_NE(p, nullptr) << "NTT alloc failed at " << i;
		slots.PushBack(p);
	}
	EXPECT_EQ(pool.NttSlotsUsed(), 8u);

	// 9th should fail
	EXPECT_EQ(pool.AllocNttSlot(), nullptr);

	// Verify 256B alignment
	for (void* p : slots) {
		EXPECT_EQ(reinterpret_cast<OaUsize>(p) % 256, 0u) << "NTT slot not 256B-aligned";
	}

	for (void* p : slots) pool.FreeNttSlot(p);
	EXPECT_EQ(pool.NttSlotsUsed(), 0u);

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(PqcPool, HashStatePool) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	OaVkPqcPoolConfig cfg{};
	cfg.MaxConcurrentSigns = 4;
	cfg.MaxConcurrentHashes = 4;
	cfg.NttSlotSize = 1024;
	cfg.HashSlotSize = 200;

	OaVkPqcPool pool;
	ASSERT_TRUE(pool.Init(rt.Allocator, cfg).IsOk());

	EXPECT_EQ(pool.HashSlotsUsed(), 0u);

	OaVec<void*> slots;
	for (OaU32 i = 0; i < 4; ++i) {
		void* p = pool.AllocHashSlot();
		ASSERT_NE(p, nullptr);
		slots.PushBack(p);
	}
	EXPECT_EQ(pool.HashSlotsUsed(), 4u);
	EXPECT_EQ(pool.AllocHashSlot(), nullptr);

	for (void* p : slots) pool.FreeHashSlot(p);
	EXPECT_EQ(pool.HashSlotsUsed(), 0u);

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}

TEST(PqcPool, SecureKeyAlloc) {
	auto rtP = CreateTestEngine(); OaComputeEngine& rt = *rtP;

	OaVkPqcPoolConfig cfg{};
	cfg.MaxConcurrentSigns = 4;
	cfg.MaxConcurrentHashes = 4;

	OaVkPqcPool pool;
	ASSERT_TRUE(pool.Init(rt.Allocator, cfg).IsOk());

	auto key = pool.AllocKeyBuffer(4096);
	EXPECT_TRUE(key.IsValid());
	EXPECT_EQ(key.SizeBytes(), 4096u);

	// Write to key buffer
	OaMemset(key.Data(), 0xFF, key.SizeBytes());

	// Move out — zeroes the old one on destruction
	{
		OaSecureBuffer key2 = std::move(key);
		EXPECT_TRUE(key2.IsValid());
		EXPECT_FALSE(key.IsValid());
		// key2 destructor will zero the memory
	}

	pool.Destroy(rt.Allocator);
	rt.Destroy();
}
