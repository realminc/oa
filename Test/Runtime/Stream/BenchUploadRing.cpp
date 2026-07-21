// End-to-end CPU-to-GPU upload benchmark.
//
// Compares the former allocate/map/copy/submit/wait/free path with one
// persistent mapped upload ring and one queue submission for the whole batch.
// The destination buffers are identical in both paths; medians are reported
// after alternating the order to reduce clock/thermal bias.

#include "../../OaTest.h"

#include <Oa/Runtime/UploadRing.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct UploadCase {
	OaU64 Bytes;
	OaU32 Count;
};

double Median(std::vector<double> InValues) {
	std::sort(InValues.begin(), InValues.end());
	return InValues[InValues.size() / 2];
}

double RunLegacy(
	OaEngine& InRt,
	OaSpan<const OaVkBuffer> InDestinations,
	const void* InData,
	OaU64 InBytes)
{
	const auto begin = Clock::now();
	for (const OaVkBuffer& destination : InDestinations) {
		auto stagingResult = InRt.AllocBuffer(InBytes);
		if (!stagingResult) return -1.0;
		auto staging = std::move(*stagingResult);
		OaMemcpy(staging.MappedPtr, InData, static_cast<OaUsize>(InBytes));
		if (not InRt.Allocator.FlushHostBuffer(staging, 0, InBytes)) {
			InRt.FreeBuffer(staging);
			return -1.0;
		}
		auto copy = InRt.CopyBufferAsync(staging, destination, InBytes);
		if (not copy.IsOk() or not copy->Wait().IsOk()) {
			InRt.FreeBuffer(staging);
			return -1.0;
		}
		InRt.FreeBuffer(staging);
	}
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double RunRing(
	OaUploadRing& InRing,
	OaSpan<const OaVkBuffer> InDestinations,
	const void* InData,
	OaU64 InBytes)
{
	const auto begin = Clock::now();
	if (!InRing.BeginBatch().IsOk()) return -1.0;
	for (const OaVkBuffer& destination : InDestinations) {
		if (!InRing.Upload(destination, 0, InData, InBytes).IsOk()) return -1.0;
	}
	auto completion = InRing.Submit();
	if (!completion || !completion->Wait().IsOk()) return -1.0;
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double RunLegacyReadback(
	OaEngine& InRt,
	const OaVkBuffer& InSource,
	void* OutData,
	OaU64 InBytes,
	OaU32 InCount)
{
	const auto begin = Clock::now();
	for (OaU32 index = 0; index < InCount; ++index) {
		auto stagingResult = InRt.Allocator.AllocHostReadback(InBytes);
		if (!stagingResult) return -1.0;
		auto staging = std::move(*stagingResult);
		auto streamResult = OaVkStream::Create(
			InRt.Device,
			InRt.Device.Queues.ComputeQueueFamily,
			InRt.Device.Queues.ComputeQueue);
		if (!streamResult) {
			InRt.Allocator.Free(staging);
			return -1.0;
		}
		auto stream = std::move(*streamResult);
		OaStatus status = stream.Begin(InRt.Device);
		if (status.IsOk()) {
			stream.RecordTransferReadBarrier(InSource, 0U, InBytes);
			stream.RecordCopyBuffer(InSource, staging, InBytes);
			stream.RecordTransferWriteBarrier(staging, 0U, InBytes);
			status = stream.SubmitAndWait(InRt);
		}
		if (status.IsOk() && !InRt.Allocator.InvalidateHostBuffer(staging, 0, InBytes)) {
			status = OaStatus::Error("legacy readback invalidate failed");
		}
		if (status.IsOk()) OaMemcpy(OutData, staging.MappedPtr, InBytes);
		stream.Destroy(InRt.Device);
		InRt.Allocator.Free(staging);
		if (!status.IsOk()) return -1.0;
	}
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double RunPersistentReadback(
	OaEngine& InRt,
	const OaVkBuffer& InSource,
	void* OutData,
	OaU64 InBytes,
	OaU32 InCount)
{
	const auto begin = Clock::now();
	for (OaU32 index = 0; index < InCount; ++index) {
		if (!InRt.ReadbackBuffer(InSource, 0, OutData, InBytes).IsOk()) return -1.0;
	}
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

} // namespace

TEST(VkUploadRingBenchmark, BatchedVsLegacyPerCopy) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr UploadCase cases[] = {
		{256, 64},
		{4096, 64},
		{65536, 16},
	};
	constexpr OaU32 kWarmup = 4;
	constexpr OaU32 kSamples = 15;

	std::printf("\nCPU-to-GPU upload batches (end-to-end median, warmup=%u samples=%u)\n",
		kWarmup, kSamples);
	std::printf("%10s %7s %14s %14s %10s\n",
		"bytes", "copies", "legacy us", "ring us", "speedup");

	for (const UploadCase& uploadCase : cases) {
		OaVec<OaVkBuffer> destinations;
		destinations.Reserve(uploadCase.Count);
		for (OaU32 index = 0; index < uploadCase.Count; ++index) {
			auto destination = rt->AllocBufferDevice(uploadCase.Bytes);
			ASSERT_TRUE(destination.IsOk());
			destinations.PushBack(std::move(*destination));
		}

		OaVec<OaU8> data(uploadCase.Bytes);
		for (OaU64 index = 0; index < uploadCase.Bytes; ++index) {
			data[index] = static_cast<OaU8>((index * 131u + 17u) & 0xFFu);
		}

		const OaU64 batchBytes = uploadCase.Bytes * uploadCase.Count;
		auto ringResult = OaUploadRing::Create(*rt, OaUploadRingConfig{
			.CapacityBytes = std::max<OaU64>(batchBytes * 3, 3 * 4096),
			.FramesInFlight = 3,
			.Alignment = 256,
		});
		ASSERT_TRUE(ringResult.IsOk());
		auto ring = std::move(*ringResult);
		const auto destinationSpan = OaSpan<const OaVkBuffer>(
			destinations.Data(), destinations.Size());

		for (OaU32 sample = 0; sample < kWarmup; ++sample) {
			ASSERT_GT(RunLegacy(*rt, destinationSpan, data.Data(), uploadCase.Bytes), 0.0);
			ASSERT_GT(RunRing(ring, destinationSpan, data.Data(), uploadCase.Bytes), 0.0);
		}

		std::vector<double> legacySamples;
		std::vector<double> ringSamples;
		legacySamples.reserve(kSamples);
		ringSamples.reserve(kSamples);
		for (OaU32 sample = 0; sample < kSamples; ++sample) {
			if ((sample & 1u) == 0) {
				legacySamples.push_back(RunLegacy(
					*rt, destinationSpan, data.Data(), uploadCase.Bytes));
				ringSamples.push_back(RunRing(
					ring, destinationSpan, data.Data(), uploadCase.Bytes));
			} else {
				ringSamples.push_back(RunRing(
					ring, destinationSpan, data.Data(), uploadCase.Bytes));
				legacySamples.push_back(RunLegacy(
					*rt, destinationSpan, data.Data(), uploadCase.Bytes));
			}
		}

		const double legacyUs = Median(std::move(legacySamples));
		const double ringUs = Median(std::move(ringSamples));
		ASSERT_GT(legacyUs, 0.0);
		ASSERT_GT(ringUs, 0.0);
		std::printf("%10llu %7u %14.2f %14.2f %9.2fx\n",
			static_cast<unsigned long long>(uploadCase.Bytes),
			uploadCase.Count, legacyUs, ringUs, legacyUs / ringUs);

		ring.Destroy();
		for (auto& destination : destinations) rt->FreeBuffer(destination);
	}
}

TEST(VkUploadRingBenchmark, PersistentReadbackVsPerCallResources) {
	auto* rt = OaEngine::GetGlobal();
	ASSERT_NE(rt, nullptr);

	constexpr UploadCase cases[] = {
		{4, 64},
		{4096, 32},
		{65536, 16},
	};
	constexpr OaU32 kWarmup = 3;
	constexpr OaU32 kSamples = 11;

	std::printf("\nGPU-to-CPU readback batches (end-to-end median, warmup=%u samples=%u)\n",
		kWarmup, kSamples);
	std::printf("%10s %7s %14s %14s %10s\n",
		"bytes", "copies", "legacy us", "reused us", "speedup");

	for (const UploadCase& readbackCase : cases) {
		auto sourceResult = rt->AllocBufferDevice(readbackCase.Bytes);
		ASSERT_TRUE(sourceResult.IsOk());
		auto source = std::move(*sourceResult);
		OaVec<OaU8> data(readbackCase.Bytes);
		OaVec<OaU8> output(readbackCase.Bytes);
		for (OaU64 index = 0; index < readbackCase.Bytes; ++index) {
			data[index] = static_cast<OaU8>((index * 73u + 11u) & 0xFFu);
		}
		ASSERT_TRUE(rt->UploadBuffer(source, 0, data.Data(), readbackCase.Bytes).IsOk());

		for (OaU32 sample = 0; sample < kWarmup; ++sample) {
			ASSERT_GT(RunLegacyReadback(*rt, source, output.Data(),
				readbackCase.Bytes, readbackCase.Count), 0.0);
			ASSERT_GT(RunPersistentReadback(*rt, source, output.Data(),
				readbackCase.Bytes, readbackCase.Count), 0.0);
		}

		std::vector<double> legacySamples;
		std::vector<double> reusedSamples;
		legacySamples.reserve(kSamples);
		reusedSamples.reserve(kSamples);
		for (OaU32 sample = 0; sample < kSamples; ++sample) {
			if ((sample & 1u) == 0) {
				legacySamples.push_back(RunLegacyReadback(*rt, source, output.Data(),
					readbackCase.Bytes, readbackCase.Count));
				reusedSamples.push_back(RunPersistentReadback(*rt, source, output.Data(),
					readbackCase.Bytes, readbackCase.Count));
			} else {
				reusedSamples.push_back(RunPersistentReadback(*rt, source, output.Data(),
					readbackCase.Bytes, readbackCase.Count));
				legacySamples.push_back(RunLegacyReadback(*rt, source, output.Data(),
					readbackCase.Bytes, readbackCase.Count));
			}
		}
		const double legacyUs = Median(std::move(legacySamples));
		const double reusedUs = Median(std::move(reusedSamples));
		ASSERT_GT(legacyUs, 0.0);
		ASSERT_GT(reusedUs, 0.0);
		EXPECT_TRUE(OaMemEqual(data.Data(), output.Data(), readbackCase.Bytes));
		std::printf("%10llu %7u %14.2f %14.2f %9.2fx\n",
			static_cast<unsigned long long>(readbackCase.Bytes),
			readbackCase.Count, legacyUs, reusedUs, legacyUs / reusedUs);
		rt->FreeBuffer(source);
	}
}
