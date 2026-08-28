// End-to-end CPU-to-GPU upload benchmark.
//
// Compares the former allocate/map/copy/submit/wait/free path with one
// persistent mapped upload ring and one queue submission for the whole batch.
// The destination buffers are identical in both paths; medians are reported
// after alternating the order to reduce clock/thermal bias.

#include "../../oaTest.h"

#include <oa/runtime/uploadRing.h>
#include <oa/runtime/engine/allocatorAccess.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct UploadCase {
	oa::U64 bytes;
	oa::U32 count;
};

double median(std::vector<double> inValues) {
	std::sort(inValues.begin(), inValues.end());
	return inValues[inValues.size() / 2];
}

double runLegacy(
	oa::Engine& inRt,
	oa::Span<const oavk::Buffer> inDestinations,
	const void* inData,
	oa::U64 inBytes)
{
	const auto begin = Clock::now();
	for (const oavk::Buffer& destination : inDestinations) {
		auto stagingResult = oa::EngineResourceAccess::allocBuffer(inRt, inBytes);
		if (!stagingResult) return -1.0;
		auto staging = std::move(*stagingResult);
		oa::memcpy(staging.mappedPtr, inData, static_cast<oa::Usize>(inBytes));
		if (not oa::EngineAllocatorAccess::get(inRt).flushHostBuffer(staging, 0, inBytes)) {
			oa::EngineResourceAccess::freeBuffer(inRt, staging);
			return -1.0;
		}
		auto copy = oa::EngineResourceAccess::copyBufferAsync(inRt, staging, destination, inBytes);
		if (not copy.isOk() or not copy->wait().isOk()) {
			oa::EngineResourceAccess::freeBuffer(inRt, staging);
			return -1.0;
		}
		oa::EngineResourceAccess::freeBuffer(inRt, staging);
	}
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double runRing(
	oa::UploadRing& inRing,
	oa::Span<const oavk::Buffer> inDestinations,
	const void* inData,
	oa::U64 inBytes)
{
	const auto begin = Clock::now();
	if (!inRing.beginBatch().isOk()) return -1.0;
	for (const oavk::Buffer& destination : inDestinations) {
		if (!inRing.upload(destination, 0, inData, inBytes).isOk()) return -1.0;
	}
	auto completion = inRing.submit();
	if (!completion || !completion->wait().isOk()) return -1.0;
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double runLegacyReadback(
	oa::Engine& inRt,
	const oavk::Buffer& inSource,
	void* outData,
	oa::U64 inBytes,
	oa::U32 inCount)
{
	const auto begin = Clock::now();
	for (oa::U32 index = 0; index < inCount; ++index) {
		auto stagingResult = oa::EngineAllocatorAccess::get(inRt).allocHostReadback(inBytes);
		if (!stagingResult) return -1.0;
		auto staging = std::move(*stagingResult);
		auto streamResult = oavk::Stream::create(
			oa::EngineDeviceAccess::get(inRt),
			oa::EngineDeviceAccess::get(inRt).queues.computeQueueFamily,
			oa::EngineDeviceAccess::get(inRt).queues.computeQueue);
		if (!streamResult) {
			oa::EngineAllocatorAccess::get(inRt).free(staging);
			return -1.0;
		}
		auto stream = std::move(*streamResult);
		oa::Status status = stream.begin(oa::EngineDeviceAccess::get(inRt));
		if (status.isOk()) {
			stream.recordTransferReadBarrier(inSource, 0U, inBytes);
			stream.recordCopyBuffer(inSource, staging, inBytes);
			stream.recordTransferWriteBarrier(staging, 0U, inBytes);
			status = stream.submitAndWait(inRt);
		}
		if (status.isOk() && !oa::EngineAllocatorAccess::get(inRt).invalidateHostBuffer(staging, 0, inBytes)) {
			status = oa::Status::error("legacy readback invalidate failed");
		}
		if (status.isOk()) oa::memcpy(outData, staging.mappedPtr, inBytes);
		stream.destroy(oa::EngineDeviceAccess::get(inRt));
		oa::EngineAllocatorAccess::get(inRt).free(staging);
		if (!status.isOk()) return -1.0;
	}
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

double runPersistentReadback(
	oa::Engine& inRt,
	const oavk::Buffer& inSource,
	void* outData,
	oa::U64 inBytes,
	oa::U32 inCount)
{
	const auto begin = Clock::now();
	for (oa::U32 index = 0; index < inCount; ++index) {
		if (!oa::EngineResourceAccess::readbackBuffer(inRt, inSource, 0, outData, inBytes).isOk()) return -1.0;
	}
	return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

} // namespace

TEST(VkUploadRingBenchmark, BatchedVsLegacyPerCopy) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr UploadCase cases[] = {
		{256, 64},
		{4096, 64},
		{65536, 16},
	};
	constexpr oa::U32 kWarmup = 4;
	constexpr oa::U32 kSamples = 15;

	std::printf("\nCPU-to-GPU upload batches (end-to-end median, warmup=%u samples=%u)\n",
		kWarmup, kSamples);
	std::printf("%10s %7s %14s %14s %10s\n",
		"bytes", "copies", "legacy us", "ring us", "speedup");

	for (const UploadCase& uploadCase : cases) {
		oa::Vector<oavk::Buffer> destinations;
		destinations.reserve(uploadCase.count);
		for (oa::U32 index = 0; index < uploadCase.count; ++index) {
			auto destination = oa::EngineResourceAccess::allocBufferDevice(*rt, uploadCase.bytes);
			ASSERT_TRUE(destination.isOk());
			destinations.pushBack(std::move(*destination));
		}

		oa::Vector<oa::U8> data(uploadCase.bytes);
		for (oa::U64 index = 0; index < uploadCase.bytes; ++index) {
			data[index] = static_cast<oa::U8>((index * 131u + 17u) & 0xFFu);
		}

		const oa::U64 batchBytes = uploadCase.bytes * uploadCase.count;
		auto ringResult = oa::UploadRing::create(*rt, oa::UploadRingConfig{
			.capacityBytes = std::max<oa::U64>(batchBytes * 3, 3 * 4096),
			.framesInFlight = 3,
			.alignment = 256,
		});
		ASSERT_TRUE(ringResult.isOk());
		auto ring = std::move(*ringResult);
		const auto destinationSpan = oa::Span<const oavk::Buffer>(
			destinations.data(), destinations.size());

		for (oa::U32 sample = 0; sample < kWarmup; ++sample) {
			ASSERT_GT(runLegacy(*rt, destinationSpan, data.data(), uploadCase.bytes), 0.0);
			ASSERT_GT(runRing(ring, destinationSpan, data.data(), uploadCase.bytes), 0.0);
		}

		std::vector<double> legacySamples;
		std::vector<double> ringSamples;
		legacySamples.reserve(kSamples);
		ringSamples.reserve(kSamples);
		for (oa::U32 sample = 0; sample < kSamples; ++sample) {
			if ((sample & 1u) == 0) {
				legacySamples.push_back(runLegacy(
					*rt, destinationSpan, data.data(), uploadCase.bytes));
				ringSamples.push_back(runRing(
					ring, destinationSpan, data.data(), uploadCase.bytes));
			} else {
				ringSamples.push_back(runRing(
					ring, destinationSpan, data.data(), uploadCase.bytes));
				legacySamples.push_back(runLegacy(
					*rt, destinationSpan, data.data(), uploadCase.bytes));
			}
		}

		const double legacyUs = median(std::move(legacySamples));
		const double ringUs = median(std::move(ringSamples));
		ASSERT_GT(legacyUs, 0.0);
		ASSERT_GT(ringUs, 0.0);
		std::printf("%10llu %7u %14.2f %14.2f %9.2fx\n",
			static_cast<unsigned long long>(uploadCase.bytes),
			uploadCase.count, legacyUs, ringUs, legacyUs / ringUs);

		ASSERT_TRUE(ring.close().isOk());
		for (auto& destination : destinations) oa::EngineResourceAccess::freeBuffer(*rt, destination);
	}
}

TEST(VkUploadRingBenchmark, PersistentReadbackVsPerCallResources) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);

	constexpr UploadCase cases[] = {
		{4, 64},
		{4096, 32},
		{65536, 16},
	};
	constexpr oa::U32 kWarmup = 3;
	constexpr oa::U32 kSamples = 11;

	std::printf("\nGPU-to-CPU readback batches (end-to-end median, warmup=%u samples=%u)\n",
		kWarmup, kSamples);
	std::printf("%10s %7s %14s %14s %10s\n",
		"bytes", "copies", "legacy us", "reused us", "speedup");

	for (const UploadCase& readbackCase : cases) {
		auto sourceResult = oa::EngineResourceAccess::allocBufferDevice(*rt, readbackCase.bytes);
		ASSERT_TRUE(sourceResult.isOk());
		auto source = std::move(*sourceResult);
		oa::Vector<oa::U8> data(readbackCase.bytes);
		oa::Vector<oa::U8> output(readbackCase.bytes);
		for (oa::U64 index = 0; index < readbackCase.bytes; ++index) {
			data[index] = static_cast<oa::U8>((index * 73u + 11u) & 0xFFu);
		}
		ASSERT_TRUE(oa::EngineResourceAccess::uploadBuffer(*rt, source, 0, data.data(), readbackCase.bytes).isOk());

		for (oa::U32 sample = 0; sample < kWarmup; ++sample) {
			ASSERT_GT(runLegacyReadback(*rt, source, output.data(),
				readbackCase.bytes, readbackCase.count), 0.0);
			ASSERT_GT(runPersistentReadback(*rt, source, output.data(),
				readbackCase.bytes, readbackCase.count), 0.0);
		}

		std::vector<double> legacySamples;
		std::vector<double> reusedSamples;
		legacySamples.reserve(kSamples);
		reusedSamples.reserve(kSamples);
		for (oa::U32 sample = 0; sample < kSamples; ++sample) {
			if ((sample & 1u) == 0) {
				legacySamples.push_back(runLegacyReadback(*rt, source, output.data(),
					readbackCase.bytes, readbackCase.count));
				reusedSamples.push_back(runPersistentReadback(*rt, source, output.data(),
					readbackCase.bytes, readbackCase.count));
			} else {
				reusedSamples.push_back(runPersistentReadback(*rt, source, output.data(),
					readbackCase.bytes, readbackCase.count));
				legacySamples.push_back(runLegacyReadback(*rt, source, output.data(),
					readbackCase.bytes, readbackCase.count));
			}
		}
		const double legacyUs = median(std::move(legacySamples));
		const double reusedUs = median(std::move(reusedSamples));
		ASSERT_GT(legacyUs, 0.0);
		ASSERT_GT(reusedUs, 0.0);
		EXPECT_TRUE(oa::memEqual(data.data(), output.data(), readbackCase.bytes));
		std::printf("%10llu %7u %14.2f %14.2f %9.2fx\n",
			static_cast<unsigned long long>(readbackCase.bytes),
			readbackCase.count, legacyUs, reusedUs, legacyUs / reusedUs);
		oa::EngineResourceAccess::freeBuffer(*rt, source);
	}
}
