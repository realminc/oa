// AsyncBatchPipeline — Worker thread + ring buffer of pinned-memory batches
//
// Dedicated prefetch thread samples random sequences from mmap'd data into a
// ring buffer of pinned-memory slots. Main thread consumes batches at zero
// latency (already in pinned memory, ready for async GPU transfer).
//
// Each slot holds [B, S+1] raw bytes (input + 1 byte shifted target).
// GPU kernel splits input/target on device after H2D transfer.

#pragma once

#include <Oa/Data/Dataset.h>
#include <Oa/Core/Thread.h>
#include <thread>

class OaAsyncBatchPipeline {
public:
	OaAsyncBatchPipeline(OaMMapByteDataset& InDataset,
		OaI32 InBatchSize, OaI32 InSeqLen,
		OaI32 InRingSize = 4);
	~OaAsyncBatchPipeline();

	OaAsyncBatchPipeline(const OaAsyncBatchPipeline&) = delete;
	OaAsyncBatchPipeline& operator=(const OaAsyncBatchPipeline&) = delete;

	[[nodiscard]] const OaU8* NextBatch();
	void ConsumeBatch();

	[[nodiscard]] OaUsize SlotBytes() const { return SlotBytes_; }
	[[nodiscard]] OaI32 RingSize() const { return RingSize_; }

private:
	void WorkerLoop();

	OaMMapByteDataset& Dataset_;
	OaI32 BatchSize_;
	OaI32 SeqLen_;
	OaI32 RingSize_;
	OaUsize SlotBytes_;

	OaVec<OaPinnedBuffer> Ring_;
	std::thread Worker_;
	OaChannel<OaI32> FreeCh_;
	OaChannel<OaI32> ReadyCh_;
	OaI32 CurrentSlot_ = -1;
	uint64_t RngState_ = 42;
};
