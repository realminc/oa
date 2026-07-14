#include <Oa/Data/AsyncPipeline.h>
#include <Oa/Core/Memory.h>

#ifdef OA_PLATFORM_LINUX
#include <sys/mman.h>
#endif

OaAsyncBatchPipeline::OaAsyncBatchPipeline(OaMMapByteDataset& InDataset,
	OaI32 InBatchSize, OaI32 InSeqLen, OaI32 InRingSize)
	: Dataset_(InDataset)
	, BatchSize_(InBatchSize)
	, SeqLen_(InSeqLen)
	, RingSize_(InRingSize)
	, SlotBytes_(static_cast<OaUsize>(InBatchSize) * static_cast<OaUsize>(InSeqLen + 1))
	, FreeCh_(InRingSize)
	, ReadyCh_(InRingSize)
{
	Ring_.Reserve(static_cast<OaUsize>(RingSize_));
	for (OaI32 i = 0; i < RingSize_; ++i) {
		Ring_.EmplaceBack(SlotBytes_);
		FreeCh_.TrySend(i);
	}

	Worker_ = std::thread(&OaAsyncBatchPipeline::WorkerLoop, this);
}

OaAsyncBatchPipeline::~OaAsyncBatchPipeline() {
	FreeCh_.Close();
	ReadyCh_.Close();
	if (Worker_.joinable()) Worker_.join();
}

const OaU8* OaAsyncBatchPipeline::NextBatch() {
	auto slot = ReadyCh_.Recv();
	if (!slot.has_value()) return nullptr;
	CurrentSlot_ = *slot;
	return Ring_[static_cast<OaUsize>(CurrentSlot_)].As<const OaU8>();
}

void OaAsyncBatchPipeline::ConsumeBatch() {
	if (CurrentSlot_ >= 0) {
		FreeCh_.TrySend(CurrentSlot_);
		CurrentSlot_ = -1;
	}
}

void OaAsyncBatchPipeline::WorkerLoop() {
	const OaU8* rawData = Dataset_.RawData();
	OaI64 dataLen = Dataset_.TotalBytes();
	OaI64 sampleBytes = static_cast<OaI64>(SeqLen_ + 1);

	while (true) {
		auto slot = FreeCh_.Recv();
		if (!slot.has_value()) break;

		OaU8* dst = Ring_[static_cast<OaUsize>(*slot)].As<OaU8>();
		OaI64 maxOffset = dataLen - sampleBytes;

		for (OaI32 b = 0; b < BatchSize_; ++b) {
			RngState_ = RngState_ * 6364136223846793005ULL + 1442695040888963407ULL;
			OaI64 offset = static_cast<OaI64>(RngState_ % static_cast<uint64_t>(maxOffset));

#ifdef OA_PLATFORM_LINUX
			madvise(const_cast<OaU8*>(rawData + offset),
				static_cast<size_t>(sampleBytes), MADV_WILLNEED);
#endif

			OaMemcpy(dst + b * sampleBytes, rawData + offset,
				static_cast<OaUsize>(sampleBytes));
		}

		if (!ReadyCh_.Send(*slot)) break;
	}
}
