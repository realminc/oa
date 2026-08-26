#include <oa/ml/callbacks.h>

#include <stdio.h>

class oa::CbCsvLogger::Impl {
public:
	explicit Impl(const oa::String& inPath) : path(inPath) {}
	~Impl() {
		if (file != nullptr) (void)::fclose(file);
	}

	oa::String path;
	::FILE* file = nullptr;
};

oa::CbCsvLogger::CbCsvLogger(const oa::String& inPath)
	: impl_(oa::makeUnique<Impl>(inPath)) {}

oa::CbCsvLogger::~CbCsvLogger() = default;

void oa::CbCsvLogger::onTrainBegin(oa::ItTraining& inIter) {
	(void)inIter;
	if (impl_->file != nullptr) (void)::fclose(impl_->file);
	impl_->file = ::fopen(impl_->path.cStr(), "w");
	if (impl_->file != nullptr) {
		(void)::fputs(
			"epoch,step,batch_size,sequence_length,sequence_unit,source_unit,"
			"loss,gpu_ms,wall_ms_per_step,wall_samples_per_second,"
			"gpu_samples_per_second,wall_units_per_second,gpu_units_per_second,"
			"wall_source_units_per_second,gpu_source_units_per_second\n",
			impl_->file);
		(void)::fflush(impl_->file);
	}
}

void oa::CbCsvLogger::onStepEnd(oa::ItTraining& inIter) {
	if (impl_->file == nullptr) return;
	(void)::fprintf(impl_->file,
		"%lld,%lld,%lld,%lld,%s,%s,%g,%g,%g,%g,%g,%g,%g,%g,%g\n",
		static_cast<long long>(inIter.epoch()),
		static_cast<long long>(inIter.stepCount()),
		static_cast<long long>(inIter.cfg().batchSize),
		static_cast<long long>(inIter.cfg().sequenceLength),
		inIter.cfg().sequenceUnit.cStr(), inIter.cfg().sourceUnit.cStr(),
		inIter.lastLoss(), inIter.lastGpuMs(), inIter.wallMsPerStep(),
		inIter.wallSamplesPerSecond(), inIter.gpuSamplesPerSecond(),
		inIter.wallUnitsPerSecond(), inIter.gpuUnitsPerSecond(),
		inIter.wallSourceUnitsPerSecond(), inIter.gpuSourceUnitsPerSecond());
}

void oa::CbCsvLogger::onEpochEnd(oa::ItTraining& inIter) {
	(void)inIter;
	if (impl_->file != nullptr) (void)::fflush(impl_->file);
}

void oa::CbCsvLogger::onTrainEnd(oa::ItTraining& inIter) {
	(void)inIter;
	if (impl_->file != nullptr) {
		(void)::fclose(impl_->file);
		impl_->file = nullptr;
	}
}
