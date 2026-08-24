#include <oa/ml/dqnTrainer.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/optim.h>
#include <oa/ml/trainingSession.h>
#include <oa/runtime/executionSession.h>

namespace {

oa::Status copyModel(oa::Module& inSource, oa::Module& inTarget) {
	auto source = inSource.allNamedParameterPtrs();
	auto target = inTarget.allNamedParameterPtrs();
	if (source.size() != target.size()) {
		return oa::Status::error(oa::StatusCode::ShapeMismatch,
			"DQN online and target modules have different parameter counts");
	}
	for (oa::Usize index = 0; index < source.size(); ++index) {
		if (source[index].path != target[index].path
			|| source[index].param == nullptr || target[index].param == nullptr
			|| source[index].param->data.getShape()
				!= target[index].param->data.getShape()
			|| source[index].param->data.getDtype()
				!= target[index].param->data.getDtype()) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"DQN online and target module schemas do not match");
		}
		target[index].param->data = oa::FnMatrix::copy(source[index].param->data);
		target[index].param->data.setRequiresGrad(false);
		target[index].param->requiresGrad = false;
	}
	return oa::Status::ok();
}

} // namespace

struct oa::DqnTrainer::Impl {
	oa::Engine& engine;
	oa::Module& online;
	oa::Module& target;
	oa::Optimizer& optimizer;
	oa::ReplayBuffer& replay;
	oa::DqnTrainerConfig config;
	oa::ItTraining training;
	oa::DqnTrainerMetrics metrics;

	Impl(oa::Engine& inEngine, oa::Module& inOnline, oa::Module& inTarget,
		oa::Optimizer& inOptimizer,
		oa::ReplayBuffer& inReplay, const oa::DqnTrainerConfig& inConfig)
		: engine(inEngine), online(inOnline), target(inTarget)
		, optimizer(inOptimizer), replay(inReplay)
		, config(inConfig)
		, training(inEngine, inOptimizer, oa::ItTrainingConfig{
			.totalSteps = static_cast<oa::I64>(inConfig.updates),
			.batchSize = static_cast<oa::I32>(inConfig.batchSize),
			.timerName = "dqn_update",
		}) {}
};

oa::DqnTrainer::DqnTrainer(oa::UniquePtr<Impl> inImpl)
	: impl_(oa::move(inImpl)) {}

oa::DqnTrainer::~DqnTrainer() = default;

oa::Result<oa::UniquePtr<oa::DqnTrainer>> oa::DqnTrainer::create(
	oa::Engine& inEngine,
	oa::Module& inOnline,
	oa::Module& inTarget,
	oa::Optimizer& inOptimizer,
	oa::ReplayBuffer& inReplay,
	const oa::DqnTrainerConfig& inConfig) {
	const auto& replay = inReplay.config();
	if (inConfig.updates == 0 || inConfig.batchSize == 0
		|| inConfig.targetUpdateInterval == 0
		|| inConfig.observationShape != replay.observationShape
		|| replay.actionShape.rank != 0
		|| replay.actionDtype != oa::ScalarType::Int32) {
		return oa::Status::invalidArgument(
			"oa::DqnTrainer expects positive update settings and scalar Int32 replay actions");
	}
	auto impl = oa::makeUnique<Impl>(
		inEngine, inOnline, inTarget, inOptimizer, inReplay, inConfig);
	OA_RETURN_IF_ERROR(copyModel(inOnline, inTarget));
	OA_RETURN_IF_ERROR(
		oa::ExecutionSession::forEngine(inEngine).submitAndWait());
	return oa::UniquePtr<oa::DqnTrainer>(new oa::DqnTrainer(oa::move(impl)));
}

oa::Status oa::DqnTrainer::update() {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::DqnTrainer is empty");
	auto& impl = *impl_;
	if (impl.training.stopRequested()) return impl.training.finish();
	if (impl.training.stepCount() >= impl.training.totalSteps()) return oa::Status::ok();
	const bool mayBegin = impl.training.session() != nullptr
		? impl.training.session()->tryBeginStep()
		: !impl.training.isDone();
	if (!mayBegin) {
		return impl.training.stopRequested()
			? impl.training.finish() : oa::Status::ok();
	}
	if (impl.replay.size() < impl.config.batchSize) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::DqnTrainer replay does not contain one complete batch");
	}
	auto sampled = impl.replay.sample(impl.config.batchSize,
		impl.config.seed + static_cast<oa::U64>(impl.training.index()));
	if (sampled.isError()) return sampled.getStatus();
	const oa::I64 observationElements = impl.config.observationShape.numElements();
	const oa::Matrix observation = oa::FnMatrix::reshape(sampled->observation,
		{static_cast<oa::I64>(impl.config.batchSize), observationElements});
	const oa::Matrix nextObservation = oa::FnMatrix::reshape(
		sampled->nextObservation,
		{static_cast<oa::I64>(impl.config.batchSize), observationElements});
	oa::Matrix nextQ;
	{
		oa::GradNo noGrad;
		nextQ = impl.target.forward(nextObservation);
	}
	impl.optimizer.zeroGrad();
	oa::GradientTape tape;
	const oa::Matrix q = impl.online.forward(observation);
	const oa::DqnLossResult loss = oa::FnLoss::dqn(
		q, sampled->action, sampled->reward, nextQ,
		sampled->terminated, sampled->truncated, impl.config.loss);
	if (!loss.isValid()) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::DqnTrainer loss construction failed");
	tape.backward(loss.loss);
	impl.training.next(loss.loss);
	if (impl.training.lastStatus().isError()) return impl.training.lastStatus();
	impl.metrics = {
		.update = static_cast<oa::U64>(impl.training.index()),
		.loss = impl.training.lastLoss(),
	};
	if (impl.metrics.update % impl.config.targetUpdateInterval == 0) {
		OA_RETURN_IF_ERROR(syncTarget());
	}
	if (impl.training.stepCount() >= impl.training.totalSteps()) {
		return impl.training.finish();
	}
	return oa::Status::ok();
}

oa::Status oa::DqnTrainer::syncTarget() {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::DqnTrainer is empty");
	OA_RETURN_IF_ERROR(copyModel(impl_->online, impl_->target));
	return oa::ExecutionSession::forEngine(impl_->engine).submitAndWait();
}

bool oa::DqnTrainer::isDone() const noexcept {
	return impl_ && (impl_->training.stopRequested()
		|| impl_->training.stepCount() >= impl_->training.totalSteps());
}

const oa::DqnTrainerMetrics& oa::DqnTrainer::metrics() const noexcept {
	return impl_->metrics;
}

oa::ItTraining& oa::DqnTrainer::trainingLoop() noexcept {
	return impl_->training;
}

const oa::ItTraining& oa::DqnTrainer::trainingLoop() const noexcept {
	return impl_->training;
}

oa::Status oa::DqnTrainer::save(const oa::String& inPath) const {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::DqnTrainer is empty");
	return impl_->online.save(impl_->engine, inPath, impl_->optimizer);
}

oa::Status oa::DqnTrainer::load(const oa::String& inPath) {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::DqnTrainer is empty");
	OA_RETURN_IF_ERROR(
		impl_->online.load(impl_->engine, inPath, impl_->optimizer));
	return syncTarget();
}
