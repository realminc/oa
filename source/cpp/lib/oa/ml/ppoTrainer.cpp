#include <oa/ml/ppoTrainer.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/optim.h>
#include <oa/ml/trainingSession.h>
#include <oa/runtime/executionSession.h>

namespace {

oa::I64 numObservationElements(const oa::MatrixShape& inShape) {
	return inShape.numElements();
}

bool validConfig(const oa::PpoTrainerConfig& inConfig) {
	return inConfig.rollouts > 0 && inConfig.horizon > 0
		&& inConfig.environments > 0 && inConfig.updateEpochs > 0
		&& inConfig.observationShape.rank > 0
		&& numObservationElements(inConfig.observationShape) > 0;
}

} // namespace

struct oa::PpoTrainer::Impl {
	oa::Engine& engine;
	oa::ActorCritic& model;
	oa::Optimizer& optimizer;
	oa::PpoTrainerConfig config;
	oa::RolloutBuffer rollout;
	oa::ItRolloutTraining training;
	oa::PpoTrainerMetrics metrics;
	oa::U64 actionIndex = 0;
	oa::U64 collectionActionIndex = 0;
	bool collecting = false;
	bool collectionAbortable = false;

	Impl(oa::Engine& inEngine, oa::ActorCritic& inModel, oa::Optimizer& inOptimizer,
		const oa::PpoTrainerConfig& inConfig, oa::RolloutBuffer&& inRollout)
		: engine(inEngine)
		, model(inModel)
		, optimizer(inOptimizer)
		, config(inConfig)
		, rollout(oa::move(inRollout))
		, training(inEngine, inOptimizer, oa::ItRolloutTrainingConfig{
			.rollouts = inConfig.rollouts,
			.horizon = inConfig.horizon,
			.environments = inConfig.environments,
			.updateEpochs = inConfig.updateEpochs,
			.timerName = "ppo_update",
		}) {}
};

oa::PpoTrainer::PpoTrainer(oa::UniquePtr<Impl> inImpl)
	: impl_(oa::move(inImpl)) {}

oa::PpoTrainer::~PpoTrainer() = default;

oa::Result<oa::UniquePtr<oa::PpoTrainer>> oa::PpoTrainer::create(
	oa::Engine& inEngine,
	oa::ActorCritic& inModel,
	oa::Optimizer& inOptimizer,
	const oa::PpoTrainerConfig& inConfig) {
	if (!validConfig(inConfig)) {
		return oa::Status::invalidArgument(
			"oa::PpoTrainer expects non-zero rollout dimensions and a non-empty observation shape");
	}
	auto rollout = oa::RolloutBuffer::create(oa::RolloutConfig{
		.time = inConfig.horizon,
		.environments = inConfig.environments,
		.observationShape = inConfig.observationShape,
	});
	if (rollout.isError()) return rollout.getStatus();
	auto impl = oa::makeUnique<Impl>(
		inEngine, inModel, inOptimizer, inConfig, oa::move(*rollout));
	if (!impl->training.isValid()) return impl->training.lastStatus();
	OA_RETURN_IF_ERROR(
		oa::ExecutionSession::forEngine(inEngine).submitAndWait());
	return oa::UniquePtr<oa::PpoTrainer>(new oa::PpoTrainer(oa::move(impl)));
}

oa::Status oa::PpoTrainer::beginCollection() {
	if (!impl_ || impl_->collecting
		|| impl_->training.phase() != oa::RolloutTrainingPhase::Collect) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer cannot begin collection in the current phase");
	}
	OA_RETURN_IF_ERROR(impl_->training.beginRollout(impl_->rollout));
	impl_->collectionActionIndex = impl_->actionIndex;
	impl_->collecting = true;
	impl_->collectionAbortable = true;
	return oa::Status::ok();
}

oa::PolicyResult oa::PpoTrainer::act(const oa::Matrix& inObservation) {
	if (!impl_ || !impl_->collecting) return {};
	oa::GradNo noGrad;
	const oa::I64 observationElements = numObservationElements(
		impl_->config.observationShape);
	const oa::Matrix flat = oa::FnMatrix::reshape(inObservation,
		{static_cast<oa::I64>(impl_->config.environments), observationElements});
	const oa::ActorCriticOutput network = impl_->model.evaluate(flat);
	if (!network.isValid()) return {};
	return oa::FnPolicy::sampleCategorical(
		network.logits, network.value,
		impl_->config.seed + ++impl_->actionIndex);
}

oa::Status oa::PpoTrainer::observe(
	const oa::Matrix& inObservation,
	const oa::Matrix& inNextObservation,
	const oa::Matrix& inReward,
	const oa::Matrix& inTerminated,
	const oa::Matrix& inTruncated,
	const oa::PolicyResult& inPolicy) {
	if (!impl_ || !impl_->collecting || !inPolicy.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer::observe requires active collection and a valid policy result");
	}
	oa::GradNo noGrad;
	const oa::I64 observationElements = numObservationElements(
		impl_->config.observationShape);
	const oa::Matrix nextFlat = oa::FnMatrix::reshape(inNextObservation,
		{static_cast<oa::I64>(impl_->config.environments), observationElements});
	const oa::ActorCriticOutput next = impl_->model.evaluate(nextFlat);
	if (!next.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer next-value evaluation failed");
	}
	return impl_->rollout.append(oa::RolloutTransition{
		.observation = inObservation,
		.action = inPolicy.action,
		.reward = inReward,
		.value = inPolicy.value,
		.nextValue = next.value,
		.logProbability = inPolicy.logProbability,
		.terminated = inTerminated,
		.truncated = inTruncated,
	});
}

oa::Status oa::PpoTrainer::endCollection() {
	if (!impl_ || !impl_->collecting) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer collection is not active");
	}
	OA_RETURN_IF_ERROR(impl_->training.finalizeRollout(
		impl_->rollout, impl_->config.gae));
	impl_->collecting = false;
	return oa::Status::ok();
}

oa::Status oa::PpoTrainer::abortCollection() {
	if (!impl_ || !impl_->collectionAbortable) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer has no unsubmitted collection to abort");
	}
	OA_RETURN_IF_ERROR(
		impl_->training.abortRollout(impl_->rollout));
	impl_->actionIndex = impl_->collectionActionIndex;
	impl_->collecting = false;
	impl_->collectionAbortable = false;
	return oa::Status::ok();
}

oa::Status oa::PpoTrainer::update() {
	if (!impl_ || impl_->training.phase() != oa::RolloutTrainingPhase::Update) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer cannot update in the current phase");
	}
	if (!impl_->training.beginUpdate()) {
		auto* session = impl_->training.updateLoop().session();
		if (impl_->training.updateLoop().stopRequested()) {
			return impl_->training.finish();
		}
		if (session != nullptr && session->state() == TrainingState::Paused) {
			return oa::Status::ok();
		}
		return impl_->training.lastStatus().isError()
			? impl_->training.lastStatus()
			: oa::Status::error(oa::StatusCode::FailedPrecondition,
				"oa::PpoTrainer update did not begin");
	}
	impl_->collectionAbortable = false;
	auto& impl = *impl_;
	const oa::I64 batch = static_cast<oa::I64>(impl.config.environments)
		* impl.config.horizon;
	const oa::I64 observationElements = numObservationElements(
		impl.config.observationShape);
	const oa::Matrix observation = oa::FnMatrix::reshape(
		impl.rollout.batch().observation, {batch, observationElements});
	const oa::Matrix action = oa::FnMatrix::reshape(
		impl.rollout.batch().action, {batch});
	const oa::Matrix oldLogProbability = oa::FnMatrix::reshape(
		impl.rollout.batch().oldLogProbability, {batch});
	const oa::Matrix advantage = oa::FnMatrix::reshape(
		impl.rollout.batch().advantage, {batch});
	const oa::Matrix targetReturn = oa::FnMatrix::reshape(
		impl.rollout.batch().ret, {batch});

	impl.optimizer.zeroGrad();
	oa::GradientTape tape;
	const oa::ActorCriticOutput network = impl.model.evaluate(observation);
	const oa::PolicyResult policy = oa::FnPolicy::evaluateCategorical(
		network.logits, action, network.value);
	if (!policy.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer action re-evaluation failed");
	}
	const oa::Matrix normalizedAdvantage = oa::FnAdvantage::normalize(advantage);
	const oa::PpoLossResult loss = oa::FnLoss::ppo(
		policy.logProbability, oldLogProbability, normalizedAdvantage,
		policy.value, targetReturn, policy.entropy, impl.config.loss);
	if (!loss.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"oa::PpoTrainer loss construction failed");
	}
	tape.backward(loss.totalLoss);
	OA_RETURN_IF_ERROR(impl.training.nextUpdate(loss.totalLoss));
	impl.metrics = {
		.rollout = impl.training.rolloutIndex(),
		.updateEpoch = impl.training.updateEpoch(),
		.totalLoss = impl.training.updateLoop().lastLoss(),
		.policyLoss = loss.policyLoss.item(),
		.valueLoss = loss.valueLoss.item(),
		.entropy = loss.entropy.item(),
	};
	if (impl.training.isDone()) {
		OA_RETURN_IF_ERROR(impl.training.finish());
	}
	return oa::Status::ok();
}

bool oa::PpoTrainer::isValid() const noexcept {
	return impl_ && impl_->training.isValid();
}

bool oa::PpoTrainer::isDone() const noexcept {
	return impl_ && impl_->training.isDone();
}

bool oa::PpoTrainer::needsCollection() const noexcept {
	return impl_ && !impl_->collecting
		&& impl_->training.phase() == oa::RolloutTrainingPhase::Collect;
}

oa::RolloutTrainingPhase oa::PpoTrainer::phase() const noexcept {
	return impl_ ? impl_->training.phase() : oa::RolloutTrainingPhase::Complete;
}

const oa::PpoTrainerConfig& oa::PpoTrainer::config() const noexcept {
	return impl_->config;
}

const oa::PpoTrainerMetrics& oa::PpoTrainer::metrics() const noexcept {
	return impl_->metrics;
}

const oa::RolloutBatch& oa::PpoTrainer::batch() const noexcept {
	return impl_->rollout.batch();
}

oa::ItTraining& oa::PpoTrainer::trainingLoop() noexcept {
	return impl_->training.updateLoop();
}

const oa::ItTraining& oa::PpoTrainer::trainingLoop() const noexcept {
	return impl_->training.updateLoop();
}

oa::Status oa::PpoTrainer::save(const oa::String& inPath) const {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::PpoTrainer is empty");
	return impl_->model.save(impl_->engine, inPath, impl_->optimizer);
}

oa::Status oa::PpoTrainer::load(const oa::String& inPath) {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::PpoTrainer is empty");
	return impl_->model.load(impl_->engine, inPath, impl_->optimizer);
}
