#include "cartPolePpo.h"

#include <ml/rl/cartPole.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/runtime/engine.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

template<typename T>
oa::Result<std::vector<T>> copy(const oa::Matrix& inMatrix) {
	std::vector<T> result(static_cast<oa::Usize>(inMatrix.numElements()));
	const oa::Status status = oa::FnMatrix::copyToHost(
		inMatrix, result.data(), result.size() * sizeof(T));
	if (status.isError()) return status;
	return result;
}

} // namespace

struct TutorialCartPolePpo::Impl {
	oa::Engine* engine = nullptr;
	TutorialCartPolePpoConfig config;
	oa::UniquePtr<oa::CategoricalActorCritic> model;
	oa::Vec<oa::Parameter*> parameters;
	oa::AdamW optimizer;
	oa::CartPole environment;
	oa::UniquePtr<oa::PpoTrainer> trainer;
	oa::UniquePtr<oa::TrainingSession> control;
	TutorialCartPolePpoMetrics metrics;

	Impl(
		oa::Engine& inEngine,
		const TutorialCartPolePpoConfig& inConfig,
		oa::UniquePtr<oa::CategoricalActorCritic> inModel,
		oa::CartPole&& inEnvironment)
		: engine(&inEngine)
		, config(inConfig)
		, model(oa::move(inModel))
		, parameters(model->allParameterPtrs())
		, optimizer(parameters, inConfig.learningRate,
			0.9F, 0.999F, 1.0e-8F, 0.0F)
		, environment(oa::move(inEnvironment)) {}
};

TutorialCartPolePpo::TutorialCartPolePpo(oa::UniquePtr<Impl> inImpl)
	: impl_(oa::move(inImpl)) {}

TutorialCartPolePpo::~TutorialCartPolePpo() = default;

oa::Result<oa::UniquePtr<TutorialCartPolePpo>> TutorialCartPolePpo::create(
	oa::Engine& inEngine,
	const TutorialCartPolePpoConfig& inConfig) {
	if (inConfig.environments == 0 || inConfig.horizon == 0
		|| inConfig.rollouts == 0 || inConfig.updateEpochs == 0
		|| !std::isfinite(inConfig.learningRate)
		|| inConfig.learningRate <= 0.0F) {
		return oa::Status::invalidArgument(
			"TutorialCartPolePpo requires non-zero dimensions and a positive finite learning rate");
	}
	oa::FnMatrix::setRngSeed(inConfig.trainingSeed);
	if (!inEngine.isReady()) return oa::Status::error(
		oa::StatusCode::FailedPrecondition,
		"TutorialCartPolePpo requires a ready engine");
	auto environment = oa::CartPole::create(inEngine, oa::CartPoleConfig{
		.environments = inConfig.environments,
		.maxEpisodeSteps = 500,
		.seed = inConfig.trainingSeed,
	});
	if (environment.isError()) return environment.getStatus();
	auto model = oa::CategoricalActorCritic::create(
		oa::CategoricalActorCriticConfig{
			.observationSize = 4,
			.actionCount = 2,
			.hiddenSize = 64,
		});
	if (model.isError()) return model.getStatus();
	auto impl = oa::makeUnique<Impl>(
		inEngine, inConfig, oa::move(*model), oa::move(*environment));
	auto trainer = oa::PpoTrainer::create(
		inEngine, *impl->model, impl->optimizer, oa::PpoTrainerConfig{
			.rollouts = inConfig.rollouts,
			.horizon = inConfig.horizon,
			.environments = inConfig.environments,
			.updateEpochs = inConfig.updateEpochs,
			.observationShape = {4},
			.seed = inConfig.trainingSeed,
			.gae = {},
			.loss = oa::PpoLossConfig{
				.clipEpsilon = 0.2F,
				.valueCoefficient = 0.5F,
				.entropyCoefficient = 0.01F,
			},
	});
	if (trainer.isError()) return trainer.getStatus();
	impl->trainer = oa::move(*trainer);
	impl->control = oa::makeUnique<oa::TrainingSession>(
		impl->trainer->trainingLoop());
	return oa::UniquePtr<TutorialCartPolePpo>(
		new TutorialCartPolePpo(oa::move(impl)));
}

oa::Status TutorialCartPolePpo::advance() {
	if (!impl_) return oa::Status::error(
		oa::StatusCode::FailedPrecondition, "CartPole PPO session is empty");
	if (impl_->trainer->isDone()) return oa::Status::ok();
	auto& impl = *impl_;
	if (impl.trainer->needsCollection()) {
		// Create batches its initial reset with the first submission. Restage it
		// when a cancelled first-rollout transaction is retried.
		if (impl.environment.submissionCount() == 0U
			&& !impl.environment.hasActiveRecording()) {
			OA_RETURN_IF_ERROR(impl.environment.reset());
		}
		const oa::Status recorded = impl.environment.recordCommands(
			[&]() -> oa::Status {
				OA_RETURN_IF_ERROR(impl.trainer->beginCollection());
				for (oa::U32 step = 0; step < impl.config.horizon; ++step) {
					const oa::PolicyResult policy = impl.trainer->act(
						impl.environment.observation());
					if (!policy.isValid()) return oa::Status::error(
						oa::StatusCode::FailedPrecondition,
						"CartPole PPO policy evaluation failed");
					auto transition = impl.environment.step(policy.action);
					if (transition.isError()) return transition.getStatus();
					OA_RETURN_IF_ERROR(impl.trainer->observe(
						transition->observation, transition->nextObservation,
						transition->reward, transition->terminated,
						transition->truncated, policy));
					OA_RETURN_IF_ERROR(impl.environment.resetDone());
				}
				return impl.trainer->endCollection();
			});
		if (recorded.isError()) {
			const oa::Status aborted = impl.trainer->abortCollection();
			return aborted.isError() ? aborted : recorded;
		}
		auto completion = impl.environment.submit();
		if (completion.isError()) {
			const oa::Status failure = completion.getStatus();
			if (impl.environment.isOpen()) {
				const oa::Status aborted = impl.trainer->abortCollection();
				if (aborted.isError()) return aborted;
			}
			return failure;
		}
		OA_RETURN_IF_ERROR(impl.environment.wait(*completion));
	}
	OA_RETURN_IF_ERROR(impl.trainer->update());
	const oa::PpoTrainerMetrics& metrics = impl.trainer->metrics();
	impl.metrics.totalLoss = metrics.totalLoss;
	impl.metrics.policyLoss = metrics.policyLoss;
	impl.metrics.valueLoss = metrics.valueLoss;
	impl.metrics.entropy = metrics.entropy;
	impl.metrics.updateEpoch = metrics.updateEpoch;
	if (impl.trainer->phase() != oa::RolloutTrainingPhase::Update) {
		impl.metrics.rollout = metrics.rollout;
		impl.metrics.lossHistory.pushBack(impl.metrics.totalLoss);
		impl.metrics.policyLossHistory.pushBack(impl.metrics.policyLoss);
		impl.metrics.valueLossHistory.pushBack(impl.metrics.valueLoss);
		impl.metrics.entropyHistory.pushBack(impl.metrics.entropy);
	}
	return oa::Status::ok();
}

bool TutorialCartPolePpo::isDone() const noexcept {
	return impl_ && impl_->trainer->isDone();
}

const TutorialCartPolePpoConfig& TutorialCartPolePpo::config() const noexcept {
	return impl_->config;
}

const TutorialCartPolePpoMetrics& TutorialCartPolePpo::metrics() const noexcept {
	return impl_->metrics;
}

oa::Result<TutorialCartPoleSnapshot> TutorialCartPolePpo::snapshotLane(
	oa::U32 inLane) {
	if (!impl_ || inLane >= impl_->config.environments) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange, "CartPole snapshot lane is out of range");
	}
	const oa::Matrix& state = impl_->environment.observation();
	if (impl_->engine == nullptr) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition, "CartPole runtime is unavailable");
	}
	oa::Vec<oa::F32> values(static_cast<oa::Usize>(state.numElements()));
	const auto readback = oa::FnMatrix::copyToHost(
		state, values.data(), values.size() * sizeof(oa::F32));
	if (readback.isError()) {
		return oa::Status::error(
			oa::StatusCode::VulkanError,
			"CartPole snapshot readback failed: " + readback.getMessage());
	}
	const oa::Usize laneOffset = static_cast<oa::Usize>(inLane) * 4U;
	return TutorialCartPoleSnapshot{
		.cartPosition = values[laneOffset + 0U],
		.cartVelocity = values[laneOffset + 1U],
		.poleAngle = values[laneOffset + 2U],
		.poleAngularVelocity = values[laneOffset + 3U],
	};
}

oa::Status TutorialCartPolePpo::demonstrate() {
	if (!impl_) return oa::Status::error(
		oa::StatusCode::FailedPrecondition, "CartPole PPO session is empty");
	const oa::Status recorded = impl_->environment.recordCommands(
		[&]() -> oa::Status {
			oa::GradNo noGrad;
			const oa::Matrix logits = impl_->model->forward(
				impl_->environment.observation());
			const oa::TopKResult best = oa::FnMatrix::topK(logits, 1, 1);
			const oa::Matrix action = oa::FnMatrix::reshape(
				best.indices,
				{static_cast<oa::I64>(impl_->config.environments)});
			auto transition = impl_->environment.step(action);
			if (transition.isError()) return transition.getStatus();
			return impl_->environment.resetDone();
		});
	if (recorded.isError()) return recorded;
	auto completion = impl_->environment.submit();
	if (completion.isError()) return completion.getStatus();
	return impl_->environment.wait(*completion);
}

oa::Result<TutorialCartPolePpoEvaluation> TutorialCartPolePpo::evaluate(
	oa::U64 inSeed,
	oa::U32 inEnvironments,
	oa::U32 inHorizon) {
	if (!impl_ || inEnvironments == 0 || inHorizon == 0) {
		return oa::Status::invalidArgument("CartPole evaluation dimensions must be non-zero");
	}
	auto environment = oa::CartPole::create(*impl_->engine,
		oa::CartPoleConfig{
		.environments = inEnvironments,
		.maxEpisodeSteps = 500,
		.seed = inSeed,
	});
	if (environment.isError()) return environment.getStatus();
	auto metrics = oa::PolicyEvaluator::evaluateCategorical(
		*environment, *impl_->model,
		{.horizon = inHorizon, .seed = inSeed});
	if (metrics.isError()) return metrics.getStatus();
	TutorialCartPolePpoEvaluation result{
		.meanCompletedReturn = metrics->meanCompletedReturn,
		.completedEpisodes = static_cast<oa::U32>(metrics->completedEpisodes),
	};
	impl_->metrics.evaluationReturnHistory.pushBack(
		static_cast<oa::F32>(result.meanCompletedReturn));
	return result;
}

oa::Status TutorialCartPolePpo::save(const oa::String& inPath) const {
	if (!impl_) return oa::Status::error(
		oa::StatusCode::FailedPrecondition, "CartPole PPO session is empty");
	return impl_->trainer->save(inPath);
}

oa::Status TutorialCartPolePpo::load(const oa::String& inPath) {
	if (!impl_) return oa::Status::error(
		oa::StatusCode::FailedPrecondition, "CartPole PPO session is empty");
	return impl_->trainer->load(inPath);
}

oa::U64 TutorialCartPolePpo::optimizerStep() const noexcept {
	return impl_ ? impl_->optimizer.getStep() : 0U;
}

oa::TrainingSession& TutorialCartPolePpo::control() noexcept {
	return *impl_->control;
}

const oa::TrainingSession& TutorialCartPolePpo::control() const noexcept {
	return *impl_->control;
}
