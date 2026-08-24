#include "lunarLander3dPpo.h"
#include "../../oaTest.h"

#include <ml/rl/lunarLander3d.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml.h>
#include <oa/ml/autograd.h>
#include <oa/ml.h>
#include <oa/runtime/executionSession.h>
#include <oa/runtime/engine.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

class LunarEvaluationDigest {
public:
	void addU32(oa::U32 inValue) noexcept {
		for (oa::U32 byteIndex = 0U; byteIndex < 4U; ++byteIndex) {
			hash_ ^= static_cast<oa::U8>(inValue >> (byteIndex * 8U));
			hash_ *= 1099511628211ULL;
		}
	}

	void addI32(oa::I32 inValue) noexcept {
		addU32(std::bit_cast<oa::U32>(inValue));
	}

	void addF32(oa::F32 inValue) noexcept {
		addU32(std::bit_cast<oa::U32>(inValue));
	}

	void addU64(oa::U64 inValue) noexcept {
		addU32(static_cast<oa::U32>(inValue));
		addU32(static_cast<oa::U32>(inValue >> 32U));
	}

	[[nodiscard]] oa::U64 value() const noexcept { return hash_; }

private:
	oa::U64 hash_ = 14695981039346656037ULL;
};

struct LunarTeacherSample {
	std::array<oa::F32, oa::kLunarObservationSize> observation;
	oa::I32 action = 0;
};

oa::Result<oa::F32> lunarEvaluateTeacherProbe(
	oa::CategoricalActorCritic& inModel,
	const std::vector<LunarTeacherSample>& inSamples,
	oa::Usize inCount) {
	if (inCount == 0U or inCount > inSamples.size()) {
		return oa::Status::invalidArgument(
			"Lunar Lander 3D teacher loss probe has an invalid extent");
	}
	std::vector<oa::F32> observations(
		inCount * oa::kLunarObservationSize);
	std::vector<oa::I32> actions(inCount);
	for (oa::Usize row = 0U; row < inCount; ++row) {
		const LunarTeacherSample& sample = inSamples[row];
		std::memcpy(
			observations.data() + row * oa::kLunarObservationSize,
			sample.observation.data(),
			oa::kLunarObservationSize * sizeof(oa::F32));
		actions[row] = sample.action;
	}
	const oa::Matrix observation = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(
			reinterpret_cast<const oa::U8*>(observations.data()),
			observations.size() * sizeof(oa::F32)),
		{static_cast<oa::I64>(inCount), oa::kLunarObservationSize},
		oa::ScalarType::Float32);
	const oa::Matrix action = oa::FnMatrix::fromInt32(
		oa::Span<const oa::I32>(actions.data(), actions.size()),
		{static_cast<oa::I64>(inCount)}, oa::ScalarType::Int32);
	if (observation.isEmpty() or action.isEmpty()) {
		return oa::Status::error(
			oa::StatusCode::OutOfMemory,
			"Lunar Lander 3D teacher loss probe upload failed");
	}
	oa::GradNo noGrad;
	const oa::ActorCriticOutput output = inModel.evaluate(observation);
	if (not output.isValid()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher loss probe policy evaluation failed");
	}
	const oa::Matrix loss = oa::FnLoss::crossEntropy(output.logits, action);
	if (loss.isEmpty()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher loss probe construction failed");
	}
	oa::F32 hostLoss = 0.0F;
	OA_RETURN_IF_ERROR(oa::FnMatrix::copyToHost(
		loss, &hostLoss, sizeof(hostLoss)));
	if (not std::isfinite(hostLoss)) {
		return oa::Status::error(
			oa::StatusCode::DataLoss,
			"Lunar Lander 3D teacher loss probe became non-finite");
	}
	return hostLoss;
}

template<typename T>
oa::Result<oa::Vec<T>> lunarCopyMatrix(const oa::Matrix& inMatrix) {
	if (inMatrix.isEmpty() or inMatrix.numElements() < 0) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D evaluation history is empty");
	}
	oa::Vec<T> result(static_cast<oa::Usize>(inMatrix.numElements()));
	const oa::Status copied = oa::FnMatrix::copyToHost(
		inMatrix, result.data(),
		static_cast<oa::U64>(result.size() * sizeof(T)));
	if (copied.isError()) return copied;
	return result;
}

oa::F64 lunarWilsonLower95(oa::U32 inSuccesses, oa::U32 inTrials) noexcept {
	if (inTrials == 0U) return 0.0;
	constexpr oa::F64 z = 1.959963984540054;
	constexpr oa::F64 zSquared = z * z;
	const oa::F64 trials = static_cast<oa::F64>(inTrials);
	const oa::F64 proportion = static_cast<oa::F64>(inSuccesses) / trials;
	const oa::F64 denominator = 1.0 + zSquared / trials;
	const oa::F64 center = proportion + zSquared / (2.0 * trials);
	const oa::F64 radius = z * std::sqrt(
		(proportion * (1.0 - proportion) + zSquared / (4.0 * trials))
		/ trials);
	return std::max(0.0, (center - radius) / denominator);
}

} // namespace

class TestLunarLander3dPpo::Impl {
public:
	oa::Engine* engine_ = nullptr;
	TestLunarLander3dPpoConfig config_;
	oa::UniquePtr<oa::ExecutionSession> context_;
	oa::UniquePtr<oa::CategoricalActorCritic> model_;
	oa::Vec<oa::Parameter*> parameters_;
	oa::AdamW optimizer_;
	oa::LunarLander3dVector environment_;
	oa::UniquePtr<oa::PpoTrainer> trainer_;
	TestLunarLander3dPpoMetrics metrics_;
	TestLunarLander3dTeacherMetrics teacherMetrics_;
	oa::I64 observationElements_ = 0;
	oa::I64 actionCount_ = 0;
	oa::F32 gaeGamma_ = 0.0F;

	Impl(
		oa::Engine& inEngine,
		const TestLunarLander3dPpoConfig& inConfig,
		oa::UniquePtr<oa::ExecutionSession> inContext,
		oa::UniquePtr<oa::CategoricalActorCritic> inModel,
		oa::LunarLander3dVector&& inEnvironment,
		oa::I64 inObservationElements,
		oa::I64 inActionCount,
		oa::F32 inGaeGamma)
		: engine_(&inEngine)
		, config_(inConfig)
		, context_(oa::move(inContext))
		, model_(oa::move(inModel))
		, parameters_(model_->allParameterPtrs())
		, optimizer_(parameters_, inConfig.learningRate_,
			0.9F, 0.999F, 1.0e-8F, 0.0F)
		, environment_(oa::move(inEnvironment))
		, observationElements_(inObservationElements)
		, actionCount_(inActionCount)
		, gaeGamma_(inGaeGamma) {}

	[[nodiscard]] oa::Status validateExecutionSession() const {
		if (engine_ == nullptr or not engine_->isReady()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"Lunar Lander 3D PPO requires its borrowed engine to remain ready");
		}
		if (not context_ or &context_->engine() != engine_) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"Lunar Lander 3D PPO execution context does not belong to its engine");
		}
		return oa::Status::ok();
	}
};

TestLunarLander3dPpo::TestLunarLander3dPpo(
	oa::UniquePtr<Impl> inImpl)
	: config_(inImpl->config_)
	, metrics_(inImpl->metrics_)
	, teacherMetrics_(inImpl->teacherMetrics_)
	, observationElements_(inImpl->observationElements_)
	, actionCount_(inImpl->actionCount_)
	, gaeGamma_(inImpl->gaeGamma_)
	, optimizerStep_(inImpl->optimizer_.getStep())
	, impl_(oa::move(inImpl)) {}

TestLunarLander3dPpo::~TestLunarLander3dPpo() = default;

oa::Result<oa::UniquePtr<TestLunarLander3dPpo>>
TestLunarLander3dPpo::create(
	oa::Engine& inEngine,
	const TestLunarLander3dPpoConfig& inConfig) {
	if (inConfig.environments_ == 0U or inConfig.horizon_ == 0U
		or inConfig.rollouts_ == 0U or inConfig.updateEpochs_ == 0U
		or inConfig.hiddenSize_ <= 0
		or not std::isfinite(inConfig.learningRate_)
		or inConfig.learningRate_ <= 0.0F) {
		return oa::Status::invalidArgument(
			"Lunar Lander 3D PPO requires non-zero dimensions, a positive hidden size, and a positive finite learning rate");
	}
	if (not inEngine.isReady()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO requires a ready engine");
	}

	oa::UniquePtr<oa::ExecutionSession> context(new oa::ExecutionSession(&inEngine));
	if (not context or &context->engine() != &inEngine) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO could not open an engine-bound execution context");
	}
	oa::ExecutionSession::RecordingScope recording(*context);
	oa::FnMatrix::setRngSeed(inConfig.trainingSeed_);
	oa::LunarLander3dVectorConfig environmentConfig;
	environmentConfig.environments_ = inConfig.environments_;
	environmentConfig.seed_ = inConfig.trainingSeed_;
	auto environment = oa::LunarLander3dVector::createFlat(
		inEngine, environmentConfig);
	if (environment.isError()) return environment.getStatus();

	const oa::EnvironmentSpec spec = environment->spec();
	const oa::I64 observationElements =
		spec.observation.elementsPerEnvironment();
	const oa::I64 actionCount = spec.action.cardinality;
	if (spec.observation.kind != oa::EnvironmentSpaceKind::Box
		or spec.observation.dtype != oa::ScalarType::Float32
		or spec.action.kind != oa::EnvironmentSpaceKind::Discrete
		or observationElements <= 0 or actionCount <= 1
		or observationElements > std::numeric_limits<oa::I32>::max()
		or actionCount > std::numeric_limits<oa::I32>::max()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO requires a finite FP32 Box observation and a bounded Discrete action spec");
	}
	const oa::F32 gaeGamma = static_cast<oa::F32>(
		environment->config().environment_.rewardGamma_);
	if (not std::isfinite(gaeGamma)
		or gaeGamma < 0.0F or gaeGamma > 1.0F) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D reward gamma cannot be represented by PPO GAE");
	}

	auto model = oa::CategoricalActorCritic::create({
		.observationSize = static_cast<oa::I32>(observationElements),
		.actionCount = static_cast<oa::I32>(actionCount),
		.hiddenSize = inConfig.hiddenSize_,
	});
	if (model.isError()) return model.getStatus();
	auto impl = oa::makeUnique<Impl>(
		inEngine, inConfig, oa::move(context), oa::move(*model),
		oa::move(*environment),
		observationElements, actionCount, gaeGamma);
	auto trainer = oa::PpoTrainer::create(
		inEngine, *impl->model_, impl->optimizer_, oa::PpoTrainerConfig{
			.rollouts = inConfig.rollouts_,
			.horizon = inConfig.horizon_,
			.environments = inConfig.environments_,
			.updateEpochs = inConfig.updateEpochs_,
			.observationShape = spec.observation.shape,
			.seed = inConfig.trainingSeed_,
			.gae = oa::GaeConfig{
				.gamma = gaeGamma,
				.lambda = 0.95F,
			},
			.loss = oa::PpoLossConfig{
				.clipEpsilon = 0.2F,
				.valueCoefficient = 0.5F,
				.entropyCoefficient = 0.01F,
			},
		});
	if (trainer.isError()) return trainer.getStatus();
	impl->trainer_ = oa::move(*trainer);
	return oa::UniquePtr<TestLunarLander3dPpo>(
		new TestLunarLander3dPpo(oa::move(impl)));
}

bool TestLunarLander3dPpo::isDone() const noexcept {
	return impl_ and impl_->trainer_ and impl_->trainer_->isDone();
}

const TestLunarLander3dPpoConfig&
TestLunarLander3dPpo::config() const noexcept {
	return config_;
}

const TestLunarLander3dPpoMetrics&
TestLunarLander3dPpo::metrics() const noexcept {
	return metrics_;
}

oa::I64 TestLunarLander3dPpo::observationElements() const noexcept {
	return observationElements_;
}

oa::I64 TestLunarLander3dPpo::actionCount() const noexcept {
	return actionCount_;
}

oa::F32 TestLunarLander3dPpo::gaeGamma() const noexcept {
	return gaeGamma_;
}

oa::U64 TestLunarLander3dPpo::optimizerStep() const noexcept {
	return impl_ ? impl_->optimizer_.getStep() : optimizerStep_;
}

oa::Status TestLunarLander3dPpo::advance() {
	if (not impl_ or not impl_->trainer_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO session is empty");
	}
	OA_RETURN_IF_ERROR(impl_->validateExecutionSession());
	oa::ExecutionSession::RecordingScope recording(*impl_->context_);
	if (impl_->trainer_->isDone()) return oa::Status::ok();
	auto& impl = *impl_;
	if (impl.trainer_->needsCollection()) {
		// CreateFlat batches its initial reset with the first submission. If an
		// earlier first-rollout transaction was cancelled, restage that reset
		// before retrying collection.
		if (impl.environment_.submissionCount() == 0U
			and not impl.environment_.hasActiveRecording()) {
			OA_RETURN_IF_ERROR(impl.environment_.reset());
		}
		const oa::Status recorded = impl.environment_.recordCommands(
			[&]() -> oa::Status {
				OA_RETURN_IF_ERROR(impl.trainer_->beginCollection());
				for (oa::U32 step = 0U; step < impl.config_.horizon_; ++step) {
					const oa::PolicyResult policy = impl.trainer_->act(
						impl.environment_.observation());
					if (not policy.isValid()) {
						return oa::Status::error(
							oa::StatusCode::FailedPrecondition,
							"Lunar Lander 3D PPO policy evaluation failed");
					}
					auto transition = impl.environment_.step(policy.action);
					if (transition.isError()) return transition.getStatus();
					OA_RETURN_IF_ERROR(impl.trainer_->observe(
						transition->observation_, transition->nextObservation_,
						transition->reward_, transition->terminated_,
						transition->truncated_, policy));
					OA_RETURN_IF_ERROR(impl.environment_.resetDone());
				}
				return impl.trainer_->endCollection();
			});
		if (recorded.isError()) {
			const oa::Status aborted = impl.trainer_->abortCollection();
			return aborted.isError() ? aborted : recorded;
		}
		auto completion = impl.environment_.submit();
		if (completion.isError()) {
			const oa::Status failure = completion.getStatus();
			if (impl.environment_.isOpen()) {
				const oa::Status aborted = impl.trainer_->abortCollection();
				if (aborted.isError()) return aborted;
			}
			return failure;
		}
		OA_RETURN_IF_ERROR(impl.environment_.wait(*completion));
	}

	OA_RETURN_IF_ERROR(impl.trainer_->update());
	const oa::PpoTrainerMetrics& metrics = impl.trainer_->metrics();
	impl.metrics_ = {
		.rollout_ = metrics.rollout,
		.updateEpoch_ = metrics.updateEpoch,
		.totalLoss_ = metrics.totalLoss,
		.policyLoss_ = metrics.policyLoss,
		.valueLoss_ = metrics.valueLoss,
		.entropy_ = metrics.entropy,
	};
	metrics_ = impl.metrics_;
	optimizerStep_ = impl.optimizer_.getStep();
	return oa::Status::ok();
}

oa::Status TestLunarLander3dPpo::pretrainScriptedTeacher(
	const TestLunarLander3dTeacherConfig& inConfig) {
	if (not impl_ or not impl_->model_ or not impl_->trainer_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher pretraining requires a live PPO session");
	}
	OA_RETURN_IF_ERROR(impl_->validateExecutionSession());
	oa::ExecutionSession::RecordingScope recording(*impl_->context_);
	if (impl_->trainer_->phase() != oa::RolloutTrainingPhase::Collect
		or impl_->optimizer_.getStep() != 0U) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher pretraining must run before PPO collection");
	}
	if (inConfig.episodes_ == 0U or inConfig.epochs_ == 0U
		or inConfig.batchSize_ == 0U or inConfig.maximumSamples_ == 0U
		or not std::isfinite(inConfig.learningRate_)
		or inConfig.learningRate_ <= 0.0F) {
		return oa::Status::invalidArgument(
			"Lunar Lander 3D teacher pretraining requires non-zero dimensions and a positive finite learning rate");
	}
	if (inConfig.environmentSeed_
		== TestLunarLander3dFirstEpisodeEvaluationConfig{}.environmentSeed_) {
		return oa::Status::invalidArgument(
			"Lunar Lander 3D teacher and held-out evaluator seeds must be disjoint");
	}

	TestLunarLander3dTeacherMetrics metrics;
	oa::LunarLander3dConfig environmentConfig;
	std::vector<LunarTeacherSample> samples;
	samples.reserve(inConfig.maximumSamples_);
	LunarEvaluationDigest datasetDigest;
	datasetDigest.addU64(environmentConfig.contractFingerprint());
	datasetDigest.addU64(inConfig.environmentSeed_);
	for (oa::U32 lane = 0U;
		lane < inConfig.episodes_ and samples.size() < inConfig.maximumSamples_;
		++lane) {
		const oa::LunarEpisodeManifest manifest = oa::LunarEpisodeManifest::derive(
			inConfig.environmentSeed_, lane, 0U,
			environmentConfig.contractFingerprint());
		auto environment = oa::LunarScalarEnvironment::createFlat(
			environmentConfig, manifest);
		if (not environment.isValid()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				oa::String("Lunar Lander 3D teacher environment creation failed: ")
					+ environment.error());
		}
		while (not environment.state().terminated_
			and not environment.state().truncated_
			and samples.size() < inConfig.maximumSamples_) {
			LunarTeacherSample sample;
			sample.observation = environment.observation();
			const oa::LunarAction action = oa::lunarScriptedLandingAction(
				environmentConfig, environment.state());
			sample.action = static_cast<oa::I32>(action);
			for (const oa::F32 value : sample.observation) {
				if (not std::isfinite(value)) {
					return oa::Status::error(
						oa::StatusCode::DataLoss,
						"Lunar Lander 3D teacher produced a non-finite observation");
				}
				datasetDigest.addF32(value);
			}
			datasetDigest.addI32(sample.action);
			++metrics.actionCounts_[static_cast<oa::Usize>(action)];
			samples.push_back(sample);
			const oa::LunarTransition transition = environment.step(
				static_cast<oa::U32>(action));
			if (not transition.valid_) {
				return oa::Status::error(
					oa::StatusCode::DataLoss,
					oa::String("Lunar Lander 3D teacher transition failed: ")
						+ transition.error_);
			}
		}
		if (environment.state().terminated_ or environment.state().truncated_) {
			++metrics.episodes_;
			switch (environment.state().endReason_) {
				case oa::LunarEndReason::SafeLanding: ++metrics.safeLandings_; break;
				case oa::LunarEndReason::BodyImpact: ++metrics.bodyImpacts_; break;
				case oa::LunarEndReason::HardFootImpact:
					++metrics.hardFootImpacts_;
					break;
				case oa::LunarEndReason::OutOfBounds: ++metrics.outOfBounds_; break;
				case oa::LunarEndReason::TimeLimit: ++metrics.timeLimits_; break;
				default: ++metrics.otherFailures_; break;
			}
		}
	}
	if (samples.empty()) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher produced no training samples");
	}
	metrics.samples_ = static_cast<oa::U32>(samples.size());
	metrics.datasetDigest_ = datasetDigest.value();
	const oa::Usize probeSamples = std::min<oa::Usize>(
		samples.size(), inConfig.batchSize_);
	auto initialLoss = lunarEvaluateTeacherProbe(
		*impl_->model_, samples, probeSamples);
	if (initialLoss.isError()) return initialLoss.getStatus();
	metrics.initialLoss_ = *initialLoss;

	oa::Vec<oa::Parameter*> policyParameters;
	for (const oa::NamedParameter& named : impl_->model_->allNamedParameterPtrs()) {
		constexpr char prefix[] = "policy";
		if (named.param != nullptr and named.path.size() >= sizeof(prefix) - 1U
			and std::memcmp(
				named.path.data(), prefix, sizeof(prefix) - 1U) == 0) {
			policyParameters.pushBack(named.param);
		}
	}
	if (policyParameters.empty()) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"Lunar Lander 3D teacher could not resolve policy parameters");
	}

	oa::AdamW imitationOptimizer(
		policyParameters, inConfig.learningRate_, 0.9F, 0.999F, 1.0e-8F, 0.0F);
	const oa::U64 ppoOptimizerStep = impl_->optimizer_.getStep();
	const oa::U32 stepsPerEpoch = static_cast<oa::U32>(
		(samples.size() + inConfig.batchSize_ - 1U) / inConfig.batchSize_);
	const oa::U64 totalSteps64 = static_cast<oa::U64>(inConfig.epochs_)
		* stepsPerEpoch;
	if (totalSteps64 > static_cast<oa::U64>(std::numeric_limits<oa::I64>::max())) {
		return oa::Status::invalidArgument(
			"Lunar Lander 3D teacher training step count is too large");
	}
	oa::ItTraining training(testEngine(), imitationOptimizer, oa::ItTrainingConfig{
		.totalSteps = static_cast<oa::I64>(totalSteps64),
		.stepsPerEpoch = static_cast<oa::I64>(stepsPerEpoch),
		.epochSteps = {},
		.batchSize = static_cast<oa::I32>(std::min<oa::U32>(
			inConfig.batchSize_, static_cast<oa::U32>(
				std::numeric_limits<oa::I32>::max()))),
		.timerName = "lunar_teacher_imitation",
		.metrics = {},
		.callbacks = {},
		.program = nullptr,
	});
	std::vector<oa::Usize> order(samples.size());
	std::iota(order.begin(), order.end(), oa::Usize{0});
	std::mt19937_64 random(inConfig.shuffleSeed_);
	while (not training.isDone()) {
		const oa::U64 zeroBasedStep = static_cast<oa::U64>(training.index() - 1);
		const oa::U32 stepInEpoch = static_cast<oa::U32>(
			zeroBasedStep % stepsPerEpoch);
		if (stepInEpoch == 0U) {
			std::shuffle(order.begin(), order.end(), random);
		}
		const oa::Usize begin = static_cast<oa::Usize>(stepInEpoch)
			* inConfig.batchSize_;
		const oa::Usize count = std::min<oa::Usize>(
			inConfig.batchSize_, samples.size() - begin);
		std::vector<oa::F32> observations(
			count * oa::kLunarObservationSize);
		std::vector<oa::I32> actions(count);
		for (oa::Usize row = 0U; row < count; ++row) {
			const LunarTeacherSample& sample = samples[order[begin + row]];
			std::memcpy(
				observations.data() + row * oa::kLunarObservationSize,
				sample.observation.data(),
				oa::kLunarObservationSize * sizeof(oa::F32));
			actions[row] = sample.action;
		}
		const oa::Matrix observation = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(
				reinterpret_cast<const oa::U8*>(observations.data()),
				observations.size() * sizeof(oa::F32)),
			{static_cast<oa::I64>(count), oa::kLunarObservationSize},
			oa::ScalarType::Float32);
		const oa::Matrix action = oa::FnMatrix::fromInt32(
			oa::Span<const oa::I32>(actions.data(), actions.size()),
			{static_cast<oa::I64>(count)}, oa::ScalarType::Int32);
		if (observation.isEmpty() or action.isEmpty()) {
			return oa::Status::error(
				oa::StatusCode::OutOfMemory,
				"Lunar Lander 3D teacher batch upload failed");
		}
		imitationOptimizer.zeroGrad();
		oa::GradientTape tape;
		const oa::ActorCriticOutput output = impl_->model_->evaluate(observation);
		if (not output.isValid()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"Lunar Lander 3D teacher policy evaluation failed");
		}
		const oa::Matrix loss = oa::FnLoss::crossEntropy(output.logits, action);
		if (loss.isEmpty()) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"Lunar Lander 3D teacher cross-entropy construction failed");
		}
		tape.backward(loss);
		training.next(loss);
		if (training.lastStatus().isError()) return training.lastStatus();
		if (not std::isfinite(training.lastLoss())) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"Lunar Lander 3D teacher loss became non-finite");
		}
		++metrics.optimizerSteps_;
	}
	OA_RETURN_IF_ERROR(training.finish());
	if (imitationOptimizer.getStep() != metrics.optimizerSteps_
		or impl_->optimizer_.getStep() != ppoOptimizerStep) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"Lunar Lander 3D teacher optimizer accounting diverged");
	}
	auto finalLoss = lunarEvaluateTeacherProbe(
		*impl_->model_, samples, probeSamples);
	if (finalLoss.isError()) return finalLoss.getStatus();
	metrics.finalLoss_ = *finalLoss;
	impl_->teacherMetrics_ = metrics;
	teacherMetrics_ = metrics;
	return oa::Status::ok();
}

const TestLunarLander3dTeacherMetrics&
TestLunarLander3dPpo::teacherMetrics() const noexcept {
	return teacherMetrics_;
}

oa::Result<TestLunarLander3dFirstEpisodeEvaluation>
TestLunarLander3dPpo::evaluateFirstEpisodes(
	const TestLunarLander3dFirstEpisodeEvaluationConfig& inConfig) {
	if (not impl_ or not impl_->model_ or not impl_->engine_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO evaluation requires a live session");
	}
	OA_RETURN_IF_ERROR(impl_->validateExecutionSession());
	oa::ExecutionSession::RecordingScope recording(*impl_->context_);
	if (inConfig.environments_ == 0U or inConfig.horizon_ == 0U
		or inConfig.submissionChunkSteps_ == 0U) {
		return oa::Status::invalidArgument(
			"Lunar Lander 3D first-episode evaluation dimensions must be non-zero");
	}

	oa::LunarLander3dVectorConfig environmentConfig;
	environmentConfig.environments_ = inConfig.environments_;
	environmentConfig.seed_ = inConfig.environmentSeed_;
	auto created = oa::LunarLander3dVector::createFlat(
		*impl_->engine_, environmentConfig);
	if (created.isError()) return created.getStatus();
	auto environment = oa::move(*created);

	// Keep cleanup outside the worker lambda so every result path explicitly
	// closes the fresh environment, including submission/readback failures.
	auto evaluation = [&]()
		-> oa::Result<TestLunarLander3dFirstEpisodeEvaluation> {
		TestLunarLander3dFirstEpisodeEvaluation result;
		result.expectedEpisodes_ = inConfig.environments_;
		LunarEvaluationDigest actionDigest;
		LunarEvaluationDigest valueDigest;
		oa::Vec<oa::U8> completed(inConfig.environments_, 0U);
		oa::Vec<oa::U32> terminalReasons(inConfig.environments_, 0U);
		oa::Vec<oa::F32> accumulatedReturns(inConfig.environments_, 0.0F);

		for (oa::U32 chunkStart = 0U; chunkStart < inConfig.horizon_;) {
			const oa::U32 chunkSteps = std::min(
				inConfig.submissionChunkSteps_, inConfig.horizon_ - chunkStart);
			oa::Matrix actionHistory;
			oa::Matrix valueHistory;
			oa::Matrix rewardHistory;
			oa::Matrix endReasonHistory;
			oa::Vec<oa::Matrix> actions;
			oa::Vec<oa::Matrix> values;
			oa::Vec<oa::Matrix> rewards;
			oa::Vec<oa::Matrix> endReasons;
			actions.reserve(chunkSteps);
			values.reserve(chunkSteps);
			rewards.reserve(chunkSteps);
			endReasons.reserve(chunkSteps);
			const oa::Status recorded = environment.recordCommands(
				[&]() -> oa::Status {
					oa::GradNo noGrad;
					for (oa::U32 step = 0U; step < chunkSteps; ++step) {
						const oa::ActorCriticOutput policy =
							impl_->model_->evaluate(environment.observation());
						if (not policy.isValid()
							or policy.logits.getDtype() != oa::ScalarType::Float32
							or policy.logits.getShape()
								!= oa::MatrixShape{
									static_cast<oa::I64>(inConfig.environments_),
									impl_->actionCount_}
							or policy.value.getDtype() != oa::ScalarType::Float32
							or policy.value.getShape()
								!= oa::MatrixShape{
									static_cast<oa::I64>(inConfig.environments_)}) {
							return oa::Status::error(
								oa::StatusCode::FailedPrecondition,
								"Lunar Lander 3D greedy evaluation produced an invalid policy output");
						}
						const oa::TopKResult best = oa::FnMatrix::topK(
							policy.logits, 1, 1);
						if (best.indices.isEmpty()
							or best.indices.getDtype() != oa::ScalarType::Int32) {
							return oa::Status::error(
								oa::StatusCode::FailedPrecondition,
								"Lunar Lander 3D greedy topK evaluation failed");
						}
						const oa::Matrix action = oa::FnMatrix::reshape(
							best.indices,
							{static_cast<oa::I64>(inConfig.environments_)});
						auto transition = environment.step(action);
						if (transition.isError()) return transition.getStatus();

						actions.pushBack(action.clone());
						values.pushBack(policy.value.clone());
						rewards.pushBack(transition->reward_.clone());
						endReasons.pushBack(transition->endReason_.clone());
						if (actions.back().isEmpty() or values.back().isEmpty()
							or rewards.back().isEmpty()
							or endReasons.back().isEmpty()) {
							return oa::Status::error(
								oa::StatusCode::OutOfMemory,
								"Lunar Lander 3D evaluation could not retain its bounded history");
						}
					}
					actionHistory = oa::FnMatrix::concat(
						oa::Span<oa::Matrix>(actions), 0);
					valueHistory = oa::FnMatrix::concat(
						oa::Span<oa::Matrix>(values), 0);
					rewardHistory = oa::FnMatrix::concat(
						oa::Span<oa::Matrix>(rewards), 0);
					endReasonHistory = oa::FnMatrix::concat(
						oa::Span<oa::Matrix>(endReasons), 0);
					if (actionHistory.isEmpty() or valueHistory.isEmpty()
						or rewardHistory.isEmpty()
						or endReasonHistory.isEmpty()) {
						return oa::Status::error(
							oa::StatusCode::OutOfMemory,
							"Lunar Lander 3D evaluation history concatenation failed");
					}
					return oa::Status::ok();
				});
			if (recorded.isError()) return recorded;
			auto completion = environment.submit();
			if (completion.isError()) return completion.getStatus();
			const oa::Status waited = environment.wait(*completion);
			if (waited.isError()) return waited;
			++result.submissions_;

			auto actionsResult = lunarCopyMatrix<oa::I32>(actionHistory);
			if (actionsResult.isError()) return actionsResult.getStatus();
			auto valuesResult = lunarCopyMatrix<oa::F32>(valueHistory);
			if (valuesResult.isError()) return valuesResult.getStatus();
			auto rewardsResult = lunarCopyMatrix<oa::F32>(rewardHistory);
			if (rewardsResult.isError()) return rewardsResult.getStatus();
			auto endReasonsResult = lunarCopyMatrix<oa::U32>(endReasonHistory);
			if (endReasonsResult.isError()) return endReasonsResult.getStatus();

			oa::Vec<oa::I32> actionHost = oa::move(actionsResult).getValue();
			oa::Vec<oa::F32> valueHost = oa::move(valuesResult).getValue();
			oa::Vec<oa::F32> rewardHost = oa::move(rewardsResult).getValue();
			oa::Vec<oa::U32> endReasonHost =
				oa::move(endReasonsResult).getValue();
			const oa::Usize expectedElements =
				static_cast<oa::Usize>(chunkSteps) * inConfig.environments_;
			if (actionHost.size() != expectedElements
				or valueHost.size() != expectedElements
				or rewardHost.size() != expectedElements
				or endReasonHost.size() != expectedElements) {
				return oa::Status::error(
					oa::StatusCode::DataLoss,
					"Lunar Lander 3D evaluation history has an unexpected extent");
			}

			for (oa::U32 step = 0U; step < chunkSteps; ++step) {
				for (oa::U32 lane = 0U; lane < inConfig.environments_; ++lane) {
					const oa::Usize index = static_cast<oa::Usize>(step)
						* inConfig.environments_ + lane;
					const oa::I32 action = actionHost[index];
					const oa::F32 value = valueHost[index];
					const oa::F32 reward = rewardHost[index];
					const oa::U32 reason = endReasonHost[index];
					if (action < 0 or action >= impl_->actionCount_) {
						return oa::Status::error(
							oa::StatusCode::DataLoss,
							"Lunar Lander 3D evaluation produced an invalid greedy action");
					}
					if (not std::isfinite(value) or not std::isfinite(reward)) {
						return oa::Status::error(
							oa::StatusCode::DataLoss,
							"Lunar Lander 3D evaluation produced a non-finite value or reward");
					}
					if (reason > static_cast<oa::U32>(
						oa::LunarEndReason::InvalidAction)) {
						return oa::Status::error(
							oa::StatusCode::DataLoss,
							"Lunar Lander 3D evaluation produced an unknown end reason");
					}
					actionDigest.addI32(action);
					valueDigest.addF32(value);
					const bool wasCompleted = completed[lane] != 0U;
					const bool isCompleted = reason != static_cast<oa::U32>(
						oa::LunarEndReason::None);
					if (wasCompleted) {
						if (not isCompleted or reward != 0.0F
							or terminalReasons[lane] != reason) {
							return oa::Status::error(
								oa::StatusCode::DataLoss,
								"Lunar Lander 3D completed lane did not remain terminal and reward-free");
						}
					} else {
						++result.actionCounts_[static_cast<oa::Usize>(action)];
						accumulatedReturns[lane] += reward;
						if (not std::isfinite(accumulatedReturns[lane])) {
							return oa::Status::error(
								oa::StatusCode::DataLoss,
								"Lunar Lander 3D evaluation return became non-finite");
						}
						if (isCompleted) {
							completed[lane] = 1U;
							terminalReasons[lane] = reason;
						}
					}
				}
			}
			result.recordedEnvironmentSteps_ +=
				static_cast<oa::U64>(chunkSteps) * inConfig.environments_;
			chunkStart += chunkSteps;
			bool allCompleted = true;
			for (const oa::U8 laneCompleted : completed) {
				allCompleted = allCompleted and laneCompleted != 0U;
			}
			if (allCompleted) break;
		}

		if (result.submissions_ != environment.submissionCount()) {
			return oa::Status::error(
				oa::StatusCode::Internal,
				"Lunar Lander 3D evaluation submission accounting diverged");
		}
		auto telemetryResult = environment.copyEpisodeTelemetry();
		if (telemetryResult.isError()) return telemetryResult.getStatus();
		oa::Vec<oa::LunarLander3dEpisodeTelemetry> telemetry =
			oa::move(telemetryResult).getValue();
		if (telemetry.size() != inConfig.environments_) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"Lunar Lander 3D evaluation telemetry has an unexpected lane count");
		}

		oa::F64 returnSum = 0.0;
		oa::F64 stepSum = 0.0;
		oa::F64 fuelSum = 0.0;
		oa::F64 linearSpeedSum = 0.0;
		oa::F64 angularSpeedSum = 0.0;
		oa::F64 footImpulseSum = 0.0;
		result.minReturn_ = std::numeric_limits<oa::F64>::infinity();
		result.maxReturn_ = -std::numeric_limits<oa::F64>::infinity();
		for (oa::U32 lane = 0U; lane < inConfig.environments_; ++lane) {
			const oa::LunarLander3dEpisodeTelemetry& episode = telemetry[lane];
			const bool isCompleted = episode.terminated_ or episode.truncated_;
			if (not episode.isFinite()
				or episode.episodeStep_ > inConfig.horizon_
				or (completed[lane] != 0U) != isCompleted
				or (isCompleted and terminalReasons[lane]
					!= static_cast<oa::U32>(episode.endReason_))) {
				return oa::Status::error(
					oa::StatusCode::DataLoss,
					"Lunar Lander 3D final telemetry disagrees with the recorded first episode");
			}
			const oa::F64 returnDifference = std::abs(
				static_cast<oa::F64>(episode.episodeReturn_)
				- accumulatedReturns[lane]);
			const oa::F64 returnTolerance = 1.0e-3
				+ 2.0e-5 * std::abs(
					static_cast<oa::F64>(episode.episodeReturn_));
			if (returnDifference > returnTolerance) {
				return oa::Status::error(
					oa::StatusCode::DataLoss,
					"Lunar Lander 3D final return disagrees with transition history");
			}

			if (isCompleted) ++result.completedEpisodes_;
			switch (episode.endReason_) {
				case oa::LunarEndReason::None:
					++result.incompleteEpisodes_;
					break;
				case oa::LunarEndReason::SafeLanding:
					++result.safeLandings_;
					break;
				case oa::LunarEndReason::BodyImpact:
					++result.bodyImpacts_;
					break;
				case oa::LunarEndReason::HardFootImpact:
					++result.hardFootImpacts_;
					break;
				case oa::LunarEndReason::OutOfBounds:
					++result.outOfBounds_;
					break;
				case oa::LunarEndReason::NumericalFailure:
					++result.numericalFailures_;
					break;
				case oa::LunarEndReason::TimeLimit:
					++result.timeLimits_;
					break;
				case oa::LunarEndReason::ExternalStop:
					++result.externalStops_;
					break;
				case oa::LunarEndReason::InvalidAction:
					++result.invalidActions_;
					break;
			}
			const oa::F64 episodeReturn = episode.episodeReturn_;
			returnSum += episodeReturn;
			stepSum += episode.episodeStep_;
			fuelSum += episode.fuelRemaining_;
			linearSpeedSum += episode.terminalLinearSpeed_;
			angularSpeedSum += episode.terminalAngularSpeed_;
			footImpulseSum += episode.maximumFootImpulse_;
			result.minReturn_ = std::min(result.minReturn_, episodeReturn);
			result.maxReturn_ = std::max(result.maxReturn_, episodeReturn);
		}
		const oa::U32 reasonCount = result.safeLandings_
			+ result.bodyImpacts_ + result.hardFootImpacts_
			+ result.outOfBounds_ + result.numericalFailures_
			+ result.timeLimits_ + result.externalStops_
			+ result.invalidActions_;
		if (reasonCount != result.completedEpisodes_
			or result.completedEpisodes_ + result.incompleteEpisodes_
				!= result.expectedEpisodes_) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"Lunar Lander 3D evaluation reason accounting diverged");
		}

		const oa::F64 episodes = static_cast<oa::F64>(result.expectedEpisodes_);
		result.safeLandingRate_ =
			static_cast<oa::F64>(result.safeLandings_) / episodes;
		result.wilsonLower95_ = lunarWilsonLower95(
			result.safeLandings_, result.expectedEpisodes_);
		result.meanReturn_ = returnSum / episodes;
		result.meanEpisodeSteps_ = stepSum / episodes;
		result.meanFuelRemaining_ = fuelSum / episodes;
		result.meanTerminalLinearSpeed_ = linearSpeedSum / episodes;
		result.meanTerminalAngularSpeed_ = angularSpeedSum / episodes;
		result.meanMaximumFootImpulse_ = footImpulseSum / episodes;
		result.actionTraceDigest_ = actionDigest.value();
		result.valueTraceDigest_ = valueDigest.value();
		if (not std::isfinite(result.safeLandingRate_)
			or not std::isfinite(result.wilsonLower95_)
			or not std::isfinite(result.meanReturn_)
			or not std::isfinite(result.minReturn_)
			or not std::isfinite(result.maxReturn_)
			or not std::isfinite(result.meanEpisodeSteps_)
			or not std::isfinite(result.meanFuelRemaining_)
			or not std::isfinite(result.meanTerminalLinearSpeed_)
			or not std::isfinite(result.meanTerminalAngularSpeed_)
			or not std::isfinite(result.meanMaximumFootImpulse_)) {
			return oa::Status::error(
				oa::StatusCode::DataLoss,
				"Lunar Lander 3D evaluation aggregate became non-finite");
		}
		return result;
	}();

	const oa::Status closeStatus = environment.close();
	if (evaluation.isError()) return evaluation.getStatus();
	if (closeStatus.isError()) return closeStatus;
	return oa::move(evaluation).getValue();
}

oa::Status TestLunarLander3dPpo::save(
	const oa::String& inPath) const {
	if (not impl_ or not impl_->trainer_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO session is empty");
	}
	OA_RETURN_IF_ERROR(impl_->validateExecutionSession());
	oa::ExecutionSession::RecordingScope recording(*impl_->context_);
	return impl_->trainer_->save(inPath);
}

oa::Status TestLunarLander3dPpo::load(const oa::String& inPath) {
	if (not impl_ or not impl_->trainer_) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO session is empty");
	}
	OA_RETURN_IF_ERROR(impl_->validateExecutionSession());
	oa::ExecutionSession::RecordingScope recording(*impl_->context_);
	return impl_->trainer_->load(inPath);
}

oa::Status TestLunarLander3dPpo::close() {
	if (not impl_) return oa::Status::ok();
	OA_RETURN_IF_ERROR(impl_->validateExecutionSession());
	oa::Status closeStatus;
	{
		oa::ExecutionSession::RecordingScope recording(*impl_->context_);
		closeStatus = impl_->environment_.close();
		if (closeStatus.isOk()) closeStatus = testSubmitAndWait(*impl_->context_);
	}
	if (closeStatus.isError()) return closeStatus;
	// A closed session must release every engine-owned matrix and private
	// context before its borrowed engine may be closed.
	config_ = impl_->config_;
	metrics_ = impl_->metrics_;
	teacherMetrics_ = impl_->teacherMetrics_;
	observationElements_ = impl_->observationElements_;
	actionCount_ = impl_->actionCount_;
	gaeGamma_ = impl_->gaeGamma_;
	optimizerStep_ = impl_->optimizer_.getStep();
	impl_.reset();
	return oa::Status::ok();
}
