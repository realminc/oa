#include "LunarLander3dPpo.h"

#include "LunarLander3d.h"
#include "LunarLander3dVector.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Rl.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

class OaLunarEvaluationDigest {
public:
	void AddU32(OaU32 InValue) noexcept {
		for (OaU32 byteIndex = 0U; byteIndex < 4U; ++byteIndex) {
			Hash_ ^= static_cast<OaU8>(InValue >> (byteIndex * 8U));
			Hash_ *= 1099511628211ULL;
		}
	}

	void AddI32(OaI32 InValue) noexcept {
		AddU32(std::bit_cast<OaU32>(InValue));
	}

	void AddF32(OaF32 InValue) noexcept {
		AddU32(std::bit_cast<OaU32>(InValue));
	}

	void AddU64(OaU64 InValue) noexcept {
		AddU32(static_cast<OaU32>(InValue));
		AddU32(static_cast<OaU32>(InValue >> 32U));
	}

	[[nodiscard]] OaU64 Value() const noexcept { return Hash_; }

private:
	OaU64 Hash_ = 14695981039346656037ULL;
};

struct OaLunarTeacherSample {
	std::array<OaF32, OA_LUNAR_OBSERVATION_SIZE> Observation;
	OaI32 Action = 0;
};

OaResult<OaF32> OaLunarEvaluateTeacherProbe(
	OaCategoricalActorCritic& InModel,
	const std::vector<OaLunarTeacherSample>& InSamples,
	OaUsize InCount) {
	if (InCount == 0U or InCount > InSamples.size()) {
		return OaStatus::InvalidArgument(
			"Lunar Lander 3D teacher loss probe has an invalid extent");
	}
	std::vector<OaF32> observations(
		InCount * OA_LUNAR_OBSERVATION_SIZE);
	std::vector<OaI32> actions(InCount);
	for (OaUsize row = 0U; row < InCount; ++row) {
		const OaLunarTeacherSample& sample = InSamples[row];
		std::memcpy(
			observations.data() + row * OA_LUNAR_OBSERVATION_SIZE,
			sample.Observation.data(),
			OA_LUNAR_OBSERVATION_SIZE * sizeof(OaF32));
		actions[row] = sample.Action;
	}
	const OaMatrix observation = OaFnMatrix::FromBytes(
		OaSpan<const OaU8>(
			reinterpret_cast<const OaU8*>(observations.data()),
			observations.size() * sizeof(OaF32)),
		{static_cast<OaI64>(InCount), OA_LUNAR_OBSERVATION_SIZE},
		OaScalarType::Float32);
	const OaMatrix action = OaFnMatrix::FromInt32(
		OaSpan<const OaI32>(actions.data(), actions.size()),
		{static_cast<OaI64>(InCount)}, OaScalarType::Int32);
	if (observation.IsEmpty() or action.IsEmpty()) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"Lunar Lander 3D teacher loss probe upload failed");
	}
	OaGradNo noGrad;
	const OaRlActorCriticOutput output = InModel.Evaluate(observation);
	if (not output.IsValid()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher loss probe policy evaluation failed");
	}
	const OaMatrix loss = OaFnLoss::CrossEntropy(output.Logits, action);
	if (loss.IsEmpty()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher loss probe construction failed");
	}
	OaF32 hostLoss = 0.0F;
	OA_RETURN_IF_ERROR(OaFnMatrix::CopyToHost(
		loss, &hostLoss, sizeof(hostLoss)));
	if (not std::isfinite(hostLoss)) {
		return OaStatus::Error(
			OaStatusCode::DataLoss,
			"Lunar Lander 3D teacher loss probe became non-finite");
	}
	return hostLoss;
}

template<typename T>
OaResult<OaVec<T>> OaLunarCopyMatrix(const OaMatrix& InMatrix) {
	if (InMatrix.IsEmpty() or InMatrix.NumElements() < 0) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D evaluation history is empty");
	}
	OaVec<T> result(static_cast<OaUsize>(InMatrix.NumElements()));
	const OaStatus copied = OaFnMatrix::CopyToHost(
		InMatrix, result.Data(),
		static_cast<OaU64>(result.Size() * sizeof(T)));
	if (copied.IsError()) return copied;
	return result;
}

OaF64 OaLunarWilsonLower95(OaU32 InSuccesses, OaU32 InTrials) noexcept {
	if (InTrials == 0U) return 0.0;
	constexpr OaF64 z = 1.959963984540054;
	constexpr OaF64 zSquared = z * z;
	const OaF64 trials = static_cast<OaF64>(InTrials);
	const OaF64 proportion = static_cast<OaF64>(InSuccesses) / trials;
	const OaF64 denominator = 1.0 + zSquared / trials;
	const OaF64 center = proportion + zSquared / (2.0 * trials);
	const OaF64 radius = z * std::sqrt(
		(proportion * (1.0 - proportion) + zSquared / (4.0 * trials))
		/ trials);
	return std::max(0.0, (center - radius) / denominator);
}

} // namespace

class OaTutorialLunarLander3dPpo::Impl {
public:
	OaEngine* Engine_ = nullptr;
	OaTutorialLunarLander3dPpoConfig Config_;
	OaUniquePtr<OaContext> Context_;
	OaUniquePtr<OaCategoricalActorCritic> Model_;
	OaVec<OaParameter*> Parameters_;
	OaAdamW Optimizer_;
	OaLunarLander3dVector Environment_;
	OaUniquePtr<OaPpoTrainer> Trainer_;
	OaTutorialLunarLander3dPpoMetrics Metrics_;
	OaTutorialLunarLander3dTeacherMetrics TeacherMetrics_;
	OaI64 ObservationElements_ = 0;
	OaI64 ActionCount_ = 0;
	OaF32 GaeGamma_ = 0.0F;

	Impl(
		OaEngine& InEngine,
		const OaTutorialLunarLander3dPpoConfig& InConfig,
		OaUniquePtr<OaContext> InContext,
		OaUniquePtr<OaCategoricalActorCritic> InModel,
		OaLunarLander3dVector&& InEnvironment,
		OaI64 InObservationElements,
		OaI64 InActionCount,
		OaF32 InGaeGamma)
		: Engine_(&InEngine)
		, Config_(InConfig)
		, Context_(OaStdMove(InContext))
		, Model_(OaStdMove(InModel))
		, Parameters_(Model_->AllParameterPtrs())
		, Optimizer_(Parameters_, InConfig.LearningRate_,
			0.9F, 0.999F, 1.0e-8F, 0.0F)
		, Environment_(OaStdMove(InEnvironment))
		, ObservationElements_(InObservationElements)
		, ActionCount_(InActionCount)
		, GaeGamma_(InGaeGamma) {}

	[[nodiscard]] OaStatus ValidateExecutionContext() const {
		if (Engine_ == nullptr or not Engine_->IsReady()) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"Lunar Lander 3D PPO requires its borrowed engine to remain ready");
		}
		if (not Context_ or Context_->GetEngine() != Engine_) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"Lunar Lander 3D PPO execution context does not belong to its engine");
		}
		return OaStatus::Ok();
	}
};

OaTutorialLunarLander3dPpo::OaTutorialLunarLander3dPpo(
	OaUniquePtr<Impl> InImpl)
	: Impl_(OaStdMove(InImpl)) {}

OaTutorialLunarLander3dPpo::~OaTutorialLunarLander3dPpo() = default;

OaResult<OaUniquePtr<OaTutorialLunarLander3dPpo>>
OaTutorialLunarLander3dPpo::Create(
	OaEngine& InEngine,
	const OaTutorialLunarLander3dPpoConfig& InConfig) {
	if (InConfig.Environments_ == 0U or InConfig.Horizon_ == 0U
		or InConfig.Rollouts_ == 0U or InConfig.UpdateEpochs_ == 0U
		or InConfig.HiddenSize_ <= 0
		or not std::isfinite(InConfig.LearningRate_)
		or InConfig.LearningRate_ <= 0.0F) {
		return OaStatus::InvalidArgument(
			"Lunar Lander 3D PPO requires non-zero dimensions, a positive hidden size, and a positive finite learning rate");
	}
	if (not InEngine.IsReady()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO requires a ready engine");
	}

	OaUniquePtr<OaContext> context(OaContext::Create(&InEngine));
	if (not context or context->GetEngine() != &InEngine) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO could not open an engine-bound execution context");
	}
	OaContext::RecordingScope recording(*context);
	OaFnMatrix::SetRngSeed(InConfig.TrainingSeed_);
	OaLunarLander3dVectorConfig environmentConfig;
	environmentConfig.Environments_ = InConfig.Environments_;
	environmentConfig.Seed_ = InConfig.TrainingSeed_;
	auto environment = OaLunarLander3dVector::CreateFlat(
		InEngine, environmentConfig);
	if (environment.IsError()) return environment.GetStatus();

	const OaRlEnvironmentSpec spec = environment->Spec();
	const OaI64 observationElements =
		spec.Observation.ElementsPerEnvironment();
	const OaI64 actionCount = spec.Action.Cardinality;
	if (spec.Observation.Kind != OaRlSpaceKind::Box
		or spec.Observation.Dtype != OaScalarType::Float32
		or spec.Action.Kind != OaRlSpaceKind::Discrete
		or observationElements <= 0 or actionCount <= 1
		or observationElements > std::numeric_limits<OaI32>::max()
		or actionCount > std::numeric_limits<OaI32>::max()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO requires a finite FP32 Box observation and a bounded Discrete action spec");
	}
	const OaF32 gaeGamma = static_cast<OaF32>(
		environment->Config().Environment_.RewardGamma_);
	if (not std::isfinite(gaeGamma)
		or gaeGamma < 0.0F or gaeGamma > 1.0F) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D reward gamma cannot be represented by PPO GAE");
	}

	auto model = OaCategoricalActorCritic::Create({
		.ObservationSize = static_cast<OaI32>(observationElements),
		.ActionCount = static_cast<OaI32>(actionCount),
		.HiddenSize = InConfig.HiddenSize_,
	});
	if (model.IsError()) return model.GetStatus();
	auto impl = OaMakeUniquePtr<Impl>(
		InEngine, InConfig, OaStdMove(context), OaStdMove(*model),
		OaStdMove(*environment),
		observationElements, actionCount, gaeGamma);
	auto trainer = OaPpoTrainer::Create(
		*impl->Model_, impl->Optimizer_, OaPpoTrainerConfig{
			.Rollouts = InConfig.Rollouts_,
			.Horizon = InConfig.Horizon_,
			.Environments = InConfig.Environments_,
			.UpdateEpochs = InConfig.UpdateEpochs_,
			.ObservationShape = spec.Observation.Shape,
			.Seed = InConfig.TrainingSeed_,
			.Gae = OaGaeConfig{
				.Gamma = gaeGamma,
				.Lambda = 0.95F,
			},
			.Loss = OaPpoLossConfig{
				.ClipEpsilon = 0.2F,
				.ValueCoefficient = 0.5F,
				.EntropyCoefficient = 0.01F,
			},
		});
	if (trainer.IsError()) return trainer.GetStatus();
	impl->Trainer_ = OaStdMove(*trainer);
	return OaUniquePtr<OaTutorialLunarLander3dPpo>(
		new OaTutorialLunarLander3dPpo(OaStdMove(impl)));
}

bool OaTutorialLunarLander3dPpo::IsDone() const noexcept {
	return Impl_ and Impl_->Trainer_ and Impl_->Trainer_->IsDone();
}

const OaTutorialLunarLander3dPpoConfig&
OaTutorialLunarLander3dPpo::Config() const noexcept {
	return Impl_->Config_;
}

const OaTutorialLunarLander3dPpoMetrics&
OaTutorialLunarLander3dPpo::Metrics() const noexcept {
	return Impl_->Metrics_;
}

OaI64 OaTutorialLunarLander3dPpo::ObservationElements() const noexcept {
	return Impl_ ? Impl_->ObservationElements_ : 0;
}

OaI64 OaTutorialLunarLander3dPpo::ActionCount() const noexcept {
	return Impl_ ? Impl_->ActionCount_ : 0;
}

OaF32 OaTutorialLunarLander3dPpo::GaeGamma() const noexcept {
	return Impl_ ? Impl_->GaeGamma_ : 0.0F;
}

OaU64 OaTutorialLunarLander3dPpo::OptimizerStep() const noexcept {
	return Impl_ ? Impl_->Optimizer_.GetStep() : 0U;
}

OaStatus OaTutorialLunarLander3dPpo::Advance() {
	if (not Impl_ or not Impl_->Trainer_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO session is empty");
	}
	OA_RETURN_IF_ERROR(Impl_->ValidateExecutionContext());
	OaContext::RecordingScope recording(*Impl_->Context_);
	if (Impl_->Trainer_->IsDone()) return OaStatus::Ok();
	auto& impl = *Impl_;
	if (impl.Trainer_->NeedsCollection()) {
		const OaStatus recorded = impl.Environment_.RecordCommands(
			[&]() -> OaStatus {
				OA_RETURN_IF_ERROR(impl.Trainer_->BeginCollection());
				for (OaU32 step = 0U; step < impl.Config_.Horizon_; ++step) {
					const OaRlPolicyResult policy = impl.Trainer_->Act(
						impl.Environment_.Observation());
					if (not policy.IsValid()) {
						return OaStatus::Error(
							OaStatusCode::FailedPrecondition,
							"Lunar Lander 3D PPO policy evaluation failed");
					}
					auto transition = impl.Environment_.Step(policy.Action);
					if (transition.IsError()) return transition.GetStatus();
					OA_RETURN_IF_ERROR(impl.Trainer_->Observe(
						transition->Observation_, transition->NextObservation_,
						transition->Reward_, transition->Terminated_,
						transition->Truncated_, policy));
					OA_RETURN_IF_ERROR(impl.Environment_.ResetDone());
				}
				return impl.Trainer_->EndCollection();
			});
		if (recorded.IsError()) return recorded;
		auto completion = impl.Environment_.Submit();
		if (completion.IsError()) return completion.GetStatus();
		OA_RETURN_IF_ERROR(impl.Environment_.Wait(*completion));
	}

	OA_RETURN_IF_ERROR(impl.Trainer_->Update());
	const OaPpoTrainerMetrics& metrics = impl.Trainer_->Metrics();
	impl.Metrics_ = {
		.Rollout_ = metrics.Rollout,
		.UpdateEpoch_ = metrics.UpdateEpoch,
		.TotalLoss_ = metrics.TotalLoss,
		.PolicyLoss_ = metrics.PolicyLoss,
		.ValueLoss_ = metrics.ValueLoss,
		.Entropy_ = metrics.Entropy,
	};
	return OaStatus::Ok();
}

OaStatus OaTutorialLunarLander3dPpo::PretrainScriptedTeacher(
	const OaTutorialLunarLander3dTeacherConfig& InConfig) {
	if (not Impl_ or not Impl_->Model_ or not Impl_->Trainer_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher pretraining requires a live PPO session");
	}
	OA_RETURN_IF_ERROR(Impl_->ValidateExecutionContext());
	OaContext::RecordingScope recording(*Impl_->Context_);
	if (Impl_->Trainer_->Phase() != OaRlTrainingPhase::Collect
		or Impl_->Optimizer_.GetStep() != 0U) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher pretraining must run before PPO collection");
	}
	if (InConfig.Episodes_ == 0U or InConfig.Epochs_ == 0U
		or InConfig.BatchSize_ == 0U or InConfig.MaximumSamples_ == 0U
		or not std::isfinite(InConfig.LearningRate_)
		or InConfig.LearningRate_ <= 0.0F) {
		return OaStatus::InvalidArgument(
			"Lunar Lander 3D teacher pretraining requires non-zero dimensions and a positive finite learning rate");
	}
	if (InConfig.EnvironmentSeed_
		== OaTutorialLunarLander3dFirstEpisodeEvaluationConfig{}.EnvironmentSeed_) {
		return OaStatus::InvalidArgument(
			"Lunar Lander 3D teacher and held-out evaluator seeds must be disjoint");
	}

	OaTutorialLunarLander3dTeacherMetrics metrics;
	OaLunarLander3dConfig environmentConfig;
	std::vector<OaLunarTeacherSample> samples;
	samples.reserve(InConfig.MaximumSamples_);
	OaLunarEvaluationDigest datasetDigest;
	datasetDigest.AddU64(environmentConfig.ContractFingerprint());
	datasetDigest.AddU64(InConfig.EnvironmentSeed_);
	for (OaU32 lane = 0U;
		lane < InConfig.Episodes_ and samples.size() < InConfig.MaximumSamples_;
		++lane) {
		const OaLunarEpisodeManifest manifest = OaLunarEpisodeManifest::Derive(
			InConfig.EnvironmentSeed_, lane, 0U,
			environmentConfig.ContractFingerprint());
		auto environment = OaLunarScalarEnvironment::CreateFlat(
			environmentConfig, manifest);
		if (not environment.IsValid()) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				OaString("Lunar Lander 3D teacher environment creation failed: ")
					+ environment.Error());
		}
		while (not environment.State().Terminated_
			and not environment.State().Truncated_
			and samples.size() < InConfig.MaximumSamples_) {
			OaLunarTeacherSample sample;
			sample.Observation = environment.Observation();
			const OaLunarAction action = OaLunarScriptedLandingAction(
				environmentConfig, environment.State());
			sample.Action = static_cast<OaI32>(action);
			for (const OaF32 value : sample.Observation) {
				if (not std::isfinite(value)) {
					return OaStatus::Error(
						OaStatusCode::DataLoss,
						"Lunar Lander 3D teacher produced a non-finite observation");
				}
				datasetDigest.AddF32(value);
			}
			datasetDigest.AddI32(sample.Action);
			++metrics.ActionCounts_[static_cast<OaUsize>(action)];
			samples.push_back(sample);
			const OaLunarTransition transition = environment.Step(
				static_cast<OaU32>(action));
			if (not transition.Valid_) {
				return OaStatus::Error(
					OaStatusCode::DataLoss,
					OaString("Lunar Lander 3D teacher transition failed: ")
						+ transition.Error_);
			}
		}
		if (environment.State().Terminated_ or environment.State().Truncated_) {
			++metrics.Episodes_;
			switch (environment.State().EndReason_) {
				case OaLunarEndReason::SafeLanding: ++metrics.SafeLandings_; break;
				case OaLunarEndReason::BodyImpact: ++metrics.BodyImpacts_; break;
				case OaLunarEndReason::HardFootImpact:
					++metrics.HardFootImpacts_;
					break;
				case OaLunarEndReason::OutOfBounds: ++metrics.OutOfBounds_; break;
				case OaLunarEndReason::TimeLimit: ++metrics.TimeLimits_; break;
				default: ++metrics.OtherFailures_; break;
			}
		}
	}
	if (samples.empty()) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D teacher produced no training samples");
	}
	metrics.Samples_ = static_cast<OaU32>(samples.size());
	metrics.DatasetDigest_ = datasetDigest.Value();
	const OaUsize probeSamples = std::min<OaUsize>(
		samples.size(), InConfig.BatchSize_);
	auto initialLoss = OaLunarEvaluateTeacherProbe(
		*Impl_->Model_, samples, probeSamples);
	if (initialLoss.IsError()) return initialLoss.GetStatus();
	metrics.InitialLoss_ = *initialLoss;

	OaVec<OaParameter*> policyParameters;
	for (const OaNamedParameter& named : Impl_->Model_->AllNamedParameterPtrs()) {
		constexpr char prefix[] = "policy";
		if (named.Param != nullptr and named.Path.Size() >= sizeof(prefix) - 1U
			and std::memcmp(
				named.Path.Data(), prefix, sizeof(prefix) - 1U) == 0) {
			policyParameters.PushBack(named.Param);
		}
	}
	if (policyParameters.Empty()) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"Lunar Lander 3D teacher could not resolve policy parameters");
	}

	OaAdamW imitationOptimizer(
		policyParameters, InConfig.LearningRate_, 0.9F, 0.999F, 1.0e-8F, 0.0F);
	const OaU64 ppoOptimizerStep = Impl_->Optimizer_.GetStep();
	const OaU32 stepsPerEpoch = static_cast<OaU32>(
		(samples.size() + InConfig.BatchSize_ - 1U) / InConfig.BatchSize_);
	const OaU64 totalSteps64 = static_cast<OaU64>(InConfig.Epochs_)
		* stepsPerEpoch;
	if (totalSteps64 > static_cast<OaU64>(std::numeric_limits<OaI64>::max())) {
		return OaStatus::InvalidArgument(
			"Lunar Lander 3D teacher training step count is too large");
	}
	OaItTraining training(imitationOptimizer, OaItTrainingConfig{
		.TotalSteps = static_cast<OaI64>(totalSteps64),
		.StepsPerEpoch = static_cast<OaI64>(stepsPerEpoch),
		.EpochSteps = {},
		.BatchSize = static_cast<OaI32>(std::min<OaU32>(
			InConfig.BatchSize_, static_cast<OaU32>(
				std::numeric_limits<OaI32>::max()))),
		.TimerName = "lunar_teacher_imitation",
		.Metrics = {},
		.Callbacks = {},
		.Program = nullptr,
	});
	std::vector<OaUsize> order(samples.size());
	std::iota(order.begin(), order.end(), OaUsize{0});
	std::mt19937_64 random(InConfig.ShuffleSeed_);
	while (not training.IsDone()) {
		const OaU64 zeroBasedStep = static_cast<OaU64>(training.Index() - 1);
		const OaU32 stepInEpoch = static_cast<OaU32>(
			zeroBasedStep % stepsPerEpoch);
		if (stepInEpoch == 0U) {
			std::shuffle(order.begin(), order.end(), random);
		}
		const OaUsize begin = static_cast<OaUsize>(stepInEpoch)
			* InConfig.BatchSize_;
		const OaUsize count = std::min<OaUsize>(
			InConfig.BatchSize_, samples.size() - begin);
		std::vector<OaF32> observations(
			count * OA_LUNAR_OBSERVATION_SIZE);
		std::vector<OaI32> actions(count);
		for (OaUsize row = 0U; row < count; ++row) {
			const OaLunarTeacherSample& sample = samples[order[begin + row]];
			std::memcpy(
				observations.data() + row * OA_LUNAR_OBSERVATION_SIZE,
				sample.Observation.data(),
				OA_LUNAR_OBSERVATION_SIZE * sizeof(OaF32));
			actions[row] = sample.Action;
		}
		const OaMatrix observation = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(
				reinterpret_cast<const OaU8*>(observations.data()),
				observations.size() * sizeof(OaF32)),
			{static_cast<OaI64>(count), OA_LUNAR_OBSERVATION_SIZE},
			OaScalarType::Float32);
		const OaMatrix action = OaFnMatrix::FromInt32(
			OaSpan<const OaI32>(actions.data(), actions.size()),
			{static_cast<OaI64>(count)}, OaScalarType::Int32);
		if (observation.IsEmpty() or action.IsEmpty()) {
			return OaStatus::Error(
				OaStatusCode::OutOfMemory,
				"Lunar Lander 3D teacher batch upload failed");
		}
		imitationOptimizer.ZeroGrad();
		OaGradientTape tape;
		const OaRlActorCriticOutput output = Impl_->Model_->Evaluate(observation);
		if (not output.IsValid()) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"Lunar Lander 3D teacher policy evaluation failed");
		}
		const OaMatrix loss = OaFnLoss::CrossEntropy(output.Logits, action);
		if (loss.IsEmpty()) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"Lunar Lander 3D teacher cross-entropy construction failed");
		}
		tape.Backward(loss);
		training.Next(loss);
		if (training.LastStatus().IsError()) return training.LastStatus();
		if (not std::isfinite(training.LastLoss())) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"Lunar Lander 3D teacher loss became non-finite");
		}
		++metrics.OptimizerSteps_;
	}
	OA_RETURN_IF_ERROR(training.Finish());
	if (imitationOptimizer.GetStep() != metrics.OptimizerSteps_
		or Impl_->Optimizer_.GetStep() != ppoOptimizerStep) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"Lunar Lander 3D teacher optimizer accounting diverged");
	}
	auto finalLoss = OaLunarEvaluateTeacherProbe(
		*Impl_->Model_, samples, probeSamples);
	if (finalLoss.IsError()) return finalLoss.GetStatus();
	metrics.FinalLoss_ = *finalLoss;
	Impl_->TeacherMetrics_ = metrics;
	return OaStatus::Ok();
}

const OaTutorialLunarLander3dTeacherMetrics&
OaTutorialLunarLander3dPpo::TeacherMetrics() const noexcept {
	return Impl_->TeacherMetrics_;
}

OaResult<OaTutorialLunarLander3dFirstEpisodeEvaluation>
OaTutorialLunarLander3dPpo::EvaluateFirstEpisodes(
	const OaTutorialLunarLander3dFirstEpisodeEvaluationConfig& InConfig) {
	if (not Impl_ or not Impl_->Model_ or not Impl_->Engine_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO evaluation requires a live session");
	}
	OA_RETURN_IF_ERROR(Impl_->ValidateExecutionContext());
	OaContext::RecordingScope recording(*Impl_->Context_);
	if (InConfig.Environments_ == 0U or InConfig.Horizon_ == 0U
		or InConfig.SubmissionChunkSteps_ == 0U) {
		return OaStatus::InvalidArgument(
			"Lunar Lander 3D first-episode evaluation dimensions must be non-zero");
	}

	OaLunarLander3dVectorConfig environmentConfig;
	environmentConfig.Environments_ = InConfig.Environments_;
	environmentConfig.Seed_ = InConfig.EnvironmentSeed_;
	auto created = OaLunarLander3dVector::CreateFlat(
		*Impl_->Engine_, environmentConfig);
	if (created.IsError()) return created.GetStatus();
	auto environment = OaStdMove(*created);

	// Keep cleanup outside the worker lambda so every result path explicitly
	// closes the fresh environment, including submission/readback failures.
	auto evaluation = [&]()
		-> OaResult<OaTutorialLunarLander3dFirstEpisodeEvaluation> {
		OaTutorialLunarLander3dFirstEpisodeEvaluation result;
		result.ExpectedEpisodes_ = InConfig.Environments_;
		OaLunarEvaluationDigest actionDigest;
		OaLunarEvaluationDigest valueDigest;
		OaVec<OaU8> completed(InConfig.Environments_, 0U);
		OaVec<OaU32> terminalReasons(InConfig.Environments_, 0U);
		OaVec<OaF32> accumulatedReturns(InConfig.Environments_, 0.0F);

		for (OaU32 chunkStart = 0U; chunkStart < InConfig.Horizon_;) {
			const OaU32 chunkSteps = std::min(
				InConfig.SubmissionChunkSteps_, InConfig.Horizon_ - chunkStart);
			OaMatrix actionHistory;
			OaMatrix valueHistory;
			OaMatrix rewardHistory;
			OaMatrix endReasonHistory;
			OaVec<OaMatrix> actions;
			OaVec<OaMatrix> values;
			OaVec<OaMatrix> rewards;
			OaVec<OaMatrix> endReasons;
			actions.Reserve(chunkSteps);
			values.Reserve(chunkSteps);
			rewards.Reserve(chunkSteps);
			endReasons.Reserve(chunkSteps);
			const OaStatus recorded = environment.RecordCommands(
				[&]() -> OaStatus {
					OaGradNo noGrad;
					for (OaU32 step = 0U; step < chunkSteps; ++step) {
						const OaRlActorCriticOutput policy =
							Impl_->Model_->Evaluate(environment.Observation());
						if (not policy.IsValid()
							or policy.Logits.GetDtype() != OaScalarType::Float32
							or policy.Logits.GetShape()
								!= OaMatrixShape{
									static_cast<OaI64>(InConfig.Environments_),
									Impl_->ActionCount_}
							or policy.Value.GetDtype() != OaScalarType::Float32
							or policy.Value.GetShape()
								!= OaMatrixShape{
									static_cast<OaI64>(InConfig.Environments_)}) {
							return OaStatus::Error(
								OaStatusCode::FailedPrecondition,
								"Lunar Lander 3D greedy evaluation produced an invalid policy output");
						}
						const OaTopKResult best = OaFnMatrix::TopK(
							policy.Logits, 1, 1);
						if (best.Indices.IsEmpty()
							or best.Indices.GetDtype() != OaScalarType::Int32) {
							return OaStatus::Error(
								OaStatusCode::FailedPrecondition,
								"Lunar Lander 3D greedy TopK evaluation failed");
						}
						const OaMatrix action = OaFnMatrix::Reshape(
							best.Indices,
							{static_cast<OaI64>(InConfig.Environments_)});
						auto transition = environment.Step(action);
						if (transition.IsError()) return transition.GetStatus();

						actions.PushBack(action.Clone());
						values.PushBack(policy.Value.Clone());
						rewards.PushBack(transition->Reward_.Clone());
						endReasons.PushBack(transition->EndReason_.Clone());
						if (actions.Back().IsEmpty() or values.Back().IsEmpty()
							or rewards.Back().IsEmpty()
							or endReasons.Back().IsEmpty()) {
							return OaStatus::Error(
								OaStatusCode::OutOfMemory,
								"Lunar Lander 3D evaluation could not retain its bounded history");
						}
					}
					actionHistory = OaFnMatrix::Concat(
						OaSpan<OaMatrix>(actions), 0);
					valueHistory = OaFnMatrix::Concat(
						OaSpan<OaMatrix>(values), 0);
					rewardHistory = OaFnMatrix::Concat(
						OaSpan<OaMatrix>(rewards), 0);
					endReasonHistory = OaFnMatrix::Concat(
						OaSpan<OaMatrix>(endReasons), 0);
					if (actionHistory.IsEmpty() or valueHistory.IsEmpty()
						or rewardHistory.IsEmpty()
						or endReasonHistory.IsEmpty()) {
						return OaStatus::Error(
							OaStatusCode::OutOfMemory,
							"Lunar Lander 3D evaluation history concatenation failed");
					}
					return OaStatus::Ok();
				});
			if (recorded.IsError()) return recorded;
			auto completion = environment.Submit();
			if (completion.IsError()) return completion.GetStatus();
			const OaStatus waited = environment.Wait(*completion);
			if (waited.IsError()) return waited;
			++result.Submissions_;

			auto actionsResult = OaLunarCopyMatrix<OaI32>(actionHistory);
			if (actionsResult.IsError()) return actionsResult.GetStatus();
			auto valuesResult = OaLunarCopyMatrix<OaF32>(valueHistory);
			if (valuesResult.IsError()) return valuesResult.GetStatus();
			auto rewardsResult = OaLunarCopyMatrix<OaF32>(rewardHistory);
			if (rewardsResult.IsError()) return rewardsResult.GetStatus();
			auto endReasonsResult = OaLunarCopyMatrix<OaU32>(endReasonHistory);
			if (endReasonsResult.IsError()) return endReasonsResult.GetStatus();

			OaVec<OaI32> actionHost = OaStdMove(actionsResult).GetValue();
			OaVec<OaF32> valueHost = OaStdMove(valuesResult).GetValue();
			OaVec<OaF32> rewardHost = OaStdMove(rewardsResult).GetValue();
			OaVec<OaU32> endReasonHost =
				OaStdMove(endReasonsResult).GetValue();
			const OaUsize expectedElements =
				static_cast<OaUsize>(chunkSteps) * InConfig.Environments_;
			if (actionHost.Size() != expectedElements
				or valueHost.Size() != expectedElements
				or rewardHost.Size() != expectedElements
				or endReasonHost.Size() != expectedElements) {
				return OaStatus::Error(
					OaStatusCode::DataLoss,
					"Lunar Lander 3D evaluation history has an unexpected extent");
			}

			for (OaU32 step = 0U; step < chunkSteps; ++step) {
				for (OaU32 lane = 0U; lane < InConfig.Environments_; ++lane) {
					const OaUsize index = static_cast<OaUsize>(step)
						* InConfig.Environments_ + lane;
					const OaI32 action = actionHost[index];
					const OaF32 value = valueHost[index];
					const OaF32 reward = rewardHost[index];
					const OaU32 reason = endReasonHost[index];
					if (action < 0 or action >= Impl_->ActionCount_) {
						return OaStatus::Error(
							OaStatusCode::DataLoss,
							"Lunar Lander 3D evaluation produced an invalid greedy action");
					}
					if (not std::isfinite(value) or not std::isfinite(reward)) {
						return OaStatus::Error(
							OaStatusCode::DataLoss,
							"Lunar Lander 3D evaluation produced a non-finite value or reward");
					}
					if (reason > static_cast<OaU32>(
						OaLunarEndReason::InvalidAction)) {
						return OaStatus::Error(
							OaStatusCode::DataLoss,
							"Lunar Lander 3D evaluation produced an unknown end reason");
					}
					actionDigest.AddI32(action);
					valueDigest.AddF32(value);
					const bool wasCompleted = completed[lane] != 0U;
					const bool isCompleted = reason != static_cast<OaU32>(
						OaLunarEndReason::None);
					if (wasCompleted) {
						if (not isCompleted or reward != 0.0F
							or terminalReasons[lane] != reason) {
							return OaStatus::Error(
								OaStatusCode::DataLoss,
								"Lunar Lander 3D completed lane did not remain terminal and reward-free");
						}
					} else {
						++result.ActionCounts_[static_cast<OaUsize>(action)];
						accumulatedReturns[lane] += reward;
						if (not std::isfinite(accumulatedReturns[lane])) {
							return OaStatus::Error(
								OaStatusCode::DataLoss,
								"Lunar Lander 3D evaluation return became non-finite");
						}
						if (isCompleted) {
							completed[lane] = 1U;
							terminalReasons[lane] = reason;
						}
					}
				}
			}
			result.RecordedEnvironmentSteps_ +=
				static_cast<OaU64>(chunkSteps) * InConfig.Environments_;
			chunkStart += chunkSteps;
			bool allCompleted = true;
			for (const OaU8 laneCompleted : completed) {
				allCompleted = allCompleted and laneCompleted != 0U;
			}
			if (allCompleted) break;
		}

		if (result.Submissions_ != environment.SubmissionCount()) {
			return OaStatus::Error(
				OaStatusCode::Internal,
				"Lunar Lander 3D evaluation submission accounting diverged");
		}
		auto telemetryResult = environment.CopyEpisodeTelemetry();
		if (telemetryResult.IsError()) return telemetryResult.GetStatus();
		OaVec<OaLunarLander3dEpisodeTelemetry> telemetry =
			OaStdMove(telemetryResult).GetValue();
		if (telemetry.Size() != InConfig.Environments_) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"Lunar Lander 3D evaluation telemetry has an unexpected lane count");
		}

		OaF64 returnSum = 0.0;
		OaF64 stepSum = 0.0;
		OaF64 fuelSum = 0.0;
		OaF64 linearSpeedSum = 0.0;
		OaF64 angularSpeedSum = 0.0;
		OaF64 footImpulseSum = 0.0;
		result.MinReturn_ = std::numeric_limits<OaF64>::infinity();
		result.MaxReturn_ = -std::numeric_limits<OaF64>::infinity();
		for (OaU32 lane = 0U; lane < InConfig.Environments_; ++lane) {
			const OaLunarLander3dEpisodeTelemetry& episode = telemetry[lane];
			const bool isCompleted = episode.Terminated_ or episode.Truncated_;
			if (not episode.IsFinite()
				or episode.EpisodeStep_ > InConfig.Horizon_
				or (completed[lane] != 0U) != isCompleted
				or (isCompleted and terminalReasons[lane]
					!= static_cast<OaU32>(episode.EndReason_))) {
				return OaStatus::Error(
					OaStatusCode::DataLoss,
					"Lunar Lander 3D final telemetry disagrees with the recorded first episode");
			}
			const OaF64 returnDifference = std::abs(
				static_cast<OaF64>(episode.EpisodeReturn_)
				- accumulatedReturns[lane]);
			const OaF64 returnTolerance = 1.0e-3
				+ 2.0e-5 * std::abs(
					static_cast<OaF64>(episode.EpisodeReturn_));
			if (returnDifference > returnTolerance) {
				return OaStatus::Error(
					OaStatusCode::DataLoss,
					"Lunar Lander 3D final return disagrees with transition history");
			}

			if (isCompleted) ++result.CompletedEpisodes_;
			switch (episode.EndReason_) {
				case OaLunarEndReason::None:
					++result.IncompleteEpisodes_;
					break;
				case OaLunarEndReason::SafeLanding:
					++result.SafeLandings_;
					break;
				case OaLunarEndReason::BodyImpact:
					++result.BodyImpacts_;
					break;
				case OaLunarEndReason::HardFootImpact:
					++result.HardFootImpacts_;
					break;
				case OaLunarEndReason::OutOfBounds:
					++result.OutOfBounds_;
					break;
				case OaLunarEndReason::NumericalFailure:
					++result.NumericalFailures_;
					break;
				case OaLunarEndReason::TimeLimit:
					++result.TimeLimits_;
					break;
				case OaLunarEndReason::ExternalStop:
					++result.ExternalStops_;
					break;
				case OaLunarEndReason::InvalidAction:
					++result.InvalidActions_;
					break;
			}
			const OaF64 episodeReturn = episode.EpisodeReturn_;
			returnSum += episodeReturn;
			stepSum += episode.EpisodeStep_;
			fuelSum += episode.FuelRemaining_;
			linearSpeedSum += episode.TerminalLinearSpeed_;
			angularSpeedSum += episode.TerminalAngularSpeed_;
			footImpulseSum += episode.MaximumFootImpulse_;
			result.MinReturn_ = std::min(result.MinReturn_, episodeReturn);
			result.MaxReturn_ = std::max(result.MaxReturn_, episodeReturn);
		}
		const OaU32 reasonCount = result.SafeLandings_
			+ result.BodyImpacts_ + result.HardFootImpacts_
			+ result.OutOfBounds_ + result.NumericalFailures_
			+ result.TimeLimits_ + result.ExternalStops_
			+ result.InvalidActions_;
		if (reasonCount != result.CompletedEpisodes_
			or result.CompletedEpisodes_ + result.IncompleteEpisodes_
				!= result.ExpectedEpisodes_) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"Lunar Lander 3D evaluation reason accounting diverged");
		}

		const OaF64 episodes = static_cast<OaF64>(result.ExpectedEpisodes_);
		result.SafeLandingRate_ =
			static_cast<OaF64>(result.SafeLandings_) / episodes;
		result.WilsonLower95_ = OaLunarWilsonLower95(
			result.SafeLandings_, result.ExpectedEpisodes_);
		result.MeanReturn_ = returnSum / episodes;
		result.MeanEpisodeSteps_ = stepSum / episodes;
		result.MeanFuelRemaining_ = fuelSum / episodes;
		result.MeanTerminalLinearSpeed_ = linearSpeedSum / episodes;
		result.MeanTerminalAngularSpeed_ = angularSpeedSum / episodes;
		result.MeanMaximumFootImpulse_ = footImpulseSum / episodes;
		result.ActionTraceDigest_ = actionDigest.Value();
		result.ValueTraceDigest_ = valueDigest.Value();
		if (not std::isfinite(result.SafeLandingRate_)
			or not std::isfinite(result.WilsonLower95_)
			or not std::isfinite(result.MeanReturn_)
			or not std::isfinite(result.MinReturn_)
			or not std::isfinite(result.MaxReturn_)
			or not std::isfinite(result.MeanEpisodeSteps_)
			or not std::isfinite(result.MeanFuelRemaining_)
			or not std::isfinite(result.MeanTerminalLinearSpeed_)
			or not std::isfinite(result.MeanTerminalAngularSpeed_)
			or not std::isfinite(result.MeanMaximumFootImpulse_)) {
			return OaStatus::Error(
				OaStatusCode::DataLoss,
				"Lunar Lander 3D evaluation aggregate became non-finite");
		}
		return result;
	}();

	const OaStatus closeStatus = environment.Close();
	if (evaluation.IsError()) return evaluation.GetStatus();
	if (closeStatus.IsError()) return closeStatus;
	return OaStdMove(evaluation).GetValue();
}

OaStatus OaTutorialLunarLander3dPpo::Save(
	const OaString& InPath) const {
	if (not Impl_ or not Impl_->Trainer_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO session is empty");
	}
	OA_RETURN_IF_ERROR(Impl_->ValidateExecutionContext());
	OaContext::RecordingScope recording(*Impl_->Context_);
	return Impl_->Trainer_->Save(InPath);
}

OaStatus OaTutorialLunarLander3dPpo::Load(const OaString& InPath) {
	if (not Impl_ or not Impl_->Trainer_) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"Lunar Lander 3D PPO session is empty");
	}
	OA_RETURN_IF_ERROR(Impl_->ValidateExecutionContext());
	OaContext::RecordingScope recording(*Impl_->Context_);
	return Impl_->Trainer_->Load(InPath);
}

OaStatus OaTutorialLunarLander3dPpo::Close() {
	if (not Impl_) return OaStatus::Ok();
	OA_RETURN_IF_ERROR(Impl_->ValidateExecutionContext());
	OaStatus closeStatus;
	{
		OaContext::RecordingScope recording(*Impl_->Context_);
		closeStatus = Impl_->Environment_.Close();
		if (closeStatus.IsOk()) closeStatus = Impl_->Context_->Sync();
	}
	if (closeStatus.IsError()) return closeStatus;
	// A closed session must release every engine-owned matrix and private
	// context before its borrowed engine may be closed.
	Impl_.Reset();
	return OaStatus::Ok();
}
