#include "CartPolePpo.h"

#include "CartPole.h"

#include <Oa/Core/FnMatrix.h>
#include <Oa/Ml.h>
#include <Oa/Ml/Autograd.h>
#include <Oa/Ml/Rl.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Runtime/Engine.h>

#include <cmath>
#include <numeric>
#include <vector>

namespace {

template<typename T>
OaResult<std::vector<T>> Copy(const OaMatrix& InMatrix) {
	std::vector<T> result(static_cast<OaUsize>(InMatrix.NumElements()));
	const OaStatus status = OaFnMatrix::CopyToHost(
		InMatrix, result.data(), result.size() * sizeof(T));
	if (status.IsError()) return status;
	return result;
}

} // namespace

struct OaTutorialCartPolePpo::Impl {
	OaTutorialCartPolePpoConfig Config;
	OaUniquePtr<OaCategoricalActorCritic> Model;
	OaVec<OaParameter*> Parameters;
	OaAdamW Optimizer;
	OaTutorialCartPole Environment;
	OaUniquePtr<OaPpoTrainer> Trainer;
	OaUniquePtr<OaTrainingSession> Control;
	OaTutorialCartPolePpoMetrics Metrics;

	Impl(
		const OaTutorialCartPolePpoConfig& InConfig,
		OaUniquePtr<OaCategoricalActorCritic> InModel,
		OaTutorialCartPole&& InEnvironment)
		: Config(InConfig)
		, Model(OaStdMove(InModel))
		, Parameters(Model->AllParameterPtrs())
		, Optimizer(Parameters, InConfig.LearningRate,
			0.9F, 0.999F, 1.0e-8F, 0.0F)
		, Environment(OaStdMove(InEnvironment)) {}
};

OaTutorialCartPolePpo::OaTutorialCartPolePpo(OaUniquePtr<Impl> InImpl)
	: Impl_(OaStdMove(InImpl)) {}

OaTutorialCartPolePpo::~OaTutorialCartPolePpo() = default;

OaResult<OaUniquePtr<OaTutorialCartPolePpo>> OaTutorialCartPolePpo::Create(
	const OaTutorialCartPolePpoConfig& InConfig) {
	if (InConfig.Environments == 0 || InConfig.Horizon == 0
		|| InConfig.Rollouts == 0 || InConfig.UpdateEpochs == 0
		|| !std::isfinite(InConfig.LearningRate)
		|| InConfig.LearningRate <= 0.0F) {
		return OaStatus::InvalidArgument(
			"OaTutorialCartPolePpo requires non-zero dimensions and a positive finite learning rate");
	}
	OaFnMatrix::SetRngSeed(InConfig.TrainingSeed);
	auto environment = OaTutorialCartPole::Create(OaTutorialCartPoleConfig{
		.Environments = InConfig.Environments,
		.MaxEpisodeSteps = 500,
		.Seed = InConfig.TrainingSeed,
	});
	if (environment.IsError()) return environment.GetStatus();
	auto model = OaCategoricalActorCritic::Create(
		OaCategoricalActorCriticConfig{
			.ObservationSize = 4,
			.ActionCount = 2,
			.HiddenSize = 64,
		});
	if (model.IsError()) return model.GetStatus();
	auto impl = OaMakeUniquePtr<Impl>(
		InConfig, OaStdMove(*model), OaStdMove(*environment));
	auto trainer = OaPpoTrainer::Create(
		*impl->Model, impl->Optimizer, OaPpoTrainerConfig{
			.Rollouts = InConfig.Rollouts,
			.Horizon = InConfig.Horizon,
			.Environments = InConfig.Environments,
			.UpdateEpochs = InConfig.UpdateEpochs,
			.ObservationShape = {4},
			.Seed = InConfig.TrainingSeed,
			.Gae = {},
			.Loss = OaPpoLossConfig{
				.ClipEpsilon = 0.2F,
				.ValueCoefficient = 0.5F,
				.EntropyCoefficient = 0.01F,
			},
	});
	if (trainer.IsError()) return trainer.GetStatus();
	impl->Trainer = OaStdMove(*trainer);
	impl->Control = OaMakeUniquePtr<OaTrainingSession>(
		impl->Trainer->TrainingLoop());
	return OaUniquePtr<OaTutorialCartPolePpo>(
		new OaTutorialCartPolePpo(OaStdMove(impl)));
}

OaStatus OaTutorialCartPolePpo::Advance() {
	if (!Impl_) return OaStatus::Error(
		OaStatusCode::FailedPrecondition, "CartPole PPO session is empty");
	if (Impl_->Trainer->IsDone()) return OaStatus::Ok();
	auto& impl = *Impl_;
	if (impl.Trainer->NeedsCollection()) {
		OA_RETURN_IF_ERROR(impl.Trainer->BeginCollection());
		for (OaU32 step = 0; step < impl.Config.Horizon; ++step) {
			const OaRlPolicyResult policy = impl.Trainer->Act(
				impl.Environment.Observation());
			if (!policy.IsValid()) return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"CartPole PPO policy evaluation failed");
			auto transition = impl.Environment.Step(policy.Action);
			if (transition.IsError()) return transition.GetStatus();
			OA_RETURN_IF_ERROR(impl.Trainer->Observe(
				transition->Observation, transition->NextObservation,
				transition->Reward, transition->Terminated,
				transition->Truncated, policy));
			impl.Environment.ResetDone();
		}
		OA_RETURN_IF_ERROR(impl.Trainer->EndCollection());
	}
	OA_RETURN_IF_ERROR(impl.Trainer->Update());
	const OaPpoTrainerMetrics& metrics = impl.Trainer->Metrics();
	impl.Metrics.TotalLoss = metrics.TotalLoss;
	impl.Metrics.PolicyLoss = metrics.PolicyLoss;
	impl.Metrics.ValueLoss = metrics.ValueLoss;
	impl.Metrics.Entropy = metrics.Entropy;
	impl.Metrics.UpdateEpoch = metrics.UpdateEpoch;
	if (impl.Trainer->Phase() != OaRlTrainingPhase::Update) {
		impl.Metrics.Rollout = metrics.Rollout;
		impl.Metrics.LossHistory.PushBack(impl.Metrics.TotalLoss);
		impl.Metrics.PolicyLossHistory.PushBack(impl.Metrics.PolicyLoss);
		impl.Metrics.ValueLossHistory.PushBack(impl.Metrics.ValueLoss);
		impl.Metrics.EntropyHistory.PushBack(impl.Metrics.Entropy);
	}
	return OaStatus::Ok();
}

bool OaTutorialCartPolePpo::IsDone() const noexcept {
	return Impl_ && Impl_->Trainer->IsDone();
}

const OaTutorialCartPolePpoConfig& OaTutorialCartPolePpo::Config() const noexcept {
	return Impl_->Config;
}

const OaTutorialCartPolePpoMetrics& OaTutorialCartPolePpo::Metrics() const noexcept {
	return Impl_->Metrics;
}

OaResult<OaTutorialCartPoleSnapshot> OaTutorialCartPolePpo::SnapshotLane(
	OaU32 InLane) {
	if (!Impl_ || InLane >= Impl_->Config.Environments) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange, "CartPole snapshot lane is out of range");
	}
	const OaMatrix& state = Impl_->Environment.Observation();
	auto* runtime = OaEngine::GetGlobal();
	const OaU64 offset = static_cast<OaU64>(InLane) * 4U * sizeof(OaF32);
	if (runtime == nullptr || !runtime->Allocator.InvalidateHostBuffer(
		state.GetVkBuffer(), offset, 4U * sizeof(OaF32))) {
		return OaStatus::Error(
			OaStatusCode::VulkanError, "CartPole snapshot invalidate failed");
	}
	const auto* values = state.DataAs<OaF32>() + InLane * 4U;
	return OaTutorialCartPoleSnapshot{
		.CartPosition = values[0],
		.CartVelocity = values[1],
		.PoleAngle = values[2],
		.PoleAngularVelocity = values[3],
	};
}

OaStatus OaTutorialCartPolePpo::Demonstrate() {
	if (!Impl_) return OaStatus::Error(
		OaStatusCode::FailedPrecondition, "CartPole PPO session is empty");
	{
		OaGradNo noGrad;
		const OaMatrix logits = Impl_->Model->Forward(
			Impl_->Environment.Observation());
		const OaTopKResult best = OaFnMatrix::TopK(logits, 1, 1);
		const OaMatrix action = OaFnMatrix::Reshape(
			best.Indices,
			{static_cast<OaI64>(Impl_->Config.Environments)});
		auto transition = Impl_->Environment.Step(action);
		if (transition.IsError()) return transition.GetStatus();
		Impl_->Environment.ResetDone();
	}
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	return OaContext::GetDefault().Sync();
}

OaResult<OaTutorialCartPolePpoEvaluation> OaTutorialCartPolePpo::Evaluate(
	OaU64 InSeed,
	OaU32 InEnvironments,
	OaU32 InHorizon) {
	if (!Impl_ || InEnvironments == 0 || InHorizon == 0) {
		return OaStatus::InvalidArgument("CartPole evaluation dimensions must be non-zero");
	}
	auto environment = OaTutorialCartPole::Create(OaTutorialCartPoleConfig{
		.Environments = InEnvironments,
		.MaxEpisodeSteps = 500,
		.Seed = InSeed,
	});
	if (environment.IsError()) return environment.GetStatus();
	auto rollout = OaRlRolloutBuffer::Create(OaRlRolloutConfig{
		.Time = InHorizon,
		.Environments = InEnvironments,
		.ObservationShape = {4},
	});
	if (rollout.IsError()) return rollout.GetStatus();
	const OaMatrix zero = OaFnMatrix::Zeros(
		{static_cast<OaI64>(InEnvironments)}, OaScalarType::Float32);
	{
		OaGradNo noGrad;
		for (OaU32 step = 0; step < InHorizon; ++step) {
			const OaMatrix logits = Impl_->Model->Forward(environment->Observation());
			const OaTopKResult best = OaFnMatrix::TopK(logits, 1, 1);
			const OaMatrix action = OaFnMatrix::Reshape(
				best.Indices, {static_cast<OaI64>(InEnvironments)});
			auto transition = environment->Step(action);
			if (transition.IsError()) return transition.GetStatus();
			OA_RETURN_IF_ERROR(rollout->Append(OaRlTransition{
				.Observation = transition->Observation,
				.Action = action,
				.Reward = transition->Reward,
				.Value = zero,
				.NextValue = zero,
				.LogProbability = zero,
				.Terminated = transition->Terminated,
				.Truncated = transition->Truncated,
			}));
			environment->ResetDone();
		}
	}
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Execute());
	OA_RETURN_IF_ERROR(OaContext::GetDefault().Sync());
	auto reward = Copy<OaF32>(rollout->Batch().Reward);
	if (reward.IsError()) return reward.GetStatus();
	auto terminated = Copy<OaU8>(rollout->Batch().Terminated);
	if (terminated.IsError()) return terminated.GetStatus();
	auto truncated = Copy<OaU8>(rollout->Batch().Truncated);
	if (truncated.IsError()) return truncated.GetStatus();
	std::vector<OaF64> running(InEnvironments, 0.0);
	std::vector<OaF64> completed;
	for (OaU32 step = 0; step < InHorizon; ++step) {
		for (OaU32 lane = 0; lane < InEnvironments; ++lane) {
			const OaUsize index = static_cast<OaUsize>(step) * InEnvironments + lane;
			running[lane] += (*reward)[index];
			if ((*terminated)[index] != 0 || (*truncated)[index] != 0) {
				completed.push_back(running[lane]);
				running[lane] = 0.0;
			}
		}
	}
	const OaF64 sum = std::accumulate(completed.begin(), completed.end(), 0.0);
	OaTutorialCartPolePpoEvaluation result{
		.MeanCompletedReturn = completed.empty()
			? 0.0 : sum / static_cast<OaF64>(completed.size()),
		.CompletedEpisodes = static_cast<OaU32>(completed.size()),
	};
	Impl_->Metrics.EvaluationReturnHistory.PushBack(
		static_cast<OaF32>(result.MeanCompletedReturn));
	return result;
}

OaStatus OaTutorialCartPolePpo::Save(const OaString& InPath) const {
	if (!Impl_) return OaStatus::Error(
		OaStatusCode::FailedPrecondition, "CartPole PPO session is empty");
	return Impl_->Trainer->Save(InPath);
}

OaStatus OaTutorialCartPolePpo::Load(const OaString& InPath) {
	if (!Impl_) return OaStatus::Error(
		OaStatusCode::FailedPrecondition, "CartPole PPO session is empty");
	return Impl_->Trainer->Load(InPath);
}

OaU64 OaTutorialCartPolePpo::OptimizerStep() const noexcept {
	return Impl_ ? Impl_->Optimizer.GetStep() : 0U;
}

OaTrainingSession& OaTutorialCartPolePpo::Control() noexcept {
	return *Impl_->Control;
}

const OaTrainingSession& OaTutorialCartPolePpo::Control() const noexcept {
	return *Impl_->Control;
}
