#include <oa/ml/sacTrainer.h>

#include <oa/core/fnMatrix.h>
#include <oa/ml/autograd.h>
#include <oa/ml/optim.h>
#include <oa/ml/policy.h>
#include <oa/ml/trainingSession.h>
#include <oa/runtime/executionSession.h>
#include <oa/core/std/scalarMath.h>

namespace {

oa::Status copyModel(oa::Module& inSource, oa::Module& inTarget) {
	auto source = inSource.allNamedParameterPtrs();
	auto target = inTarget.allNamedParameterPtrs();
	if (source.size() != target.size()) return oa::Status::error(
		oa::StatusCode::ShapeMismatch, "SAC critic schemas differ");
	for (oa::Usize index = 0; index < source.size(); ++index) {
		if (source[index].path != target[index].path
			|| source[index].param == nullptr || target[index].param == nullptr
			|| source[index].param->data.getShape()
				!= target[index].param->data.getShape()
			|| source[index].param->data.getDtype()
				!= target[index].param->data.getDtype()) {
			return oa::Status::error(oa::StatusCode::ShapeMismatch,
				"SAC critic schemas differ");
		}
		target[index].param->data = oa::FnMatrix::copy(source[index].param->data);
		target[index].param->data.setRequiresGrad(false);
		target[index].param->requiresGrad = false;
	}
	return oa::Status::ok();
}

oa::Matrix vectorQ(const oa::Matrix& inQ, oa::U32 inBatch) {
	if (inQ.getDtype() != oa::ScalarType::Float32) return {};
	if (inQ.getShape() == oa::MatrixShape{static_cast<oa::I64>(inBatch)}) return inQ;
	if (inQ.getShape() == oa::MatrixShape{static_cast<oa::I64>(inBatch), 1}) {
		return oa::FnMatrix::reshape(inQ, {static_cast<oa::I64>(inBatch)});
	}
	return {};
}

oa::Matrix criticInput(const oa::Matrix& inObservation, const oa::Matrix& inAction) {
	oa::Matrix parts[] = {inObservation, inAction};
	return oa::FnMatrix::concat(oa::Span<oa::Matrix>(parts, 2), 1);
}

oa::ContinuousPolicyResult actorPolicy(
	oa::Module& inActor,
	const oa::Matrix& inObservation,
	oa::U32 inBatch,
	oa::U32 inActionDimensions,
	oa::F32 inMinimum,
	oa::F32 inMaximum,
	oa::U64 inSeed) {
	const oa::Matrix output = inActor.forward(inObservation);
	if (output.getShape() != oa::MatrixShape{
		static_cast<oa::I64>(inBatch),
		static_cast<oa::I64>(2U * inActionDimensions)}
		|| output.getDtype() != oa::ScalarType::Float32) return {};
	oa::I64 sizes[] = {
		static_cast<oa::I64>(inActionDimensions),
		static_cast<oa::I64>(inActionDimensions)};
	auto split = oa::FnMatrix::split(output, oa::Span<oa::I64>(sizes, 2), 1);
	if (split.size() != 2) return {};
	const oa::Matrix value = oa::FnMatrix::zeros(
		{static_cast<oa::I64>(inBatch)}, oa::ScalarType::Float32);
	return oa::FnPolicy::sampleTanhNormal(
		split[0], split[1], value, inMinimum, inMaximum, inSeed);
}

} // namespace

struct oa::SacTrainer::Impl {
	oa::Engine& engine;
	oa::Module& actor;
	oa::Module& critic1;
	oa::Module& critic2;
	oa::Module& targetCritic1;
	oa::Module& targetCritic2;
	oa::Optimizer& actorOptimizer;
	oa::Optimizer& criticOptimizer;
	oa::ReplayBuffer& replay;
	oa::SacTrainerConfig config;
	oa::ItTraining criticTraining;
	oa::ItTraining actorTraining;
	oa::SacTrainerMetrics metrics;

	Impl(oa::Engine& inEngine, oa::Module& inActor,
		oa::Module& inCritic1, oa::Module& inCritic2,
		oa::Module& inTargetCritic1, oa::Module& inTargetCritic2,
		oa::Optimizer& inActorOptimizer, oa::Optimizer& inCriticOptimizer,
		oa::ReplayBuffer& inReplay, const oa::SacTrainerConfig& inConfig)
		: engine(inEngine), actor(inActor), critic1(inCritic1), critic2(inCritic2)
		, targetCritic1(inTargetCritic1), targetCritic2(inTargetCritic2)
		, actorOptimizer(inActorOptimizer), criticOptimizer(inCriticOptimizer)
		, replay(inReplay), config(inConfig)
		, criticTraining(inEngine, inCriticOptimizer, oa::ItTrainingConfig{
			.totalSteps = static_cast<oa::I64>(inConfig.updates),
			.batchSize = static_cast<oa::I32>(inConfig.batchSize),
			.timerName = "sac_critic_update",
		})
		, actorTraining(inEngine, inActorOptimizer, oa::ItTrainingConfig{
			.totalSteps = static_cast<oa::I64>(inConfig.updates),
			.batchSize = static_cast<oa::I32>(inConfig.batchSize),
			.timerName = "sac_actor_update",
		}) {}
};

oa::SacTrainer::SacTrainer(oa::UniquePtr<Impl> inImpl)
	: impl_(oa::move(inImpl)) {}

oa::SacTrainer::~SacTrainer() = default;

oa::Result<oa::UniquePtr<oa::SacTrainer>> oa::SacTrainer::create(
	oa::Engine& inEngine,
	oa::Module& inActor, oa::Module& inCritic1, oa::Module& inCritic2,
	oa::Module& inTargetCritic1, oa::Module& inTargetCritic2,
	oa::Optimizer& inActorOptimizer, oa::Optimizer& inCriticOptimizer,
	oa::ReplayBuffer& inReplay, const oa::SacTrainerConfig& inConfig) {
	const auto& replay = inReplay.config();
	if (inConfig.updates == 0 || inConfig.batchSize == 0
		|| inConfig.actionDimensions == 0 || inConfig.targetUpdateInterval == 0
		|| inConfig.observationShape != replay.observationShape
		|| replay.actionDtype != oa::ScalarType::Float32
		|| replay.actionShape != oa::MatrixShape{
			static_cast<oa::I64>(inConfig.actionDimensions)}
		|| !oa::isFinite(inConfig.actionMinimum)
		|| !oa::isFinite(inConfig.actionMaximum)
		|| inConfig.actionMinimum >= inConfig.actionMaximum) {
		return oa::Status::invalidArgument(
			"oa::SacTrainer configuration does not match continuous replay storage");
	}
	auto impl = oa::makeUnique<Impl>(inEngine, inActor, inCritic1, inCritic2,
		inTargetCritic1, inTargetCritic2, inActorOptimizer,
		inCriticOptimizer, inReplay, inConfig);
	OA_RETURN_IF_ERROR(copyModel(inCritic1, inTargetCritic1));
	OA_RETURN_IF_ERROR(copyModel(inCritic2, inTargetCritic2));
	OA_RETURN_IF_ERROR(
		oa::ExecutionSession::forEngine(inEngine).submitAndWait());
	return oa::UniquePtr<oa::SacTrainer>(new oa::SacTrainer(oa::move(impl)));
}

oa::Status oa::SacTrainer::update() {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::SacTrainer is empty");
	auto& impl = *impl_;
	if (impl.criticTraining.stopRequested()) {
		OA_RETURN_IF_ERROR(impl.criticTraining.finish());
		impl.actorTraining.requestStop();
		return impl.actorTraining.finish();
	}
	if (isDone()) return oa::Status::ok();
	const bool mayBegin = impl.criticTraining.session() != nullptr
		? impl.criticTraining.session()->tryBeginStep()
		: !impl.criticTraining.isDone();
	if (!mayBegin) {
		if (!impl.criticTraining.stopRequested()) return oa::Status::ok();
		OA_RETURN_IF_ERROR(impl.criticTraining.finish());
		impl.actorTraining.requestStop();
		return impl.actorTraining.finish();
	}
	if (impl.replay.size() < impl.config.batchSize) return oa::Status::error(
		oa::StatusCode::FailedPrecondition,
		"oa::SacTrainer replay does not contain one complete batch");
	auto sampled = impl.replay.sample(impl.config.batchSize,
		impl.config.seed + impl.metrics.update + 1U);
	if (sampled.isError()) return sampled.getStatus();
	const oa::I64 observationElements = impl.config.observationShape.numElements();
	const oa::Matrix observation = oa::FnMatrix::reshape(sampled->observation,
		{static_cast<oa::I64>(impl.config.batchSize), observationElements});
	const oa::Matrix nextObservation = oa::FnMatrix::reshape(
		sampled->nextObservation,
		{static_cast<oa::I64>(impl.config.batchSize), observationElements});

	oa::ContinuousPolicyResult nextPolicy;
	oa::Matrix nextQ1;
	oa::Matrix nextQ2;
	{
		oa::GradNo noGrad;
		nextPolicy = actorPolicy(impl.actor, nextObservation,
			impl.config.batchSize, impl.config.actionDimensions,
			impl.config.actionMinimum, impl.config.actionMaximum,
			impl.config.seed + 0x100000001ULL + impl.metrics.update);
		if (!nextPolicy.isValid()) return oa::Status::error(
			oa::StatusCode::ShapeMismatch,
			"oa::SacTrainer actor must return [B,2*action-dim]");
		const oa::Matrix input = criticInput(nextObservation, nextPolicy.action);
		nextQ1 = vectorQ(impl.targetCritic1.forward(input), impl.config.batchSize);
		nextQ2 = vectorQ(impl.targetCritic2.forward(input), impl.config.batchSize);
	}
	if (nextQ1.isEmpty() || nextQ2.isEmpty()) return oa::Status::error(
		oa::StatusCode::ShapeMismatch,
		"oa::SacTrainer critics must return [B] or [B,1]");

	impl.criticOptimizer.zeroGrad();
	oa::GradientTape criticTape;
	const oa::Matrix storedInput = criticInput(observation, sampled->action);
	const oa::Matrix q1 = vectorQ(
		impl.critic1.forward(storedInput), impl.config.batchSize);
	const oa::Matrix q2 = vectorQ(
		impl.critic2.forward(storedInput), impl.config.batchSize);
	const oa::SacCriticLossResult criticLoss = oa::FnLoss::sacCritic(
		q1, q2, sampled->reward, nextQ1, nextQ2,
		nextPolicy.logProbability, sampled->terminated,
		sampled->truncated, impl.config.loss);
	if (!criticLoss.isValid()) return oa::Status::error(
		oa::StatusCode::FailedPrecondition, "oa::SacTrainer critic loss failed");
	criticTape.backward(criticLoss.totalLoss);
	impl.criticTraining.next(criticLoss.totalLoss);
	if (impl.criticTraining.lastStatus().isError()) {
		return impl.criticTraining.lastStatus();
	}
	impl.metrics.criticLoss = impl.criticTraining.lastLoss();

	if (impl.actorTraining.isDone()) return oa::Status::error(
		oa::StatusCode::FailedPrecondition,
		"oa::SacTrainer actor iterator completed before the critic iterator");
	impl.actorOptimizer.zeroGrad();
	impl.criticOptimizer.zeroGrad();
	oa::GradientTape actorTape;
	const oa::ContinuousPolicyResult policy = actorPolicy(
		impl.actor, observation, impl.config.batchSize,
		impl.config.actionDimensions, impl.config.actionMinimum,
		impl.config.actionMaximum,
		impl.config.seed + 0x200000001ULL + impl.metrics.update);
	if (!policy.isValid()) return oa::Status::error(
		oa::StatusCode::FailedPrecondition, "oa::SacTrainer actor policy failed");
	const oa::Matrix policyInput = criticInput(observation, policy.action);
	const oa::Matrix actorQ1 = vectorQ(
		impl.critic1.forward(policyInput), impl.config.batchSize);
	const oa::Matrix actorQ2 = vectorQ(
		impl.critic2.forward(policyInput), impl.config.batchSize);
	const oa::Matrix actorLoss = oa::FnLoss::sacActor(
		actorQ1, actorQ2, policy.logProbability,
		impl.config.loss.entropyCoefficient);
	if (actorLoss.isEmpty()) return oa::Status::error(
		oa::StatusCode::FailedPrecondition, "oa::SacTrainer actor loss failed");
	actorTape.backward(actorLoss);
	impl.actorTraining.next(actorLoss);
	if (impl.actorTraining.lastStatus().isError()) {
		return impl.actorTraining.lastStatus();
	}
	impl.metrics.actorLoss = impl.actorTraining.lastLoss();
	impl.metrics.update = static_cast<oa::U64>(impl.criticTraining.stepCount());
	if (impl.metrics.update % impl.config.targetUpdateInterval == 0) {
		OA_RETURN_IF_ERROR(syncTargets());
	}
	if (impl.metrics.update >= impl.config.updates) {
		OA_RETURN_IF_ERROR(impl.criticTraining.finish());
		OA_RETURN_IF_ERROR(impl.actorTraining.finish());
	}
	return oa::Status::ok();
}

oa::Status oa::SacTrainer::syncTargets() {
	if (!impl_) return oa::Status::error(oa::StatusCode::FailedPrecondition,
		"oa::SacTrainer is empty");
	OA_RETURN_IF_ERROR(copyModel(impl_->critic1, impl_->targetCritic1));
	OA_RETURN_IF_ERROR(copyModel(impl_->critic2, impl_->targetCritic2));
	return oa::ExecutionSession::forEngine(impl_->engine).submitAndWait();
}

bool oa::SacTrainer::isDone() const noexcept {
	return impl_ && (impl_->criticTraining.stopRequested()
		|| impl_->criticTraining.stepCount() >= impl_->criticTraining.totalSteps());
}

const oa::SacTrainerMetrics& oa::SacTrainer::metrics() const noexcept {
	return impl_->metrics;
}

oa::ItTraining& oa::SacTrainer::trainingLoop() noexcept {
	return impl_->criticTraining;
}

const oa::ItTraining& oa::SacTrainer::trainingLoop() const noexcept {
	return impl_->criticTraining;
}

oa::ItTraining& oa::SacTrainer::actorTrainingLoop() noexcept {
	return impl_->actorTraining;
}

const oa::ItTraining& oa::SacTrainer::actorTrainingLoop() const noexcept {
	return impl_->actorTraining;
}
