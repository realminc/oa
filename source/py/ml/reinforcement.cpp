// OA Python bindings -- reinforcement-learning contracts and categorical PPO.
#include "../binding.h"

#include <oa/ml.h>
#include <oa/ml/optim.h>

#include <limits>
#include <string>

namespace {

void bindFnAdvantageOps(nb::module_& m) {
	#include "fnAdvantageOps.gen.inl"
}

void bindFnEnvironmentOps(nb::module_& m) {
	#include "fnEnvironmentOps.gen.inl"
}

void bindFnPolicyOps(nb::module_& m) {
	#include "fnPolicyOps.gen.inl"
}

} // namespace

void bindMlReinforcement(
	nb::module_& m,
	nb::module_& inFnAdvantage,
	nb::module_& inFnEnvironment,
	nb::module_& inFnPolicy) {
	nb::enum_<oa::EnvironmentSpaceKind>(m, "EnvironmentSpaceKind")
		.value("Box", oa::EnvironmentSpaceKind::Box)
		.value("Discrete", oa::EnvironmentSpaceKind::Discrete)
		.value("Binary", oa::EnvironmentSpaceKind::Binary);

	nb::class_<oa::EnvironmentSpace>(m, "EnvironmentSpace")
		.def(nb::init<>())
		.def_static("box", [](const std::string& name,
			const std::vector<oa::I64>& shape, oa::ScalarType dtype,
			oa::F64 minimum, oa::F64 maximum) {
			return oa::EnvironmentSpace::box(
				oa::StringView(name.c_str()), shapeFromVector(shape), dtype,
				minimum, maximum);
		}, nb::arg("name"), nb::arg("shape"),
			nb::arg("dtype") = oa::ScalarType::Float32,
			nb::arg("minimum") = -std::numeric_limits<oa::F64>::infinity(),
			nb::arg("maximum") = std::numeric_limits<oa::F64>::infinity())
		.def_static("discrete", [](const std::string& name,
			oa::I64 cardinality, oa::ScalarType dtype) {
			return oa::EnvironmentSpace::discrete(
				oa::StringView(name.c_str()), cardinality, dtype);
		}, nb::arg("name"), nb::arg("cardinality"),
			nb::arg("dtype") = oa::ScalarType::Int32)
		.def_static("binary", [](const std::string& name,
			const std::vector<oa::I64>& shape, oa::ScalarType dtype) {
			return oa::EnvironmentSpace::binary(
				oa::StringView(name.c_str()), shapeFromVector(shape), dtype);
		}, nb::arg("name"), nb::arg("shape") = std::vector<oa::I64>{},
			nb::arg("dtype") = oa::ScalarType::UInt8)
		.def_prop_rw("name",
			[](const oa::EnvironmentSpace& self) { return std::string(self.name.cStr()); },
			[](oa::EnvironmentSpace& self, const std::string& value) {
				self.name = oa::String(value.c_str());
			})
		.def_rw("kind", &oa::EnvironmentSpace::kind)
		.def_prop_rw("shape",
			[](const oa::EnvironmentSpace& self) { return shapeToVector(self.shape); },
			[](oa::EnvironmentSpace& self, const std::vector<oa::I64>& value) {
				self.shape = shapeFromVector(value);
			})
		.def_rw("dtype", &oa::EnvironmentSpace::dtype)
		.def_rw("minimum", &oa::EnvironmentSpace::minimum)
		.def_rw("maximum", &oa::EnvironmentSpace::maximum)
		.def_rw("cardinality", &oa::EnvironmentSpace::cardinality)
		.def("validateDefinition", [](const oa::EnvironmentSpace& self) {
			throwIfError(self.validateDefinition());
		})
		.def("elementsPerEnvironment", &oa::EnvironmentSpace::elementsPerEnvironment)
		.def("batchedShape", [](const oa::EnvironmentSpace& self,
			oa::U32 environments) {
			auto result = self.batchedShape(environments);
			throwIfError(result.getStatus());
			return shapeToVector(*result);
		}, nb::arg("environments"))
		.def("validateMatrix", [](const oa::EnvironmentSpace& self,
			const oa::Matrix& matrix, oa::U32 environments) {
			throwIfError(self.validateMatrix(matrix, environments));
		}, nb::arg("matrix"), nb::arg("environments"));

	nb::class_<oa::EnvironmentSpec>(m, "EnvironmentSpec")
		.def(nb::init<>())
		.def_rw("observation", &oa::EnvironmentSpec::observation)
		.def_rw("action", &oa::EnvironmentSpec::action)
		.def_rw("reward", &oa::EnvironmentSpec::reward)
		.def_rw("terminated", &oa::EnvironmentSpec::terminated)
		.def_rw("truncated", &oa::EnvironmentSpec::truncated)
		.def("validateDefinition", [](const oa::EnvironmentSpec& self) {
			throwIfError(self.validateDefinition());
		})
		.def("validateReset", [](const oa::EnvironmentSpec& self,
			const oa::Matrix& observation, oa::U32 environments) {
			throwIfError(self.validateReset(observation, environments));
		}, nb::arg("observation"), nb::arg("environments"))
		.def("validateAction", [](const oa::EnvironmentSpec& self,
			const oa::Matrix& action, oa::U32 environments) {
			throwIfError(self.validateAction(action, environments));
		}, nb::arg("action"), nb::arg("environments"))
		.def("validateTransition", [](const oa::EnvironmentSpec& self,
			const oa::Matrix& observation, const oa::Matrix& action,
			const oa::Matrix& nextObservation, const oa::Matrix& reward,
			const oa::Matrix& terminated, const oa::Matrix& truncated,
			oa::U32 environments) {
			throwIfError(self.validateTransition(
				observation, action, nextObservation, reward,
				terminated, truncated, environments));
		}, nb::arg("observation"), nb::arg("action"),
			nb::arg("nextObservation"), nb::arg("reward"),
			nb::arg("terminated"), nb::arg("truncated"),
			nb::arg("environments"));

	nb::class_<oa::GaeConfig>(m, "GaeConfig")
		.def(nb::init<>())
		.def_rw("gamma", &oa::GaeConfig::gamma)
		.def_rw("lambda", &oa::GaeConfig::lambda);

	nb::class_<oa::GaeResult>(m, "GaeResult")
		.def_prop_ro("advantage", [](oa::GaeResult& self) -> oa::Matrix& {
			return self.advantage;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("return", [](oa::GaeResult& self) -> oa::Matrix& {
			return self.ret;
		}, nb::rv_policy::reference_internal)
		.def("isValid", &oa::GaeResult::isValid);

	nb::class_<oa::PolicyResult>(m, "PolicyResult")
		.def_prop_ro("action", [](oa::PolicyResult& self) -> oa::Matrix& {
			return self.action;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("logProbability", [](oa::PolicyResult& self) -> oa::Matrix& {
			return self.logProbability;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("entropy", [](oa::PolicyResult& self) -> oa::Matrix& {
			return self.entropy;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("value", [](oa::PolicyResult& self) -> oa::Matrix& {
			return self.value;
		}, nb::rv_policy::reference_internal)
		.def("isValid", &oa::PolicyResult::isValid);

	nb::class_<oa::ContinuousPolicyResult>(m, "ContinuousPolicyResult")
		.def_prop_ro("action", [](oa::ContinuousPolicyResult& self) -> oa::Matrix& { return self.action; }, nb::rv_policy::reference_internal)
		.def_prop_ro("rawAction", [](oa::ContinuousPolicyResult& self) -> oa::Matrix& { return self.rawAction; }, nb::rv_policy::reference_internal)
		.def_prop_ro("logProbability", [](oa::ContinuousPolicyResult& self) -> oa::Matrix& { return self.logProbability; }, nb::rv_policy::reference_internal)
		.def_prop_ro("entropy", [](oa::ContinuousPolicyResult& self) -> oa::Matrix& { return self.entropy; }, nb::rv_policy::reference_internal)
		.def_prop_ro("value", [](oa::ContinuousPolicyResult& self) -> oa::Matrix& { return self.value; }, nb::rv_policy::reference_internal)
		.def("isValid", &oa::ContinuousPolicyResult::isValid);

	nb::class_<oa::PpoLossConfig>(m, "PpoLossConfig")
		.def(nb::init<>())
		.def_rw("clipEpsilon", &oa::PpoLossConfig::clipEpsilon)
		.def_rw("valueCoefficient", &oa::PpoLossConfig::valueCoefficient)
		.def_rw("entropyCoefficient", &oa::PpoLossConfig::entropyCoefficient);

	nb::class_<oa::PpoLossResult>(m, "PpoLossResult")
		.def_prop_ro("policyLoss", [](oa::PpoLossResult& self) -> oa::Matrix& {
			return self.policyLoss;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("valueLoss", [](oa::PpoLossResult& self) -> oa::Matrix& {
			return self.valueLoss;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("entropy", [](oa::PpoLossResult& self) -> oa::Matrix& {
			return self.entropy;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("totalLoss", [](oa::PpoLossResult& self) -> oa::Matrix& {
			return self.totalLoss;
		}, nb::rv_policy::reference_internal)
		.def("isValid", &oa::PpoLossResult::isValid);

	nb::class_<oa::DqnLossConfig>(m, "DqnLossConfig")
		.def(nb::init<>())
		.def_rw("discount", &oa::DqnLossConfig::discount);

	nb::class_<oa::DqnLossResult>(m, "DqnLossResult")
		.def_prop_ro("selectedQ", [](oa::DqnLossResult& self) -> oa::Matrix& { return self.selectedQ; }, nb::rv_policy::reference_internal)
		.def_prop_ro("targetQ", [](oa::DqnLossResult& self) -> oa::Matrix& { return self.targetQ; }, nb::rv_policy::reference_internal)
		.def_prop_ro("loss", [](oa::DqnLossResult& self) -> oa::Matrix& { return self.loss; }, nb::rv_policy::reference_internal)
		.def("isValid", &oa::DqnLossResult::isValid);

	nb::class_<oa::SacLossConfig>(m, "SacLossConfig")
		.def(nb::init<>())
		.def_rw("discount", &oa::SacLossConfig::discount)
		.def_rw("entropyCoefficient", &oa::SacLossConfig::entropyCoefficient);

	nb::class_<oa::SacCriticLossResult>(m, "SacCriticLossResult")
		.def_prop_ro("targetQ", [](oa::SacCriticLossResult& self) -> oa::Matrix& { return self.targetQ; }, nb::rv_policy::reference_internal)
		.def_prop_ro("q1Loss", [](oa::SacCriticLossResult& self) -> oa::Matrix& { return self.q1Loss; }, nb::rv_policy::reference_internal)
		.def_prop_ro("q2Loss", [](oa::SacCriticLossResult& self) -> oa::Matrix& { return self.q2Loss; }, nb::rv_policy::reference_internal)
		.def_prop_ro("totalLoss", [](oa::SacCriticLossResult& self) -> oa::Matrix& { return self.totalLoss; }, nb::rv_policy::reference_internal)
		.def("isValid", &oa::SacCriticLossResult::isValid);

	bindFnAdvantageOps(inFnAdvantage);
	bindFnEnvironmentOps(inFnEnvironment);
	bindFnPolicyOps(inFnPolicy);

	nb::class_<oa::ReplayConfig>(m, "ReplayConfig")
		.def(nb::init<>())
		.def_rw("capacity", &oa::ReplayConfig::capacity)
		.def_prop_rw("observationShape",
			[](const oa::ReplayConfig& self) { return shapeToVector(self.observationShape); },
			[](oa::ReplayConfig& self, const std::vector<oa::I64>& value) { self.observationShape = shapeFromVector(value); })
		.def_prop_rw("actionShape",
			[](const oa::ReplayConfig& self) { return shapeToVector(self.actionShape); },
			[](oa::ReplayConfig& self, const std::vector<oa::I64>& value) { self.actionShape = shapeFromVector(value); })
		.def_rw("actionDtype", &oa::ReplayConfig::actionDtype);

	nb::class_<oa::ReplayBatch>(m, "ReplayBatch")
		.def_prop_ro("observation", [](oa::ReplayBatch& self) -> oa::Matrix& { return self.observation; }, nb::rv_policy::reference_internal)
		.def_prop_ro("action", [](oa::ReplayBatch& self) -> oa::Matrix& { return self.action; }, nb::rv_policy::reference_internal)
		.def_prop_ro("nextObservation", [](oa::ReplayBatch& self) -> oa::Matrix& { return self.nextObservation; }, nb::rv_policy::reference_internal)
		.def_prop_ro("reward", [](oa::ReplayBatch& self) -> oa::Matrix& { return self.reward; }, nb::rv_policy::reference_internal)
		.def_prop_ro("terminated", [](oa::ReplayBatch& self) -> oa::Matrix& { return self.terminated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("truncated", [](oa::ReplayBatch& self) -> oa::Matrix& { return self.truncated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("index", [](oa::ReplayBatch& self) -> oa::Matrix& { return self.index; }, nb::rv_policy::reference_internal)
		.def("isValid", &oa::ReplayBatch::isValid);

	nb::class_<oa::ReplayBuffer>(m, "ReplayBuffer")
		.def_static("create", [](const oa::ReplayConfig& config) {
			auto created = oa::ReplayBuffer::create(config);
			throwIfError(created.getStatus());
			return new oa::ReplayBuffer(oa::move(*created));
		}, nb::arg("config"), nb::rv_policy::take_ownership)
		.def("append", [](oa::ReplayBuffer& self,
			const oa::Matrix& observation, const oa::Matrix& action,
			const oa::Matrix& nextObservation, const oa::Matrix& reward,
			const oa::Matrix& terminated, const oa::Matrix& truncated) {
			throwIfError(self.append({observation, action, nextObservation,
				reward, terminated, truncated}));
		}, nb::arg("observation"), nb::arg("action"),
			nb::arg("nextObservation"), nb::arg("reward"),
			nb::arg("terminated"), nb::arg("truncated"))
		.def("sample", [](const oa::ReplayBuffer& self,
			oa::U32 batchSize, oa::U64 seed) {
			auto sampled = self.sample(batchSize, seed);
			throwIfError(sampled.getStatus());
			return new oa::ReplayBatch(oa::move(*sampled));
		}, nb::arg("batchSize"), nb::arg("seed"),
			nb::rv_policy::take_ownership)
		.def("reset", &oa::ReplayBuffer::reset)
		.def("isValid", &oa::ReplayBuffer::isValid)
		.def("isFull", &oa::ReplayBuffer::isFull)
		.def("size", &oa::ReplayBuffer::size)
		.def("capacity", &oa::ReplayBuffer::capacity)
		.def("cursor", &oa::ReplayBuffer::cursor);

	nb::class_<oa::RolloutConfig>(m, "RolloutConfig")
		.def(nb::init<>())
		.def_rw("time", &oa::RolloutConfig::time)
		.def_rw("environments", &oa::RolloutConfig::environments)
		.def_prop_rw("observationShape",
			[](const oa::RolloutConfig& self) {
				return shapeToVector(self.observationShape);
			},
			[](oa::RolloutConfig& self, const std::vector<oa::I64>& value) {
				self.observationShape = shapeFromVector(value);
			});

	nb::class_<oa::RolloutBatch>(m, "RolloutBatch")
		.def_prop_ro("observation", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.observation; }, nb::rv_policy::reference_internal)
		.def_prop_ro("action", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.action; }, nb::rv_policy::reference_internal)
		.def_prop_ro("reward", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.reward; }, nb::rv_policy::reference_internal)
		.def_prop_ro("value", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.value; }, nb::rv_policy::reference_internal)
		.def_prop_ro("nextValue", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.nextValue; }, nb::rv_policy::reference_internal)
		.def_prop_ro("oldLogProbability", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.oldLogProbability; }, nb::rv_policy::reference_internal)
		.def_prop_ro("terminated", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.terminated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("truncated", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.truncated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("valid", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.valid; }, nb::rv_policy::reference_internal)
		.def_prop_ro("advantage", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.advantage; }, nb::rv_policy::reference_internal)
		.def_prop_ro("return", [](oa::RolloutBatch& self) -> oa::Matrix& { return self.ret; }, nb::rv_policy::reference_internal)
		.def("isValid", &oa::RolloutBatch::isValid);

	nb::class_<oa::RolloutBuffer>(m, "RolloutBuffer")
		.def_static("create", [](const oa::RolloutConfig& config) {
			auto created = oa::RolloutBuffer::create(config);
			throwIfError(created.getStatus());
			return new oa::RolloutBuffer(oa::move(*created));
		}, nb::arg("config"), nb::rv_policy::take_ownership)
		.def("append", [](oa::RolloutBuffer& self,
			const oa::Matrix& observation, const oa::Matrix& action,
			const oa::Matrix& reward, const oa::Matrix& value,
			const oa::Matrix& nextValue, const oa::Matrix& logProbability,
			const oa::Matrix& terminated, const oa::Matrix& truncated) {
			throwIfError(self.append(oa::RolloutTransition{
				.observation = observation,
				.action = action,
				.reward = reward,
				.value = value,
				.nextValue = nextValue,
				.logProbability = logProbability,
				.terminated = terminated,
				.truncated = truncated,
			}));
		}, nb::arg("observation"), nb::arg("action"), nb::arg("reward"),
			nb::arg("value"), nb::arg("nextValue"),
			nb::arg("logProbability"), nb::arg("terminated"),
			nb::arg("truncated"))
		.def("finalize", [](oa::RolloutBuffer& self,
			const oa::GaeConfig& config) {
			throwIfError(self.finalize(config));
		}, nb::arg("config") = oa::GaeConfig())
		.def("reset", &oa::RolloutBuffer::reset)
		.def("isValid", &oa::RolloutBuffer::isValid)
		.def("isFull", &oa::RolloutBuffer::isFull)
		.def("isFinalized", &oa::RolloutBuffer::isFinalized)
		.def("size", &oa::RolloutBuffer::size)
		.def("capacity", &oa::RolloutBuffer::capacity)
		.def_prop_ro("batch", [](oa::RolloutBuffer& self) -> const oa::RolloutBatch& {
			return self.batch();
		}, nb::rv_policy::reference_internal);

	nb::enum_<oa::RolloutTrainingPhase>(m, "RolloutTrainingPhase")
		.value("Collect", oa::RolloutTrainingPhase::Collect)
		.value("Update", oa::RolloutTrainingPhase::Update)
		.value("Complete", oa::RolloutTrainingPhase::Complete);

	nb::class_<oa::ItRolloutTrainingConfig>(m, "ItRolloutTrainingConfig")
		.def(nb::init<>())
		.def_rw("rollouts", &oa::ItRolloutTrainingConfig::rollouts)
		.def_rw("horizon", &oa::ItRolloutTrainingConfig::horizon)
		.def_rw("environments", &oa::ItRolloutTrainingConfig::environments)
		.def_rw("updateEpochs", &oa::ItRolloutTrainingConfig::updateEpochs);

	nb::class_<oa::ItRolloutTraining>(m, "ItRolloutTraining")
		.def("__init__", [](oa::ItRolloutTraining* self, oa::Optimizer& optimizer,
			const oa::ItRolloutTrainingConfig& config) {
			new (self) oa::ItRolloutTraining(pythonEngine(), optimizer, config);
		}, nb::arg("optimizer"), nb::arg("config"), nb::keep_alive<1, 2>())
		.def("beginRollout", [](oa::ItRolloutTraining& self,
			oa::RolloutBuffer& rollout) {
			throwIfError(self.beginRollout(rollout));
		}, nb::arg("rollout"))
		.def("finalizeRollout", [](oa::ItRolloutTraining& self,
			oa::RolloutBuffer& rollout, const oa::GaeConfig& config) {
			throwIfError(self.finalizeRollout(rollout, config));
		}, nb::arg("rollout"), nb::arg("config") = oa::GaeConfig())
		.def("beginUpdate", &oa::ItRolloutTraining::beginUpdate)
		.def("nextUpdate", [](oa::ItRolloutTraining& self, const oa::Matrix& loss) {
			throwIfError(self.nextUpdate(loss));
		}, nb::arg("loss"))
		.def("finish", [](oa::ItRolloutTraining& self) {
			throwIfError(self.finish());
		})
		.def("isValid", &oa::ItRolloutTraining::isValid)
		.def("isDone", &oa::ItRolloutTraining::isDone)
		.def("phase", &oa::ItRolloutTraining::phase)
		.def("rolloutIndex", &oa::ItRolloutTraining::rolloutIndex)
		.def("updateEpoch", &oa::ItRolloutTraining::updateEpoch)
		.def("updateLoop", nb::overload_cast<>(&oa::ItRolloutTraining::updateLoop),
			nb::rv_policy::reference_internal);
}
