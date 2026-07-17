// OA Python bindings -- reinforcement-learning contracts and categorical PPO.
#include "../Binding.h"

#include <Oa/Ml/Rl.h>
#include <Oa/Ml/Optim.h>

#include <limits>
#include <string>

void BindMlRl(nb::module_& m) {
	nb::enum_<OaRlSpaceKind>(m, "OaRlSpaceKind")
		.value("Box", OaRlSpaceKind::Box)
		.value("Discrete", OaRlSpaceKind::Discrete)
		.value("Binary", OaRlSpaceKind::Binary);

	nb::class_<OaRlFieldSpec>(m, "OaRlFieldSpec")
		.def(nb::init<>())
		.def_static("Box", [](const std::string& name,
			const std::vector<OaI64>& shape, OaScalarType dtype,
			OaF64 minimum, OaF64 maximum) {
			return OaRlFieldSpec::Box(
				OaStringView(name.c_str()), shape_from_vector(shape), dtype,
				minimum, maximum);
		}, nb::arg("name"), nb::arg("shape"),
			nb::arg("dtype") = OaScalarType::Float32,
			nb::arg("minimum") = -std::numeric_limits<OaF64>::infinity(),
			nb::arg("maximum") = std::numeric_limits<OaF64>::infinity())
		.def_static("Discrete", [](const std::string& name,
			OaI64 cardinality, OaScalarType dtype) {
			return OaRlFieldSpec::Discrete(
				OaStringView(name.c_str()), cardinality, dtype);
		}, nb::arg("name"), nb::arg("cardinality"),
			nb::arg("dtype") = OaScalarType::Int32)
		.def_static("Binary", [](const std::string& name,
			const std::vector<OaI64>& shape, OaScalarType dtype) {
			return OaRlFieldSpec::Binary(
				OaStringView(name.c_str()), shape_from_vector(shape), dtype);
		}, nb::arg("name"), nb::arg("shape") = std::vector<OaI64>{},
			nb::arg("dtype") = OaScalarType::UInt8)
		.def_prop_rw("Name",
			[](const OaRlFieldSpec& self) { return std::string(self.Name.c_str()); },
			[](OaRlFieldSpec& self, const std::string& value) {
				self.Name = OaString(value.c_str());
			})
		.def_rw("Kind", &OaRlFieldSpec::Kind)
		.def_prop_rw("Shape",
			[](const OaRlFieldSpec& self) { return shape_to_vector(self.Shape); },
			[](OaRlFieldSpec& self, const std::vector<OaI64>& value) {
				self.Shape = shape_from_vector(value);
			})
		.def_rw("Dtype", &OaRlFieldSpec::Dtype)
		.def_rw("Minimum", &OaRlFieldSpec::Minimum)
		.def_rw("Maximum", &OaRlFieldSpec::Maximum)
		.def_rw("Cardinality", &OaRlFieldSpec::Cardinality)
		.def("ValidateDefinition", [](const OaRlFieldSpec& self) {
			throw_if_error(self.ValidateDefinition());
		})
		.def("ElementsPerEnvironment", &OaRlFieldSpec::ElementsPerEnvironment)
		.def("BatchedShape", [](const OaRlFieldSpec& self,
			OaU32 environments) {
			auto result = self.BatchedShape(environments);
			throw_if_error(result.GetStatus());
			return shape_to_vector(*result);
		}, nb::arg("environments"))
		.def("ValidateMatrix", [](const OaRlFieldSpec& self,
			const OaMatrix& matrix, OaU32 environments) {
			throw_if_error(self.ValidateMatrix(matrix, environments));
		}, nb::arg("matrix"), nb::arg("environments"));

	nb::class_<OaRlEnvironmentSpec>(m, "OaRlEnvironmentSpec")
		.def(nb::init<>())
		.def_rw("Observation", &OaRlEnvironmentSpec::Observation)
		.def_rw("Action", &OaRlEnvironmentSpec::Action)
		.def_rw("Reward", &OaRlEnvironmentSpec::Reward)
		.def_rw("Terminated", &OaRlEnvironmentSpec::Terminated)
		.def_rw("Truncated", &OaRlEnvironmentSpec::Truncated)
		.def("ValidateDefinition", [](const OaRlEnvironmentSpec& self) {
			throw_if_error(self.ValidateDefinition());
		})
		.def("ValidateReset", [](const OaRlEnvironmentSpec& self,
			const OaMatrix& observation, OaU32 environments) {
			throw_if_error(self.ValidateReset(observation, environments));
		}, nb::arg("observation"), nb::arg("environments"))
		.def("ValidateAction", [](const OaRlEnvironmentSpec& self,
			const OaMatrix& action, OaU32 environments) {
			throw_if_error(self.ValidateAction(action, environments));
		}, nb::arg("action"), nb::arg("environments"))
		.def("ValidateTransition", [](const OaRlEnvironmentSpec& self,
			const OaMatrix& observation, const OaMatrix& action,
			const OaMatrix& nextObservation, const OaMatrix& reward,
			const OaMatrix& terminated, const OaMatrix& truncated,
			OaU32 environments) {
			throw_if_error(self.ValidateTransition(
				observation, action, nextObservation, reward,
				terminated, truncated, environments));
		}, nb::arg("observation"), nb::arg("action"),
			nb::arg("next_observation"), nb::arg("reward"),
			nb::arg("terminated"), nb::arg("truncated"),
			nb::arg("environments"));

	nb::class_<OaGaeConfig>(m, "OaGaeConfig")
		.def(nb::init<>())
		.def_rw("Gamma", &OaGaeConfig::Gamma)
		.def_rw("Lambda", &OaGaeConfig::Lambda);

	nb::class_<OaGaeResult>(m, "OaGaeResult")
		.def_prop_ro("Advantage", [](OaGaeResult& self) -> OaMatrix& {
			return self.Advantage;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("Return", [](OaGaeResult& self) -> OaMatrix& {
			return self.Return;
		}, nb::rv_policy::reference_internal)
		.def("IsValid", &OaGaeResult::IsValid);

	nb::class_<OaRlPolicyResult>(m, "OaRlPolicyResult")
		.def_prop_ro("Action", [](OaRlPolicyResult& self) -> OaMatrix& {
			return self.Action;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("LogProbability", [](OaRlPolicyResult& self) -> OaMatrix& {
			return self.LogProbability;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("Entropy", [](OaRlPolicyResult& self) -> OaMatrix& {
			return self.Entropy;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("Value", [](OaRlPolicyResult& self) -> OaMatrix& {
			return self.Value;
		}, nb::rv_policy::reference_internal)
		.def("IsValid", &OaRlPolicyResult::IsValid);

	nb::class_<OaRlContinuousPolicyResult>(m, "OaRlContinuousPolicyResult")
		.def_prop_ro("Action", [](OaRlContinuousPolicyResult& self) -> OaMatrix& { return self.Action; }, nb::rv_policy::reference_internal)
		.def_prop_ro("RawAction", [](OaRlContinuousPolicyResult& self) -> OaMatrix& { return self.RawAction; }, nb::rv_policy::reference_internal)
		.def_prop_ro("LogProbability", [](OaRlContinuousPolicyResult& self) -> OaMatrix& { return self.LogProbability; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Entropy", [](OaRlContinuousPolicyResult& self) -> OaMatrix& { return self.Entropy; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Value", [](OaRlContinuousPolicyResult& self) -> OaMatrix& { return self.Value; }, nb::rv_policy::reference_internal)
		.def("IsValid", &OaRlContinuousPolicyResult::IsValid);

	nb::class_<OaPpoLossConfig>(m, "OaPpoLossConfig")
		.def(nb::init<>())
		.def_rw("ClipEpsilon", &OaPpoLossConfig::ClipEpsilon)
		.def_rw("ValueCoefficient", &OaPpoLossConfig::ValueCoefficient)
		.def_rw("EntropyCoefficient", &OaPpoLossConfig::EntropyCoefficient);

	nb::class_<OaPpoLossResult>(m, "OaPpoLossResult")
		.def_prop_ro("PolicyLoss", [](OaPpoLossResult& self) -> OaMatrix& {
			return self.PolicyLoss;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("ValueLoss", [](OaPpoLossResult& self) -> OaMatrix& {
			return self.ValueLoss;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("Entropy", [](OaPpoLossResult& self) -> OaMatrix& {
			return self.Entropy;
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("TotalLoss", [](OaPpoLossResult& self) -> OaMatrix& {
			return self.TotalLoss;
		}, nb::rv_policy::reference_internal)
		.def("IsValid", &OaPpoLossResult::IsValid);

	nb::class_<OaDqnLossConfig>(m, "OaDqnLossConfig")
		.def(nb::init<>())
		.def_rw("Discount", &OaDqnLossConfig::Discount);

	nb::class_<OaDqnLossResult>(m, "OaDqnLossResult")
		.def_prop_ro("SelectedQ", [](OaDqnLossResult& self) -> OaMatrix& { return self.SelectedQ; }, nb::rv_policy::reference_internal)
		.def_prop_ro("TargetQ", [](OaDqnLossResult& self) -> OaMatrix& { return self.TargetQ; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Loss", [](OaDqnLossResult& self) -> OaMatrix& { return self.Loss; }, nb::rv_policy::reference_internal)
		.def("IsValid", &OaDqnLossResult::IsValid);

	nb::class_<OaSacLossConfig>(m, "OaSacLossConfig")
		.def(nb::init<>())
		.def_rw("Discount", &OaSacLossConfig::Discount)
		.def_rw("EntropyCoefficient", &OaSacLossConfig::EntropyCoefficient);

	nb::class_<OaSacCriticLossResult>(m, "OaSacCriticLossResult")
		.def_prop_ro("TargetQ", [](OaSacCriticLossResult& self) -> OaMatrix& { return self.TargetQ; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Q1Loss", [](OaSacCriticLossResult& self) -> OaMatrix& { return self.Q1Loss; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Q2Loss", [](OaSacCriticLossResult& self) -> OaMatrix& { return self.Q2Loss; }, nb::rv_policy::reference_internal)
		.def_prop_ro("TotalLoss", [](OaSacCriticLossResult& self) -> OaMatrix& { return self.TotalLoss; }, nb::rv_policy::reference_internal)
		.def("IsValid", &OaSacCriticLossResult::IsValid);

	m.def("NormalizeAdvantages", [](const OaMatrix& advantage, OaF32 epsilon) {
		auto result = OaFnRl::NormalizeAdvantages(advantage, epsilon);
		if (result.IsEmpty()) {
			throw std::runtime_error("NormalizeAdvantages rejected its input");
		}
		return matrix_ptr(OaStdMove(result));
	}, nb::arg("advantage"), nb::arg("epsilon") = 1.0e-8F,
		nb::rv_policy::take_ownership);

	m.def("Gae", [](const OaMatrix& reward, const OaMatrix& value,
		const OaMatrix& nextValue, const OaMatrix& terminated,
		const OaMatrix& truncated, const OaGaeConfig& config) {
		auto result = OaFnRl::Gae(
			reward, value, nextValue, terminated, truncated, config);
		if (!result.IsValid()) {
			throw std::runtime_error("Gae rejected its input");
		}
		return new OaGaeResult(OaStdMove(result));
	}, nb::arg("reward"), nb::arg("value"), nb::arg("next_value"),
		nb::arg("terminated"), nb::arg("truncated"),
		nb::arg("config") = OaGaeConfig(), nb::rv_policy::take_ownership);

	m.def("SampleCategoricalPolicy", [](const OaMatrix& logits,
		const OaMatrix& value, OaU64 seed) {
		auto result = OaFnRl::SampleCategoricalPolicy(logits, value, seed);
		if (!result.IsValid()) {
			throw std::runtime_error("SampleCategoricalPolicy rejected its input");
		}
		return new OaRlPolicyResult(OaStdMove(result));
	}, nb::arg("logits"), nb::arg("value"), nb::arg("seed") = 0,
		nb::rv_policy::take_ownership);

	m.def("EvaluateCategoricalPolicy", [](const OaMatrix& logits,
		const OaMatrix& action, const OaMatrix& value) {
		auto result = OaFnRl::EvaluateCategoricalPolicy(logits, action, value);
		if (!result.IsValid()) {
			throw std::runtime_error("EvaluateCategoricalPolicy rejected its input");
		}
		return new OaRlPolicyResult(OaStdMove(result));
	}, nb::arg("logits"), nb::arg("action"), nb::arg("value"),
		nb::rv_policy::take_ownership);

	m.def("SampleTanhNormalPolicy", [](const OaMatrix& mean,
		const OaMatrix& logStddev, const OaMatrix& value,
		OaF32 minimum, OaF32 maximum, OaU64 seed, OaF32 epsilon) {
		auto result = OaFnRl::SampleTanhNormalPolicy(
			mean, logStddev, value, minimum, maximum, seed, epsilon);
		if (!result.IsValid()) throw std::runtime_error(
			"SampleTanhNormalPolicy rejected its input");
		return new OaRlContinuousPolicyResult(OaStdMove(result));
	}, nb::arg("mean"), nb::arg("log_stddev"), nb::arg("value"),
		nb::arg("minimum") = -1.0F, nb::arg("maximum") = 1.0F,
		nb::arg("seed") = 0, nb::arg("epsilon") = 1.0e-6F,
		nb::rv_policy::take_ownership);

	m.def("EvaluateTanhNormalPolicy", [](const OaMatrix& mean,
		const OaMatrix& logStddev, const OaMatrix& rawAction,
		const OaMatrix& value, OaF32 minimum, OaF32 maximum, OaF32 epsilon) {
		auto result = OaFnRl::EvaluateTanhNormalPolicy(
			mean, logStddev, rawAction, value, minimum, maximum, epsilon);
		if (!result.IsValid()) throw std::runtime_error(
			"EvaluateTanhNormalPolicy rejected its input");
		return new OaRlContinuousPolicyResult(OaStdMove(result));
	}, nb::arg("mean"), nb::arg("log_stddev"), nb::arg("raw_action"),
		nb::arg("value"), nb::arg("minimum") = -1.0F,
		nb::arg("maximum") = 1.0F, nb::arg("epsilon") = 1.0e-6F,
		nb::rv_policy::take_ownership);

	m.def("NormalizeObservation", [](const OaMatrix& observation,
		const OaMatrix& mean, const OaMatrix& stddev,
		OaF32 epsilon, OaF32 clip) {
		auto result = OaFnRl::NormalizeObservation(
			observation, mean, stddev, epsilon, clip);
		if (result.IsEmpty()) throw std::runtime_error(
			"NormalizeObservation rejected its input");
		return matrix_ptr(OaStdMove(result));
	}, nb::arg("observation"), nb::arg("mean"), nb::arg("stddev"),
		nb::arg("epsilon") = 1.0e-6F, nb::arg("clip") = 10.0F,
		nb::rv_policy::take_ownership);

	m.def("ScaleAction", [](const OaMatrix& action, OaF32 sourceMinimum,
		OaF32 sourceMaximum, OaF32 targetMinimum, OaF32 targetMaximum,
		bool clamp) {
		auto result = OaFnRl::ScaleAction(action, sourceMinimum, sourceMaximum,
			targetMinimum, targetMaximum, clamp);
		if (result.IsEmpty()) throw std::runtime_error("ScaleAction rejected its input");
		return matrix_ptr(OaStdMove(result));
	}, nb::arg("action"), nb::arg("source_minimum"),
		nb::arg("source_maximum"), nb::arg("target_minimum"),
		nb::arg("target_maximum"), nb::arg("clamp") = true,
		nb::rv_policy::take_ownership);

	m.def("ClipReward", [](const OaMatrix& reward,
		OaF32 minimum, OaF32 maximum) {
		auto result = OaFnRl::ClipReward(reward, minimum, maximum);
		if (result.IsEmpty()) throw std::runtime_error("ClipReward rejected its input");
		return matrix_ptr(OaStdMove(result));
	}, nb::arg("reward"), nb::arg("minimum") = -1.0F,
		nb::arg("maximum") = 1.0F, nb::rv_policy::take_ownership);

	m.def("PpoClippedPolicyBwd", [](const OaMatrix& newLogProbability,
		const OaMatrix& oldLogProbability, const OaMatrix& advantage,
		OaF32 clipEpsilon) {
		auto result = OaFnLoss::PpoClippedPolicyBwd(
			newLogProbability, oldLogProbability, advantage, clipEpsilon);
		if (result.IsEmpty()) {
			throw std::runtime_error("PpoClippedPolicyBwd rejected its input");
		}
		return matrix_ptr(OaStdMove(result));
	}, nb::arg("new_log_probability"), nb::arg("old_log_probability"),
		nb::arg("advantage"), nb::arg("clip_epsilon") = 0.2F,
		nb::rv_policy::take_ownership);

	m.def("PpoClippedPolicy", [](const OaMatrix& newLogProbability,
		const OaMatrix& oldLogProbability, const OaMatrix& advantage,
		OaF32 clipEpsilon) {
		return matrix_ptr(OaFnLoss::PpoClippedPolicy(
			newLogProbability, oldLogProbability, advantage, clipEpsilon));
	}, nb::arg("new_log_probability"), nb::arg("old_log_probability"),
		nb::arg("advantage"), nb::arg("clip_epsilon") = 0.2F,
		nb::rv_policy::take_ownership);

	m.def("Ppo", [](const OaMatrix& newLogProbability,
		const OaMatrix& oldLogProbability, const OaMatrix& advantage,
		const OaMatrix& value, const OaMatrix& targetReturn,
		const OaMatrix& entropy, const OaPpoLossConfig& config) {
		auto result = OaFnLoss::Ppo(
			newLogProbability, oldLogProbability, advantage, value,
			targetReturn, entropy, config);
		if (!result.IsValid()) {
			throw std::runtime_error("Ppo rejected its input");
		}
		return new OaPpoLossResult(OaStdMove(result));
	}, nb::arg("new_log_probability"), nb::arg("old_log_probability"),
		nb::arg("advantage"), nb::arg("value"), nb::arg("target_return"),
		nb::arg("entropy"), nb::arg("config") = OaPpoLossConfig(),
		nb::rv_policy::take_ownership);

	m.def("Dqn", [](const OaMatrix& q, const OaMatrix& action,
		const OaMatrix& reward, const OaMatrix& nextQ,
		const OaMatrix& terminated, const OaMatrix& truncated,
		const OaDqnLossConfig& config) {
		auto result = OaFnLoss::Dqn(
			q, action, reward, nextQ, terminated, truncated, config);
		if (!result.IsValid()) throw std::runtime_error("Dqn rejected its input");
		return new OaDqnLossResult(OaStdMove(result));
	}, nb::arg("q"), nb::arg("action"), nb::arg("reward"),
		nb::arg("next_q"), nb::arg("terminated"), nb::arg("truncated"),
		nb::arg("config") = OaDqnLossConfig(), nb::rv_policy::take_ownership);

	m.def("SacCritic", [](const OaMatrix& q1, const OaMatrix& q2,
		const OaMatrix& reward, const OaMatrix& nextQ1,
		const OaMatrix& nextQ2, const OaMatrix& nextLogProbability,
		const OaMatrix& terminated, const OaMatrix& truncated,
		const OaSacLossConfig& config) {
		auto result = OaFnLoss::SacCritic(q1, q2, reward, nextQ1, nextQ2,
			nextLogProbability, terminated, truncated, config);
		if (!result.IsValid()) throw std::runtime_error("SacCritic rejected its input");
		return new OaSacCriticLossResult(OaStdMove(result));
	}, nb::arg("q1"), nb::arg("q2"), nb::arg("reward"),
		nb::arg("next_q1"), nb::arg("next_q2"),
		nb::arg("next_log_probability"), nb::arg("terminated"),
		nb::arg("truncated"), nb::arg("config") = OaSacLossConfig(),
		nb::rv_policy::take_ownership);

	m.def("SacActor", [](const OaMatrix& q1, const OaMatrix& q2,
		const OaMatrix& logProbability, OaF32 entropyCoefficient) {
		auto result = OaFnLoss::SacActor(q1, q2, logProbability,
			entropyCoefficient);
		if (result.IsEmpty()) throw std::runtime_error("SacActor rejected its input");
		return matrix_ptr(OaStdMove(result));
	}, nb::arg("q1"), nb::arg("q2"), nb::arg("log_probability"),
		nb::arg("entropy_coefficient") = 0.2F,
		nb::rv_policy::take_ownership);

	nb::class_<OaRlReplayConfig>(m, "OaRlReplayConfig")
		.def(nb::init<>())
		.def_rw("Capacity", &OaRlReplayConfig::Capacity)
		.def_prop_rw("ObservationShape",
			[](const OaRlReplayConfig& self) { return shape_to_vector(self.ObservationShape); },
			[](OaRlReplayConfig& self, const std::vector<OaI64>& value) { self.ObservationShape = shape_from_vector(value); })
		.def_prop_rw("ActionShape",
			[](const OaRlReplayConfig& self) { return shape_to_vector(self.ActionShape); },
			[](OaRlReplayConfig& self, const std::vector<OaI64>& value) { self.ActionShape = shape_from_vector(value); })
		.def_rw("ActionDtype", &OaRlReplayConfig::ActionDtype);

	nb::class_<OaRlReplayBatch>(m, "OaRlReplayBatch")
		.def_prop_ro("Observation", [](OaRlReplayBatch& self) -> OaMatrix& { return self.Observation; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Action", [](OaRlReplayBatch& self) -> OaMatrix& { return self.Action; }, nb::rv_policy::reference_internal)
		.def_prop_ro("NextObservation", [](OaRlReplayBatch& self) -> OaMatrix& { return self.NextObservation; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Reward", [](OaRlReplayBatch& self) -> OaMatrix& { return self.Reward; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Terminated", [](OaRlReplayBatch& self) -> OaMatrix& { return self.Terminated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Truncated", [](OaRlReplayBatch& self) -> OaMatrix& { return self.Truncated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Index", [](OaRlReplayBatch& self) -> OaMatrix& { return self.Index; }, nb::rv_policy::reference_internal)
		.def("IsValid", &OaRlReplayBatch::IsValid);

	nb::class_<OaRlReplayBuffer>(m, "OaRlReplayBuffer")
		.def_static("Create", [](const OaRlReplayConfig& config) {
			auto created = OaRlReplayBuffer::Create(config);
			throw_if_error(created.GetStatus());
			return new OaRlReplayBuffer(OaStdMove(*created));
		}, nb::arg("config"), nb::rv_policy::take_ownership)
		.def("Append", [](OaRlReplayBuffer& self,
			const OaMatrix& observation, const OaMatrix& action,
			const OaMatrix& nextObservation, const OaMatrix& reward,
			const OaMatrix& terminated, const OaMatrix& truncated) {
			throw_if_error(self.Append({observation, action, nextObservation,
				reward, terminated, truncated}));
		}, nb::arg("observation"), nb::arg("action"),
			nb::arg("next_observation"), nb::arg("reward"),
			nb::arg("terminated"), nb::arg("truncated"))
		.def("Sample", [](const OaRlReplayBuffer& self,
			OaU32 batchSize, OaU64 seed) {
			auto sampled = self.Sample(batchSize, seed);
			throw_if_error(sampled.GetStatus());
			return new OaRlReplayBatch(OaStdMove(*sampled));
		}, nb::arg("batch_size"), nb::arg("seed"),
			nb::rv_policy::take_ownership)
		.def("Reset", &OaRlReplayBuffer::Reset)
		.def("IsValid", &OaRlReplayBuffer::IsValid)
		.def("IsFull", &OaRlReplayBuffer::IsFull)
		.def("Size", &OaRlReplayBuffer::Size)
		.def("Capacity", &OaRlReplayBuffer::Capacity)
		.def("Cursor", &OaRlReplayBuffer::Cursor);

	nb::class_<OaRlRolloutConfig>(m, "OaRlRolloutConfig")
		.def(nb::init<>())
		.def_rw("Time", &OaRlRolloutConfig::Time)
		.def_rw("Environments", &OaRlRolloutConfig::Environments)
		.def_prop_rw("ObservationShape",
			[](const OaRlRolloutConfig& self) {
				return shape_to_vector(self.ObservationShape);
			},
			[](OaRlRolloutConfig& self, const std::vector<OaI64>& value) {
				self.ObservationShape = shape_from_vector(value);
			});

	nb::class_<OaRlRolloutBatch>(m, "OaRlRolloutBatch")
		.def_prop_ro("Observation", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Observation; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Action", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Action; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Reward", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Reward; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Value", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Value; }, nb::rv_policy::reference_internal)
		.def_prop_ro("NextValue", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.NextValue; }, nb::rv_policy::reference_internal)
		.def_prop_ro("OldLogProbability", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.OldLogProbability; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Terminated", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Terminated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Truncated", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Truncated; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Valid", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Valid; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Advantage", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Advantage; }, nb::rv_policy::reference_internal)
		.def_prop_ro("Return", [](OaRlRolloutBatch& self) -> OaMatrix& { return self.Return; }, nb::rv_policy::reference_internal)
		.def("IsValid", &OaRlRolloutBatch::IsValid);

	nb::class_<OaRlRolloutBuffer>(m, "OaRlRolloutBuffer")
		.def_static("Create", [](const OaRlRolloutConfig& config) {
			auto created = OaRlRolloutBuffer::Create(config);
			throw_if_error(created.GetStatus());
			return new OaRlRolloutBuffer(OaStdMove(*created));
		}, nb::arg("config"), nb::rv_policy::take_ownership)
		.def("Append", [](OaRlRolloutBuffer& self,
			const OaMatrix& observation, const OaMatrix& action,
			const OaMatrix& reward, const OaMatrix& value,
			const OaMatrix& nextValue, const OaMatrix& logProbability,
			const OaMatrix& terminated, const OaMatrix& truncated) {
			throw_if_error(self.Append(OaRlTransition{
				.Observation = observation,
				.Action = action,
				.Reward = reward,
				.Value = value,
				.NextValue = nextValue,
				.LogProbability = logProbability,
				.Terminated = terminated,
				.Truncated = truncated,
			}));
		}, nb::arg("observation"), nb::arg("action"), nb::arg("reward"),
			nb::arg("value"), nb::arg("next_value"),
			nb::arg("log_probability"), nb::arg("terminated"),
			nb::arg("truncated"))
		.def("Finalize", [](OaRlRolloutBuffer& self,
			const OaGaeConfig& config) {
			throw_if_error(self.Finalize(config));
		}, nb::arg("config") = OaGaeConfig())
		.def("Reset", &OaRlRolloutBuffer::Reset)
		.def("IsValid", &OaRlRolloutBuffer::IsValid)
		.def("IsFull", &OaRlRolloutBuffer::IsFull)
		.def("IsFinalized", &OaRlRolloutBuffer::IsFinalized)
		.def("Size", &OaRlRolloutBuffer::Size)
		.def("Capacity", &OaRlRolloutBuffer::Capacity)
		.def_prop_ro("Batch", [](OaRlRolloutBuffer& self) -> const OaRlRolloutBatch& {
			return self.Batch();
		}, nb::rv_policy::reference_internal);

	nb::enum_<OaRlTrainingPhase>(m, "OaRlTrainingPhase")
		.value("Collect", OaRlTrainingPhase::Collect)
		.value("Update", OaRlTrainingPhase::Update)
		.value("Complete", OaRlTrainingPhase::Complete);

	nb::class_<OaItRlTrainingConfig>(m, "OaItRlTrainingConfig")
		.def(nb::init<>())
		.def_rw("Rollouts", &OaItRlTrainingConfig::Rollouts)
		.def_rw("Horizon", &OaItRlTrainingConfig::Horizon)
		.def_rw("Environments", &OaItRlTrainingConfig::Environments)
		.def_rw("UpdateEpochs", &OaItRlTrainingConfig::UpdateEpochs);

	nb::class_<OaItRlTraining>(m, "OaItRlTraining")
		.def("__init__", [](OaItRlTraining* self, OaOptimizer& optimizer,
			const OaItRlTrainingConfig& config) {
			new (self) OaItRlTraining(optimizer, config);
		}, nb::arg("optimizer"), nb::arg("config"), nb::keep_alive<1, 2>())
		.def("BeginRollout", [](OaItRlTraining& self,
			OaRlRolloutBuffer& rollout) {
			throw_if_error(self.BeginRollout(rollout));
		}, nb::arg("rollout"))
		.def("FinalizeRollout", [](OaItRlTraining& self,
			OaRlRolloutBuffer& rollout, const OaGaeConfig& config) {
			throw_if_error(self.FinalizeRollout(rollout, config));
		}, nb::arg("rollout"), nb::arg("config") = OaGaeConfig())
		.def("BeginUpdate", &OaItRlTraining::BeginUpdate)
		.def("NextUpdate", [](OaItRlTraining& self, const OaMatrix& loss) {
			throw_if_error(self.NextUpdate(loss));
		}, nb::arg("loss"))
		.def("Finish", [](OaItRlTraining& self) {
			throw_if_error(self.Finish());
		})
		.def("IsValid", &OaItRlTraining::IsValid)
		.def("IsDone", &OaItRlTraining::IsDone)
		.def("Phase", &OaItRlTraining::Phase)
		.def("RolloutIndex", &OaItRlTraining::RolloutIndex)
		.def("UpdateEpoch", &OaItRlTraining::UpdateEpoch)
		.def("UpdateLoop", nb::overload_cast<>(&OaItRlTraining::UpdateLoop),
			nb::rv_policy::reference_internal);
}
