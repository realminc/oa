// OA Tutorial - deterministic scalar Lunar Lander 3D headless render.
//
// Usage:
//   TutorialRlLunarLander3dHeadless [output-directory]
//
// The bounded run writes a binary PPM sequence and manifest.json. The scripted
// controller must reach a safe landing and its scalar trajectory has a frozen
// same-build digest. Rendered-image digests are scoped to the Vulkan device and
// driver recorded in the manifest.

#include <Oa/Core/Filesystem.h>
#include <Oa/Core/Paths.h>
#include <Oa/Core/Log.h>
#include <Oa/Runtime/Engine.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>

#include "LunarLander3d.h"
#include "LunarLander3dRender.h"

class OaLunarHeadlessDigest {
public:
	void AddBytes(OaSpan<const OaU8> InBytes) noexcept {
		for (const OaU8 byte : InBytes) {
			Hash_ ^= byte;
			Hash_ *= 1099511628211ULL;
		}
	}

	void AddU64(OaU64 InValue) noexcept {
		for (OaU32 byteIndex = 0U; byteIndex < 8U; ++byteIndex) {
			const OaU8 byte = static_cast<OaU8>(
				(InValue >> (byteIndex * 8U)) & 0xffU);
			AddBytes(OaSpan<const OaU8>(&byte, 1U));
		}
	}

	void AddU32(OaU32 InValue) noexcept { AddU64(InValue); }
	void AddBool(bool InValue) noexcept { AddU64(InValue ? 1U : 0U); }

	void AddDouble(OaF64 InValue) noexcept {
		AddU64(static_cast<OaU64>(static_cast<OaI64>(
			std::llround(InValue * 1.0e9))));
	}

	void AddFloat(OaF32 InValue) noexcept {
		AddU64(static_cast<OaU64>(static_cast<OaI64>(
			std::llround(static_cast<OaF64>(InValue) * 1.0e6))));
	}

	[[nodiscard]] OaU64 Value() const noexcept { return Hash_; }

private:
	OaU64 Hash_ = 1469598103934665603ULL;
};

class OaLunarHeadlessFrameRecord {
public:
	OaString Filename_;
	OaU32 EpisodeStep_ = 0U;
	OaU64 PpmDigest_ = 0U;
};

class OaLunarHeadlessSummary {
public:
	OaU32 FrameCount_ = 0U;
	OaU32 EpisodeSteps_ = 0U;
	OaLunarEndReason EndReason_ = OaLunarEndReason::None;
	OaI64 EpisodeReturnQ1e9_ = 0;
	OaU64 TraceDigest_ = 0U;
	OaU64 ImageSequenceDigest_ = 0U;
	OaPath ManifestPath_;
};

static void OaLunarHeadlessDigestVector(
	OaLunarHeadlessDigest& InOutDigest,
	const OaLunarVec3& InVector) noexcept {
	InOutDigest.AddDouble(InVector.ComponentX_);
	InOutDigest.AddDouble(InVector.ComponentY_);
	InOutDigest.AddDouble(InVector.ComponentZ_);
}

static void OaLunarHeadlessDigestQuaternion(
	OaLunarHeadlessDigest& InOutDigest,
	const OaLunarQuat& InQuaternion) noexcept {
	InOutDigest.AddDouble(InQuaternion.Scalar_);
	InOutDigest.AddDouble(InQuaternion.ComponentX_);
	InOutDigest.AddDouble(InQuaternion.ComponentY_);
	InOutDigest.AddDouble(InQuaternion.ComponentZ_);
}

static void OaLunarHeadlessDigestManifest(
	OaLunarHeadlessDigest& InOutDigest,
	const OaLunarEpisodeManifest& InManifest) noexcept {
	InOutDigest.AddU32(InManifest.EnvironmentVersion_);
	InOutDigest.AddU32(InManifest.RandomVersion_);
	InOutDigest.AddU32(InManifest.TerrainVersion_);
	InOutDigest.AddU32(InManifest.PhysicsVersion_);
	InOutDigest.AddU32(InManifest.ObservationVersion_);
	InOutDigest.AddU32(InManifest.RewardVersion_);
	InOutDigest.AddU64(InManifest.ConfigFingerprint_);
	InOutDigest.AddU64(InManifest.BaseSeed_);
	InOutDigest.AddU32(InManifest.EnvironmentLane_);
	InOutDigest.AddU64(InManifest.EpisodeIndex_);
	InOutDigest.AddU64(InManifest.TerrainSeed_);
	InOutDigest.AddU64(InManifest.SpawnSeed_);
	InOutDigest.AddU64(InManifest.DomainSeed_);
}

static void OaLunarHeadlessDigestState(
	OaLunarHeadlessDigest& InOutDigest,
	const OaLunarLander3dState& InState) noexcept {
	OaLunarHeadlessDigestVector(InOutDigest, InState.Position_);
	OaLunarHeadlessDigestVector(InOutDigest, InState.LinearVelocity_);
	OaLunarHeadlessDigestQuaternion(InOutDigest, InState.Orientation_);
	OaLunarHeadlessDigestVector(InOutDigest, InState.AngularVelocityBody_);
	InOutDigest.AddDouble(InState.Fuel_);
	InOutDigest.AddU32(static_cast<OaU32>(InState.LastAction_));
	InOutDigest.AddDouble(InState.MainThrottle_);
	OaLunarHeadlessDigestVector(InOutDigest, InState.AttitudeCommandBody_);
	for (const bool contact : InState.BodyContacts_) {
		InOutDigest.AddBool(contact);
	}
	for (const OaF64 impulse : InState.BodyContactImpulses_) {
		InOutDigest.AddDouble(impulse);
	}
	for (const bool contact : InState.FootContacts_) {
		InOutDigest.AddBool(contact);
	}
	for (const bool onPad : InState.FeetOnPad_) {
		InOutDigest.AddBool(onPad);
	}
	for (const OaF64 impulse : InState.FootContactImpulses_) {
		InOutDigest.AddDouble(impulse);
	}
	for (const bool rewarded : InState.FootContactRewarded_) {
		InOutDigest.AddBool(rewarded);
	}
	InOutDigest.AddU32(InState.EpisodeStep_);
	InOutDigest.AddU32(InState.StableDwell_);
	InOutDigest.AddBool(InState.Terminated_);
	InOutDigest.AddBool(InState.Truncated_);
	InOutDigest.AddU32(static_cast<OaU32>(InState.EndReason_));
	InOutDigest.AddDouble(InState.EpisodeReturn_);
}

static void OaLunarHeadlessDigestReward(
	OaLunarHeadlessDigest& InOutDigest,
	const OaLunarRewardTerms& InReward) noexcept {
	InOutDigest.AddDouble(InReward.PotentialBefore_);
	InOutDigest.AddDouble(InReward.PotentialAfter_);
	InOutDigest.AddDouble(InReward.Shaping_);
	InOutDigest.AddDouble(InReward.MainFuelCost_);
	InOutDigest.AddDouble(InReward.AttitudeFuelCost_);
	InOutDigest.AddDouble(InReward.SoftFootContact_);
	InOutDigest.AddDouble(InReward.StableDwell_);
	InOutDigest.AddDouble(InReward.Terminal_);
	InOutDigest.AddDouble(InReward.Total_);
}

static const char* OaLunarHeadlessEndReasonName(
	OaLunarEndReason InReason) noexcept {
	switch (InReason) {
		case OaLunarEndReason::None: return "none";
		case OaLunarEndReason::SafeLanding: return "safe_landing";
		case OaLunarEndReason::BodyImpact: return "body_impact";
		case OaLunarEndReason::HardFootImpact: return "hard_foot_impact";
		case OaLunarEndReason::OutOfBounds: return "out_of_bounds";
		case OaLunarEndReason::NumericalFailure: return "numerical_failure";
		case OaLunarEndReason::TimeLimit: return "time_limit";
		case OaLunarEndReason::ExternalStop: return "external_stop";
		case OaLunarEndReason::InvalidAction: return "invalid_action";
	}
	return "unknown";
}

static void OaLunarHeadlessAppendJsonString(
	OaString& InOutText,
	OaStringView InValue) {
	static constexpr char OA_HEX[] = "0123456789abcdef";
	InOutText += '"';
	for (const char character : InValue) {
		switch (character) {
			case '"': InOutText += "\\\""; break;
			case '\\': InOutText += "\\\\"; break;
			case '\b': InOutText += "\\b"; break;
			case '\f': InOutText += "\\f"; break;
			case '\n': InOutText += "\\n"; break;
			case '\r': InOutText += "\\r"; break;
			case '\t': InOutText += "\\t"; break;
			default: {
				const OaU8 byte = static_cast<OaU8>(character);
				if (byte < 0x20U) {
					InOutText += "\\u00";
					InOutText += OA_HEX[byte >> 4U];
					InOutText += OA_HEX[byte & 0x0fU];
				} else {
					InOutText += character;
				}
				break;
			}
		}
	}
	InOutText += '"';
}

static OaStatus OaLunarHeadlessBuildPpm(
	const OaLunarLander3dReadback& InReadback,
	OaVec<OaU8>& OutPpm) {
	const OaU64 pixelCount = static_cast<OaU64>(InReadback.Width_)
		* static_cast<OaU64>(InReadback.Height_);
	if (pixelCount == 0U
		or pixelCount > std::numeric_limits<OaUsize>::max() / 4U
		or InReadback.ColorRgba8_.size()
			!= static_cast<OaUsize>(pixelCount * 4U)) {
		return OaStatus::Error(
			OaStatusCode::DataLoss,
			"lunar render returned an invalid RGBA8 extent");
	}

	char header[64];
	const int headerSize = std::snprintf(
		header, sizeof(header), "P6\n%u %u\n255\n",
		InReadback.Width_, InReadback.Height_);
	if (headerSize <= 0
		or static_cast<OaUsize>(headerSize) >= sizeof(header)
		or pixelCount > (std::numeric_limits<OaUsize>::max()
			- static_cast<OaUsize>(headerSize)) / 3U) {
		return OaStatus::Error(
			OaStatusCode::OutOfRange,
			"lunar PPM extent exceeds host addressability");
	}

	OutPpm.Resize(static_cast<OaUsize>(headerSize) + pixelCount * 3U);
	OaMemcpy(OutPpm.Data(), header, static_cast<OaUsize>(headerSize));
	OaU8* destination = OutPpm.Data() + headerSize;
	for (OaU64 pixel = 0U; pixel < pixelCount; ++pixel) {
		const OaUsize source = static_cast<OaUsize>(pixel * 4U);
		const OaUsize target = static_cast<OaUsize>(pixel * 3U);
		destination[target] = InReadback.ColorRgba8_[source];
		destination[target + 1U] = InReadback.ColorRgba8_[source + 1U];
		destination[target + 2U] = InReadback.ColorRgba8_[source + 2U];
	}
	return OaStatus::Ok();
}

static OaStatus OaLunarHeadlessRenderFrame(
	OaLunarLander3dRenderSession& InSession,
	const OaLunarLander3dState& InState,
	const OaCameraState& InCamera,
	const OaPath& InOutputDirectory,
	OaU32 InFrameIndex,
	OaLunarHeadlessDigest& InOutImageSequenceDigest,
	OaVec<OaLunarHeadlessFrameRecord>& InOutRecords) {
	OA_RETURN_IF_ERROR(InSession.BeginFrame(InState, InCamera));
	auto frameResult = InSession.SubmitFrame();
	if (frameResult.IsError()) {
		const OaStatus submitStatus = frameResult.GetStatus();
		const OaStatus cancelStatus = InSession.CancelFrame();
		if (cancelStatus.IsError()) {
			OA_LOG_ERROR(
				OaLogComponent::App,
				"Lunar headless frame cancellation failed after submit error: %s",
				cancelStatus.ToString().CStr());
		}
		return submitStatus;
	}
	auto readbackResult = InSession.ConsumeReadback(*frameResult);
	if (readbackResult.IsError()) return readbackResult.GetStatus();

	OaVec<OaU8> ppm;
	OA_RETURN_IF_ERROR(OaLunarHeadlessBuildPpm(*readbackResult, ppm));
	char filename[32];
	const int filenameSize = std::snprintf(
		filename, sizeof(filename), "frame_%04u.ppm", InFrameIndex);
	if (filenameSize <= 0
		or static_cast<OaUsize>(filenameSize) >= sizeof(filename)) {
		return OaStatus::Error(
			OaStatusCode::Internal,
			"lunar frame filename formatting failed");
	}
	OA_RETURN_IF_ERROR(OaFilesystem::WriteBinary(
		InOutputDirectory / filename,
		OaSpan<const OaU8>(ppm.Data(), ppm.Size())));

	OaLunarHeadlessDigest frameDigest;
	frameDigest.AddBytes(OaSpan<const OaU8>(ppm.Data(), ppm.Size()));
	InOutImageSequenceDigest.AddU32(InFrameIndex);
	InOutImageSequenceDigest.AddBytes(
		OaSpan<const OaU8>(ppm.Data(), ppm.Size()));
	OaLunarHeadlessFrameRecord record;
	record.Filename_ = filename;
	record.EpisodeStep_ = InState.EpisodeStep_;
	record.PpmDigest_ = frameDigest.Value();
	InOutRecords.PushBack(OaStdMove(record));
	return OaStatus::Ok();
}

static OaString OaLunarHeadlessBuildManifestJson(
	const OaLunarLander3dConfig& InConfig,
	const OaLunarEpisodeManifest& InEpisodeManifest,
	const OaLunarLander3dState& InFinalState,
	const OaEngine& InEngine,
	OaU32 InWidth,
	OaU32 InHeight,
	OaU64 InTraceDigest,
	OaU64 InImageSequenceDigest,
	const OaVec<OaLunarHeadlessFrameRecord>& InRecords) {
	char values[1024];
	const int valueSize = std::snprintf(
		values, sizeof(values),
		"{\n"
		"  \"format_version\": 1,\n"
		"  \"environment\": \"oa_lunar_lander_3d\",\n"
		"  \"environment_version\": %u,\n"
		"  \"physics_version\": %u,\n"
		"  \"observation_version\": %u,\n"
		"  \"reward_version\": %u,\n"
		"  \"base_seed_hex\": \"%016llx\",\n"
		"  \"environment_lane\": %u,\n"
		"  \"episode_index\": %llu,\n"
		"  \"config_fingerprint_hex\": \"%016llx\",\n"
		"  \"controller\": \"scripted_descent_lateral_attitude_pd_v3\",\n"
		"  \"episode_steps\": %u,\n"
		"  \"end_reason\": \"%s\",\n"
		"  \"episode_return_q1e9\": %lld,\n"
		"  \"width\": %u,\n"
		"  \"height\": %u,\n"
		"  \"trace_digest_algorithm\": \"oa_lunar_quantized_fnv64_v1\",\n"
		"  \"trace_digest_hex\": \"%016llx\",\n"
		"  \"image_digest_algorithm\": \"ppm_bytes_fnv64_v1\",\n"
		"  \"image_sequence_digest_hex\": \"%016llx\",\n"
		"  \"device\": {\n"
		"    \"name\": ",
		InConfig.EnvironmentVersion_, InConfig.PhysicsVersion_,
		InConfig.ObservationVersion_, InConfig.RewardVersion_,
		static_cast<unsigned long long>(InEpisodeManifest.BaseSeed_),
		InEpisodeManifest.EnvironmentLane_,
		static_cast<unsigned long long>(InEpisodeManifest.EpisodeIndex_),
		static_cast<unsigned long long>(InEpisodeManifest.ConfigFingerprint_),
		InFinalState.EpisodeStep_,
		OaLunarHeadlessEndReasonName(InFinalState.EndReason_),
		static_cast<long long>(std::llround(InFinalState.EpisodeReturn_ * 1.0e9)),
		InWidth, InHeight,
		static_cast<unsigned long long>(InTraceDigest),
		static_cast<unsigned long long>(InImageSequenceDigest));
	OaString result;
	if (valueSize > 0 and static_cast<OaUsize>(valueSize) < sizeof(values)) {
		result += OaStringView(values, static_cast<OaUsize>(valueSize));
	}
	OaLunarHeadlessAppendJsonString(result, InEngine.DeviceName());
	result += ",\n    \"driver_name\": ";
	OaLunarHeadlessAppendJsonString(
		result, OaStringView(InEngine.Device.Info.Software.DriverName));
	result += ",\n    \"driver_version\": ";
	OaLunarHeadlessAppendJsonString(
		result, OaStringView(InEngine.Device.Info.Software.DriverVersion));
	result += ",\n    \"vulkan_api_version\": ";
	OaLunarHeadlessAppendJsonString(
		result, OaStringView(InEngine.Device.Info.Software.ApiVersion));
	result += "\n  },\n  \"frames\": [\n";
	for (OaUsize index = 0U; index < InRecords.Size(); ++index) {
		const OaLunarHeadlessFrameRecord& record = InRecords[index];
		char line[256];
		const int lineSize = std::snprintf(
			line, sizeof(line),
			"    {\"file\": \"%s\", \"episode_step\": %u, "
			"\"ppm_digest_hex\": \"%016llx\"}%s\n",
			record.Filename_.CStr(), record.EpisodeStep_,
			static_cast<unsigned long long>(record.PpmDigest_),
			index + 1U == InRecords.Size() ? "" : ",");
		if (lineSize > 0 and static_cast<OaUsize>(lineSize) < sizeof(line)) {
			result += OaStringView(line, static_cast<OaUsize>(lineSize));
		}
	}
	result += "  ]\n}\n";
	return result;
}

static OaResult<OaLunarHeadlessSummary> OaLunarHeadlessRun(
	const OaPath& InOutputDirectory) {
	constexpr OaU32 OA_WIDTH = 256U;
	constexpr OaU32 OA_HEIGHT = 192U;
	constexpr OaU32 OA_EPISODE_STEPS = 1200U;
	constexpr OaU32 OA_RENDER_STRIDE = 8U;
	constexpr OaU64 OA_BASE_SEED = 0x123456789abcdef0ULL;
	constexpr OaU32 OA_ENVIRONMENT_LANE = 3U;
	constexpr OaU64 OA_EPISODE_INDEX = 7U;
	constexpr OaU64 OA_EXPECTED_TRACE_DIGEST = 0x03deb1940a0f80cfULL;

	if (InOutputDirectory.Empty()) {
		return OaStatus::InvalidArgument(
			"lunar headless output directory must not be empty");
	}
	OA_RETURN_IF_ERROR(OaFilesystem::CreateDirectories(InOutputDirectory));

	OaLunarLander3dConfig landerConfig;
	landerConfig.SafeDwellSteps_ = 12U;
	landerConfig.MaxEpisodeSteps_ = OA_EPISODE_STEPS;
	const OaLunarEpisodeManifest episodeManifest =
		OaLunarEpisodeManifest::DeriveVersioned(
			OA_BASE_SEED, OA_ENVIRONMENT_LANE, OA_EPISODE_INDEX,
			landerConfig.EnvironmentVersion_, OA_LUNAR_TERRAIN_VERSION,
			landerConfig.PhysicsVersion_, landerConfig.ObservationVersion_,
			landerConfig.RewardVersion_, landerConfig.ContractFingerprint());
	auto environment = OaLunarScalarEnvironment::CreateSeeded(
		landerConfig, episodeManifest);
	if (not environment.IsValid()) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			OaString("lunar scalar environment creation failed: ")
				+ environment.Error());
	}
	OaLunarLander3dState initialState;
	initialState.Position_ = {0.0, 4.0, 0.0};
	initialState.LinearVelocity_.ComponentY_ = -0.2;
	initialState.Fuel_ = landerConfig.FuelCapacity_;
	if (not environment.SetState(initialState)) {
		return OaStatus::Error(
			OaStatusCode::FailedPrecondition,
			"lunar scripted initial state was rejected");
	}

	OaEngineConfig engineConfig;
	engineConfig.PresentationMode = OaPresentationMode::Headless;
	engineConfig.NumericMode = OaNumericMode::Deterministic;
	engineConfig.RegisterAsGlobal = false;
	engineConfig.PreloadEmbeddedPipelines = false;
	engineConfig.EnablePipelineCache = false;
	engineConfig.AppName = "TutorialRlLunarLander3dHeadless";
	auto engineResult = OaEngine::Create(engineConfig);
	if (engineResult.IsError()) return engineResult.GetStatus();
	OaUniquePtr<OaEngine> engine = OaStdMove(*engineResult);

	OaLunarLander3dRenderConfig renderConfig;
	renderConfig.Width_ = OA_WIDTH;
	renderConfig.Height_ = OA_HEIGHT;
	renderConfig.TargetSlotCount_ = 1U;
	auto sessionResult = OaLunarLander3dRenderSession::Create(
		*engine, landerConfig, environment.Terrain(), renderConfig);
	if (sessionResult.IsError()) {
		const OaStatus createStatus = sessionResult.GetStatus();
		const OaStatus closeStatus = engine->Close();
		if (closeStatus.IsError()) {
			OA_LOG_ERROR(OaLogComponent::App,
				"Lunar headless engine close failed: %s",
				closeStatus.ToString().CStr());
		}
		return createStatus;
	}
	OaUniquePtr<OaLunarLander3dRenderSession> session =
		OaStdMove(*sessionResult);

	OaStatus runStatus = OaStatus::Ok();
	OaLunarHeadlessDigest traceDigest;
	OaLunarHeadlessDigest imageSequenceDigest;
	OaVec<OaLunarHeadlessFrameRecord> records;
	OaLunarHeadlessDigestManifest(traceDigest, episodeManifest);
	for (const OaF64 height : environment.Terrain().Heights()) {
		traceDigest.AddDouble(height);
	}
	OaLunarHeadlessDigestState(traceDigest, environment.State());
	const OaCameraState camera =
		OaLunarLander3dRenderSession::DefaultCamera(OA_WIDTH, OA_HEIGHT);
	runStatus = OaLunarHeadlessRenderFrame(
		*session, environment.State(), camera, InOutputDirectory,
		0U, imageSequenceDigest, records);

	for (OaU32 step = 0U;
		step < OA_EPISODE_STEPS and runStatus.IsOk()
			and not environment.State().Terminated_
			and not environment.State().Truncated_;
		++step) {
		const OaLunarAction action = OaLunarScriptedLandingAction(
			landerConfig, environment.State());
		traceDigest.AddU32(static_cast<OaU32>(action));
		const OaLunarTransition transition = environment.Step(
			static_cast<OaU32>(action));
		if (not transition.Valid_) {
			runStatus = OaStatus::Error(
				OaStatusCode::DataLoss,
				OaString("lunar scalar transition failed: ")
					+ transition.Error_);
			break;
		}
		OaLunarHeadlessDigestState(traceDigest, environment.State());
		OaLunarHeadlessDigestReward(traceDigest, transition.RewardTerms_);
		traceDigest.AddDouble(transition.Reward_);
		traceDigest.AddBool(transition.Terminated_);
		traceDigest.AddBool(transition.Truncated_);
		traceDigest.AddU32(static_cast<OaU32>(transition.EndReason_));
		for (const OaF32 observation : transition.Observation_) {
			traceDigest.AddFloat(observation);
		}
		if ((step + 1U) % OA_RENDER_STRIDE == 0U
			or transition.Terminated_ or transition.Truncated_) {
			runStatus = OaLunarHeadlessRenderFrame(
				*session, environment.State(), camera, InOutputDirectory,
				static_cast<OaU32>(records.Size()),
				imageSequenceDigest, records);
		}
	}

	if (runStatus.IsOk()
		and (not environment.State().Terminated_
			or environment.State().Truncated_
			or environment.State().EndReason_ != OaLunarEndReason::SafeLanding)) {
		runStatus = OaStatus::Error(
			OaStatusCode::DataLoss,
			"lunar scripted controller did not reach a safe landing");
	}
	if (runStatus.IsOk() and OA_EXPECTED_TRACE_DIGEST != 0U
		and traceDigest.Value() != OA_EXPECTED_TRACE_DIGEST) {
		char digestError[192];
		const int digestErrorSize = std::snprintf(
			digestError, sizeof(digestError),
			"lunar frozen trace digest mismatch: expected=%016llx actual=%016llx",
			static_cast<unsigned long long>(OA_EXPECTED_TRACE_DIGEST),
			static_cast<unsigned long long>(traceDigest.Value()));
		runStatus = OaStatus::Error(
			OaStatusCode::DataLoss,
			digestErrorSize > 0
				? OaString(digestError)
				: OaString("lunar frozen trace digest mismatch"));
	}

	const OaString manifestJson = runStatus.IsOk()
		? OaLunarHeadlessBuildManifestJson(
			landerConfig, episodeManifest, environment.State(), *engine,
			OA_WIDTH, OA_HEIGHT, traceDigest.Value(),
			imageSequenceDigest.Value(), records)
		: OaString{};
	const OaStatus sessionCloseStatus = session->Close();
	const OaStatus engineCloseStatus = engine->Close();
	if (runStatus.IsOk() and sessionCloseStatus.IsError()) {
		runStatus = sessionCloseStatus;
	} else if (sessionCloseStatus.IsError()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"Lunar headless renderer close failed: %s",
			sessionCloseStatus.ToString().CStr());
	}
	if (runStatus.IsOk() and engineCloseStatus.IsError()) {
		runStatus = engineCloseStatus;
	} else if (engineCloseStatus.IsError()) {
		OA_LOG_ERROR(OaLogComponent::App,
			"Lunar headless engine close failed: %s",
			engineCloseStatus.ToString().CStr());
	}
	if (runStatus.IsError()) return runStatus;

	const OaPath manifestPath = InOutputDirectory / "manifest.json";
	OA_RETURN_IF_ERROR(OaFilesystem::WriteText(manifestPath, manifestJson));
	OaLunarHeadlessSummary summary;
	summary.FrameCount_ = static_cast<OaU32>(records.Size());
	summary.EpisodeSteps_ = environment.State().EpisodeStep_;
	summary.EndReason_ = environment.State().EndReason_;
	summary.EpisodeReturnQ1e9_ = static_cast<OaI64>(std::llround(
		environment.State().EpisodeReturn_ * 1.0e9));
	summary.TraceDigest_ = traceDigest.Value();
	summary.ImageSequenceDigest_ = imageSequenceDigest.Value();
	summary.ManifestPath_ = manifestPath;
	return summary;
}

int main(int argc, char** argv) {
	if (argc == 2
		and (OaStringView(argv[1]) == "--help"
			or OaStringView(argv[1]) == "-h")) {
		OA_CLI("Usage: TutorialRlLunarLander3dHeadless [output-directory]");
		return 0;
	}
	if (argc > 2) {
		OA_LOG_ERROR(OaLogComponent::App,
			"Expected at most one output-directory argument");
		return 1;
	}
	const OaPath outputDirectory = argc == 2
		? OaPath(argv[1])
		: OaPaths::Var("tutorial") / "lunar_lander_3d_headless";
	auto result = OaLunarHeadlessRun(outputDirectory);
	if (result.IsError()) {
		const OaStatus& status = result.GetStatus();
		OA_LOG_ERROR(OaLogComponent::App,
			"Lunar headless tutorial failed: %s", status.ToString().CStr());
		if (status.GetCode() == OaStatusCode::DeviceNotFound
			or status.GetCode() == OaStatusCode::Unavailable) {
			return 125;
		}
		return 1;
	}
	OA_LOG_INFO(OaLogComponent::App,
		"Lunar headless tutorial wrote %u frames and %s "
		"(trace=%016llx, images=%016llx)",
		result->FrameCount_, result->ManifestPath_.CStr(),
		static_cast<unsigned long long>(result->TraceDigest_),
		static_cast<unsigned long long>(result->ImageSequenceDigest_));
	return 0;
}
