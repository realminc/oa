// OA Tutorial - deterministic scalar Lunar Lander 3D headless render.
//
// usage:
//   TutorialRlLunarLander3dHeadless [output-directory]
//
// The bounded run writes a binary PPM sequence and manifest.json. The scripted
// controller must reach a safe landing and its scalar trajectory has a frozen
// same-build digest. Rendered-image digests are scoped to the vulkan device and
// driver recorded in the manifest.

#include <oa/core/filesystem.h>
#include <oa/core/paths.h>
#include <oa/core/log.h>
#include <oa/runtime/engine.h>

#include <core/streamText.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>

#include <ml/rl/lunarLander3d.h>
#include "lunarLander3dRender.h"

class LunarHeadlessDigest {
public:
	void addBytes(oa::Span<const oa::U8> inBytes) noexcept {
		for (const oa::U8 byte : inBytes) {
			hash_ ^= byte;
			hash_ *= 1099511628211ULL;
		}
	}

	void addU64(oa::U64 inValue) noexcept {
		for (oa::U32 byteIndex = 0U; byteIndex < 8U; ++byteIndex) {
			const oa::U8 byte = static_cast<oa::U8>(
				(inValue >> (byteIndex * 8U)) & 0xffU);
			addBytes(oa::Span<const oa::U8>(&byte, 1U));
		}
	}

	void addU32(oa::U32 inValue) noexcept { addU64(inValue); }
	void addBool(bool inValue) noexcept { addU64(inValue ? 1U : 0U); }

	void addDouble(oa::F64 inValue) noexcept {
		addU64(static_cast<oa::U64>(static_cast<oa::I64>(
			std::llround(inValue * 1.0e9))));
	}

	void addFloat(oa::F32 inValue) noexcept {
		addU64(static_cast<oa::U64>(static_cast<oa::I64>(
			std::llround(static_cast<oa::F64>(inValue) * 1.0e6))));
	}

	[[nodiscard]] oa::U64 value() const noexcept { return hash_; }

private:
	oa::U64 hash_ = 1469598103934665603ULL;
};

class LunarHeadlessFrameRecord {
public:
	oa::String filename_;
	oa::U32 episodeStep_ = 0U;
	oa::U64 ppmDigest_ = 0U;
};

class LunarHeadlessSummary {
public:
	oa::U32 frameCount_ = 0U;
	oa::U32 episodeSteps_ = 0U;
	oa::LunarEndReason endReason_ = oa::LunarEndReason::None;
	oa::I64 episodeReturnQ1e9_ = 0;
	oa::U64 traceDigest_ = 0U;
	oa::U64 imageSequenceDigest_ = 0U;
	oa::Path manifestPath_;
};

static void lunarHeadlessDigestVector(
	LunarHeadlessDigest& inOutDigest,
	const oa::vlm::DVec3& inVector) noexcept {
	inOutDigest.addDouble(inVector.x);
	inOutDigest.addDouble(inVector.y);
	inOutDigest.addDouble(inVector.z);
}

static void lunarHeadlessDigestQuaternion(
	LunarHeadlessDigest& inOutDigest,
	const oa::vlm::DQuat& inQuaternion) noexcept {
	inOutDigest.addDouble(inQuaternion.w);
	inOutDigest.addDouble(inQuaternion.x);
	inOutDigest.addDouble(inQuaternion.y);
	inOutDigest.addDouble(inQuaternion.z);
}

static void lunarHeadlessDigestManifest(
	LunarHeadlessDigest& inOutDigest,
	const oa::LunarEpisodeManifest& inManifest) noexcept {
	inOutDigest.addU32(inManifest.environmentVersion_);
	inOutDigest.addU32(inManifest.randomVersion_);
	inOutDigest.addU32(inManifest.terrainVersion_);
	inOutDigest.addU32(inManifest.physicsVersion_);
	inOutDigest.addU32(inManifest.observationVersion_);
	inOutDigest.addU32(inManifest.rewardVersion_);
	inOutDigest.addU64(inManifest.configFingerprint_);
	inOutDigest.addU64(inManifest.baseSeed_);
	inOutDigest.addU32(inManifest.environmentLane_);
	inOutDigest.addU64(inManifest.episodeIndex_);
	inOutDigest.addU64(inManifest.terrainSeed_);
	inOutDigest.addU64(inManifest.spawnSeed_);
	inOutDigest.addU64(inManifest.domainSeed_);
}

static void lunarHeadlessDigestState(
	LunarHeadlessDigest& inOutDigest,
	const oa::LunarLander3dState& inState) noexcept {
	lunarHeadlessDigestVector(inOutDigest, inState.position_);
	lunarHeadlessDigestVector(inOutDigest, inState.linearVelocity_);
	lunarHeadlessDigestQuaternion(inOutDigest, inState.orientation_);
	lunarHeadlessDigestVector(inOutDigest, inState.angularVelocityBody_);
	inOutDigest.addDouble(inState.fuel_);
	inOutDigest.addU32(static_cast<oa::U32>(inState.lastAction_));
	inOutDigest.addDouble(inState.mainThrottle_);
	lunarHeadlessDigestVector(inOutDigest, inState.attitudeCommandBody_);
	for (const bool contact : inState.bodyContacts_) {
		inOutDigest.addBool(contact);
	}
	for (const oa::F64 impulse : inState.bodyContactImpulses_) {
		inOutDigest.addDouble(impulse);
	}
	for (const bool contact : inState.footContacts_) {
		inOutDigest.addBool(contact);
	}
	for (const bool onPad : inState.feetOnPad_) {
		inOutDigest.addBool(onPad);
	}
	for (const oa::F64 impulse : inState.footContactImpulses_) {
		inOutDigest.addDouble(impulse);
	}
	for (const bool rewarded : inState.footContactRewarded_) {
		inOutDigest.addBool(rewarded);
	}
	inOutDigest.addU32(inState.episodeStep_);
	inOutDigest.addU32(inState.stableDwell_);
	inOutDigest.addBool(inState.terminated_);
	inOutDigest.addBool(inState.truncated_);
	inOutDigest.addU32(static_cast<oa::U32>(inState.endReason_));
	inOutDigest.addDouble(inState.episodeReturn_);
}

static void lunarHeadlessDigestReward(
	LunarHeadlessDigest& inOutDigest,
	const oa::LunarRewardTerms& inReward) noexcept {
	inOutDigest.addDouble(inReward.potentialBefore_);
	inOutDigest.addDouble(inReward.potentialAfter_);
	inOutDigest.addDouble(inReward.shaping_);
	inOutDigest.addDouble(inReward.mainFuelCost_);
	inOutDigest.addDouble(inReward.attitudeFuelCost_);
	inOutDigest.addDouble(inReward.softFootContact_);
	inOutDigest.addDouble(inReward.stableDwell_);
	inOutDigest.addDouble(inReward.terminal_);
	inOutDigest.addDouble(inReward.total_);
}

static const char* lunarHeadlessEndReasonName(
	oa::LunarEndReason inReason) noexcept {
	switch (inReason) {
		case oa::LunarEndReason::None: return "none";
		case oa::LunarEndReason::SafeLanding: return "safe_landing";
		case oa::LunarEndReason::BodyImpact: return "body_impact";
		case oa::LunarEndReason::HardFootImpact: return "hard_foot_impact";
		case oa::LunarEndReason::OutOfBounds: return "out_of_bounds";
		case oa::LunarEndReason::NumericalFailure: return "numerical_failure";
		case oa::LunarEndReason::TimeLimit: return "time_limit";
		case oa::LunarEndReason::ExternalStop: return "external_stop";
		case oa::LunarEndReason::InvalidAction: return "invalid_action";
	}
	return "unknown";
}

static void lunarHeadlessAppendJsonString(
	oa::String& inOutText,
	oa::StringView inValue) {
	static constexpr char OA_HEX[] = "0123456789abcdef";
	inOutText += '"';
	for (const char character : inValue) {
		switch (character) {
			case '"': inOutText += "\\\""; break;
			case '\\': inOutText += "\\\\"; break;
			case '\b': inOutText += "\\b"; break;
			case '\f': inOutText += "\\f"; break;
			case '\n': inOutText += "\\n"; break;
			case '\r': inOutText += "\\r"; break;
			case '\t': inOutText += "\\t"; break;
			default: {
				const oa::U8 byte = static_cast<oa::U8>(character);
				if (byte < 0x20U) {
					inOutText += "\\u00";
					inOutText += OA_HEX[byte >> 4U];
					inOutText += OA_HEX[byte & 0x0fU];
				} else {
					inOutText += character;
				}
				break;
			}
		}
	}
	inOutText += '"';
}

static oa::Status lunarHeadlessBuildPpm(
	const oa::RenderReadback& inReadback,
	oa::Vector<oa::U8>& outPpm) {
	const oa::U64 pixelCount = static_cast<oa::U64>(inReadback.width_)
		* static_cast<oa::U64>(inReadback.height_);
	if (pixelCount == 0U
		or pixelCount > std::numeric_limits<oa::Usize>::max() / 4U
		or inReadback.colorRgba8_.size()
			!= static_cast<oa::Usize>(pixelCount * 4U)) {
		return oa::Status::error(
			oa::StatusCode::DataLoss,
			"lunar render returned an invalid RGBA8 extent");
	}

	char header[64];
	const int headerSize = std::snprintf(
		header, sizeof(header), "P6\n%u %u\n255\n",
		inReadback.width_, inReadback.height_);
	if (headerSize <= 0
		or static_cast<oa::Usize>(headerSize) >= sizeof(header)
		or pixelCount > (std::numeric_limits<oa::Usize>::max()
			- static_cast<oa::Usize>(headerSize)) / 3U) {
		return oa::Status::error(
			oa::StatusCode::OutOfRange,
			"lunar PPM extent exceeds host addressability");
	}

	outPpm.resize(static_cast<oa::Usize>(headerSize) + pixelCount * 3U);
	oa::memcpy(outPpm.data(), header, static_cast<oa::Usize>(headerSize));
	oa::U8* destination = outPpm.data() + headerSize;
	for (oa::U64 pixel = 0U; pixel < pixelCount; ++pixel) {
		const oa::Usize source = static_cast<oa::Usize>(pixel * 4U);
		const oa::Usize target = static_cast<oa::Usize>(pixel * 3U);
		destination[target] = inReadback.colorRgba8_[source];
		destination[target + 1U] = inReadback.colorRgba8_[source + 1U];
		destination[target + 2U] = inReadback.colorRgba8_[source + 2U];
	}
	return oa::Status::ok();
}

static oa::Status lunarHeadlessRenderFrame(
	LunarLander3dRenderSession& inSession,
	const oa::LunarLander3dState& inState,
	const oa::CameraState& inCamera,
	const oa::Path& inOutputDirectory,
	oa::U32 inFrameIndex,
	LunarHeadlessDigest& inOutImageSequenceDigest,
	oa::Vector<LunarHeadlessFrameRecord>& inOutRecords) {
	OA_RETURN_IF_ERROR(inSession.beginFrame(inState, inCamera));
	auto frameResult = inSession.submitFrame();
	if (frameResult.isError()) {
		const oa::Status submitStatus = frameResult.getStatus();
		const oa::Status cancelStatus = inSession.cancelFrame();
		if (cancelStatus.isError()) {
			OaLogError(
				oa::LogComponent::App,
				"Lunar headless frame cancellation failed after submit error: %s",
				cancelStatus.toString().cStr());
		}
		return submitStatus;
	}
	auto readbackResult = inSession.consumeReadback(*frameResult);
	if (readbackResult.isError()) return readbackResult.getStatus();

	oa::Vector<oa::U8> ppm;
	OA_RETURN_IF_ERROR(lunarHeadlessBuildPpm(*readbackResult, ppm));
	char filename[32];
	const int filenameSize = std::snprintf(
		filename, sizeof(filename), "frame_%04u.ppm", inFrameIndex);
	if (filenameSize <= 0
		or static_cast<oa::Usize>(filenameSize) >= sizeof(filename)) {
		return oa::Status::error(
			oa::StatusCode::Internal,
			"lunar frame filename formatting failed");
	}
	OA_RETURN_IF_ERROR(oa::Filesystem::writeBinary(
		inOutputDirectory / filename,
		oa::Span<const oa::U8>(ppm.data(), ppm.size())));

	LunarHeadlessDigest frameDigest;
	frameDigest.addBytes(oa::Span<const oa::U8>(ppm.data(), ppm.size()));
	inOutImageSequenceDigest.addU32(inFrameIndex);
	inOutImageSequenceDigest.addBytes(
		oa::Span<const oa::U8>(ppm.data(), ppm.size()));
	LunarHeadlessFrameRecord record;
	record.filename_ = filename;
	record.episodeStep_ = inState.episodeStep_;
	record.ppmDigest_ = frameDigest.value();
	inOutRecords.pushBack(oa::move(record));
	return oa::Status::ok();
}

static oa::String lunarHeadlessBuildManifestJson(
	const oa::LunarLander3dConfig& inConfig,
	const oa::LunarEpisodeManifest& inEpisodeManifest,
	const oa::LunarLander3dState& inFinalState,
	const oa::Engine& inEngine,
	oa::U32 inWidth,
	oa::U32 inHeight,
	oa::U64 inTraceDigest,
	oa::U64 inImageSequenceDigest,
	const oa::Vector<LunarHeadlessFrameRecord>& inRecords) {
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
		inConfig.environmentVersion_, inConfig.physicsVersion_,
		inConfig.observationVersion_, inConfig.rewardVersion_,
		static_cast<unsigned long long>(inEpisodeManifest.baseSeed_),
		inEpisodeManifest.environmentLane_,
		static_cast<unsigned long long>(inEpisodeManifest.episodeIndex_),
		static_cast<unsigned long long>(inEpisodeManifest.configFingerprint_),
		inFinalState.episodeStep_,
		lunarHeadlessEndReasonName(inFinalState.endReason_),
		static_cast<long long>(std::llround(inFinalState.episodeReturn_ * 1.0e9)),
		inWidth, inHeight,
		static_cast<unsigned long long>(inTraceDigest),
		static_cast<unsigned long long>(inImageSequenceDigest));
	oa::String result;
	if (valueSize > 0 and static_cast<oa::Usize>(valueSize) < sizeof(values)) {
		result += oa::StringView(values, static_cast<oa::Usize>(valueSize));
	}
	lunarHeadlessAppendJsonString(result, inEngine.deviceName());
	result += ",\n    \"driver_name\": ";
	lunarHeadlessAppendJsonString(
		result, inEngine.driverName());
	result += ",\n    \"driver_version\": ";
	lunarHeadlessAppendJsonString(
		result, inEngine.driverVersion());
	result += ",\n    \"vulkan_api_version\": ";
	lunarHeadlessAppendJsonString(
		result, inEngine.vulkanApiVersion());
	result += "\n  },\n  \"frames\": [\n";
	for (oa::Usize index = 0U; index < inRecords.size(); ++index) {
		const LunarHeadlessFrameRecord& record = inRecords[index];
		char line[256];
		const int lineSize = std::snprintf(
			line, sizeof(line),
			"    {\"file\": \"%s\", \"episode_step\": %u, "
			"\"ppm_digest_hex\": \"%016llx\"}%s\n",
			record.filename_.cStr(), record.episodeStep_,
			static_cast<unsigned long long>(record.ppmDigest_),
			index + 1U == inRecords.size() ? "" : ",");
		if (lineSize > 0 and static_cast<oa::Usize>(lineSize) < sizeof(line)) {
			result += oa::StringView(line, static_cast<oa::Usize>(lineSize));
		}
	}
	result += "  ]\n}\n";
	return result;
}

static oa::Result<LunarHeadlessSummary> lunarHeadlessRun(
	const oa::Path& inOutputDirectory) {
	constexpr oa::U32 OA_WIDTH = 256U;
	constexpr oa::U32 OA_HEIGHT = 192U;
	constexpr oa::U32 OA_EPISODE_STEPS = 1200U;
	constexpr oa::U32 OA_RENDER_STRIDE = 8U;
	constexpr oa::U64 OA_BASE_SEED = 0x123456789abcdef0ULL;
	constexpr oa::U32 OA_ENVIRONMENT_LANE = 3U;
	constexpr oa::U64 OA_EPISODE_INDEX = 7U;
	constexpr oa::U64 OA_EXPECTED_TRACE_DIGEST = 0x03deb1940a0f80cfULL;

	if (inOutputDirectory.empty()) {
		return oa::Status::invalidArgument(
			"lunar headless output directory must not be empty");
	}
	OA_RETURN_IF_ERROR(oa::Filesystem::createDirectories(inOutputDirectory));

	oa::LunarLander3dConfig landerConfig;
	landerConfig.safeDwellSteps_ = 12U;
	landerConfig.maxEpisodeSteps_ = OA_EPISODE_STEPS;
	const oa::LunarEpisodeManifest episodeManifest =
		oa::LunarEpisodeManifest::deriveVersioned(
			OA_BASE_SEED, OA_ENVIRONMENT_LANE, OA_EPISODE_INDEX,
			landerConfig.environmentVersion_, oa::kLunarTerrainVersion,
			landerConfig.physicsVersion_, landerConfig.observationVersion_,
			landerConfig.rewardVersion_, landerConfig.contractFingerprint());
	auto environment = oa::LunarScalarEnvironment::createSeeded(
		landerConfig, episodeManifest);
	if (not environment.isValid()) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			oa::String("lunar scalar environment creation failed: ")
				+ oa::sdk::fromStdString(environment.error()));
	}
	oa::LunarLander3dState initialState;
	initialState.position_ = {0.0, 4.0, 0.0};
	initialState.linearVelocity_.y = -0.2;
	initialState.fuel_ = landerConfig.fuelCapacity_;
	if (not environment.setState(initialState)) {
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"lunar scripted initial state was rejected");
	}

	oa::EngineConfig engineConfig;
	engineConfig.presentationMode = oa::PresentationMode::Headless;
	engineConfig.numericMode = oa::NumericMode::Deterministic;
	engineConfig.selectForThread = false;
	engineConfig.preloadEmbeddedPipelines = false;
	engineConfig.enablePipelineCache = false;
	engineConfig.appName = "TutorialRlLunarLander3dHeadless";
	auto engineResult = oa::Engine::create(engineConfig);
	if (engineResult.isError()) return engineResult.getStatus();
	oa::UniquePtr<oa::Engine> engine = oa::move(*engineResult);

	LunarLander3dRenderConfig renderConfig;
	renderConfig.width_ = OA_WIDTH;
	renderConfig.height_ = OA_HEIGHT;
	renderConfig.targetSlotCount_ = 1U;
	auto sessionResult = LunarLander3dRenderSession::create(
		*engine, landerConfig, environment.terrain(), renderConfig);
	if (sessionResult.isError()) {
		const oa::Status createStatus = sessionResult.getStatus();
		const oa::Status closeStatus = engine->close();
		if (closeStatus.isError()) {
			OaLogError(oa::LogComponent::App,
				"Lunar headless engine close failed: %s",
				closeStatus.toString().cStr());
		}
		return createStatus;
	}
	oa::UniquePtr<LunarLander3dRenderSession> session =
		oa::move(*sessionResult);

	oa::Status runStatus = oa::Status::ok();
	LunarHeadlessDigest traceDigest;
	LunarHeadlessDigest imageSequenceDigest;
	oa::Vector<LunarHeadlessFrameRecord> records;
	lunarHeadlessDigestManifest(traceDigest, episodeManifest);
	for (const oa::F64 height : environment.terrain().heights()) {
		traceDigest.addDouble(height);
	}
	lunarHeadlessDigestState(traceDigest, environment.state());
	const oa::CameraState camera =
		LunarLander3dRenderSession::defaultCamera(OA_WIDTH, OA_HEIGHT);
	runStatus = lunarHeadlessRenderFrame(
		*session, environment.state(), camera, inOutputDirectory,
		0U, imageSequenceDigest, records);

	for (oa::U32 step = 0U;
		step < OA_EPISODE_STEPS and runStatus.isOk()
			and not environment.state().terminated_
			and not environment.state().truncated_;
		++step) {
		const oa::LunarAction action = oa::lunarScriptedLandingAction(
			landerConfig, environment.state());
		traceDigest.addU32(static_cast<oa::U32>(action));
		const oa::LunarTransition transition = environment.step(
			static_cast<oa::U32>(action));
		if (not transition.valid_) {
			runStatus = oa::Status::error(
				oa::StatusCode::DataLoss,
				oa::String("lunar scalar transition failed: ")
					+ oa::sdk::fromStdString(transition.error_));
			break;
		}
		lunarHeadlessDigestState(traceDigest, environment.state());
		lunarHeadlessDigestReward(traceDigest, transition.rewardTerms_);
		traceDigest.addDouble(transition.reward_);
		traceDigest.addBool(transition.terminated_);
		traceDigest.addBool(transition.truncated_);
		traceDigest.addU32(static_cast<oa::U32>(transition.endReason_));
		for (const oa::F32 observation : transition.observation_) {
			traceDigest.addFloat(observation);
		}
		if ((step + 1U) % OA_RENDER_STRIDE == 0U
			or transition.terminated_ or transition.truncated_) {
			runStatus = lunarHeadlessRenderFrame(
				*session, environment.state(), camera, inOutputDirectory,
				static_cast<oa::U32>(records.size()),
				imageSequenceDigest, records);
		}
	}

	if (runStatus.isOk()
		and (not environment.state().terminated_
			or environment.state().truncated_
			or environment.state().endReason_ != oa::LunarEndReason::SafeLanding)) {
		runStatus = oa::Status::error(
			oa::StatusCode::DataLoss,
			"lunar scripted controller did not reach a safe landing");
	}
	if (runStatus.isOk() and OA_EXPECTED_TRACE_DIGEST != 0U
		and traceDigest.value() != OA_EXPECTED_TRACE_DIGEST) {
		char digestError[192];
		const int digestErrorSize = std::snprintf(
			digestError, sizeof(digestError),
			"lunar frozen trace digest mismatch: expected=%016llx actual=%016llx",
			static_cast<unsigned long long>(OA_EXPECTED_TRACE_DIGEST),
			static_cast<unsigned long long>(traceDigest.value()));
		runStatus = oa::Status::error(
			oa::StatusCode::DataLoss,
			digestErrorSize > 0
				? oa::String(digestError)
				: oa::String("lunar frozen trace digest mismatch"));
	}

	const oa::String manifestJson = runStatus.isOk()
		? lunarHeadlessBuildManifestJson(
			landerConfig, episodeManifest, environment.state(), *engine,
			OA_WIDTH, OA_HEIGHT, traceDigest.value(),
			imageSequenceDigest.value(), records)
		: oa::String{};
	const oa::Status sessionCloseStatus = session->close();
	const oa::Status engineCloseStatus = engine->close();
	if (runStatus.isOk() and sessionCloseStatus.isError()) {
		runStatus = sessionCloseStatus;
	} else if (sessionCloseStatus.isError()) {
		OaLogError(oa::LogComponent::App,
			"Lunar headless renderer close failed: %s",
			sessionCloseStatus.toString().cStr());
	}
	if (runStatus.isOk() and engineCloseStatus.isError()) {
		runStatus = engineCloseStatus;
	} else if (engineCloseStatus.isError()) {
		OaLogError(oa::LogComponent::App,
			"Lunar headless engine close failed: %s",
			engineCloseStatus.toString().cStr());
	}
	if (runStatus.isError()) return runStatus;

	const oa::Path manifestPath = inOutputDirectory / "manifest.json";
	OA_RETURN_IF_ERROR(oa::Filesystem::writeText(manifestPath, manifestJson));
	LunarHeadlessSummary summary;
	summary.frameCount_ = static_cast<oa::U32>(records.size());
	summary.episodeSteps_ = environment.state().episodeStep_;
	summary.endReason_ = environment.state().endReason_;
	summary.episodeReturnQ1e9_ = static_cast<oa::I64>(std::llround(
		environment.state().episodeReturn_ * 1.0e9));
	summary.traceDigest_ = traceDigest.value();
	summary.imageSequenceDigest_ = imageSequenceDigest.value();
	summary.manifestPath_ = manifestPath;
	return summary;
}

int main(int argc, char** argv) {
	if (argc == 2
		and (oa::StringView(argv[1]) == "--help"
			or oa::StringView(argv[1]) == "-h")) {
		OA_CLI("usage: TutorialRlLunarLander3dHeadless [output-directory]");
		return 0;
	}
	if (argc > 2) {
		OaLogError(oa::LogComponent::App,
			"expected at most one output-directory argument");
		return 1;
	}
	const oa::Path outputDirectory = argc == 2
		? oa::Path(argv[1])
		: oa::Paths::var("tutorial") / "lunar_lander_3d_headless";
	auto result = lunarHeadlessRun(outputDirectory);
	if (result.isError()) {
		const oa::Status& status = result.getStatus();
		OaLogError(oa::LogComponent::App,
			"Lunar headless tutorial failed: %s", status.toString().cStr());
		if (status.getCode() == oa::StatusCode::DeviceNotFound
			or status.getCode() == oa::StatusCode::Unavailable) {
			return 125;
		}
		return 1;
	}
	OaLogInfo(oa::LogComponent::App,
		"Lunar headless tutorial wrote %u frames and %s "
		"(trace=%016llx, images=%016llx)",
		result->frameCount_, result->manifestPath_.cStr(),
		static_cast<unsigned long long>(result->traceDigest_),
		static_cast<unsigned long long>(result->imageSequenceDigest_));
	return 0;
}
