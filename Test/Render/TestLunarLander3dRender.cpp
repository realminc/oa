#include "LunarLander3dRender.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVk.h>

#include "Oa/Runtime/Engine/BorrowedServiceRetirement.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>

namespace {

struct ReadbackOracle {
	std::size_t NonClearPixels = 0U;
	std::size_t PadPixels = 0U;
	std::size_t LanderPixels = 0U;
	std::size_t CoveredDepthPixels = 0U;
	OaF32 MinimumDepth = 1.0F;
};

class RetiredTimelineGate {
public:
	OaEngine* Engine = nullptr;
	OaVkTimelineSemaphore Gate;

	[[nodiscard]] static OaStatus Complete(void* InPayload) {
		auto* payload = static_cast<RetiredTimelineGate*>(InPayload);
		if (payload == nullptr or payload->Engine == nullptr) {
			return OaStatus::InvalidArgument(
				"retired timeline gate payload is invalid");
		}
		payload->Gate.Destroy(payload->Engine->Device);
		return OaStatus::Ok();
	}

	static void Release(void* InPayload) {
		delete static_cast<RetiredTimelineGate*>(InPayload);
	}
};

[[nodiscard]] ReadbackOracle Inspect(
	const OaLunarLander3dReadback& InReadback) {
	ReadbackOracle result;
	const std::size_t pixelCount =
		static_cast<std::size_t>(InReadback.Width_) * InReadback.Height_;
	for (std::size_t pixel = 0U; pixel < pixelCount; ++pixel) {
		const OaU8 red = InReadback.ColorRgba8_[pixel * 4U];
		const OaU8 green = InReadback.ColorRgba8_[pixel * 4U + 1U];
		const OaU8 blue = InReadback.ColorRgba8_[pixel * 4U + 2U];
		const bool clear = red <= 8U and green <= 12U and blue <= 18U;
		result.NonClearPixels += clear ? 0U : 1U;
		result.PadPixels +=
			(red > 70U and green > 28U and red > blue * 2U) ? 1U : 0U;
		result.LanderPixels +=
			(blue > 55U and blue > red * 3U / 2U
				and green > red) ? 1U : 0U;
		const OaF32 depth = InReadback.Depth32_[pixel];
		if (std::isfinite(depth) and depth < 0.9999F) {
			++result.CoveredDepthPixels;
			result.MinimumDepth = std::min(result.MinimumDepth, depth);
		}
	}
	return result;
}

void ExpectSceneOracle(const OaLunarLander3dReadback& InReadback) {
	ASSERT_EQ(
		InReadback.ColorRgba8_.size(),
		static_cast<std::size_t>(InReadback.Width_) * InReadback.Height_ * 4U);
	ASSERT_EQ(
		InReadback.Depth32_.size(),
		static_cast<std::size_t>(InReadback.Width_) * InReadback.Height_);
	for (OaF32 depth : InReadback.Depth32_) {
		EXPECT_TRUE(std::isfinite(depth));
		EXPECT_GE(depth, 0.0F);
		EXPECT_LE(depth, 1.0F);
	}
	const ReadbackOracle oracle = Inspect(InReadback);
	EXPECT_GT(oracle.NonClearPixels, 250U);
	EXPECT_GT(oracle.PadPixels, 20U);
	EXPECT_GT(oracle.LanderPixels, 5U);
	EXPECT_GT(oracle.CoveredDepthPixels, 250U);
	// The default camera uses a 0.1/100 perspective range and views the scene
	// from roughly 25 world units away, so valid Vulkan depth is intentionally
	// concentrated near 1.0.  Require a real depth write, not an arbitrary
	// linear-depth cutoff that this projection cannot satisfy.
	EXPECT_LT(oracle.MinimumDepth, 0.9999F);
}

[[nodiscard]] OaResult<OaLunarLander3dReadback> RenderAndRead(
	OaLunarLander3dRenderSession& InSession,
	const OaLunarLander3dState& InState,
	const OaCameraState& InCamera,
	OaLunarLander3dRenderFrame* OutFrame = nullptr) {
	const OaStatus begin = InSession.BeginFrame(InState, InCamera);
	if (not begin.IsOk()) return begin;
	auto frame = InSession.SubmitFrame();
	if (not frame.IsOk()) {
		(void)InSession.CancelFrame();
		return frame.GetStatus();
	}
	if (OutFrame != nullptr) *OutFrame = *frame;
	return InSession.ConsumeReadback(*frame);
}

} // namespace

TEST(LunarLander3dRender, HeadlessReadbackAndSlotLifecycle) {
	OaEngineConfig engineConfig;
	engineConfig.PresentationMode = OaPresentationMode::Headless;
	engineConfig.RegisterAsGlobal = false;
	engineConfig.PreloadEmbeddedPipelines = false;
	engineConfig.EnablePipelineCache = false;
	engineConfig.AppName = "OaLunarLander3dRenderTest";
	auto engineResult = OaEngine::Create(engineConfig);
	if (not engineResult.IsOk()
		and (engineResult.GetStatus().GetCode() == OaStatusCode::DeviceNotFound
			or engineResult.GetStatus().GetCode() == OaStatusCode::Unavailable)) {
		GTEST_SKIP() << engineResult.GetStatus().ToString().c_str();
	}
	ASSERT_TRUE(engineResult.IsOk())
		<< engineResult.GetStatus().ToString().c_str();
	OaUniquePtr<OaEngine> engine = OaStdMove(*engineResult);

	OaLunarLander3dConfig landerConfig;
	const OaLunarEpisodeManifest manifest = OaLunarEpisodeManifest::Derive(
		0x4f415f4c554e4152ULL, 0U, 0U,
		landerConfig.ContractFingerprint());
	const OaLunarTerrain terrain = OaLunarTerrain::CreateSeeded(
		landerConfig.Terrain_, manifest);
	ASSERT_TRUE(terrain.IsValid()) << terrain.Error();
	OaLunarLander3dConfig oversizedLanderConfig = landerConfig;
	oversizedLanderConfig.BodySupports_[0].Radius_ =
		static_cast<double>(std::numeric_limits<OaF32>::max()) * 2.0;
	const OaLunarTerrain oversizedTerrain = OaLunarTerrain::CreateFlat(
		oversizedLanderConfig.Terrain_);
	ASSERT_TRUE(oversizedTerrain.IsValid()) << oversizedTerrain.Error();
	OaLunarLander3dRenderConfig oversizedRenderConfig;
	oversizedRenderConfig.Width_ = 160U;
	oversizedRenderConfig.Height_ = 120U;
	oversizedRenderConfig.TargetSlotCount_ = 1U;
	auto oversizedSession = OaLunarLander3dRenderSession::Create(
		*engine, oversizedLanderConfig, oversizedTerrain,
		oversizedRenderConfig);
	ASSERT_TRUE(oversizedSession.IsError());
	EXPECT_EQ(
		oversizedSession.GetStatus().GetCode(), OaStatusCode::OutOfRange);

	OaLunarLander3dRenderConfig renderConfig;
	renderConfig.Width_ = 160U;
	renderConfig.Height_ = 120U;
	renderConfig.TargetSlotCount_ = 1U;
	auto sessionResult = OaLunarLander3dRenderSession::Create(
		*engine, landerConfig, terrain, renderConfig);
	if (not sessionResult.IsOk()
		and sessionResult.GetStatus().GetCode() == OaStatusCode::Unavailable) {
		const OaStatus engineClose = engine->Close();
		EXPECT_TRUE(engineClose.IsOk()) << engineClose.ToString().c_str();
		GTEST_SKIP() << sessionResult.GetStatus().ToString().c_str();
	}
	ASSERT_TRUE(sessionResult.IsOk())
		<< sessionResult.GetStatus().ToString().c_str();
	OaUniquePtr<OaLunarLander3dRenderSession> session =
		OaStdMove(*sessionResult);

	OaLunarLander3dState state;
	state.Position_ = {0.0, 4.0, 0.0};
	state.Orientation_ = OaLunarQuat::Identity();
	const OaCameraState camera =
		OaLunarLander3dRenderSession::DefaultCamera(160U, 120U);
	OaLunarLander3dState oversizedState = state;
	oversizedState.Position_.ComponentX_ =
		static_cast<double>(std::numeric_limits<OaF32>::max()) * 2.0;
	EXPECT_EQ(
		session->BeginFrame(oversizedState, camera).GetCode(),
		OaStatusCode::OutOfRange);

	// Cancellation never submits and the same bounded slot is immediately reusable.
	ASSERT_TRUE(session->BeginFrame(state, camera).IsOk());
	EXPECT_TRUE(session->CancelFrame().IsOk());
	EXPECT_EQ(
		session->SubmitFrame().GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);

	auto firstReadback = RenderAndRead(*session, state, camera);
	ASSERT_TRUE(firstReadback.IsOk())
		<< firstReadback.GetStatus().ToString().c_str();
	ExpectSceneOracle(*firstReadback);
	auto secondReadback = RenderAndRead(*session, state, camera);
	ASSERT_TRUE(secondReadback.IsOk())
		<< secondReadback.GetStatus().ToString().c_str();
	ExpectSceneOracle(*secondReadback);
	EXPECT_EQ(firstReadback->ColorRgba8_, secondReadback->ColorRgba8_);
	ASSERT_EQ(firstReadback->Depth32_.size(), secondReadback->Depth32_.size());
	EXPECT_EQ(
		std::memcmp(
			firstReadback->Depth32_.data(),
			secondReadback->Depth32_.data(),
			firstReadback->Depth32_.size() * sizeof(OaF32)),
		0);

	// A submitted old-generation slot makes resize Busy without mutating it.
	ASSERT_TRUE(session->BeginFrame(state, camera).IsOk());
	auto liveFrame = session->SubmitFrame();
	ASSERT_TRUE(liveFrame.IsOk())
		<< liveFrame.GetStatus().ToString().c_str();
	auto forgedExtent = *liveFrame;
	forgedExtent.Width_ = std::numeric_limits<OaU32>::max();
	EXPECT_EQ(
		session->ConsumeReadback(forgedExtent).GetStatus().GetCode(),
		OaStatusCode::InvalidArgument);
	const OaStatus busyResize = session->Resize(192U, 128U);
	EXPECT_EQ(busyResize.GetCode(), OaStatusCode::FailedPrecondition);
	EXPECT_EQ(liveFrame->Width_, 160U);
	EXPECT_EQ(liveFrame->Height_, 120U);
	auto liveReadback = session->ConsumeReadback(*liveFrame);
	ASSERT_TRUE(liveReadback.IsOk())
		<< liveReadback.GetStatus().ToString().c_str();
	EXPECT_TRUE(session->Resize(192U, 128U).IsOk());
	EXPECT_EQ(
		session->ConsumeReadback(*liveFrame).GetStatus().GetCode(),
		OaStatusCode::InvalidArgument);
	const OaCameraState resizedCamera =
		OaLunarLander3dRenderSession::DefaultCamera(192U, 128U);
	auto resizedReadback = RenderAndRead(*session, state, resizedCamera);
	ASSERT_TRUE(resizedReadback.IsOk())
		<< resizedReadback.GetStatus().ToString().c_str();
	EXPECT_EQ(resizedReadback->Width_, 192U);
	EXPECT_EQ(resizedReadback->Height_, 128U);
	ExpectSceneOracle(*resizedReadback);

	// A sampled frame exposes only non-owning handles. Registering the exact
	// consumer completion retires the slot until both producer and consumer are
	// complete; producer completion alone must not permit early reuse.
	auto consumerGateResult = OaVkTimelineSemaphore::Create(engine->Device, 0U);
	ASSERT_TRUE(consumerGateResult.IsOk())
		<< consumerGateResult.GetStatus().ToString().c_str();
	OaVkTimelineSemaphore consumerGate = OaStdMove(*consumerGateResult);
	const OaEvent consumerCompletion(
		engine->Device, consumerGate, 1U,
		engine->Device.Queues.GraphicsQueueFamily);
	ASSERT_TRUE(session->BeginFrame(state, resizedCamera).IsOk());
	auto sampledFrame = session->SubmitFrame();
	ASSERT_TRUE(sampledFrame.IsOk())
		<< sampledFrame.GetStatus().ToString().c_str();
	EXPECT_NE(sampledFrame->Image_, VK_NULL_HANDLE);
	EXPECT_NE(sampledFrame->ImageView_, VK_NULL_HANDLE);
	EXPECT_EQ(
		sampledFrame->ImageLayout_,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	EXPECT_EQ(
		session->MarkConsumed(*sampledFrame, sampledFrame->Producer_).GetCode(),
		OaStatusCode::InvalidArgument);
	ASSERT_TRUE(sampledFrame->Producer_.Wait().IsOk());
	ASSERT_TRUE(session->MarkConsumed(
		*sampledFrame, consumerCompletion).IsOk());
	EXPECT_EQ(
		session->MarkConsumed(*sampledFrame, consumerCompletion).GetCode(),
		OaStatusCode::InvalidArgument);
	EXPECT_TRUE(session->Collect().IsOk());
	EXPECT_EQ(
		session->BeginFrame(state, resizedCamera).GetCode(),
		OaStatusCode::ResourceExhausted);
	VkSemaphoreSignalInfo consumerSignal{};
	consumerSignal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	consumerSignal.semaphore =
		static_cast<VkSemaphore>(consumerGate.Semaphore);
	consumerSignal.value = 1U;
	ASSERT_EQ(
		vkSignalSemaphore(
			static_cast<VkDevice>(engine->Device.Device), &consumerSignal),
		VK_SUCCESS);
	ASSERT_TRUE(consumerCompletion.IsComplete());
	EXPECT_TRUE(session->Collect().IsOk());
	ASSERT_TRUE(session->BeginFrame(state, resizedCamera).IsOk());
	EXPECT_TRUE(session->CancelFrame().IsOk());
	consumerGate.Destroy(engine->Device);

	// A fabricated foreign-family dependency is rejected before submission and
	// leaves the exact recording cancellable.
	auto gateResult = OaVkTimelineSemaphore::Create(engine->Device, 0U);
	ASSERT_TRUE(gateResult.IsOk())
		<< gateResult.GetStatus().ToString().c_str();
	OaVkTimelineSemaphore gate = OaStdMove(*gateResult);
	const OaEvent crossFamilyDependency(
		engine->Device, gate, 1U,
		engine->Device.Queues.GraphicsQueueFamily + 1U);
	ASSERT_TRUE(session->BeginFrame(state, resizedCamera).IsOk());
	const OaEvent crossFamilyDependencies[1] = {crossFamilyDependency};
	auto rejectedSubmission = session->SubmitFrame(
		OaSpan<const OaEvent>(crossFamilyDependencies, 1U));
	EXPECT_EQ(
		rejectedSubmission.GetStatus().GetCode(),
		OaStatusCode::FailedPrecondition);
	EXPECT_TRUE(session->CancelFrame().IsOk());

	// Abandon is non-waiting. The sole target remains unavailable until its
	// exact producer (gated here) completes, then Collect makes it reusable.
	const OaEvent graphicsGate(
		engine->Device, gate, 1U,
		engine->Device.Queues.GraphicsQueueFamily);
	ASSERT_TRUE(session->BeginFrame(state, resizedCamera).IsOk());
	const OaEvent graphicsDependencies[1] = {graphicsGate};
	auto abandonedFrame = session->SubmitFrame(
		OaSpan<const OaEvent>(graphicsDependencies, 1U));
	ASSERT_TRUE(abandonedFrame.IsOk())
		<< abandonedFrame.GetStatus().ToString().c_str();
	EXPECT_TRUE(session->AbandonFrame(*abandonedFrame).IsOk());
	EXPECT_EQ(
		session->BeginFrame(state, resizedCamera).GetCode(),
		OaStatusCode::ResourceExhausted);
	VkSemaphoreSignalInfo signalInfo{};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	signalInfo.semaphore = static_cast<VkSemaphore>(gate.Semaphore);
	signalInfo.value = 1U;
	ASSERT_EQ(
		vkSignalSemaphore(
			static_cast<VkDevice>(engine->Device.Device), &signalInfo),
		VK_SUCCESS);
	for (OaU32 poll = 0U;
		poll < 100000U and not abandonedFrame->Producer_.IsComplete();
		++poll) {
		std::this_thread::yield();
	}
	ASSERT_TRUE(abandonedFrame->Producer_.IsComplete());
	EXPECT_TRUE(session->Collect().IsOk());
	ASSERT_TRUE(session->BeginFrame(state, resizedCamera).IsOk());
	EXPECT_EQ(
		session->AbandonFrame(*abandonedFrame).GetCode(),
		OaStatusCode::InvalidArgument);
	EXPECT_TRUE(session->CancelFrame().IsOk());
	gate.Destroy(engine->Device);

	const OaStatus sessionClose = session->Close();
	EXPECT_TRUE(sessionClose.IsOk()) << sessionClose.ToString().c_str();
	const OaStatus engineClose = engine->Close();
	EXPECT_TRUE(engineClose.IsOk()) << engineClose.ToString().c_str();
}

TEST(LunarLander3dRender, AbandonedSessionRetiresThroughEngineClose) {
	OaEngineConfig engineConfig;
	engineConfig.PresentationMode = OaPresentationMode::Headless;
	engineConfig.RegisterAsGlobal = false;
	engineConfig.PreloadEmbeddedPipelines = false;
	engineConfig.EnablePipelineCache = false;
	engineConfig.AppName = "OaLunarLander3dRetirementTest";
	auto engineResult = OaEngine::Create(engineConfig);
	if (not engineResult.IsOk()
		and (engineResult.GetStatus().GetCode() == OaStatusCode::DeviceNotFound
			or engineResult.GetStatus().GetCode() == OaStatusCode::Unavailable)) {
		GTEST_SKIP() << engineResult.GetStatus().ToString().c_str();
	}
	ASSERT_TRUE(engineResult.IsOk())
		<< engineResult.GetStatus().ToString().c_str();
	OaUniquePtr<OaEngine> engine = OaStdMove(*engineResult);

	OaLunarLander3dConfig landerConfig;
	const OaLunarEpisodeManifest manifest = OaLunarEpisodeManifest::Derive(
		0x5245544952453344ULL, 0U, 0U,
		landerConfig.ContractFingerprint());
	const OaLunarTerrain terrain = OaLunarTerrain::CreateSeeded(
		landerConfig.Terrain_, manifest);
	ASSERT_TRUE(terrain.IsValid()) << terrain.Error();
	OaLunarLander3dRenderConfig renderConfig;
	renderConfig.Width_ = 96U;
	renderConfig.Height_ = 72U;
	renderConfig.TargetSlotCount_ = 1U;
	auto sessionResult = OaLunarLander3dRenderSession::Create(
		*engine, landerConfig, terrain, renderConfig);
	if (not sessionResult.IsOk()
		and sessionResult.GetStatus().GetCode() == OaStatusCode::Unavailable) {
		const OaStatus engineClose = engine->Close();
		EXPECT_TRUE(engineClose.IsOk()) << engineClose.ToString().c_str();
		GTEST_SKIP() << sessionResult.GetStatus().ToString().c_str();
	}
	ASSERT_TRUE(sessionResult.IsOk())
		<< sessionResult.GetStatus().ToString().c_str();
	OaUniquePtr<OaLunarLander3dRenderSession> session =
		OaStdMove(*sessionResult);

	auto gateResult = OaVkTimelineSemaphore::Create(engine->Device, 0U);
	ASSERT_TRUE(gateResult.IsOk())
		<< gateResult.GetStatus().ToString().c_str();
	OaVkTimelineSemaphore gate = OaStdMove(*gateResult);
	const OaEvent gateEvent(
		engine->Device, gate, 1U,
		engine->Device.Queues.GraphicsQueueFamily);
	OaLunarLander3dState state;
	state.Position_ = {0.0, 4.0, 0.0};
	const OaCameraState camera =
		OaLunarLander3dRenderSession::DefaultCamera(96U, 72U);
	ASSERT_TRUE(session->BeginFrame(state, camera).IsOk());
	const OaEvent dependencies[1] = {gateEvent};
	auto frame = session->SubmitFrame(
		OaSpan<const OaEvent>(dependencies, 1U));
	ASSERT_TRUE(frame.IsOk()) << frame.GetStatus().ToString().c_str();
	EXPECT_FALSE(frame->Producer_.IsComplete());

	// No Close and no host wait: destruction transfers the exact event plus all
	// referenced targets/pipeline/buffers into OaEngine retirement.
	session.Reset();
	VkSemaphoreSignalInfo signalInfo{};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	signalInfo.semaphore = static_cast<VkSemaphore>(gate.Semaphore);
	signalInfo.value = 1U;
	ASSERT_EQ(
		vkSignalSemaphore(
			static_cast<VkDevice>(engine->Device.Device), &signalInfo),
		VK_SUCCESS);
	auto retiredGate = OaMakeUniquePtr<RetiredTimelineGate>();
	retiredGate->Engine = engine.Get();
	retiredGate->Gate.Semaphore = gate.Semaphore;
	gate.Semaphore = nullptr;
	OaBorrowedServiceRetirement::Retire(
		*engine,
		retiredGate.Release(),
		&RetiredTimelineGate::Complete,
		&RetiredTimelineGate::Release);
	const OaStatus engineClose = engine->Close();
	EXPECT_TRUE(engineClose.IsOk()) << engineClose.ToString().c_str();
}
