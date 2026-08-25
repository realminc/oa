#include "lunarLander3dRender.h"

#include <oa/render/renderer.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/deviceAccess.h>
#include <oa/runtime/oaVk.h>

#include "oa/runtime/engine/borrowedServiceRetirement.h"
#include "oa/runtime/textureAccess.h"

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
	std::size_t nonClearPixels = 0U;
	std::size_t padPixels = 0U;
	std::size_t landerPixels = 0U;
	std::size_t coveredDepthPixels = 0U;
	oa::F32 minimumDepth = 1.0F;
};

class RetiredTimelineGate {
public:
	oa::Engine* engine = nullptr;
	oavk::TimelineSemaphore gate;

	[[nodiscard]] static oa::Status complete(void* inPayload) {
		auto* payload = static_cast<RetiredTimelineGate*>(inPayload);
		if (payload == nullptr or payload->engine == nullptr) {
			return oa::Status::invalidArgument(
				"retired timeline gate payload is invalid");
		}
		payload->gate.destroy(
			oa::EngineDeviceAccess::get(*payload->engine));
		return oa::Status::ok();
	}

	static void release(void* inPayload) {
		delete static_cast<RetiredTimelineGate*>(inPayload);
	}
};

[[nodiscard]] ReadbackOracle inspect(
	const oa::RenderReadback& inReadback) {
	ReadbackOracle result;
	const std::size_t pixelCount =
		static_cast<std::size_t>(inReadback.width_) * inReadback.height_;
	for (std::size_t pixel = 0U; pixel < pixelCount; ++pixel) {
		const oa::U8 red = inReadback.colorRgba8_[pixel * 4U];
		const oa::U8 green = inReadback.colorRgba8_[pixel * 4U + 1U];
		const oa::U8 blue = inReadback.colorRgba8_[pixel * 4U + 2U];
		const bool clear = red <= 8U and green <= 12U and blue <= 18U;
		result.nonClearPixels += clear ? 0U : 1U;
		result.padPixels +=
			(red > 70U and green > 28U and red > blue * 2U) ? 1U : 0U;
		result.landerPixels +=
			(blue > 55U and blue > red * 3U / 2U
				and green > red) ? 1U : 0U;
		const oa::F32 depth = inReadback.depth32_[pixel];
		if (std::isfinite(depth) and depth < 0.9999F) {
			++result.coveredDepthPixels;
			result.minimumDepth = std::min(result.minimumDepth, depth);
		}
	}
	return result;
}

void expectSceneOracle(const oa::RenderReadback& inReadback) {
	ASSERT_EQ(
		inReadback.colorRgba8_.size(),
		static_cast<std::size_t>(inReadback.width_) * inReadback.height_ * 4U);
	ASSERT_EQ(
		inReadback.depth32_.size(),
		static_cast<std::size_t>(inReadback.width_) * inReadback.height_);
	for (oa::F32 depth : inReadback.depth32_) {
		EXPECT_TRUE(std::isfinite(depth));
		EXPECT_GE(depth, 0.0F);
		EXPECT_LE(depth, 1.0F);
	}
	const ReadbackOracle oracle = inspect(inReadback);
	EXPECT_GT(oracle.nonClearPixels, 250U);
	EXPECT_GT(oracle.padPixels, 20U);
	EXPECT_GT(oracle.landerPixels, 5U);
	EXPECT_GT(oracle.coveredDepthPixels, 250U);
	// The default camera uses a 0.1/100 perspective range and views the scene
	// from roughly 25 world units away, so valid vulkan depth is intentionally
	// concentrated near 1.0.  Require a real depth write, not an arbitrary
	// linear-depth cutoff that this projection cannot satisfy.
	EXPECT_LT(oracle.minimumDepth, 0.9999F);
}

[[nodiscard]] oa::Result<oa::RenderReadback> renderAndRead(
	LunarLander3dRenderSession& inSession,
	const oa::LunarLander3dState& inState,
	const oa::CameraState& inCamera,
	oa::RenderFrame* outFrame = nullptr) {
	const oa::Status begin = inSession.beginFrame(inState, inCamera);
	if (not begin.isOk()) return begin;
	auto frame = inSession.submitFrame();
	if (not frame.isOk()) {
		(void)inSession.cancelFrame();
		return frame.getStatus();
	}
	if (outFrame != nullptr) *outFrame = *frame;
	return inSession.consumeReadback(*frame);
}

} // namespace

TEST(LunarLander3dRender, HeadlessReadbackAndSlotLifecycle) {
	oa::EngineConfig engineConfig;
	engineConfig.presentationMode = oa::PresentationMode::Headless;
	engineConfig.selectForThread = false;
	engineConfig.preloadEmbeddedPipelines = false;
	engineConfig.enablePipelineCache = false;
	engineConfig.appName = "OaLunarLander3dRenderTest";
	auto engineResult = oa::Engine::create(engineConfig);
	if (not engineResult.isOk()
		and (engineResult.getStatus().getCode() == oa::StatusCode::DeviceNotFound
			or engineResult.getStatus().getCode() == oa::StatusCode::Unavailable)) {
		GTEST_SKIP() << engineResult.getStatus().toString().cStr();
	}
	ASSERT_TRUE(engineResult.isOk())
		<< engineResult.getStatus().toString().cStr();
	oa::UniquePtr<oa::Engine> engine = oa::move(*engineResult);

	oa::RendererConfig rayTracingConfig;
	rayTracingConfig.mode_ = oa::RendererMode::RayTracing;
	auto rayTracingRenderer = oa::Renderer::create(*engine, rayTracingConfig);
	ASSERT_TRUE(rayTracingRenderer.isError());
	EXPECT_EQ(
		rayTracingRenderer.getStatus().getCode(),
		oa::StatusCode::Unavailable);
	oa::RendererConfig invalidModeConfig;
	invalidModeConfig.mode_ = static_cast<oa::RendererMode>(255U);
	auto invalidModeRenderer = oa::Renderer::create(*engine, invalidModeConfig);
	ASSERT_TRUE(invalidModeRenderer.isError());
	EXPECT_EQ(
		invalidModeRenderer.getStatus().getCode(),
		oa::StatusCode::InvalidArgument);

	oa::LunarLander3dConfig landerConfig;
	const oa::LunarEpisodeManifest manifest = oa::LunarEpisodeManifest::derive(
		0x4f415f4c554e4152ULL, 0U, 0U,
		landerConfig.contractFingerprint());
	const oa::LunarTerrain terrain = oa::LunarTerrain::createSeeded(
		landerConfig.terrain_, manifest);
	ASSERT_TRUE(terrain.isValid()) << terrain.error();
	oa::LunarLander3dConfig oversizedLanderConfig = landerConfig;
	oversizedLanderConfig.bodySupports_[0].radius_ =
		static_cast<double>(std::numeric_limits<oa::F32>::max()) * 2.0;
	const oa::LunarTerrain oversizedTerrain = oa::LunarTerrain::createFlat(
		oversizedLanderConfig.terrain_);
	ASSERT_TRUE(oversizedTerrain.isValid()) << oversizedTerrain.error();
	LunarLander3dRenderConfig oversizedRenderConfig;
	oversizedRenderConfig.width_ = 160U;
	oversizedRenderConfig.height_ = 120U;
	oversizedRenderConfig.targetSlotCount_ = 1U;
	auto oversizedSession = LunarLander3dRenderSession::create(
		*engine, oversizedLanderConfig, oversizedTerrain,
		oversizedRenderConfig);
	ASSERT_TRUE(oversizedSession.isError());
	EXPECT_EQ(
		oversizedSession.getStatus().getCode(), oa::StatusCode::OutOfRange);

	LunarLander3dRenderConfig renderConfig;
	renderConfig.width_ = 160U;
	renderConfig.height_ = 120U;
	renderConfig.targetSlotCount_ = 1U;
	auto sessionResult = LunarLander3dRenderSession::create(
		*engine, landerConfig, terrain, renderConfig);
	if (not sessionResult.isOk()
		and sessionResult.getStatus().getCode() == oa::StatusCode::Unavailable) {
		const oa::Status engineClose = engine->close();
		EXPECT_TRUE(engineClose.isOk()) << engineClose.toString().cStr();
		GTEST_SKIP() << sessionResult.getStatus().toString().cStr();
	}
	ASSERT_TRUE(sessionResult.isOk())
		<< sessionResult.getStatus().toString().cStr();
	oa::UniquePtr<LunarLander3dRenderSession> session =
		oa::move(*sessionResult);

	oa::LunarLander3dState state;
	state.position_ = {0.0, 4.0, 0.0};
	state.orientation_ = oa::vlm::DQuat::identity();
	const oa::CameraState camera =
		LunarLander3dRenderSession::defaultCamera(160U, 120U);
	oa::LunarLander3dState oversizedState = state;
	oversizedState.position_.x =
		static_cast<double>(std::numeric_limits<oa::F32>::max()) * 2.0;
	EXPECT_EQ(
		session->beginFrame(oversizedState, camera).getCode(),
		oa::StatusCode::OutOfRange);

	// Cancellation never submits and the same bounded slot is immediately reusable.
	ASSERT_TRUE(session->beginFrame(state, camera).isOk());
	EXPECT_TRUE(session->cancelFrame().isOk());
	EXPECT_EQ(
		session->submitFrame().getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);

	auto firstReadback = renderAndRead(*session, state, camera);
	ASSERT_TRUE(firstReadback.isOk())
		<< firstReadback.getStatus().toString().cStr();
	expectSceneOracle(*firstReadback);
	auto secondReadback = renderAndRead(*session, state, camera);
	ASSERT_TRUE(secondReadback.isOk())
		<< secondReadback.getStatus().toString().cStr();
	expectSceneOracle(*secondReadback);
	EXPECT_EQ(firstReadback->colorRgba8_, secondReadback->colorRgba8_);
	ASSERT_EQ(firstReadback->depth32_.size(), secondReadback->depth32_.size());
	EXPECT_EQ(
		std::memcmp(
			firstReadback->depth32_.data(),
			secondReadback->depth32_.data(),
			firstReadback->depth32_.size() * sizeof(oa::F32)),
		0);

	// A submitted old-generation slot makes resize Busy without mutating it.
	ASSERT_TRUE(session->beginFrame(state, camera).isOk());
	auto liveFrame = session->submitFrame();
	ASSERT_TRUE(liveFrame.isOk())
		<< liveFrame.getStatus().toString().cStr();
	const oa::RenderFrame invalidFrame;
	EXPECT_EQ(
		session->consumeReadback(invalidFrame).getStatus().getCode(),
		oa::StatusCode::InvalidArgument);
	const oa::Status busyResize = session->resize(192U, 128U);
	EXPECT_EQ(busyResize.getCode(), oa::StatusCode::FailedPrecondition);
	EXPECT_EQ(liveFrame->width(), 160U);
	EXPECT_EQ(liveFrame->height(), 120U);
	auto liveReadback = session->consumeReadback(*liveFrame);
	ASSERT_TRUE(liveReadback.isOk())
		<< liveReadback.getStatus().toString().cStr();
	EXPECT_TRUE(session->resize(192U, 128U).isOk());
	EXPECT_EQ(
		session->consumeReadback(*liveFrame).getStatus().getCode(),
		oa::StatusCode::InvalidArgument);
	const oa::CameraState resizedCamera =
		LunarLander3dRenderSession::defaultCamera(192U, 128U);
	auto resizedReadback = renderAndRead(*session, state, resizedCamera);
	ASSERT_TRUE(resizedReadback.isOk())
		<< resizedReadback.getStatus().toString().cStr();
	EXPECT_EQ(resizedReadback->width_, 192U);
	EXPECT_EQ(resizedReadback->height_, 128U);
	expectSceneOracle(*resizedReadback);

	// A sampled frame exposes only non-owning handles. Registering the exact
	// consumer completion retires the slot until both producer and consumer are
	// complete; producer completion alone must not permit early reuse.
	auto consumerGateResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*engine), 0U);
	ASSERT_TRUE(consumerGateResult.isOk())
		<< consumerGateResult.getStatus().toString().cStr();
	oavk::TimelineSemaphore consumerGate = oa::move(*consumerGateResult);
	const oa::Event consumerCompletion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine), consumerGate, 1U,
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily);
	ASSERT_TRUE(session->beginFrame(state, resizedCamera).isOk());
	auto sampledFrame = session->submitFrame();
	ASSERT_TRUE(sampledFrame.isOk())
		<< sampledFrame.getStatus().toString().cStr();
	EXPECT_NE(oa::TextureAccess::image(sampledFrame->color()), VK_NULL_HANDLE);
	EXPECT_NE(oa::TextureAccess::view(sampledFrame->color()), VK_NULL_HANDLE);
	EXPECT_EQ(
		oa::TextureAccess::layout(sampledFrame->color()),
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	EXPECT_EQ(
		session->markConsumed(*sampledFrame, sampledFrame->producer()).getCode(),
		oa::StatusCode::InvalidArgument);
	ASSERT_TRUE(sampledFrame->producer().wait().isOk());
	ASSERT_TRUE(session->markConsumed(
		*sampledFrame, consumerCompletion).isOk());
	EXPECT_EQ(
		session->markConsumed(*sampledFrame, consumerCompletion).getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_TRUE(session->collect().isOk());
	EXPECT_EQ(
		session->beginFrame(state, resizedCamera).getCode(),
		oa::StatusCode::ResourceExhausted);
	VkSemaphoreSignalInfo consumerSignal{};
	consumerSignal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	consumerSignal.semaphore =
		static_cast<VkSemaphore>(consumerGate.semaphore);
	consumerSignal.value = 1U;
	ASSERT_EQ(
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device), &consumerSignal),
		VK_SUCCESS);
	ASSERT_TRUE(consumerCompletion.isComplete());
	EXPECT_TRUE(session->collect().isOk());
	ASSERT_TRUE(session->beginFrame(state, resizedCamera).isOk());
	EXPECT_TRUE(session->cancelFrame().isOk());
	consumerGate.destroy(oa::EngineDeviceAccess::get(*engine));

	// A fabricated foreign-family dependency is rejected before submission and
	// leaves the exact recording cancellable.
	auto gateResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*engine), 0U);
	ASSERT_TRUE(gateResult.isOk())
		<< gateResult.getStatus().toString().cStr();
	oavk::TimelineSemaphore gate = oa::move(*gateResult);
	const oa::Event crossFamilyDependency = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine), gate, 1U,
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily + 1U);
	ASSERT_TRUE(session->beginFrame(state, resizedCamera).isOk());
	const oa::Event crossFamilyDependencies[1] = {crossFamilyDependency};
	auto rejectedSubmission = session->submitFrame(
		oa::Span<const oa::Event>(crossFamilyDependencies, 1U));
	EXPECT_EQ(
		rejectedSubmission.getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);
	EXPECT_TRUE(session->cancelFrame().isOk());

	// abandon is non-waiting. The sole target remains unavailable until its
	// exact producer (gated here) completes, then Collect makes it reusable.
	const oa::Event graphicsGate = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine), gate, 1U,
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily);
	ASSERT_TRUE(session->beginFrame(state, resizedCamera).isOk());
	const oa::Event graphicsDependencies[1] = {graphicsGate};
	auto abandonedFrame = session->submitFrame(
		oa::Span<const oa::Event>(graphicsDependencies, 1U));
	ASSERT_TRUE(abandonedFrame.isOk())
		<< abandonedFrame.getStatus().toString().cStr();
	EXPECT_TRUE(session->abandonFrame(*abandonedFrame).isOk());
	EXPECT_EQ(
		session->beginFrame(state, resizedCamera).getCode(),
		oa::StatusCode::ResourceExhausted);
	VkSemaphoreSignalInfo signalInfo{};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	signalInfo.semaphore = static_cast<VkSemaphore>(gate.semaphore);
	signalInfo.value = 1U;
	ASSERT_EQ(
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device), &signalInfo),
		VK_SUCCESS);
	for (oa::U32 poll = 0U;
		poll < 100000U and not abandonedFrame->producer().isComplete();
		++poll) {
		std::this_thread::yield();
	}
	ASSERT_TRUE(abandonedFrame->producer().isComplete());
	EXPECT_TRUE(session->collect().isOk());
	ASSERT_TRUE(session->beginFrame(state, resizedCamera).isOk());
	EXPECT_EQ(
		session->abandonFrame(*abandonedFrame).getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_TRUE(session->cancelFrame().isOk());
	gate.destroy(oa::EngineDeviceAccess::get(*engine));

	const oa::Status sessionClose = session->close();
	EXPECT_TRUE(sessionClose.isOk()) << sessionClose.toString().cStr();

	LunarLander3dRenderConfig invalidSamples = renderConfig;
	invalidSamples.sampleCount_ = 3U;
	auto invalidSampleSession = LunarLander3dRenderSession::create(
		*engine, landerConfig, terrain, invalidSamples);
	ASSERT_TRUE(invalidSampleSession.isError());
	EXPECT_EQ(
		invalidSampleSession.getStatus().getCode(),
		oa::StatusCode::InvalidArgument);

	// Exercise the first supported multisample count. Devices with no common
	// color/depth MSAA remain a valid 1x capability pack; unsupported counts
	// must report Unavailable rather than silently changing the request.
	bool exercisedMultisampling = false;
	for (oa::U32 sampleCount : {4U, 2U, 8U, 16U, 32U, 64U}) {
		LunarLander3dRenderConfig multisampleConfig = renderConfig;
		multisampleConfig.width_ = 160U;
		multisampleConfig.height_ = 120U;
		multisampleConfig.sampleCount_ = sampleCount;
		auto multisampleResult = LunarLander3dRenderSession::create(
			*engine, landerConfig, terrain, multisampleConfig);
		if (multisampleResult.getStatus().getCode()
			== oa::StatusCode::Unavailable) {
			continue;
		}
		ASSERT_TRUE(multisampleResult.isOk())
			<< multisampleResult.getStatus().toString().cStr();
		auto multisampleSession = oa::move(*multisampleResult);
		auto multisampleReadback = renderAndRead(
			*multisampleSession, state, camera);
		ASSERT_TRUE(multisampleReadback.isOk())
			<< multisampleReadback.getStatus().toString().cStr();
		expectSceneOracle(*multisampleReadback);
		ASSERT_TRUE(multisampleSession->close().isOk());
		exercisedMultisampling = true;
		break;
	}
	RecordProperty(
		"multisample_path_exercised",
		exercisedMultisampling ? "true" : "unsupported");
	const oa::Status engineClose = engine->close();
	EXPECT_TRUE(engineClose.isOk()) << engineClose.toString().cStr();
}

TEST(LunarLander3dRender, AbandonedSessionRetiresThroughEngineClose) {
	oa::EngineConfig engineConfig;
	engineConfig.presentationMode = oa::PresentationMode::Headless;
	engineConfig.selectForThread = false;
	engineConfig.preloadEmbeddedPipelines = false;
	engineConfig.enablePipelineCache = false;
	engineConfig.appName = "OaLunarLander3dRetirementTest";
	auto engineResult = oa::Engine::create(engineConfig);
	if (not engineResult.isOk()
		and (engineResult.getStatus().getCode() == oa::StatusCode::DeviceNotFound
			or engineResult.getStatus().getCode() == oa::StatusCode::Unavailable)) {
		GTEST_SKIP() << engineResult.getStatus().toString().cStr();
	}
	ASSERT_TRUE(engineResult.isOk())
		<< engineResult.getStatus().toString().cStr();
	oa::UniquePtr<oa::Engine> engine = oa::move(*engineResult);

	oa::LunarLander3dConfig landerConfig;
	const oa::LunarEpisodeManifest manifest = oa::LunarEpisodeManifest::derive(
		0x5245544952453344ULL, 0U, 0U,
		landerConfig.contractFingerprint());
	const oa::LunarTerrain terrain = oa::LunarTerrain::createSeeded(
		landerConfig.terrain_, manifest);
	ASSERT_TRUE(terrain.isValid()) << terrain.error();
	LunarLander3dRenderConfig renderConfig;
	renderConfig.width_ = 96U;
	renderConfig.height_ = 72U;
	renderConfig.targetSlotCount_ = 1U;
	auto sessionResult = LunarLander3dRenderSession::create(
		*engine, landerConfig, terrain, renderConfig);
	if (not sessionResult.isOk()
		and sessionResult.getStatus().getCode() == oa::StatusCode::Unavailable) {
		const oa::Status engineClose = engine->close();
		EXPECT_TRUE(engineClose.isOk()) << engineClose.toString().cStr();
		GTEST_SKIP() << sessionResult.getStatus().toString().cStr();
	}
	ASSERT_TRUE(sessionResult.isOk())
		<< sessionResult.getStatus().toString().cStr();
	oa::UniquePtr<LunarLander3dRenderSession> session =
		oa::move(*sessionResult);

	auto gateResult = oavk::TimelineSemaphore::create(oa::EngineDeviceAccess::get(*engine), 0U);
	ASSERT_TRUE(gateResult.isOk())
		<< gateResult.getStatus().toString().cStr();
	oavk::TimelineSemaphore gate = oa::move(*gateResult);
	const oa::Event gateEvent = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine), gate, 1U,
		oa::EngineDeviceAccess::get(*engine).queues.graphicsQueueFamily);
	oa::LunarLander3dState state;
	state.position_ = {0.0, 4.0, 0.0};
	const oa::CameraState camera =
		LunarLander3dRenderSession::defaultCamera(96U, 72U);
	ASSERT_TRUE(session->beginFrame(state, camera).isOk());
	const oa::Event dependencies[1] = {gateEvent};
	auto frame = session->submitFrame(
		oa::Span<const oa::Event>(dependencies, 1U));
	ASSERT_TRUE(frame.isOk()) << frame.getStatus().toString().cStr();
	EXPECT_FALSE(frame->producer().isComplete());

	// No Close and no host wait: destruction transfers the exact event plus all
	// referenced targets/pipeline/buffers into oa::Engine retirement.
	session.reset();
	VkSemaphoreSignalInfo signalInfo{};
	signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	signalInfo.semaphore = static_cast<VkSemaphore>(gate.semaphore);
	signalInfo.value = 1U;
	ASSERT_EQ(
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device), &signalInfo),
		VK_SUCCESS);
	auto retiredGate = oa::makeUnique<RetiredTimelineGate>();
	retiredGate->engine = engine.get();
	retiredGate->gate.semaphore = gate.semaphore;
	gate.semaphore = nullptr;
	oa::BorrowedServiceRetirement::retire(
		*engine,
		retiredGate.release(),
		&RetiredTimelineGate::complete,
		&RetiredTimelineGate::release);
	const oa::Status engineClose = engine->close();
	EXPECT_TRUE(engineClose.isOk()) << engineClose.toString().cStr();
}
