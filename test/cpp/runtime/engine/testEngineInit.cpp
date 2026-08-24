#include "../../oaTest.h"
#include <oa/runtime/executableGraph.h>
#include <oa/runtime/externalMemory.h>
#include <oa/runtime/timer.h>
#include <oa/runtime/init.h>
#include <oa/runtime/engine.h>
#include <oa/runtime/presenter.h>
#include <oa/runtime/timestamp.h>
#include <oa/runtime/uploadRing.h>
#include <oa/runtime/storageDtype.h>
#include <oa/vision/videoPlayer.h>
#include <oa/vision/videoEncoder.h>
#include <cstdlib>
#include <cstdio>
#include <type_traits>

static_assert(!std::is_base_of_v<oa::Engine, oa::Presenter>);
static_assert(std::is_aggregate_v<oa::EngineConfig>,
	"oa::EngineConfig must remain compatible with designated initialization");
static_assert(sizeof(oa::Engine) == sizeof(oa::UniquePtr<oa::Engine>),
	"oa::Engine's installed layout must remain one opaque implementation owner");

template <typename T>
concept HasStatusDiscardingDestroy = requires(T& inSession) {
	inSession.destroy();
};

static_assert(not HasStatusDiscardingDestroy<oa::Presenter>);
static_assert(not HasStatusDiscardingDestroy<oa::UploadRing>);
static_assert(not HasStatusDiscardingDestroy<oa::ImportedDmaBufImage>);

template <typename T>
concept VideoConfigHasLegacyPath = requires(T& inConfig) {
	inConfig.path;
};

template <typename T>
concept VideoHasLegacyCreate = requires(
	oa::Engine& inEngine,
	const oa::VideoPlayerConfig& inConfig)
{
	T::create(inEngine, inConfig);
};

template <typename T>
concept VideoHasLegacyStepForward = requires(T& inVideo) {
	inVideo.stepForward();
};

template <typename T>
concept DecoderHasPublicPhysicalDecode = requires(
	T& inDecoder,
	const oa::Span<const oa::U8>& inBitstream,
	oa::VideoFrame& outFrame)
{
	inDecoder.decodeFrame(inBitstream, outFrame);
};

template <typename T>
concept EncoderHasPublicPhysicalUpload = requires(
	T& inEncoder,
	const oavk::Buffer& inBuffer)
{
	inEncoder.uploadInputRgba(inBuffer, 1U, 1U);
};

template <typename T>
concept EncoderHasPublicPhysicalEncode = requires(
	T& inEncoder,
	oa::EncodedVideoPacket& outFrame)
{
	inEncoder.encodeFrame(VK_NULL_HANDLE, 0U, outFrame);
};

static_assert(not VideoConfigHasLegacyPath<oa::VideoPlayerConfig>);
static_assert(not VideoHasLegacyCreate<oa::VideoPlayer>);
static_assert(not VideoHasLegacyStepForward<oa::VideoPlayer>);
static_assert(not DecoderHasPublicPhysicalDecode<oa::VideoDecoder>);
static_assert(not EncoderHasPublicPhysicalUpload<oa::VideoEncoder>);
static_assert(not EncoderHasPublicPhysicalEncode<oa::VideoEncoder>);

TEST(EngineInit, Fp64FailsClosedWithoutACompleteKernelAndTrainingPack) {
	auto config = testEngineConfig(oa::Precision::FP64);
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto result = oa::Engine::create(config);
	ASSERT_FALSE(result.isOk());
	EXPECT_EQ(result.getStatus().getCode(), oa::StatusCode::Unimplemented);
	EXPECT_NE(result.getStatus().getMessage().find("never substitutes FP32"),
		oa::String::Npos);
}

TEST(EngineInit, GenericDispatchStorageDoesNotAliasFp16OrFp64) {
	const auto fp16 = oavk::resolveStorageDtypeSpecConstant(oa::ScalarType::Float16);
	const auto fp64 = oavk::resolveStorageDtypeSpecConstant(oa::ScalarType::Float64);
	ASSERT_FALSE(fp16.isOk());
	ASSERT_FALSE(fp64.isOk());
	EXPECT_EQ(fp16.getStatus().getCode(), oa::StatusCode::DtypeMismatch);
	EXPECT_EQ(fp64.getStatus().getCode(), oa::StatusCode::Unimplemented);
}

template <typename T>
concept HasPublicEnsurePipeline = requires(
	T& inEngine,
	oa::StringView inName,
	oa::Span<const oa::U8> inSpirv,
	const oa::PipelineSpec& inSpec)
{
	inEngine.ensurePipeline(inName, inSpirv, inSpec);
};

template <typename T>
concept HasPublicRawDevice = requires(T& inEngine) {
	inEngine.device;
};

template <typename T>
concept HasPublicAcquireStream = requires(T& inEngine) {
	inEngine.acquireStream();
};

template <typename T>
concept HasPublicReleaseStream = requires(
	T& inEngine,
	oavk::Stream* inStream)
{
	inEngine.releaseStream(inStream);
};

template <typename T>
concept HasPublicRegisterBuffer = requires(
	T& inEngine,
	oavk::Buffer& inBuffer)
{
	inEngine.registerBuffer(inBuffer);
};

template <typename T>
concept HasPublicUpdateBufferDescriptor = requires(
	T& inEngine,
	const oavk::Buffer& inBuffer)
{
	inEngine.updateBufferDescriptor(inBuffer);
};

template <typename T>
concept HasPublicDeregisterBuffer = requires(
	T& inEngine,
	oavk::Buffer& inBuffer)
{
	inEngine.deregisterBuffer(inBuffer);
};

template <typename T>
concept HasPublicAllocBuffer = requires(T& inEngine) {
	inEngine.allocBuffer(16U);
};

template <typename T>
concept HasPublicUploadBuffer = requires(
	T& inEngine,
	const oavk::Buffer& inBuffer,
	const void* inData)
{
	inEngine.uploadBuffer(inBuffer, 0U, inData, 4U);
};

template <typename T>
concept HasPublicReadbackBuffer = requires(
	T& inEngine,
	const oavk::Buffer& inBuffer,
	void* outData)
{
	inEngine.readbackBuffer(inBuffer, 0U, outData, 4U);
};

template <typename T>
concept HasPublicFreeBuffer = requires(
	T& inEngine,
	oavk::Buffer& inBuffer)
{
	inEngine.freeBuffer(inBuffer);
};

template <typename T>
concept HasPublicCopyBufferAsync = requires(
	T& inEngine,
	const oavk::Buffer& inSource,
	const oavk::Buffer& inDestination)
{
	inEngine.copyBufferAsync(inSource, inDestination, 4U);
};

template <typename T>
concept HasPublicDefaultMatrixPlacement = requires(T& inEngine) {
	inEngine.defaultMatrixPlacement();
};

template <typename T>
concept HasPublicInitInPlace = requires(
	T& inEngine,
	const oa::EngineConfig& inConfig)
{
	inEngine.initInPlace(inConfig);
};

template <typename T>
concept HasLegacySurfaceConfig = requires(T& inConfig) {
	inConfig.surface;
};

#define OA_DECLARE_ENGINE_QUERY_CONCEPT(inConcept, inMethod) \
	template <typename T>                                      \
	concept inConcept = requires(const T& inEngine) {           \
		inEngine.inMethod();                                     \
	}

OA_DECLARE_ENGINE_QUERY_CONCEPT(HasPublicAsyncCompute, hasAsyncCompute);
OA_DECLARE_ENGINE_QUERY_CONCEPT(HasPublicSam, hasSAM);
OA_DECLARE_ENGINE_QUERY_CONCEPT(
	HasPublicCooperativeMatrix, hasCooperativeMatrix);
OA_DECLARE_ENGINE_QUERY_CONCEPT(
	HasPublicCooperativeMatrix2, hasCooperativeMatrix2);
OA_DECLARE_ENGINE_QUERY_CONCEPT(
	HasPublicNativeBfloat16CooperativeMatrix,
	HasNativeBfloat16CooperativeMatrix);
OA_DECLARE_ENGINE_QUERY_CONCEPT(HasPublicPipelineLibrary, hasPipelineLibrary);
OA_DECLARE_ENGINE_QUERY_CONCEPT(
	HasPublicDeviceGeneratedCommands, hasDeviceGeneratedCommands);
OA_DECLARE_ENGINE_QUERY_CONCEPT(HasPublicBlackwellPolicy, IsBlackwell);
OA_DECLARE_ENGINE_QUERY_CONCEPT(HasPublicSubgroupSize, subgroupSize);
OA_DECLARE_ENGINE_QUERY_CONCEPT(HasPublicGemmCapsMask, GemmCapsMask);

#undef OA_DECLARE_ENGINE_QUERY_CONCEPT

template <typename T>
concept AcceptsRawDeviceDestroy = requires(
	T& inOwner,
	const oavk::Device& inDevice)
{
	inOwner.destroy(inDevice);
};

template <typename T>
concept AcceptsEngineDestroy = requires(
	T& inOwner,
	const oa::Engine& inEngine)
{
	inOwner.destroy(inEngine);
};

template <typename T>
concept AcceptsRawDeviceCommit = requires(
	T& inTimer,
	const oavk::Device& inDevice)
{
	inTimer.commit(inDevice, 1.0);
};

template <typename T>
concept AcceptsRawDeviceGraphWait = requires(
	T& inGraph,
	const oavk::Device& inDevice)
{
	inGraph.waitForPendingReplay(inDevice);
};

static_assert(not HasPublicEnsurePipeline<oa::Engine>);
static_assert(not HasPublicRawDevice<oa::Engine>);
static_assert(not HasPublicAcquireStream<oa::Engine>);
static_assert(not HasPublicReleaseStream<oa::Engine>);
static_assert(not HasPublicRegisterBuffer<oa::Engine>);
static_assert(not HasPublicUpdateBufferDescriptor<oa::Engine>);
static_assert(not HasPublicDeregisterBuffer<oa::Engine>);
static_assert(not HasPublicAllocBuffer<oa::Engine>);
static_assert(not HasPublicUploadBuffer<oa::Engine>);
static_assert(not HasPublicReadbackBuffer<oa::Engine>);
static_assert(not HasPublicFreeBuffer<oa::Engine>);
static_assert(not HasPublicCopyBufferAsync<oa::Engine>);
static_assert(not HasPublicDefaultMatrixPlacement<oa::Engine>);
static_assert(not HasPublicInitInPlace<oa::Engine>);
static_assert(not HasLegacySurfaceConfig<oa::EngineConfig>);
static_assert(not std::is_default_constructible_v<oa::Engine>);
static_assert(not HasPublicAsyncCompute<oa::Engine>);
static_assert(not HasPublicSam<oa::Engine>);
static_assert(not HasPublicCooperativeMatrix<oa::Engine>);
static_assert(not HasPublicCooperativeMatrix2<oa::Engine>);
static_assert(not HasPublicNativeBfloat16CooperativeMatrix<oa::Engine>);
static_assert(not HasPublicPipelineLibrary<oa::Engine>);
static_assert(not HasPublicDeviceGeneratedCommands<oa::Engine>);
static_assert(not HasPublicBlackwellPolicy<oa::Engine>);
static_assert(not HasPublicSubgroupSize<oa::Engine>);
static_assert(not HasPublicGemmCapsMask<oa::Engine>);
static_assert(not AcceptsRawDeviceDestroy<oa::Timer>);
static_assert(not AcceptsRawDeviceDestroy<oavk::Timestamp>);
static_assert(not AcceptsRawDeviceDestroy<oa::ExecutableGraph>);
static_assert(not AcceptsEngineDestroy<oa::Timer>);
static_assert(not AcceptsEngineDestroy<oavk::Timestamp>);
static_assert(not AcceptsEngineDestroy<oa::ExecutableGraph>);
static_assert(not AcceptsRawDeviceCommit<oa::Timer>);
static_assert(not AcceptsRawDeviceGraphWait<oa::ExecutableGraph>);

TEST(EngineInit, SuiteEngineFromEnvironment) {
	ASSERT_TRUE(vkTestEngineOk());
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	EXPECT_NE(oa::EngineDeviceAccess::get(*rt).device, nullptr);
	EXPECT_NE(oa::EngineDeviceAccess::get(*rt).physicalDevice, nullptr);
	EXPECT_NE(oa::EngineDeviceAccess::get(*rt).instance, nullptr);
	EXPECT_FALSE(oa::EngineDeviceAccess::get(*rt).info.hardware.deviceName.empty());
}

TEST(EngineInit, OptionalExtensionsReport) {
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	const oavk::Device& d = oa::EngineDeviceAccess::get(*rt);
	fprintf(stderr,
		"  [EngineInit] optional caps: CoopMat=%d PipLib=%d ExtMemFd=%d 16bit=%d SAM=%d\n",
		d.info.software.hasCooperativeMatrix ? 1 : 0,
		d.info.software.hasPipelineLibrary ? 1 : 0,
		d.info.software.hasExternalMemoryFd ? 1 : 0,
		d.info.software.has16BitStorage ? 1 : 0,
		d.info.hardware.hasSAM ? 1 : 0);
	fprintf(stderr,
		"  [EngineInit] extensions policy: %s (enable), %s (query-only fallback), %s, %s+%s\n",
		oavk::ExtKhrCooperativeMatrix,
		oavk::ExtNvCooperativeMatrix,
		oavk::ExtKhrPipelineLibrary,
		oavk::ExtKhrExternalMemory,
		oavk::ExtKhrExternalMemoryFd);
	SUCCEED();
}

TEST(EngineInit, CooperativeMatrixStrictEnv) {
	const char* req = std::getenv("OA_REQUIRE_COOPMAT");
	if (!req || req[0] == '\0' || std::strcmp(req, "0") == 0) {
		GTEST_SKIP() << "set OA_REQUIRE_COOPMAT=1 to require VK_KHR cooperative matrix path";
	}
	auto* rt = testEnginePtr();
	ASSERT_NE(rt, nullptr);
	EXPECT_TRUE(oa::EngineDeviceAccess::get(*rt).info.software.hasCooperativeMatrix)
		<< "Driver must expose usable 16x16x16 cooperative matrices + "
		<< oavk::ExtKhrCooperativeMatrix;
}

TEST_VK(VkEngineTestFixture, UsesSharedEngine) {
	EXPECT_NE(oa::EngineDeviceAccess::get(rt()).queues.computeQueue, nullptr);
}

TEST_VK(VkEngineTestFixture, PresenterBorrowsEngine) {
	oa::Engine& engine = rt();
	{
		oa::Presenter presenter(engine);
		EXPECT_EQ(&presenter.engine(), &engine);
		EXPECT_FALSE(presenter.hasPresent());
		EXPECT_TRUE(presenter.close().isOk());
		EXPECT_TRUE(presenter.close().isOk());
	}
	EXPECT_TRUE(engine.isReady());
	EXPECT_NE(oa::EngineDeviceAccess::get(engine).device, nullptr);
}

TEST(EngineInit, HeadlessGraphicsDoesNotEnableWsi) {
	auto* primaryEngine = testEnginePtr();
	ASSERT_NE(primaryEngine, nullptr);
	const auto primaryQueueSubmit =
		oa::EngineDeviceAccess::get(*primaryEngine).deviceDispatch.vkQueueSubmit;
	ASSERT_NE(primaryQueueSubmit, nullptr);

	auto config = testEngineConfig(oa::Precision::FP32);
	config.presentationMode = oa::PresentationMode::Headless;
	config.selectForThread = false;
	config.preloadEmbeddedPipelines = false;
	config.enablePipelineCache = false;
	auto result = oa::Engine::create(config);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	auto engine = oa::move(*result);
	EXPECT_NE(oa::EngineDeviceAccess::get(*engine).queues.graphicsQueue, nullptr);
	EXPECT_EQ(oa::EngineDeviceAccess::get(*engine).queues.presentQueue, nullptr);
	EXPECT_FALSE(oa::EngineDeviceAccess::get(*engine).queues.hasPresentation);
	EXPECT_FALSE(oa::EngineDeviceAccess::get(*engine).info.software.hasSwapchainMaintenance1);
	for (const auto& extension : oa::EngineDeviceAccess::get(*engine).info.software.enabledDeviceExtensions) {
		EXPECT_NE(extension, oa::StringView(oavk::ExtKhrSwapchain));
	}
	EXPECT_EQ(oa::EngineDeviceAccess::get(*primaryEngine).deviceDispatch.vkQueueSubmit,
		primaryQueueSubmit);
	auto primaryBufferResult = oa::EngineResourceAccess::allocBuffer(*primaryEngine, 16U);
	ASSERT_TRUE(primaryBufferResult.isOk())
		<< primaryBufferResult.getStatus().toString();
	auto primaryBuffer = oa::move(*primaryBufferResult);
	oa::EngineResourceAccess::freeBuffer(*primaryEngine, primaryBuffer);
	EXPECT_TRUE(engine->close().isOk());
}
