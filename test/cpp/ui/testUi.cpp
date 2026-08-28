#include "../oaTest.h"

#include <oa/runtime/engine.h>
#include <oa/runtime/eventAccess.h>
#include <oa/runtime/engine/allocatorAccess.h>
#include <oa/runtime/engine/bindlessAccess.h>
#include <oa/runtime/engine/engineAccess.h>
#include <oa/runtime/executionSession.h>
#include <vma/vma.hpp>
#include <oa/runtime/stream.h>
#include <oa/core/fnMatrix.h>
#include <oa/core/filesystem.h>
#include <oa/render/renderer.h>
#include <oa/ui/renderConfig.h>
#include <oa/ui/image.h>
#include <oa/ui/cv.h>
#include <oa/ui/detectionOverlay.h>
#include <oa/ui/input.h>
#include <oa/ui/navigation.h>
#include <oa/ui/platformInput.h>
#include <oa/ui/plot/plot.h>
#include <oa/ui/text.h>
#include <oa/ui/ui.h>
#include <oa/ui/viewer.h>
#include <oa/ui/viewport.h>
#include <oa/vision/fnImage.h>

#include <oa/runtime/textureAccess.h>

#include <oa/ui/windowDecoration.h>

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_scancode.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

template<typename T>
concept HasPublicDestroy = requires(T& inValue) {
	inValue.destroy();
};

template<typename T>
concept HasPublicEngineDestroy = requires(T& inValue, oa::Engine& inEngine) {
	inValue.destroy(inEngine);
};

template<typename T>
concept HasPlanesMatrixBridge = requires(
	const T& inValue,
	oa::Engine& inEngine,
	const oa::Matrix& inMatrix)
{
	inValue.toMatrix();
	T::fromMatrix(inEngine, inMatrix);
};

template<typename T>
concept HasSaveToPath = requires(T& inValue, const char* inPath) {
	inValue.saveTo(inPath);
};

template<typename T>
concept HasSaveFigPath = requires(T& inValue, const char* inPath) {
	inValue.saveFig(inPath);
};

template<typename T>
concept HasCvEdges = requires(T& inValue) { inValue.addEdges(); };

template<typename T>
concept HasCvBlobs = requires(T& inValue) { inValue.addBlobs(); };

template<typename T>
concept HasCvMasks = requires(T& inValue) { inValue.addMasks({}); };

template<typename T>
concept HasCvFlow = requires(T& inValue) { inValue.addFlow(nullptr, nullptr); };

template<typename T>
concept HasCvStats = requires(T& inValue) { inValue.addStats(); };

template<typename TSession, typename TFrame>
concept HasViewerFrameShow = requires(
	oa::Engine& inEngine,
	TSession& inSession,
	const TFrame& inFrame)
{
	oa::Viewer::show(inEngine, inSession, inFrame);
};

static_assert(not HasPublicDestroy<oa::DetectionBuffer>);
static_assert(not HasPublicDestroy<oa::GlyphBuffer>);
static_assert(not HasPublicDestroy<oa::DetectionOverlay>);
static_assert(not HasPublicDestroy<oa::TextAtlas>);
static_assert(not HasPublicDestroy<oa::Ui>);
static_assert(not HasPublicDestroy<oa::Renderer>);
static_assert(not HasPublicEngineDestroy<oa::ImagePlanes>);
static_assert(not HasPlanesMatrixBridge<oa::ImagePlanes>);
static_assert(not HasPublicEngineDestroy<oa::Texture>);
static_assert(not HasPlanesMatrixBridge<oa::Texture>);
static_assert(std::is_copy_constructible_v<oa::Texture>);
static_assert(std::is_copy_assignable_v<oa::Texture>);
static_assert(not std::is_copy_constructible_v<oa::ImagePlanes>);
static_assert(not std::is_copy_assignable_v<oa::ImagePlanes>);
static_assert(std::is_nothrow_move_constructible_v<oa::ImagePlanes>);
static_assert(std::is_nothrow_move_assignable_v<oa::ImagePlanes>);
static_assert(HasSaveToPath<oa::plot::Figure>);
static_assert(not HasSaveFigPath<oa::plot::Figure>);
static_assert(not HasCvEdges<oa::CvFrame>);
static_assert(not HasCvBlobs<oa::CvFrame>);
static_assert(not HasCvMasks<oa::CvFrame>);
static_assert(not HasCvFlow<oa::CvFrame>);
static_assert(not HasCvStats<oa::CvFrame>);
static_assert(HasViewerFrameShow<oa::Renderer, oa::RenderFrame>);

class TestUi : public ::testing::Test {};

class ExplicitLiveSource final : public oa::ViewerLiveSource {
public:
	[[nodiscard]] oa::ViewerLiveCapabilities capabilities() const noexcept override {
		return {};
	}
	oa::Status open(oa::Engine&) override { return oa::Status::ok(); }
	oa::Status init(oa::InputSystem&, oa::Fn<void(bool)>) override {
		return oa::Status::ok();
	}
	oa::Status update(oa::F32) override { return oa::Status::ok(); }
	oa::Status render(oa::Ui&, const oa::TextAtlas&, oa::U32, oa::U32) override {
		return oa::Status::ok();
	}
	oa::Status close() override { return oa::Status::ok(); }
};

static_assert(std::is_abstract_v<oa::ViewerLiveSource>);

TEST_F(TestUi, LiveSourcesDeclareOptionalCapabilitiesWithoutSilentHooks)
{
	ExplicitLiveSource source;
	const oa::ViewerLiveCapabilities capabilities = source.capabilities();
	EXPECT_FALSE(capabilities.receivesEvents);
	EXPECT_FALSE(capabilities.publishesRenderDependency);
	EXPECT_FALSE(capabilities.retainsConsumerCompletion);
	EXPECT_EQ(source.event({}).getCode(), oa::StatusCode::Unimplemented);
	const auto ready = source.renderReady();
	ASSERT_FALSE(ready.isOk());
	EXPECT_EQ(ready.getStatus().getCode(), oa::StatusCode::Unimplemented);
	EXPECT_EQ(
		source.markConsumed({}).getCode(),
		oa::StatusCode::Unimplemented);
}

TEST_F(TestUi, ThemePresetsAreDefinedValidatedAndDisplayEncoded)
{
	const oa::UiStyle realmDark = oa::UiStyle::realmDark();
	const oa::UiStyle realmLight = oa::UiStyle::realmLight();
	const oa::UiStyle editorDark = oa::UiStyle::editorDark();
	const oa::UiStyle editorLight = oa::UiStyle::editorLight();
	EXPECT_TRUE(realmDark.validate().isOk());
	EXPECT_TRUE(realmLight.validate().isOk());
	EXPECT_TRUE(editorDark.validate().isOk());
	EXPECT_TRUE(editorLight.validate().isOk());
	EXPECT_EQ(editorDark.background.toU32(), 0x181818FFU);
	EXPECT_EQ(editorDark.surface.toU32(), 0x313131FFU);
	EXPECT_EQ(editorDark.accent.toU32(), 0x0078D4FFU);
	EXPECT_EQ(editorDark.text.toU32(), 0xCCCCCCFFU);
	EXPECT_EQ(editorLight.background.toU32(), 0xF8F8F8FFU);
	EXPECT_EQ(editorLight.accent.toU32(), 0x005FB8FFU);

	oa::UiStyle invalid = editorDark;
	invalid.fontSize = std::numeric_limits<oa::F32>::quiet_NaN();
	EXPECT_FALSE(invalid.validate().isOk());
	invalid = editorDark;
	invalid.text.r = 1.1F;
	EXPECT_FALSE(invalid.validate().isOk());

	const oa::ViewerConfig viewer;
	EXPECT_EQ(viewer.style.background.toU32(), editorDark.background.toU32());
	EXPECT_EQ(viewer.style.accent.toU32(), editorDark.accent.toU32());
	EXPECT_FALSE(viewer.customWindowDecoration);
}

TEST_VK(TestUi, UiRejectsInvalidBaseAndStackStylesAtTheirAdmissionBoundary)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::UiStyle invalid = oa::UiStyle::editorDark();
	invalid.padding = -1.0F;
	oa::Ui rejected;
	EXPECT_FALSE(rejected.init(*engine, invalid).isOk());

	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine, oa::UiStyle::editorDark()).isOk());
	ui.beginFrame(16.0F, {0, 0, 16, 16});
	invalid = oa::UiStyle::editorDark();
	invalid.accent.a = std::numeric_limits<oa::F32>::infinity();
	ui.pushStyle(invalid);
	EXPECT_FALSE(ui.recordRender(VK_NULL_HANDLE, OA_BINDLESS_INVALID).isOk());
	ui.endFrame();

	ui.beginFrame(16.0F, {0, 0, 16, 16});
	ui.popStyle();
	EXPECT_FALSE(ui.recordRender(VK_NULL_HANDLE, OA_BINDLESS_INVALID).isOk());
	ui.endFrame();

	ui.beginFrame(
		16.0F,
		{0, 0, 16, 16},
		std::numeric_limits<oa::F32>::max());
	EXPECT_FALSE(ui.recordRender(VK_NULL_HANDLE, OA_BINDLESS_INVALID).isOk());
	ui.endFrame();

	ui.beginFrame(16.0F, {0, 0, 16, 16});
	ui.textAt("invalid alignment", {0, 0, 16, 16}, {
		.horizontalAlign = static_cast<oa::UiAlign>(255U),
	});
	EXPECT_FALSE(ui.recordRender(VK_NULL_HANDLE, OA_BINDLESS_INVALID).isOk());
	ui.endFrame();

	ui.beginFrame(16.0F, {0, 0, 16, 16});
	ui.textAt("invalid direction", {0, 0, 16, 16}, {
		.direction = static_cast<oa::UiTextDirection>(255U),
	});
	EXPECT_FALSE(ui.recordRender(VK_NULL_HANDLE, OA_BINDLESS_INVALID).isOk());
	ui.endFrame();

	ui.beginFrame(16.0F, {0, 0, 16, 16});
	ui.grid({0, 0, 16, 16}, {.opacity = 1.01F});
	EXPECT_FALSE(ui.recordRender(VK_NULL_HANDLE, OA_BINDLESS_INVALID).isOk());
	ui.endFrame();

	ui.beginFrame(16.0F, {0, 0, 16, 16});
	ui.rect(
		{0, 0, 16, 16},
		oa::Color{1.0F, 1.0F, 1.0F, 1.0F},
		std::numeric_limits<oa::F32>::quiet_NaN());
	EXPECT_FALSE(ui.recordRender(VK_NULL_HANDLE, OA_BINDLESS_INVALID).isOk());
	ui.endFrame();
	EXPECT_TRUE(ui.close().isOk());
}

struct UiStorageTarget {
	oa::Engine* engine = nullptr;
	VkImage image = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	vma::Allocation allocation = VK_NULL_HANDLE;
	oavk::Buffer readback;
	oa::U32 bindlessIndex = OA_BINDLESS_INVALID;
	oa::U32 width = 0U;
	oa::U32 height = 0U;

	~UiStorageTarget() { destroy(); }

	[[nodiscard]] oa::Status init(oa::Engine& inEngine, oa::U32 inWidth, oa::U32 inHeight) {
		engine = &inEngine;
		width = inWidth;
		height = inHeight;

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		imageInfo.extent = {width, height, 1U};
		imageInfo.mipLevels = 1U;
		imageInfo.arrayLayers = 1U;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT
			| VK_IMAGE_USAGE_TRANSFER_SRC_BIT
			| VK_IMAGE_USAGE_TRANSFER_DST_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		vma::AllocationCreateInfo allocationInfo{};
		allocationInfo.usage = vma::memoryUsageGpuOnly;
		if (vma::createImage(
				static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(*engine).allocator),
				&imageInfo,
				&allocationInfo,
				&image,
				&allocation,
				nullptr) != VK_SUCCESS) {
			return oa::Status::error(
				oa::StatusCode::OutOfMemory,
				"TestUi: storage target allocation failed");
		}

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.levelCount = 1U;
		viewInfo.subresourceRange.layerCount = 1U;
		if (oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkCreateImageView(
				static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device),
				&viewInfo,
				nullptr,
				&view) != VK_SUCCESS) {
			return oa::Status::error(
				oa::StatusCode::VulkanError,
				"TestUi: storage target view creation failed");
		}

		bindlessIndex = oa::EngineBindlessAccess::get(*engine).registerStorageImage(
			oa::EngineDeviceAccess::get(*engine),
			view,
			VK_IMAGE_LAYOUT_GENERAL);
		if (bindlessIndex == OA_BINDLESS_INVALID) {
			return oa::Status::error(
				oa::StatusCode::ResourceExhausted,
				"TestUi: storage target bindless registration failed");
		}

		auto readbackResult = oa::EngineAllocatorAccess::get(*engine).allocHostVisible(
			static_cast<oa::U64>(width) * height * 4U);
		if (readbackResult.isError()) return readbackResult.getStatus();
		readback = *readbackResult;
		return oa::Status::ok();
	}

	void recordInitialize(VkCommandBuffer inCommandBuffer) const {
		VkImageMemoryBarrier transition{};
		transition.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		transition.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		transition.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		transition.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		transition.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		transition.image = image;
		transition.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		transition.subresourceRange.levelCount = 1U;
		transition.subresourceRange.layerCount = 1U;
		transition.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkCmdPipelineBarrier(
			inCommandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0U,
			0U, nullptr,
			0U, nullptr,
			1U, &transition);

		VkClearColorValue clear{};
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkCmdClearColorImage(
			inCommandBuffer,
			image,
			VK_IMAGE_LAYOUT_GENERAL,
			&clear,
			1U,
			&transition.subresourceRange);

		VkImageMemoryBarrier ready = transition;
		ready.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT
			| VK_ACCESS_SHADER_WRITE_BIT;
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkCmdPipelineBarrier(
			inCommandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			0U,
			0U, nullptr,
			0U, nullptr,
			1U, &ready);
	}

	void recordReadback(oavk::Stream& inStream) const {
		const VkCommandBuffer commandBuffer =
			static_cast<VkCommandBuffer>(inStream.commandBuffer);
		VkImageMemoryBarrier readable{};
		readable.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		readable.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		readable.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		readable.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		readable.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		readable.image = image;
		readable.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		readable.subresourceRange.levelCount = 1U;
		readable.subresourceRange.layerCount = 1U;
		readable.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		readable.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkCmdPipelineBarrier(
			commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0U,
			0U, nullptr,
			0U, nullptr,
			1U, &readable);

		VkBufferImageCopy copy{};
		copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		copy.imageSubresource.layerCount = 1U;
		copy.imageExtent = {width, height, 1U};
		oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkCmdCopyImageToBuffer(
			commandBuffer,
			image,
			VK_IMAGE_LAYOUT_GENERAL,
			static_cast<VkBuffer>(readback.buffer),
			1U,
			&copy);
		inStream.recordTransferWriteBarrier(
			readback,
			0U,
			static_cast<oa::U64>(width) * height * 4U);
	}

	void destroy() {
		if (engine == nullptr) return;
		if (bindlessIndex != OA_BINDLESS_INVALID) {
			oa::EngineBindlessAccess::get(*engine)
				.deregisterStorageImage(bindlessIndex);
			bindlessIndex = OA_BINDLESS_INVALID;
		}
		if (readback.buffer != nullptr) oa::EngineResourceAccess::freeBuffer(*engine, readback);
		if (view != VK_NULL_HANDLE) {
			oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkDestroyImageView(
				static_cast<VkDevice>(oa::EngineDeviceAccess::get(*engine).device),
				view,
				nullptr);
			view = VK_NULL_HANDLE;
		}
		if (image != VK_NULL_HANDLE) {
			vma::destroyImage(
				static_cast<vma::Allocator>(oa::EngineAllocatorAccess::get(*engine).allocator),
				image,
				allocation);
			image = VK_NULL_HANDLE;
			allocation = VK_NULL_HANDLE;
		}
		engine = nullptr;
	}
};

} // namespace

TEST_VK(TestUi, TimelinePointerDragAndRelease)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	const oa::PixelRect rect{10, 10, 100, 12};
	oa::F32 fraction = 0.25F;

	ui.beginFrame(16.0F);
	oa::UiEvent down;
	down.type = oa::UiEventType::MouseDown;
	down.button = 1;
	down.mouseX = 85.0F;
	down.mouseY = 16.0F;
	EXPECT_FALSE(ui.routeEvent(down));
	EXPECT_TRUE(ui.timeline("transport", rect, fraction));
	EXPECT_FLOAT_EQ(fraction, 0.75F);
	EXPECT_NE(ui.input().activeId, 0U);
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::UiEvent move;
	move.type = oa::UiEventType::MouseMove;
	move.mouseX = 35.0F;
	move.mouseY = 16.0F;
	EXPECT_TRUE(ui.routeEvent(move));
	EXPECT_TRUE(ui.timeline("transport", rect, fraction));
	EXPECT_FLOAT_EQ(fraction, 0.25F);
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::UiEvent up;
	up.type = oa::UiEventType::MouseUp;
	up.button = 1;
	up.mouseX = 60.0F;
	up.mouseY = 16.0F;
	EXPECT_TRUE(ui.routeEvent(up));
	EXPECT_TRUE(ui.timeline("transport", rect, fraction));
	EXPECT_FLOAT_EQ(fraction, 0.5F);
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();
}

TEST_VK(TestUi, TimelineClampsFraction)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ui.beginFrame(16.0F);
	oa::F32 fraction = 2.0F;
	EXPECT_FALSE(ui.timeline("transport", {10, 10, 100, 12}, fraction));
	EXPECT_FLOAT_EQ(fraction, 1.0F);
	ui.endFrame();
}

TEST_F(TestUi, NodeCanvasNavigationHitTestingAndGridMetricsAreTransactional)
{
	const oa::PixelRect nearMaximum{
		std::numeric_limits<oa::I32>::max() - 2, 0, 4, 4};
	const oa::PixelRect atMaximum{
		std::numeric_limits<oa::I32>::max(), 1, 1, 1};
	const oa::PixelRect invalidPixelRect{0, 0, -1, 4};
	const oa::PixelRect ordinaryPixelRect{0, 0, 4, 4};
	const oa::WorldAabb originBounds{{-1.0F, -1.0F}, {1.0F, 1.0F}};
	const oa::WorldAabb invertedBounds{{1.0F, 0.0F}, {-1.0F, 1.0F}};
	EXPECT_TRUE(nearMaximum.intersects(atMaximum));
	EXPECT_FALSE(invalidPixelRect.intersects(ordinaryPixelRect));
	EXPECT_TRUE(originBounds.contains({0.0F, 0.0F}));
	EXPECT_FALSE(invertedBounds.isValid());

	oa::NodeCanvas canvas;
	ASSERT_TRUE(canvas.setViewSize(200.0F, 100.0F).isOk());
	const oa::vlm::Vec2 center = canvas.worldToScreen({0.0F, 0.0F});
	EXPECT_FLOAT_EQ(center.x, 100.0F);
	EXPECT_FLOAT_EQ(center.y, 50.0F);
	const oa::vlm::Vec2 focus{125.0F, 75.0F};
	const oa::vlm::Vec2 before = canvas.screenToWorld(focus);
	ASSERT_TRUE(canvas.zoomAt(2.0F, focus).isOk());
	const oa::vlm::Vec2 after = canvas.screenToWorld(focus);
	EXPECT_NEAR(after.x, before.x, 1.0e-5F);
	EXPECT_NEAR(after.y, before.y, 1.0e-5F);

	ASSERT_TRUE(canvas.setState({
		.pan = {0.0F, 0.0F},
		.zoom = 1.0F,
		.viewSize = {200.0F, 100.0F},
	}).isOk());
	const oa::Array<oa::NodeCanvasHitItem, 4> items{
		oa::NodeCanvasHitItem{
			.id = 11U,
			.bounds = {{-10.0F, -10.0F}, {10.0F, 10.0F}},
			.layer = 1},
		oa::NodeCanvasHitItem{
			.id = 22U,
			.bounds = {{-5.0F, -5.0F}, {5.0F, 5.0F}},
			.layer = 2},
		oa::NodeCanvasHitItem{
			.id = 33U,
			.bounds = {{-2.0F, -2.0F}, {2.0F, 2.0F}},
			.layer = 3,
			.enabled = false},
		oa::NodeCanvasHitItem{
			.id = 44U,
			.bounds = {{-4.0F, -4.0F}, {4.0F, 4.0F}},
			.layer = 2},
	};
	auto hit = canvas.hitTest(
		{100.0F, 50.0F},
		oa::Span<const oa::NodeCanvasHitItem>(items.data(), items.size()));
	ASSERT_TRUE(hit.isOk()) << hit.getStatus().toString();
	ASSERT_TRUE(hit->hasValue());
	EXPECT_EQ((*hit)->id, 44U);
	EXPECT_EQ((*hit)->index, 3U);
	EXPECT_FLOAT_EQ((*hit)->worldPoint.x, 0.0F);
	auto miss = canvas.hitTest(
		{0.0F, 0.0F},
		oa::Span<const oa::NodeCanvasHitItem>(items.data(), items.size()));
	ASSERT_TRUE(miss.isOk());
	EXPECT_FALSE(miss->hasValue());
	auto invalidItems = items;
	invalidItems[3].id = 0U;
	auto invalidHitItems = canvas.hitTest(
		{100.0F, 50.0F},
		oa::Span<const oa::NodeCanvasHitItem>(
			invalidItems.data(), invalidItems.size()));
	EXPECT_FALSE(invalidHitItems.isOk());

	auto grid = canvas.grid(8.0F, 10.0F, 10U, 100U);
	ASSERT_TRUE(grid.isOk()) << grid.getStatus().toString();
	EXPECT_FLOAT_EQ(grid->minorWorldStep, 10.0F);
	EXPECT_FLOAT_EQ(grid->minorScreenStep, 10.0F);
	EXPECT_FLOAT_EQ(grid->originScreen.x, 100.0F);
	EXPECT_FLOAT_EQ(grid->originScreen.y, 50.0F);
	EXPECT_EQ(grid->majorEvery, 10U);
	EXPECT_EQ(grid->superMajorEvery, 100U);

	const oa::NodeCanvasState beforeInvalid = canvas.state();
	EXPECT_FALSE(canvas.setState({
		.pan = {std::numeric_limits<oa::F32>::quiet_NaN(), 0.0F},
		.zoom = 1.0F,
		.viewSize = {200.0F, 100.0F},
	}).isOk());
	EXPECT_EQ(canvas.state().pan.x, beforeInvalid.pan.x);
	EXPECT_EQ(canvas.state().zoom, beforeInvalid.zoom);
	EXPECT_FALSE(canvas.zoomAt(
		std::numeric_limits<oa::F32>::quiet_NaN(), focus).isOk());
	EXPECT_EQ(canvas.state().zoom, beforeInvalid.zoom);

	ASSERT_TRUE(canvas.fitToView(
		{{-50.0F, -25.0F}, {50.0F, 25.0F}}, 20.0F).isOk());
	ASSERT_TRUE(canvas.isAnimating());
	ASSERT_TRUE(canvas.stepAnimation(100.0F).isOk());
	EXPECT_NEAR(canvas.state().zoom, 1.1F, 1.0e-5F);
	ASSERT_TRUE(canvas.stepAnimation(100.0F).isOk());
	EXPECT_FALSE(canvas.isAnimating());
	EXPECT_NEAR(canvas.state().zoom, 1.2F, 1.0e-5F);
	EXPECT_FLOAT_EQ(canvas.state().pan.x, 0.0F);
	EXPECT_FLOAT_EQ(canvas.state().pan.y, 0.0F);
	EXPECT_FALSE(canvas.stepAnimation(-1.0F).isOk());
}

TEST_F(TestUi, ViewerNavigationIsTransactionalAnchoredAndFramePartitionIndependent)
{
	oa::Navigation navigation;
	ASSERT_TRUE(navigation.validate().isOk());
	ASSERT_TRUE(navigation.setContentSize(400.0F, 200.0F).isOk());
	ASSERT_TRUE(navigation.setWindowSize(800.0F, 600.0F).isOk());
	ASSERT_TRUE(navigation.fitToWindow(false).isOk());
	EXPECT_FLOAT_EQ(navigation.zoom(), 2.0F);
	EXPECT_FLOAT_EQ(navigation.panX(), 0.0F);
	EXPECT_FLOAT_EQ(navigation.panY(), 100.0F);

	const oa::vlm::Vec2 anchor{400.0F, 300.0F};
	const oa::vlm::Vec2 contentBefore{
		(anchor.x - navigation.panX()) / navigation.zoom(),
		(anchor.y - navigation.panY()) / navigation.zoom(),
	};
	ASSERT_TRUE(navigation.zoomTo(4.0F, anchor.x, anchor.y, false).isOk());
	const oa::vlm::Vec2 contentAfter{
		(anchor.x - navigation.panX()) / navigation.zoom(),
		(anchor.y - navigation.panY()) / navigation.zoom(),
	};
	EXPECT_NEAR(contentAfter.x, contentBefore.x, 1.0e-4F);
	EXPECT_NEAR(contentAfter.y, contentBefore.y, 1.0e-4F);

	const oa::vlm::Vec3 beforeInvalid = navigation.movement();
	const oa::F32 panBeforeInvalid = navigation.panX();
	EXPECT_FALSE(navigation.setWindowSize(
		std::numeric_limits<oa::F32>::quiet_NaN(), 600.0F).isOk());
	EXPECT_EQ(navigation.movement().x, beforeInvalid.x);
	EXPECT_EQ(navigation.movement().y, beforeInvalid.y);
	EXPECT_EQ(navigation.movement().z, beforeInvalid.z);
	EXPECT_EQ(navigation.panX(), panBeforeInvalid);
	EXPECT_FALSE(navigation.panBy(
		std::numeric_limits<oa::F32>::infinity(), 0.0F).isOk());
	EXPECT_EQ(navigation.panX(), panBeforeInvalid);

	oa::UiEvent malformedPinch;
	malformedPinch.type = oa::UiEventType::Pinch;
	malformedPinch.pinchPhase = oa::UiPinchPhase::Update;
	malformedPinch.gestureScale = 0.0F;
	auto malformedResult = navigation.handleEvent(malformedPinch);
	EXPECT_FALSE(malformedResult.isOk());
	EXPECT_FALSE(navigation.isPinching());

	oa::Navigation oneStep;
	oa::Navigation partitioned;
	for (oa::Navigation* nav : {&oneStep, &partitioned}) {
		ASSERT_TRUE(nav->setContentSize(1600.0F, 900.0F).isOk());
		ASSERT_TRUE(nav->setWindowSize(800.0F, 600.0F).isOk());
		ASSERT_TRUE(nav->fitToWindow(false).isOk());
		ASSERT_TRUE(nav->keyboardZoomTo100().isOk());
		ASSERT_TRUE(nav->isAnimating());
	}
	ASSERT_TRUE(oneStep.update(100.0F).isOk());
	for (oa::U32 i = 0U; i < 4U; ++i) {
		ASSERT_TRUE(partitioned.update(25.0F).isOk());
	}
	EXPECT_NEAR(oneStep.movement().x, partitioned.movement().x, 1.0e-6F);
	EXPECT_NEAR(oneStep.movement().y, partitioned.movement().y, 1.0e-6F);
	EXPECT_NEAR(oneStep.movement().z, partitioned.movement().z, 1.0e-6F);
	EXPECT_NEAR(oneStep.panX(), partitioned.panX(), 1.0e-4F);
	EXPECT_NEAR(oneStep.panY(), partitioned.panY(), 1.0e-4F);
	EXPECT_FALSE(oneStep.update(-1.0F).isOk());

	oa::NavigationConfig invalidConfig;
	invalidConfig.keyboardZoomStep = 1.0F;
	oa::Navigation invalidNavigation(invalidConfig);
	EXPECT_FALSE(invalidNavigation.validate().isOk());
	oa::InputSystem input;
	EXPECT_FALSE(registerViewportShortcuts(input, invalidNavigation).isOk());
}

TEST_F(TestUi, PassiveViewportDescriptorsRejectInvalidGeometryTransactionally)
{
	oa::ViewportDesc descriptor{10, 20, 640, 360, 0.0F, 1.0F};
	auto aspect = descriptor.getAspectRatio();
	ASSERT_TRUE(aspect.isOk());
	EXPECT_NEAR(*aspect, 16.0F / 9.0F, 1.0e-6F);
	descriptor.height = 0;
	EXPECT_FALSE(descriptor.isValid());
	EXPECT_FALSE(descriptor.getAspectRatio().isOk());
	descriptor = {10, 20, 640, 360, 0.75F, 0.25F};
	EXPECT_FALSE(descriptor.isValid());

	const oa::ScissorDesc nearMaximum{
		std::numeric_limits<oa::I32>::max() - 2,
		std::numeric_limits<oa::I32>::max() - 2,
		10,
		10,
	};
	EXPECT_TRUE(nearMaximum.contains(
		std::numeric_limits<oa::I32>::max(),
		std::numeric_limits<oa::I32>::max()));
	EXPECT_FALSE(nearMaximum.contains(0, 0));
	EXPECT_FALSE((oa::ScissorDesc{0, 0, -1, 1}).isValid());

	oa::Viewport viewport;
	const oa::ViewportDesc initial = viewport.getViewport();
	EXPECT_FALSE(viewport.setViewport({0, 0, 0, 720}).isOk());
	EXPECT_EQ(viewport.getViewport().width, initial.width);
	EXPECT_EQ(viewport.getViewport().height, initial.height);
	EXPECT_FALSE(viewport.setScissor({0, 0, 1280, 0}).isOk());
	EXPECT_FALSE(viewport.isScissorEnabled());
	EXPECT_FALSE(viewport.setClearDepth(
		std::numeric_limits<oa::F32>::quiet_NaN()).isOk());
	EXPECT_FLOAT_EQ(viewport.getClearDepth(), 1.0F);
	EXPECT_FALSE(viewport.setClearColor({
		0.0F, std::numeric_limits<oa::F32>::infinity(), 0.0F, 1.0F}).isOk());
	EXPECT_FALSE(viewport.setMode(static_cast<oa::ViewportMode>(255U)).isOk());
	EXPECT_EQ(viewport.getMode(), oa::ViewportMode::Image2D);
	EXPECT_FALSE(viewport.setupImageAspectFit(
		1920.0F, 1080.0F, 1000.0F, 1000.0F, nullptr).isOk());
}

TEST_VK(TestUi, PassiveViewportAspectFitProducesAValidBorrowedTargetView)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 32U, 32U).isOk());
	const oa::Texture texture = oa::TextureAccess::fromBorrowedImage(
		*engine,
		target.image,
		target.view,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_LAYOUT_GENERAL,
		static_cast<oa::I32>(target.width),
		static_cast<oa::I32>(target.height));

	oa::Viewport viewport;
	ASSERT_TRUE(viewport.setupImageAspectFit(
		1920.0F, 1080.0F, 1000.0F, 1000.0F, &texture).isOk());
	ASSERT_TRUE(viewport.isValid());
	EXPECT_EQ(viewport.getMode(), oa::ViewportMode::Image2D);
	EXPECT_EQ(viewport.getTarget(), &texture);
	ASSERT_NE(viewport.getCamera(), nullptr);
	EXPECT_EQ(viewport.getViewport().x, 0);
	EXPECT_EQ(viewport.getViewport().y, 219);
	EXPECT_EQ(viewport.getViewport().width, 1000);
	EXPECT_EQ(viewport.getViewport().height, 563);
	EXPECT_FALSE(viewport.isScissorEnabled());
	EXPECT_FLOAT_EQ(viewport.getCamera()->getZoom(), 1.0F);

	const oa::ViewportDesc beforeInvalid = viewport.getViewport();
	EXPECT_FALSE(viewport.setupImageAspectFit(
		1920.0F,
		std::numeric_limits<oa::F32>::quiet_NaN(),
		1000.0F,
		1000.0F,
		&texture).isOk());
	EXPECT_EQ(viewport.getViewport().x, beforeInvalid.x);
	EXPECT_EQ(viewport.getViewport().y, beforeInvalid.y);
	EXPECT_EQ(viewport.getViewport().width, beforeInvalid.width);
	EXPECT_EQ(viewport.getViewport().height, beforeInvalid.height);
}

TEST_VK(TestUi, NodeCanvasGridRendersAdaptiveDecimalHierarchyInOneClip)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 32U, 32U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	oa::NodeCanvas canvas;
	ASSERT_TRUE(canvas.setViewSize(32.0F, 32.0F).isOk());
	oa::UiStyle style = ui.currentStyle();
	style.background = {0.0F, 0.0F, 0.0F, 1.0F};
	style.surface = {0.12F, 0.12F, 0.12F, 1.0F};
	oa::UiNodeCanvasGridConfig config;
	config.minimumScreenSpacing = 4.0F;
	config.baseWorldStep = 4.0F;
	config.majorEvery = 4U;
	config.superMajorEvery = 8U;
	config.drawAxes = true;

	ui.beginFrame(16.0F, {0, 0, 32, 32});
	ui.pushStyle(style);
	ui.beginPanel("canvas-clip", {8, 0, 16, 32});
	ui.nodeCanvasGrid(canvas, {0, 0, 32, 32}, config);
	ui.endPanel();
	ui.popStyle();

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 32U * 32U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine, target.readback, 0U, pixels.data(), pixels.size()).isOk());
	const auto pixel = [&](oa::U32 x, oa::U32 y, oa::U32 channel) {
		return pixels[(y * 32U + x) * 4U + channel];
	};
	EXPECT_EQ(pixel(7U, 18U, 3U), 0U);
	EXPECT_EQ(pixel(8U, 18U, 3U), 255U);
	EXPECT_EQ(pixel(18U, 18U, 0U), pixel(18U, 18U, 1U));
	EXPECT_EQ(pixel(18U, 18U, 1U), pixel(18U, 18U, 2U));
	EXPECT_GT(pixel(11U, 18U, 0U), pixel(18U, 18U, 0U));
	EXPECT_LT(pixel(15U, 18U, 0U), pixel(18U, 18U, 0U));
	EXPECT_EQ(pixel(24U, 18U, 3U), 0U);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, RoundedRectsAndChevronButtonsComposeAccessibleOsdGeometry)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 88U, 32U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());

	ui.beginFrame(16.0F, {0, 0, 88, 32});
	ui.rect({0, 0, 24, 24}, oa::Color{1.0F, 0.0F, 0.0F, 1.0F}, 8.0F);
	ui.rectOutline(
		{0, 0, 24, 24}, oa::Color{1.0F, 1.0F, 1.0F, 1.0F}, 1U, 8.0F);
	EXPECT_FALSE(ui.chevronButton(
		"Previous", {32, 0, 24, 24}, oa::UiChevronDirection::Previous));
	EXPECT_FALSE(ui.chevronButton(
		"Next", {64, 0, 24, 24}, oa::UiChevronDirection::Next));
	const auto nodes = ui.accessibilitySnapshot();
	ASSERT_EQ(nodes.size(), 2U);
	EXPECT_EQ(nodes[0].role, oa::UiAccessibilityRole::Button);
	EXPECT_EQ(testStdString(nodes[0].label), "Previous");
	EXPECT_EQ(testStdString(nodes[1].label), "Next");

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 88U * 32U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine, target.readback, 0U, pixels.data(), pixels.size()).isOk());
	const auto pixel = [&](oa::U32 x, oa::U32 y, oa::U32 channel) {
		return pixels[(y * 88U + x) * 4U + channel];
	};
	EXPECT_EQ(pixel(0U, 0U, 3U), 0U);
	EXPECT_EQ(pixel(12U, 12U, 0U), 255U);
	EXPECT_EQ(pixel(12U, 12U, 3U), 255U);
	EXPECT_EQ(pixel(32U, 0U, 3U), 0U);
	EXPECT_EQ(pixel(64U, 0U, 3U), 0U);
	oa::U32 previousBright = 0U;
	oa::U32 nextBright = 0U;
	for (oa::U32 y = 5U; y < 19U; ++y) {
		for (oa::U32 x = 37U; x < 51U; ++x) {
			previousBright += pixel(x, y, 0U) > 128U ? 1U : 0U;
			nextBright += pixel(x + 32U, y, 0U) > 128U ? 1U : 0U;
		}
	}
	EXPECT_GT(previousBright, 8U);
	EXPECT_GT(nextBright, 8U);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ui.beginFrame(16.0F, {0, 0, 88, 32});
	oa::UiEvent down;
	down.type = oa::UiEventType::MouseDown;
	down.button = 1;
	down.mouseX = 44.0F;
	down.mouseY = 12.0F;
	EXPECT_FALSE(ui.routeEvent(down));
	EXPECT_FALSE(ui.chevronButton(
		"Previous", {32, 0, 24, 24}, oa::UiChevronDirection::Previous));
	ui.endFrame();

	ui.beginFrame(16.0F, {0, 0, 88, 32});
	oa::UiEvent up = down;
	up.type = oa::UiEventType::MouseUp;
	EXPECT_TRUE(ui.routeEvent(up));
	EXPECT_TRUE(ui.chevronButton(
		"Previous", {32, 0, 24, 24}, oa::UiChevronDirection::Previous));
	ui.endFrame();
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, NestedScrollPanelsOwnWheelInputAndVirtualizeRows)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	oa::UiLayout scrollLayout;
	scrollLayout.padding = oa::UiEdge{};
	scrollLayout.gap = 0.0F;

	ui.beginFrame(16.0F);
	const oa::UiScrollRegion outer0 = ui.beginScrollPanel(
		"outer", {0, 0, 200, 100}, 500, scrollLayout);
	EXPECT_EQ(outer0.offsetY, 0);
	const oa::UiScrollRegion inner0 = ui.beginScrollPanel(
		"inner", {10, 10, 100, 30}, 100, scrollLayout, {.wheelStep = 10});
	EXPECT_EQ(inner0.offsetY, 0);
	EXPECT_EQ(ui.virtualRows(10, 10, 0, 0).first, 0);
	EXPECT_EQ(ui.virtualRows(10, 10, 0, 0).onePastLast, 3);
	ui.endScrollPanel();
	ui.endScrollPanel();
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::UiEvent scroll;
	scroll.type = oa::UiEventType::MouseScroll;
	scroll.mouseX = 20.0F;
	scroll.mouseY = 20.0F;
	scroll.scrollY = -1.0F;
	for (oa::I32 tick = 0; tick < 8; ++tick) {
		EXPECT_TRUE(ui.routeEvent(scroll));
	}
	const oa::UiScrollRegion outer1 = ui.beginScrollPanel(
		"outer", {0, 0, 200, 100}, 500, scrollLayout);
	const oa::UiScrollRegion inner1 = ui.beginScrollPanel(
		"inner", {10, 10, 100, 30}, 100, scrollLayout, {.wheelStep = 10});
	// Seven ticks saturate the inner region; the eighth bubbles to the outer.
	EXPECT_EQ(inner1.offsetY, 70);
	EXPECT_EQ(outer1.offsetY, 48);
	const oa::UiVirtualRange visible = ui.virtualRows(10, 10, 0, 0);
	EXPECT_EQ(visible.first, 7);
	EXPECT_EQ(visible.onePastLast, 10);
	ui.endScrollPanel();
	ui.endScrollPanel();
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::UiEvent down;
	down.type = oa::UiEventType::MouseDown;
	down.button = 1;
	down.mouseX = 195.0F;
	down.mouseY = 75.0F;
	EXPECT_FALSE(ui.routeEvent(down));
	const oa::UiScrollRegion outer2 = ui.beginScrollPanel(
		"outer", {0, 0, 200, 100}, 500, scrollLayout);
	EXPECT_GT(outer2.offsetY, 300);
	ui.endScrollPanel();
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::UiEvent move;
	move.type = oa::UiEventType::MouseMove;
	move.mouseX = 195.0F;
	move.mouseY = 100.0F;
	EXPECT_TRUE(ui.routeEvent(move));
	const oa::UiScrollRegion outer3 = ui.beginScrollPanel(
		"outer", {0, 0, 200, 100}, 500, scrollLayout);
	EXPECT_EQ(outer3.offsetY, 400);
	ui.endScrollPanel();
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::UiEvent up;
	up.type = oa::UiEventType::MouseUp;
	up.button = 1;
	up.mouseX = 195.0F;
	up.mouseY = 100.0F;
	EXPECT_TRUE(ui.routeEvent(up));
	const oa::UiScrollRegion outer4 = ui.beginScrollPanel(
		"outer", {0, 0, 200, 100}, 500, scrollLayout);
	EXPECT_EQ(outer4.offsetY, 400);
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endScrollPanel();
	ui.endFrame();
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, SplitPanesOwnRatioCaptureMinimaKeyboardAndClip)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	const oa::PixelRect rowRect{10, 20, 310, 120};
	const oa::UiSplitConfig rowConfig{
		.direction = oa::UiDirection::Row,
		.handleSize = 10,
		.minimumFirst = 40,
		.minimumSecond = 40,
		.keyboardStep = 0.02F,
	};
	oa::F32 rowRatio = 0.5F;

	ui.beginFrame(16.0F);
	oa::UiSplitRegion split = ui.splitPane(
		"main", rowRect, rowRatio, rowConfig);
	EXPECT_FALSE(split.changed);
	EXPECT_EQ(split.first.x, 10);
	EXPECT_EQ(split.first.w, 150);
	EXPECT_EQ(split.handle.x, 160);
	EXPECT_EQ(split.handle.w, 10);
	EXPECT_EQ(split.second.x, 170);
	EXPECT_EQ(split.second.w, 150);
	ui.endFrame();

	// Press establishes focus and capture on the handle.
	ui.beginFrame(16.0F);
	oa::UiEvent pointer;
	pointer.type = oa::UiEventType::MouseDown;
	pointer.button = 1;
	pointer.mouseX = 165.0F;
	pointer.mouseY = 60.0F;
	EXPECT_FALSE(ui.routeEvent(pointer));
	split = ui.splitPane("main", rowRect, rowRatio, rowConfig);
	EXPECT_FALSE(split.changed);
	ASSERT_NE(ui.input().activeId, 0U);
	EXPECT_EQ(ui.input().focusId, ui.input().activeId);
	ui.endFrame();

	// The active separator keeps pointer ownership outside its old geometry.
	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseMove;
	pointer.mouseX = 260.0F;
	EXPECT_TRUE(ui.routeEvent(pointer));
	split = ui.splitPane("main", rowRect, rowRatio, rowConfig);
	EXPECT_TRUE(split.changed);
	EXPECT_NEAR(rowRatio, 245.0F / 300.0F, 0.0001F);
	EXPECT_EQ(split.first.w, 245);
	EXPECT_EQ(split.handle.x, 255);
	ui.endFrame();

	// release position is committed before capture ends.
	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseUp;
	pointer.mouseX = 220.0F;
	EXPECT_TRUE(ui.routeEvent(pointer));
	split = ui.splitPane("main", rowRect, rowRatio, rowConfig);
	EXPECT_TRUE(split.changed);
	EXPECT_NEAR(rowRatio, 205.0F / 300.0F, 0.0001F);
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();

	// Row arrows follow screen X; the shared modifier scaling remains intact.
	ui.beginFrame(16.0F);
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Left;
	EXPECT_TRUE(ui.routeEvent(key));
	split = ui.splitPane("main", rowRect, rowRatio, rowConfig);
	EXPECT_TRUE(split.changed);
	EXPECT_NEAR(rowRatio, 205.0F / 300.0F - 0.02F, 0.0001F);
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Right;
	key.modifiers = oa::UiModifierShift;
	EXPECT_TRUE(ui.routeEvent(key));
	split = ui.splitPane("main", rowRect, rowRatio, rowConfig);
	EXPECT_TRUE(split.changed);
	EXPECT_NEAR(rowRatio, 205.0F / 300.0F + 0.18F, 0.0001F);
	ui.endFrame();
	key.modifiers = oa::UiModifierNone;

	// Column Down grows the top region, despite the adjustable route's generic
	// signed convention using Up as positive.
	oa::F32 columnRatio = 0.5F;
	const oa::PixelRect columnRect{0, 0, 100, 210};
	const oa::UiSplitConfig columnConfig{
		.direction = oa::UiDirection::Column,
		.handleSize = 10,
		.minimumFirst = 40,
		.minimumSecond = 40,
		.keyboardStep = 0.02F,
	};
	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseDown;
	pointer.mouseX = 50.0F;
	pointer.mouseY = 105.0F;
	EXPECT_FALSE(ui.routeEvent(pointer));
	split = ui.splitPane("vertical", columnRect, columnRatio, columnConfig);
	EXPECT_FALSE(split.changed);
	ui.endFrame();

	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseUp;
	EXPECT_TRUE(ui.routeEvent(pointer));
	(void)ui.splitPane("vertical", columnRect, columnRatio, columnConfig);
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Down;
	EXPECT_TRUE(ui.routeEvent(key));
	split = ui.splitPane("vertical", columnRect, columnRatio, columnConfig);
	EXPECT_TRUE(split.changed);
	EXPECT_NEAR(columnRatio, 0.52F, 0.0001F);
	ui.endFrame();

	// Caller state is clamped to satisfiable minima, and an ancestor clip owns
	// both hit testing and separator rendering.
	rowRatio = 2.0F;
	ui.beginFrame(16.0F);
	split = ui.splitPane("main", rowRect, rowRatio, rowConfig);
	EXPECT_TRUE(split.changed);
	EXPECT_NEAR(rowRatio, 260.0F / 300.0F, 0.0001F);
	EXPECT_EQ(split.second.w, 40);
	ui.endFrame();

	rowRatio = 0.5F;
	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseDown;
	pointer.mouseX = 165.0F;
	pointer.mouseY = 60.0F;
	EXPECT_FALSE(ui.routeEvent(pointer));
	ui.beginPanel("clip", {0, 0, 100, 120});
	(void)ui.splitPane("clipped", rowRect, rowRatio, rowConfig);
	ui.endPanel();
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::F32 invalidRatio = std::numeric_limits<oa::F32>::quiet_NaN();
	split = ui.splitPane("invalid", rowRect, invalidRatio, rowConfig);
	EXPECT_EQ(split.handle.w, 0);
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ui.beginFrame(16.0F);
	ASSERT_TRUE(ui.recordRender(VK_NULL_HANDLE, 0U).isOk());
	split = ui.splitPane("sealed", rowRect, rowRatio, rowConfig);
	EXPECT_EQ(split.handle.w, 0);
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, SplitPaneHandleRendersThroughInheritedClipOnGpu)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 32U, 16U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	oa::UiStyle style = ui.currentStyle();
	style.surface = {0.0F, 1.0F, 0.0F, 1.0F};
	style.borderStrong = {1.0F, 0.0F, 0.0F, 1.0F};
	oa::F32 ratio = 0.5F;

	ui.beginFrame(16.0F, {0, 0, 32, 16});
	ui.pushStyle(style);
	ui.beginPanel("split-clip", {0, 0, 16, 16});
	const oa::UiSplitRegion split = ui.splitPane(
		"main",
		{4, 2, 24, 12},
		ratio,
		{
			.direction = oa::UiDirection::Row,
			.handleSize = 4,
			.minimumFirst = 4,
			.minimumSecond = 4,
		});
	ui.endPanel();
	ui.popStyle();
	EXPECT_EQ(split.handle.x, 14);
	EXPECT_EQ(split.handle.w, 4);

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 32U * 16U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	const auto pixel = [&](oa::U32 inX, oa::U32 inY, oa::U32 inChannel) {
		return pixels[(inY * 32U + inX) * 4U + inChannel];
	};
	EXPECT_EQ(pixel(13U, 4U, 3U), 0U);
	EXPECT_EQ(pixel(14U, 4U, 0U), 0U);
	EXPECT_EQ(pixel(14U, 4U, 1U), 255U);
	EXPECT_EQ(pixel(14U, 4U, 3U), 255U);
	EXPECT_EQ(pixel(15U, 4U, 0U), 255U);
	EXPECT_EQ(pixel(15U, 4U, 1U), 0U);
	EXPECT_EQ(pixel(15U, 4U, 3U), 255U);
	EXPECT_EQ(pixel(16U, 4U, 3U), 0U);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, TreeAndPropertyRowsKeepCallerStateAndOwnKeyboardFocus)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	const oa::PixelRect rootRect{10, 20, 210, 24};
	const oa::PixelRect childRect{10, 44, 210, 24};
	const oa::UiTreeRowConfig closedRoot{
		.hasChildren = true,
		.open = false,
	};

	// Establish the prior rendered navigation order without requiring text.
	ui.beginFrame(16.0F);
	oa::UiTreeRowResult root = ui.treeRow("root", rootRect, {}, closedRoot);
	oa::UiTreeRowResult child = ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_EQ(root.disclosure.x, 16);
	EXPECT_EQ(root.disclosure.w, 18);
	EXPECT_EQ(root.label.x, 34);
	EXPECT_EQ(child.disclosure.x, 32);
	ui.endFrame();

	// Extreme but valid indentation clamps before narrowing or adding padding.
	ui.beginFrame(16.0F);
	root = ui.treeRow(
		"deep", rootRect, {},
		{.depth = std::numeric_limits<oa::I32>::max(), .indent = 1});
	EXPECT_EQ(root.disclosure.x, rootRect.x + rootRect.w);
	EXPECT_EQ(root.disclosure.w, 0);
	ui.endFrame();

	// disclosure activation requests a state transition but does not retain it.
	oa::UiEvent pointer;
	pointer.type = oa::UiEventType::MouseDown;
	pointer.button = 1;
	pointer.mouseX = 20.0F;
	pointer.mouseY = 30.0F;
	ui.beginFrame(16.0F);
	EXPECT_FALSE(ui.routeEvent(pointer));
	root = ui.treeRow("root", rootRect, {}, closedRoot);
	(void)ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_FALSE(root.openChanged);
	ui.endFrame();

	pointer.type = oa::UiEventType::MouseUp;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(pointer));
	root = ui.treeRow("root", rootRect, {}, closedRoot);
	(void)ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_TRUE(root.openChanged);
	EXPECT_TRUE(root.open);
	EXPECT_FALSE(root.activated);
	ui.endFrame();

	// Clicking the label activates selection without changing caller-owned open.
	pointer.type = oa::UiEventType::MouseDown;
	pointer.mouseX = 80.0F;
	ui.beginFrame(16.0F);
	EXPECT_FALSE(ui.routeEvent(pointer));
	(void)ui.treeRow("root", rootRect, {}, closedRoot);
	(void)ui.treeRow("child", childRect, {}, {.depth = 1});
	ui.endFrame();
	pointer.type = oa::UiEventType::MouseUp;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(pointer));
	root = ui.treeRow("root", rootRect, {}, closedRoot);
	(void)ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_TRUE(root.activated);
	EXPECT_FALSE(root.openChanged);
	ui.endFrame();

	// Right opens, Down selects the next rendered row, Up returns to the root,
	// and Left closes. The caller applies each returned transition explicitly.
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Right;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	root = ui.treeRow("root", rootRect, {}, closedRoot);
	child = ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_TRUE(root.openChanged);
	EXPECT_TRUE(root.open);
	EXPECT_FALSE(child.activated);
	ui.endFrame();

	key.key = oa::UiKey::Down;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	root = ui.treeRow("root", rootRect, {}, {.hasChildren = true, .open = true});
	child = ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_FALSE(root.activated);
	EXPECT_TRUE(child.activated);
	ui.endFrame();

	key.key = oa::UiKey::Up;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	root = ui.treeRow("root", rootRect, {}, {.hasChildren = true, .open = true});
	child = ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_TRUE(root.activated);
	EXPECT_FALSE(child.activated);
	ui.endFrame();

	key.key = oa::UiKey::Left;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	root = ui.treeRow("root", rootRect, {}, {.hasChildren = true, .open = true});
	(void)ui.treeRow("child", childRect, {}, {.depth = 1});
	EXPECT_TRUE(root.openChanged);
	EXPECT_FALSE(root.open);
	ui.endFrame();

	// Property rows return stable compact geometry for either read-only values
	// or a caller-mounted editor control.
	ui.beginFrame(16.0F);
	const oa::UiPropertyRegion property = ui.propertyRow(
		"learning-rate",
		{10, 80, 210, 22},
		{},
		{},
		{.labelFraction = 0.40F, .gap = 10, .paddingX = 4});
	EXPECT_EQ(property.label.x, 10);
	EXPECT_EQ(property.label.w, 80);
	EXPECT_EQ(property.value.x, 100);
	EXPECT_EQ(property.value.w, 120);
	ui.endFrame();

	// Ancestor clipping owns hit testing for explicit virtualized rows.
	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseDown;
	pointer.mouseX = 55.0F;
	pointer.mouseY = 30.0F;
	EXPECT_FALSE(ui.routeEvent(pointer));
	ui.beginPanel("clip", {0, 0, 20, 120});
	(void)ui.treeRow("clipped", {50, 20, 100, 24}, {}, {});
	ui.endPanel();
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();

	ui.beginFrame(16.0F);
	(void)ui.treeRow("invalid", rootRect, {}, {.depth = -1});
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ui.beginFrame(16.0F);
	(void)ui.propertyRow(
		"invalid", rootRect, {}, {},
		{.labelFraction = std::numeric_limits<oa::F32>::quiet_NaN()});
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ui.beginFrame(16.0F);
	(void)ui.propertyRow(
		"overflow", rootRect, {}, {},
		{.gap = std::numeric_limits<oa::I32>::max()});
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, TreeAndPropertyRowsRenderSelectedDisclosureAndClipOnGpu)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 40U, 28U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	oa::UiStyle style = ui.currentStyle();
	style.framePaddingX = 0.0F;
	style.accent = {0.0F, 1.0F, 0.0F, 1.0F};
	style.textSecondary = {1.0F, 0.0F, 0.0F, 1.0F};
	style.surface = {0.0F, 0.0F, 1.0F, 1.0F};
	style.borderSubtle = {1.0F, 0.0F, 0.0F, 1.0F};

	ui.beginFrame(16.0F, {0, 0, 40, 28});
	ui.pushStyle(style);
	ui.beginPanel("clip", {0, 0, 20, 28});
	(void)ui.treeRow(
		"selected",
		{4, 2, 28, 10},
		{},
		{
			.disclosureWidth = 8,
			.hasChildren = true,
			.selected = true,
		});
	(void)ui.propertyRow(
		"property",
		{4, 14, 28, 10},
		{},
		{},
		{
			.labelFraction = 0.5F,
			.gap = 4,
			.paddingX = 0,
			.alternate = true,
		});
	ui.endPanel();
	ui.popStyle();

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 40U * 28U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	const auto pixel = [&](oa::U32 inX, oa::U32 inY, oa::U32 inChannel) {
		return pixels[(inY * 40U + inX) * 4U + inChannel];
	};
	// Selected fill, disclosure plus, alternating property fill and separator.
	EXPECT_EQ(pixel(10U, 4U, 0U), 0U);
	EXPECT_EQ(pixel(10U, 4U, 1U), 71U);
	EXPECT_EQ(pixel(10U, 4U, 3U), 71U);
	EXPECT_EQ(pixel(8U, 7U, 0U), 255U);
	EXPECT_EQ(pixel(8U, 7U, 1U), 0U);
	EXPECT_EQ(pixel(8U, 7U, 3U), 255U);
	EXPECT_EQ(pixel(10U, 16U, 2U), 148U);
	EXPECT_EQ(pixel(10U, 16U, 3U), 148U);
	EXPECT_EQ(pixel(18U, 16U, 0U), 255U);
	EXPECT_EQ(pixel(18U, 16U, 2U), 0U);
	EXPECT_EQ(pixel(18U, 16U, 3U), 255U);
	EXPECT_EQ(pixel(20U, 16U, 3U), 0U);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, TabBarsKeepWorkspaceStateAndReturnCloseReorderRequests)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	const oa::PixelRect rect{10, 20, 240, 24};
	const oa::Array<oa::UiTabItem, 5> items{
		oa::UiTabItem{.id = "a", .label = {}, .dirty = true},
		oa::UiTabItem{.id = "b", .label = {}},
		oa::UiTabItem{.id = "c", .label = {}},
		oa::UiTabItem{.id = "d", .label = {}, .closable = false},
		oa::UiTabItem{.id = "e", .label = {}},
	};
	const oa::UiTabBarConfig config{
		.minimumTabWidth = 80,
		.maximumTabWidth = 120,
		.overflowButtonWidth = 20,
		.closeWidth = 20,
	};
	oa::UiTabBarState state{.selected = 0};

	ui.beginFrame(16.0F);
	oa::UiTabBarResult tabs = ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	EXPECT_EQ(tabs.tabs.x, 30);
	EXPECT_EQ(tabs.tabs.w, 200);
	EXPECT_EQ(tabs.firstVisible, 0);
	EXPECT_EQ(tabs.onePastLast, 2);
	ui.endFrame();

	// Full-tab activation changes only caller-owned selection state.
	oa::UiEvent pointer;
	pointer.type = oa::UiEventType::MouseDown;
	pointer.button = 1;
	pointer.mouseX = 130.0F;
	pointer.mouseY = 30.0F;
	ui.beginFrame(16.0F);
	EXPECT_FALSE(ui.routeEvent(pointer));
	(void)ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	ui.endFrame();
	pointer.type = oa::UiEventType::MouseUp;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(pointer));
	tabs = ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	EXPECT_TRUE(tabs.selectionChanged);
	EXPECT_EQ(tabs.activatedIndex, 1);
	EXPECT_EQ(state.selected, 1);
	ui.endFrame();

	// keyboard navigation crosses the overflow boundary and brings the target
	// into the caller-owned visible window. ctrl+W returns a close request.
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Right;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	tabs = ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	EXPECT_EQ(state.selected, 2);
	EXPECT_EQ(state.firstVisible, 1);
	EXPECT_EQ(tabs.activatedIndex, 2);
	ui.endFrame();

	key.key = oa::UiKey::W;
	key.modifiers = oa::UiModifierCtrl;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	tabs = ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	EXPECT_EQ(tabs.closeRequestedIndex, 2);
	EXPECT_EQ(state.selected, 2);
	ui.endFrame();
	key.modifiers = oa::UiModifierNone;

	// The close affordance does not also activate its tab.
	state = {.selected = 1, .firstVisible = 0};
	pointer.type = oa::UiEventType::MouseDown;
	pointer.mouseX = 100.0F;
	ui.beginFrame(16.0F);
	(void)ui.routeEvent(pointer);
	(void)ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	ui.endFrame();
	pointer.type = oa::UiEventType::MouseUp;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(pointer));
	tabs = ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	EXPECT_EQ(tabs.closeRequestedIndex, 0);
	EXPECT_EQ(tabs.activatedIndex, -1);
	EXPECT_EQ(state.selected, 1);
	ui.endFrame();

	// Dragging one visible stable item onto another returns an order request;
	// OA never mutates the item sequence.
	pointer.type = oa::UiEventType::MouseDown;
	pointer.mouseX = 50.0F;
	ui.beginFrame(16.0F);
	(void)ui.routeEvent(pointer);
	(void)ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	ui.endFrame();
	pointer.type = oa::UiEventType::MouseMove;
	pointer.mouseX = 130.0F;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(pointer));
	(void)ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	ui.endFrame();
	pointer.type = oa::UiEventType::MouseUp;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(pointer));
	tabs = ui.tabBar(
		"workspace", rect, oa::Span<const oa::UiTabItem>(items), state, config);
	EXPECT_EQ(tabs.moveFromIndex, 0);
	EXPECT_EQ(tabs.moveToIndex, 1);
	EXPECT_EQ(tabs.activatedIndex, -1);
	ui.endFrame();

	const oa::Array<oa::UiTabItem, 2> duplicate{
		oa::UiTabItem{.id = "same", .label = {}},
		oa::UiTabItem{.id = "same", .label = {}},
	};
	state = {.selected = 0};
	ui.beginFrame(16.0F);
	(void)ui.tabBar(
		"invalid", rect, oa::Span<const oa::UiTabItem>(duplicate), state);
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, TabBarsRenderSelectionDirtyCloseAndInheritedClipOnGpu)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 64U, 24U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	oa::UiStyle style = ui.currentStyle();
	style.background = {0.0F, 0.0F, 1.0F, 1.0F};
	style.surface = {0.2F, 0.2F, 0.2F, 1.0F};
	style.surfaceActive = {0.0F, 1.0F, 0.0F, 1.0F};
	style.accent = {0.0F, 1.0F, 1.0F, 1.0F};
	style.warning = {1.0F, 1.0F, 0.0F, 1.0F};
	style.textMuted = {1.0F, 0.0F, 0.0F, 1.0F};
	style.borderSubtle = {1.0F, 0.0F, 1.0F, 1.0F};
	const oa::Array<oa::UiTabItem, 2> items{
		oa::UiTabItem{.id = "selected", .label = {}, .dirty = true},
		oa::UiTabItem{.id = "other", .label = {}},
	};
	oa::UiTabBarState state{.selected = 0};

	ui.beginFrame(16.0F, {0, 0, 64, 24});
	ui.pushStyle(style);
	ui.beginPanel("clip", {0, 0, 48, 24});
	(void)ui.tabBar(
		"tabs",
		{4, 2, 56, 18},
		oa::Span<const oa::UiTabItem>(items),
		state,
		{
			.minimumTabWidth = 24,
			.maximumTabWidth = 24,
			.overflowButtonWidth = 8,
			.closeWidth = 8,
		});
	ui.endPanel();
	ui.popStyle();

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 64U * 24U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	const auto pixel = [&](oa::U32 inX, oa::U32 inY, oa::U32 inChannel) {
		return pixels[(inY * 64U + inX) * 4U + inChannel];
	};
	// Selected green body, yellow dirty mark, red close glyph, cyan selected
	// underline, magenta separator, and the ancestor's right-exclusive clip.
	EXPECT_EQ(pixel(10U, 6U, 1U), 255U);
	EXPECT_EQ(pixel(15U, 10U, 0U), 255U);
	EXPECT_EQ(pixel(15U, 10U, 1U), 255U);
	EXPECT_EQ(pixel(24U, 11U, 0U), 255U);
	EXPECT_EQ(pixel(24U, 11U, 1U), 0U);
	EXPECT_EQ(pixel(10U, 18U, 1U), 255U);
	EXPECT_EQ(pixel(10U, 18U, 2U), 255U);
	EXPECT_EQ(pixel(27U, 6U, 0U), 255U);
	EXPECT_EQ(pixel(27U, 6U, 2U), 255U);
	EXPECT_EQ(pixel(48U, 6U, 3U), 0U);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, AccessibilitySnapshotExportsStableClippedRolesStatesAndValues)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	oa::UiLayout layout;
	layout.padding = oa::UiEdge{};
	layout.gap = 0.0F;
	bool checked = true;
	oa::F32 amount = 0.25F;
	oa::String name("worker");
	const auto drawControls = [&] {
		ui.beginPanel("outer", {5, 0, 80, 120}, layout);
		ui.beginPanel("controls", {0, 0, 160, 120}, layout);
		(void)ui.button("run");
		(void)ui.checkbox("enabled", checked);
		(void)ui.sliderF32("Amount", &amount, 0.0F, 1.0F, "%.2f");
		(void)ui.inputText("Name", name);
		ui.endPanel();
		ui.endPanel();
	};
	const auto hasState = [](const oa::UiAccessibilityNode& inNode,
		oa::UiAccessibilityState inState) {
		return (static_cast<oa::U32>(inNode.state)
			& static_cast<oa::U32>(inState)) != 0U;
	};
	const auto hasAction = [](const oa::UiAccessibilityNode& inNode,
		oa::UiAccessibilityAction inAction) {
		return (static_cast<oa::U32>(inNode.actions)
			& static_cast<oa::U32>(inAction)) != 0U;
	};

	ui.beginFrame(16.0F, {0, 0, 160, 120});
	drawControls();
	auto nodes = ui.accessibilitySnapshot();
	ASSERT_EQ(nodes.size(), 4U);
	EXPECT_EQ(nodes[0].role, oa::UiAccessibilityRole::Button);
	EXPECT_EQ(testStdString(nodes[0].label), "run");
	EXPECT_TRUE(hasAction(nodes[0], oa::UiAccessibilityAction::Activate));
	EXPECT_EQ(nodes[1].role, oa::UiAccessibilityRole::Checkbox);
	EXPECT_TRUE(hasState(nodes[1], oa::UiAccessibilityState::Checked));
	EXPECT_EQ(testStdString(nodes[1].value), "true");
	EXPECT_EQ(nodes[2].role, oa::UiAccessibilityRole::Slider);
	EXPECT_TRUE(nodes[2].hasNumericValue);
	EXPECT_DOUBLE_EQ(nodes[2].minimum, 0.0);
	EXPECT_DOUBLE_EQ(nodes[2].maximum, 1.0);
	EXPECT_DOUBLE_EQ(nodes[2].current, 0.25);
	EXPECT_TRUE(hasAction(nodes[2], oa::UiAccessibilityAction::SetValue));
	EXPECT_EQ(nodes[3].role, oa::UiAccessibilityRole::TextField);
	EXPECT_TRUE(hasState(nodes[3], oa::UiAccessibilityState::Editable));
	EXPECT_EQ(testStdString(nodes[3].value), "worker");
	oa::Array<oa::U32, 4> stableIds{};
	for (oa::Usize index = 0U; index < nodes.size(); ++index) {
		stableIds[index] = nodes[index].id;
		EXPECT_NE(nodes[index].id, 0U);
		EXPECT_EQ(nodes[index].scope, nodes[0].scope);
		EXPECT_GE(nodes[index].bounds.x, 5);
		EXPECT_LE(nodes[index].bounds.x + nodes[index].bounds.w, 85);
	}
	ui.endFrame();

	oa::UiEvent tab;
	tab.type = oa::UiEventType::KeyDown;
	tab.key = oa::UiKey::Tab;
	ui.beginFrame(16.0F, {0, 0, 160, 120});
	EXPECT_TRUE(ui.routeEvent(tab));
	drawControls();
	nodes = ui.accessibilitySnapshot();
	ASSERT_EQ(nodes.size(), 4U);
	for (oa::Usize index = 0U; index < nodes.size(); ++index) {
		EXPECT_EQ(nodes[index].id, stableIds[index]);
	}
	EXPECT_TRUE(hasState(nodes[0], oa::UiAccessibilityState::Focused));
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, ContentScaleAndKeyboardTimelineUsePhysicalMetrics)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	oa::UiLayout layout;
	layout.padding = oa::UiEdge{};
	layout.gap = 0.0F;
	const auto drawButton = [&] {
		ui.beginPanel("scale", {0, 0, 200, 100}, layout);
		(void)ui.button("scale");
		ui.endPanel();
		return ui.accessibilitySnapshot()[0].bounds;
	};

	ui.beginFrame(16.0F, {0, 0, 200, 100}, 1.0F);
	EXPECT_FLOAT_EQ(ui.contentScale(), 1.0F);
	EXPECT_FLOAT_EQ(ui.currentStyle().fontSize, 14.0F);
	const oa::PixelRect unit = drawButton();
	ui.endFrame();
	ui.beginFrame(16.0F, {0, 0, 200, 100}, 2.0F);
	EXPECT_FLOAT_EQ(ui.contentScale(), 2.0F);
	EXPECT_FLOAT_EQ(ui.currentStyle().fontSize, 28.0F);
	const oa::PixelRect doubled = drawButton();
	EXPECT_GE(doubled.h, unit.h * 2 - 1);
	ui.endFrame();

	oa::F32 fraction = 0.5F;
	const auto drawTimeline = [&] {
		return ui.timeline("Transport", {0, 0, 200, 12}, fraction);
	};
	ui.beginFrame(16.0F, {0, 0, 200, 20}, 2.0F);
	EXPECT_FALSE(drawTimeline());
	ASSERT_EQ(ui.accessibilitySnapshot().size(), 1U);
	EXPECT_EQ(ui.accessibilitySnapshot()[0].role,
		oa::UiAccessibilityRole::Timeline);
	ui.endFrame();
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Tab;
	ui.beginFrame(16.0F, {0, 0, 200, 20}, 2.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawTimeline());
	ui.endFrame();
	key.key = oa::UiKey::Right;
	ui.beginFrame(16.0F, {0, 0, 200, 20}, 2.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawTimeline());
	EXPECT_NEAR(fraction, 0.51F, 1.0e-6F);
	ui.endFrame();

	ui.beginFrame(
		16.0F,
		{0, 0, 200, 100},
		std::numeric_limits<oa::F32>::quiet_NaN());
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, NestedClipPreservesGeometryAndImageSourceCoordinatesOnGpu)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const oa::Array<oa::U8, 16> sourcePixels{
		255U, 0U, 0U, 255U,
		0U, 255U, 0U, 255U,
		0U, 0U, 255U, 255U,
		255U, 255U, 255U, 255U,
	};
	auto sourceResult = oa::FnTexture::fromPixels(
		*engine,
		oa::Span<const oa::U8>(sourcePixels.data(), sourcePixels.size()),
		4,
		1);
	ASSERT_TRUE(sourceResult.isOk()) << sourceResult.getStatus().toString();
	oa::Texture source = oa::move(*sourceResult);

	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 4U, 3U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	ui.beginFrame(16.0F);
	ui.beginPanel("outer-clip", {1, 0, 2, 3});
	ui.beginPanel("negative-image", {-2, 0, 4, 1});
	ui.image(source);
	ui.endPanel();
	ui.rect({-2, 1, 4, 1}, {1.0F, 0.0F, 0.0F, 1.0F});
	ui.rectOutline({-2, 2, 4, 1}, {0.0F, 1.0F, 0.0F, 1.0F}, 1U);
	ui.endPanel();

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk())
		<< streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 48> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	const oa::Array<oa::U8, 48> expected{
		0U, 0U, 0U, 0U,
		255U, 255U, 255U, 255U,
		0U, 0U, 0U, 0U,
		0U, 0U, 0U, 0U,
		0U, 0U, 0U, 0U,
		255U, 0U, 0U, 255U,
		0U, 0U, 0U, 0U,
		0U, 0U, 0U, 0U,
		0U, 0U, 0U, 0U,
		0U, 255U, 0U, 255U,
		0U, 0U, 0U, 0U,
		0U, 0U, 0U, 0U,
	};
	EXPECT_EQ(pixels, expected);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, CompletionTrackedUploadValuesRetireThroughExactEvents)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const auto& device = oa::EngineDeviceAccess::get(*engine);
	auto gateResult = oavk::TimelineSemaphore::create(device, 0U);
	ASSERT_TRUE(gateResult.isOk()) << gateResult.getStatus().toString();
	oavk::TimelineSemaphore gate = oa::move(*gateResult);
	const oa::Event completion = oa::EventAccess::create(device, gate, 1U);

	{
		auto detections = oa::DetectionBuffer::createHostUpload(*engine, 4U);
		auto glyphs = oa::GlyphBuffer::createHostUpload(*engine, 4U);
		const oa::Array<oa::U8, 1> gray = {255U};
		const oa::Array<oa::Span<const oa::U8>, 1> planeBytes = {
			oa::Span<const oa::U8>(gray.data(), gray.size())};
		const oa::Array<oa::ImageDtype, 1> planeDtypes = {
			oa::ImageDtype::U8};
		auto planes = oa::ImagePlanes::fromPlanes(
			*engine,
			oa::Span<const oa::Span<const oa::U8>>(
				planeBytes.data(), planeBytes.size()),
			oa::Span<const oa::ImageDtype>(
				planeDtypes.data(), planeDtypes.size()),
			1,
			1);
		ASSERT_TRUE(detections.isOk()) << detections.getStatus().toString();
		ASSERT_TRUE(glyphs.isOk()) << glyphs.getStatus().toString();
		ASSERT_TRUE(planes.isOk()) << planes.getStatus().toString();
		EXPECT_EQ(
			detections->markConsumed({}).getCode(),
			oa::StatusCode::InvalidArgument);
		EXPECT_EQ(
			glyphs->markConsumed({}).getCode(),
			oa::StatusCode::InvalidArgument);
		EXPECT_EQ(
			planes->markConsumed({}).getCode(),
			oa::StatusCode::InvalidArgument);
		ASSERT_TRUE(detections->markConsumed(completion).isOk());
		ASSERT_TRUE(glyphs->markConsumed(completion).isOk());
		ASSERT_TRUE(planes->markConsumed(completion).isOk());
		EXPECT_FALSE(detections->isReady());
		EXPECT_FALSE(glyphs->isReady());
		*detections = {};
		*glyphs = {};
		*planes = {};
	}

	VkSemaphoreSignalInfo signal{};
	signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	signal.semaphore = static_cast<VkSemaphore>(gate.semaphore);
	signal.value = 1U;
	ASSERT_EQ(
		device.deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(device.device), &signal),
		VK_SUCCESS);
	ASSERT_TRUE(completion.isComplete());
	ASSERT_TRUE(
		oa::EngineAccess(*engine).completeRetiredBorrowedServices().isOk());
	gate.destroy(device);
}

TEST_VK(TestUi, PlanarImageValidatesOwnershipAndRendersWholeAndSinglePlanes)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const oa::Array<oa::U8, 2> red = {255U, 0U};
	const oa::Array<oa::U8, 2> green = {0U, 255U};
	const oa::Array<oa::U8, 2> blue = {0U, 0U};
	const oa::Array<oa::Span<const oa::U8>, 3> planeBytes = {
		oa::Span<const oa::U8>(red.data(), red.size()),
		oa::Span<const oa::U8>(green.data(), green.size()),
		oa::Span<const oa::U8>(blue.data(), blue.size()),
	};
	const oa::Array<oa::ImageDtype, 3> planeDtypes = {
		oa::ImageDtype::U8, oa::ImageDtype::U8, oa::ImageDtype::U8};
	auto planes = oa::ImagePlanes::fromPlanes(
		*engine,
		oa::Span<const oa::Span<const oa::U8>>(
			planeBytes.data(), planeBytes.size()),
		oa::Span<const oa::ImageDtype>(
			planeDtypes.data(), planeDtypes.size()),
		2,
		1);
	ASSERT_TRUE(planes.isOk()) << planes.getStatus().toString();
	EXPECT_EQ(planes->width(), 2);
	EXPECT_EQ(planes->height(), 1);
	EXPECT_EQ(planes->channelCount(), 3U);
	const oa::Array<oa::U8, 2> gray = {64U, 192U};
	const oa::Array<oa::U8, 2> alpha = {128U, 255U};
	const oa::Array<oa::Span<const oa::U8>, 2> grayAlphaBytes = {
		oa::Span<const oa::U8>(gray.data(), gray.size()),
		oa::Span<const oa::U8>(alpha.data(), alpha.size()),
	};
	const oa::Array<oa::ImageDtype, 2> grayAlphaDtypes = {
		oa::ImageDtype::U8, oa::ImageDtype::U8};
	auto grayAlpha = oa::ImagePlanes::fromPlanes(
		*engine,
		oa::Span<const oa::Span<const oa::U8>>(
			grayAlphaBytes.data(), grayAlphaBytes.size()),
		oa::Span<const oa::ImageDtype>(
			grayAlphaDtypes.data(), grayAlphaDtypes.size()),
		2,
		1);
	ASSERT_TRUE(grayAlpha.isOk()) << grayAlpha.getStatus().toString();

	const oa::Array<oa::Span<const oa::U8>, 1> shortPlane = {
		oa::Span<const oa::U8>(red.data(), 1U)};
	const oa::Array<oa::ImageDtype, 1> oneDtype = {oa::ImageDtype::U8};
	EXPECT_EQ(
		oa::ImagePlanes::fromPlanes(
			*engine,
			oa::Span<const oa::Span<const oa::U8>>(
				shortPlane.data(), shortPlane.size()),
			oa::Span<const oa::ImageDtype>(
				oneDtype.data(), oneDtype.size()),
			2,
			1).getStatus().getCode(),
		oa::StatusCode::InvalidArgument);

	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 2U, 3U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	ui.beginFrame(16.0F);
	ui.imagePlanar(*planes, 0, 0);
	ui.imagePlane(*planes, 1U, 0, 1);
	ui.imagePlanar(*grayAlpha, 0, 2);

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk())
		<< streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 24> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	const oa::Array<oa::U8, 24> expected = {
		255U, 0U, 0U, 255U,
		0U, 255U, 0U, 255U,
		0U, 0U, 0U, 255U,
		255U, 255U, 255U, 255U,
		64U, 64U, 64U, 128U,
		192U, 192U, 192U, 255U,
	};
	EXPECT_EQ(pixels, expected);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
	*planes = {};
	*grayAlpha = {};
}

TEST_VK(TestUi, RendererAbandonmentRetiresThroughExactFrameEvent)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const auto& device = oa::EngineDeviceAccess::get(*engine);
	auto gateResult = oavk::TimelineSemaphore::create(device, 0U);
	ASSERT_TRUE(gateResult.isOk()) << gateResult.getStatus().toString();
	oavk::TimelineSemaphore gate = oa::move(*gateResult);
	const oa::Event completion = oa::EventAccess::create(device, gate, 1U);

	{
		oa::Ui ui;
		ASSERT_TRUE(ui.init(*engine).isOk());
		EXPECT_EQ(ui.init(*engine).getCode(), oa::StatusCode::AlreadyExists);
		ASSERT_TRUE(ui.initBlit(nullptr).isOk());
		ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
		EXPECT_FALSE(completion.isComplete());
	}

	VkSemaphoreSignalInfo signal{};
	signal.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	signal.semaphore = static_cast<VkSemaphore>(gate.semaphore);
	signal.value = 1U;
	ASSERT_EQ(
		device.deviceDispatch.vkSignalSemaphore(
			static_cast<VkDevice>(device.device), &signal),
		VK_SUCCESS);
	ASSERT_TRUE(completion.isComplete());
	ASSERT_TRUE(
		oa::EngineAccess(*engine).completeRetiredBorrowedServices().isOk());
	gate.destroy(device);
}

TEST_VK(TestUi, RendererUiModeProducesBoundedGenerationSafeTextureFrames)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::UiRenderConfig config;
	config.width_ = 4U;
	config.height_ = 4U;
	config.targetSlotCount_ = 2U;
	config.style_ = oa::UiStyle::editorDark();
	config.style_.background = {0.0F, 0.0F, 0.0F, 1.0F};
	auto created = oa::Renderer::create(*engine, config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto renderer = oa::move(*created);
	ASSERT_NE(renderer->ui(), nullptr);
	oa::MeshData mesh;
	oa::CameraState camera;
	EXPECT_EQ(
		renderer->beginFrame(mesh, camera).getCode(),
		oa::StatusCode::FailedPrecondition);

	ASSERT_TRUE(renderer->beginFrame(16.0F).isOk());
	renderer->ui()->rect({1, 1, 2, 2}, {1.0F, 0.0F, 0.0F, 1.0F});
	auto firstResult = renderer->submitFrame();
	ASSERT_TRUE(firstResult.isOk()) << firstResult.getStatus().toString();
	const oa::RenderFrame first = *firstResult;
	EXPECT_EQ(first.width(), 4U);
	EXPECT_EQ(first.height(), 4U);
	EXPECT_TRUE(first.color().isImageBacked());
	EXPECT_TRUE(first.producer().isValid());

	ASSERT_TRUE(renderer->beginFrame(16.0F).isOk());
	renderer->ui()->rect({0, 0, 4, 4}, {0.0F, 1.0F, 0.0F, 1.0F});
	auto secondResult = renderer->submitFrame();
	ASSERT_TRUE(secondResult.isOk()) << secondResult.getStatus().toString();
	const oa::RenderFrame second = *secondResult;
	EXPECT_EQ(
		renderer->beginFrame(16.0F).getCode(),
		oa::StatusCode::ResourceExhausted);

	auto readbackResult = renderer->consumeReadback(first);
	ASSERT_TRUE(readbackResult.isOk())
		<< readbackResult.getStatus().toString();
	const oa::RenderReadback& readback = *readbackResult;
	ASSERT_EQ(readback.colorRgba8_.size(), 4U * 4U * 4U);
	const auto pixel = [&](oa::U32 inX, oa::U32 inY, oa::U32 inChannel) {
		return readback.colorRgba8_[
			(static_cast<std::size_t>(inY) * 4U + inX) * 4U + inChannel];
	};
	EXPECT_EQ(pixel(0U, 0U, 0U), 0U);
	EXPECT_EQ(pixel(0U, 0U, 3U), 255U);
	EXPECT_EQ(pixel(1U, 1U, 0U), 255U);
	EXPECT_EQ(pixel(1U, 1U, 1U), 0U);
	EXPECT_EQ(pixel(1U, 1U, 3U), 255U);
	EXPECT_EQ(
		renderer->consumeReadback(first).getStatus().getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_EQ(
		renderer->resize(8U, 6U).getCode(),
		oa::StatusCode::FailedPrecondition);

	auto consumerResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(consumerResult.isOk())
		<< consumerResult.getStatus().toString();
	oavk::Stream consumer = oa::move(*consumerResult);
	ASSERT_TRUE(consumer.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	VkImageMemoryBarrier2 consumeBarrier{};
	consumeBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
	consumeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
	consumeBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
	consumeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
	consumeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
	consumeBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	consumeBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	consumeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	consumeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	consumeBarrier.image = oa::TextureAccess::image(second.color());
	consumeBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	consumeBarrier.subresourceRange.levelCount = 1U;
	consumeBarrier.subresourceRange.layerCount = 1U;
	VkDependencyInfo dependency{};
	dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	dependency.imageMemoryBarrierCount = 1U;
	dependency.pImageMemoryBarriers = &consumeBarrier;
	oa::EngineDeviceAccess::get(*engine).deviceDispatch.vkCmdPipelineBarrier2(
		static_cast<VkCommandBuffer>(consumer.commandBuffer), &dependency);
	const oavk::TimelineWait producerWait =
		oa::EventAccess::timelineWait(second.producer());
	ASSERT_TRUE(consumer.submitWithDependencies(
		*engine,
		oa::Span<const oavk::TimelineWait>(&producerWait, 1U)).isOk());
	const oa::Event consumed = consumer.completion(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(renderer->markConsumed(second, consumed).isOk());
	ASSERT_TRUE(consumed.wait().isOk());
	ASSERT_TRUE(renderer->collect().isOk());
	ASSERT_TRUE(renderer->resize(8U, 6U).isOk());
	EXPECT_EQ(
		renderer->abandonFrame(second).getCode(),
		oa::StatusCode::InvalidArgument);

	ASSERT_TRUE(renderer->beginFrame(16.0F, 2.0F).isOk());
	EXPECT_FLOAT_EQ(renderer->ui()->contentScale(), 2.0F);
	ASSERT_TRUE(renderer->cancelFrame().isOk());
	ASSERT_TRUE(renderer->close().isOk());
	ASSERT_TRUE(renderer->close().isOk());
	ASSERT_TRUE(consumer.synchronize(
		oa::EngineDeviceAccess::get(*engine)).isOk());
	consumer.destroy(oa::EngineDeviceAccess::get(*engine));
}

TEST_F(TestUi, RgbaFileSinkValidatesAndEncodesOneExactHostImage)
{
	const oa::Array<oa::U8, 16U> rgba{
		255U, 0U, 0U, 255U,
		0U, 255U, 0U, 255U,
		0U, 0U, 255U, 255U,
		255U, 255U, 255U, 255U,
	};
	const oa::Span<const oa::U8> pixels(rgba.data(), rgba.size());
	EXPECT_EQ(
		oa::FnImage::saveRgbaFile(
			pixels, 0U, 2U, "/tmp/oa_ui_invalid.png").getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_EQ(
		oa::FnImage::saveRgbaFile(
			oa::Span<const oa::U8>(rgba.data(), rgba.size() - 1U),
			2U, 2U, "/tmp/oa_ui_invalid.png").getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_EQ(
		oa::FnImage::saveRgbaFile(
			pixels, 2U, 2U, "/tmp/oa_ui_invalid.png", 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	EXPECT_EQ(
		oa::FnImage::saveRgbaFile(
			pixels, 2U, 2U, "/tmp/oa_ui_invalid.gif").getCode(),
		oa::StatusCode::InvalidArgument);

	const char* path = "/tmp/oa_ui_rgba_sink.png";
	(void)std::remove(path);
	ASSERT_TRUE(oa::FnImage::saveRgbaFile(pixels, 2U, 2U, path).isOk());
	std::ifstream file(path, std::ios::binary);
	ASSERT_TRUE(file.good());
	oa::Array<unsigned char, 8U> signature{};
	file.read(
		reinterpret_cast<char*>(signature.data()),
		static_cast<std::streamsize>(signature.size()));
	EXPECT_EQ(
		signature,
		(oa::Array<unsigned char, 8U>{
			0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU}));
	file.close();
	EXPECT_EQ(std::remove(path), 0);
}

TEST_VK(TestUi, RendererUiModeSaveToConsumesItsExactFrame)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::UiRenderConfig config;
	config.width_ = 4U;
	config.height_ = 3U;
	config.targetSlotCount_ = 1U;
	config.style_.background = {0.0F, 0.0F, 0.0F, 1.0F};
	auto created = oa::Renderer::create(*engine, config);
	ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
	auto renderer = oa::move(*created);
	ASSERT_TRUE(renderer->beginFrame(16.0F).isOk());
	renderer->ui()->rect({0, 0, 4, 3}, {0.0F, 0.0F, 1.0F, 1.0F});
	auto submitted = renderer->submitFrame();
	ASSERT_TRUE(submitted.isOk()) << submitted.getStatus().toString();
	const oa::RenderFrame frame = *submitted;

	const char* path = "/tmp/oa_renderer_ui_save_to.png";
	(void)std::remove(path);
	ASSERT_TRUE(renderer->saveTo(frame, path).isOk());
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(file.good());
	EXPECT_GT(file.tellg(), 8);
	file.close();
	EXPECT_EQ(std::remove(path), 0);
	EXPECT_EQ(
		renderer->consumeReadback(frame).getStatus().getCode(),
		oa::StatusCode::InvalidArgument);
	ASSERT_TRUE(renderer->close().isOk());
}

TEST_VK(TestUi, RendererUiModeAbandonmentRetiresSubmittedTargetsThroughEngine)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Event producer;
	{
		oa::UiRenderConfig config;
		config.width_ = 8U;
		config.height_ = 8U;
		config.targetSlotCount_ = 1U;
		auto created = oa::Renderer::create(*engine, config);
		ASSERT_TRUE(created.isOk()) << created.getStatus().toString();
		auto renderer = oa::move(*created);
		ASSERT_TRUE(renderer->beginFrame(16.0F).isOk());
		renderer->ui()->rect(
			{0, 0, 8, 8}, {0.0F, 0.0F, 1.0F, 1.0F});
		auto frame = renderer->submitFrame();
		ASSERT_TRUE(frame.isOk()) << frame.getStatus().toString();
		producer = frame->producer();
	}
	ASSERT_TRUE(producer.wait().isOk());
	ASSERT_TRUE(
		oa::EngineAccess(*engine).completeRetiredBorrowedServices().isOk());
}

TEST_VK(TestUi, HintedTextAtlasUsesExactFontsAndUtf8Clusters)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	EXPECT_EQ(atlas.init(*engine).getCode(), oa::StatusCode::AlreadyExists);

	const oa::GlyphInfo* sans13 = atlas.findGlyph(oa::FontId::Sans, 'M', 13.0F);
	const oa::GlyphInfo* mono13 = atlas.findGlyph(oa::FontId::Mono, 'M', 13.0F);
	ASSERT_NE(sans13, nullptr);
	ASSERT_NE(mono13, nullptr);
	EXPECT_FLOAT_EQ(sans13->rasterSize, 13.0F);
	EXPECT_FLOAT_EQ(mono13->rasterSize, 13.0F);
	EXPECT_NE(atlas.findGlyph(oa::FontId::Sans, 0x03A9U, 13.0F), nullptr);
	EXPECT_EQ(atlas.findGlyph(oa::FontId::Mono, 0x03A9U, 13.0F), nullptr);
	EXPECT_NE(atlas.findGlyph(oa::FontId::Mono, 0x2500U, 13.0F), nullptr);
	EXPECT_EQ(atlas.findGlyph(oa::FontId::Sans, 0x2500U, 13.0F), nullptr);

	const oa::GlyphInfo* monoI = atlas.findGlyph(oa::FontId::Mono, 'i', 14.0F);
	const oa::GlyphInfo* monoW = atlas.findGlyph(oa::FontId::Mono, 'W', 14.0F);
	const oa::GlyphInfo* sansI = atlas.findGlyph(oa::FontId::Sans, 'i', 14.0F);
	const oa::GlyphInfo* sansW = atlas.findGlyph(oa::FontId::Sans, 'W', 14.0F);
	ASSERT_NE(monoI, nullptr);
	ASSERT_NE(monoW, nullptr);
	ASSERT_NE(sansI, nullptr);
	ASSERT_NE(sansW, nullptr);
	EXPECT_FLOAT_EQ(monoI->advance, monoW->advance);
	EXPECT_NE(sansI->advance, sansW->advance);

	oa::TextLayout layout;
	oa::TextLayoutConfig config{.font = oa::FontId::Sans, .size = 14.0F};
	oa::Vector<oa::PositionedGlyph> glyphs;
	layout.shape(
		atlas,
		oa::StringView("A\xC3\xA9" "B"),
		{},
		config,
		0xFFFFFFFFU,
		glyphs);
	ASSERT_EQ(glyphs.size(), 3U);
	EXPECT_EQ(glyphs[0].codepoint, static_cast<oa::U32>('A'));
	EXPECT_EQ(glyphs[1].codepoint, 0x00E9U);
	EXPECT_EQ(glyphs[2].codepoint, static_cast<oa::U32>('B'));
	EXPECT_EQ(glyphs[0].cluster, 0U);
	EXPECT_EQ(glyphs[1].cluster, 1U);
	EXPECT_EQ(glyphs[2].cluster, 3U);

	glyphs.clear();
	layout.shape(
		atlas,
		oa::StringView("\xCE\xA9"),
		{},
		config,
		0xFFFFFFFFU,
		glyphs);
	ASSERT_EQ(glyphs.size(), 1U);
	EXPECT_EQ(glyphs[0].codepoint, 0x03A9U);
	EXPECT_EQ(glyphs[0].cluster, 0U);
	EXPECT_EQ(glyphs[0].font, oa::FontId::Sans);

	glyphs.clear();
	layout.shape(
		atlas,
		oa::StringView("\xCE\xA9" "\xE2\x94\x80"),
		{},
		{.font = oa::FontId::Sans, .size = 14.0F, .monospace = true},
		0xFFFFFFFFU,
		glyphs);
	ASSERT_EQ(glyphs.size(), 2U);
	EXPECT_EQ(glyphs[0].codepoint, 0x03A9U);
	EXPECT_EQ(glyphs[0].font, oa::FontId::Sans);
	EXPECT_EQ(glyphs[0].cluster, 0U);
	EXPECT_EQ(glyphs[1].codepoint, 0x2500U);
	EXPECT_EQ(glyphs[1].font, oa::FontId::Mono);
	EXPECT_EQ(glyphs[1].cluster, 2U);

	glyphs.clear();
	layout.shape(
		atlas,
		oa::StringView("e\xCC\x81"),
		{},
		config,
		0xFFFFFFFFU,
		glyphs);
	ASSERT_FALSE(glyphs.empty());
	for (const oa::PositionedGlyph& glyph : glyphs) {
		EXPECT_NE(glyph.codepoint, static_cast<oa::U32>('?'));
		EXPECT_EQ(glyph.cluster, 0U);
		EXPECT_EQ(glyph.font, oa::FontId::Sans);
	}

	glyphs.clear();
	layout.shape(
		atlas,
		oa::StringView("fi"),
		{},
		config,
		0xFFFFFFFFU,
		glyphs);
	ASSERT_EQ(glyphs.size(), 1U);
	EXPECT_EQ(glyphs[0].codepoint, 0xFB01U);
	EXPECT_EQ(glyphs[0].cluster, 0U);

	glyphs.clear();
	layout.shape(
		atlas,
		oa::StringView("\xC3"),
		{},
		config,
		0xFFFFFFFFU,
		glyphs);
	ASSERT_EQ(glyphs.size(), 1U);
	EXPECT_EQ(glyphs[0].codepoint, 0xFFFDU);
	EXPECT_EQ(glyphs[0].cluster, 0U);

	glyphs.clear();
	layout.shape(
		atlas,
		oa::StringView("\xF0\x9F\x98\x80"),
		{},
		config,
		0xFFFFFFFFU,
		glyphs);
	ASSERT_EQ(glyphs.size(), 1U);
	EXPECT_EQ(glyphs[0].codepoint, static_cast<oa::U32>('?'));
	EXPECT_EQ(glyphs[0].cluster, 0U);

	const oa::F32 kernedAv = layout.measure(atlas, oa::StringView("AV"), config).x;
	const oa::F32 separateAv = layout.measure(atlas, oa::StringView("A"), config).x
		+ layout.measure(atlas, oa::StringView("V"), config).x;
	EXPECT_LT(kernedAv, separateAv);

	const oa::F32 tabbed = layout.measure(atlas, oa::StringView("A\tB"), config).x;
	const oa::F32 unTabbed = layout.measure(atlas, oa::StringView("AB"), config).x;
	EXPECT_GT(tabbed, unTabbed);

	const oa::vlm::Vec2 monoNarrow = layout.measure(
		atlas, oa::StringView("iiii"),
		{.font = oa::FontId::Sans, .size = 14.0F, .monospace = true});
	const oa::vlm::Vec2 monoWide = layout.measure(
		atlas, oa::StringView("WWWW"),
		{.font = oa::FontId::Sans, .size = 14.0F, .monospace = true});
	EXPECT_FLOAT_EQ(monoNarrow.x, monoWide.x);
	atlas = {};
}

TEST_VK(TestUi, WidgetTextRequiresExplicitOwnershipAndProducesGlyphRanges)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());

	ui.beginFrame(16.0F);
	ui.beginPanel("unbound", {0, 0, 128, 32});
	ui.label("AB");
	ui.endPanel();
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endFrame();

	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	EXPECT_EQ(ui.bindTextAtlas(atlas).getCode(), oa::StatusCode::AlreadyExists);
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.recordRender(VK_NULL_HANDLE, 0U).isOk());
	ui.beginPanel("late-empty", {0, 0, 128, 32});
	ui.label("AB");
	ui.endPanel();
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endFrame();

	ui.beginFrame(16.0F);
	ui.label("AB");
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endFrame();

	ui.beginFrame(16.0F);
	ui.beginPanel("oversized", {0, 0, 128, 32});
	oa::String oversized;
	oversized.resize(16385U, 'A');
	ui.label(oversized);
	ui.endPanel();
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::ResourceExhausted);
	ui.endFrame();

	ui.beginFrame(16.0F);
	ui.beginPanel("prepared", {0, 0, 128, 32});
	ui.label("A");
	ui.endPanel();
	EXPECT_TRUE(ui.recordRender(VK_NULL_HANDLE, 0U).isOk());
	ui.beginPanel("late", {0, 0, 128, 32});
	ui.label("B");
	ui.endPanel();
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endFrame();

	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	ui.beginFrame(16.0F);
	oa::UiLayout labelLayout;
	labelLayout.padding = oa::UiEdge{2.0F, 4.0F};
	ui.beginPanel(
		"label", {0, 0, 128, 32}, labelLayout);
	ui.labelFmt("%s", "AB");
	ui.endPanel();

	auto& maximum = oa::EngineDeviceAccess::get(*engine).info.hardware.maxComputeWorkGroupCountX;
	const oa::U32 savedMaximum = maximum;
	ASSERT_GT(savedMaximum, 1U);
	struct RestoreLimit {
		oa::U32& Value;
		oa::U32 Saved;
		~RestoreLimit() { Value = Saved; }
	} restoreMaximum{maximum, savedMaximum};
	maximum = 1U;
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::OutOfRange);
	maximum = savedMaximum;
	ui.endFrame();
	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, TextRendererExecutesDistinctGlyphRangesOnGpu)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);

	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 128U, 32U).isOk());
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());

	oa::UiLayout layout;
	layout.padding = oa::UiEdge{2.0F};
	oa::UiStyle red = ui.currentStyle();
	red.text = {1.0F, 0.0F, 0.0F, 1.0F};
	oa::UiStyle green = red;
	green.text = {0.0F, 1.0F, 0.0F, 1.0F};

	ui.beginFrame(16.0F);
	ui.pushStyle(red);
	ui.beginPanel("red", {0, 0, 64, 32}, layout);
	ui.label("M");
	ui.endPanel();
	ui.popStyle();
	ui.pushStyle(green);
	ui.beginPanel("green", {64, 0, 64, 32}, layout);
	ui.label("W");
	ui.endPanel();
	ui.popStyle();

	auto streamResult = oavk::Stream::createCompute(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk())
		<< streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 128U * 32U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	oa::U32 leftRed = 0U;
	oa::U32 leftGreen = 0U;
	oa::U32 rightRed = 0U;
	oa::U32 rightGreen = 0U;
	for (oa::U32 y = 0U; y < 32U; ++y) {
		for (oa::U32 x = 0U; x < 128U; ++x) {
			const oa::Usize offset =
				(static_cast<oa::Usize>(y) * 128U + x) * 4U;
			if (pixels[offset + 3U] == 0U) continue;
			const bool redDominant = pixels[offset] > pixels[offset + 1U];
			const bool greenDominant = pixels[offset + 1U] > pixels[offset];
			if (x < 64U) {
				leftRed += redDominant ? 1U : 0U;
				leftGreen += greenDominant ? 1U : 0U;
			} else {
				rightRed += redDominant ? 1U : 0U;
				rightGreen += greenDominant ? 1U : 0U;
			}
		}
	}
	EXPECT_GT(leftRed, 8U);
	EXPECT_EQ(leftGreen, 0U);
	EXPECT_EQ(rightRed, 0U);
	EXPECT_GT(rightGreen, 8U);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, ButtonsCheckboxesRowsAndKeyboardFocusAreInteractive)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	bool checked = false;
	ui.beginFrame(16.0F);
	oa::UiEvent unownedTab;
	unownedTab.type = oa::UiEventType::KeyDown;
	unownedTab.key = oa::UiKey::Tab;
	EXPECT_FALSE(ui.routeEvent(unownedTab));
	ui.endFrame();

	const auto drawControls = [&] {
		oa::Array<bool, 3> activated{};
		oa::UiLayout layout;
		layout.padding = oa::UiEdge{4.0F};
		layout.gap = 6.0F;
		ui.beginPanel("controls", {0, 0, 240, 80}, layout);
		ui.beginRow("actions");
		activated[0] = ui.button("A");
		activated[1] = ui.button("BBBB");
		activated[2] = ui.checkbox("enabled", checked);
		ui.endRow();
		ui.endPanel();
		return activated;
	};

	ui.beginFrame(16.0F);
	oa::UiEvent down;
	down.type = oa::UiEventType::MouseDown;
	down.button = 1;
	down.mouseX = 50.0F;
	down.mouseY = 15.0F;
	EXPECT_FALSE(ui.routeEvent(down));
	auto activated = drawControls();
	EXPECT_EQ(activated, (oa::Array<bool, 3>{false, false, false}));
	EXPECT_NE(ui.input().activeId, 0U);
	EXPECT_EQ(ui.input().focusId, ui.input().activeId);
	ui.endFrame();

	ui.beginFrame(16.0F);
	oa::UiEvent up = down;
	up.type = oa::UiEventType::MouseUp;
	EXPECT_TRUE(ui.routeEvent(up));
	activated = drawControls();
	EXPECT_EQ(activated, (oa::Array<bool, 3>{false, true, false}));
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();

	// Press the first control, then release outside it: release-inside is the
	// activation contract, so the drag-out must not click.
	ui.beginFrame(16.0F);
	down.mouseX = 10.0F;
	EXPECT_FALSE(ui.routeEvent(down));
	activated = drawControls();
	EXPECT_FALSE(activated[0]);
	ui.endFrame();

	ui.beginFrame(16.0F);
	up.mouseX = 220.0F;
	up.mouseY = 70.0F;
	EXPECT_TRUE(ui.routeEvent(up));
	activated = drawControls();
	EXPECT_EQ(activated, (oa::Array<bool, 3>{false, false, false}));
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();

	// The third row item is the checkbox; clicking anywhere in its label region
	// toggles the referenced value on release.
	ui.beginFrame(16.0F);
	down.mouseX = 120.0F;
	down.mouseY = 15.0F;
	EXPECT_FALSE(ui.routeEvent(down));
	activated = drawControls();
	ui.endFrame();

	ui.beginFrame(16.0F);
	up.mouseX = 120.0F;
	up.mouseY = 15.0F;
	EXPECT_TRUE(ui.routeEvent(up));
	activated = drawControls();
	EXPECT_TRUE(activated[2]);
	EXPECT_TRUE(checked);
	const oa::U32 checkboxFocus = ui.input().focusId;
	ASSERT_NE(checkboxFocus, 0U);
	ui.endFrame();

	// Focus order comes from the prior rendered frame. forward Tab wraps from
	// the checkbox to the first button and the next Tab selects the second.
	ui.beginFrame(16.0F);
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Tab;
	EXPECT_TRUE(ui.routeEvent(key));
	const oa::U32 firstFocus = ui.input().focusId;
	EXPECT_NE(firstFocus, 0U);
	EXPECT_NE(firstFocus, checkboxFocus);
	activated = drawControls();
	ui.endFrame();

	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	const oa::U32 secondFocus = ui.input().focusId;
	EXPECT_NE(secondFocus, 0U);
	EXPECT_NE(secondFocus, firstFocus);
	activated = drawControls();
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Return;
	EXPECT_TRUE(ui.routeEvent(key));
	activated = drawControls();
	EXPECT_EQ(activated, (oa::Array<bool, 3>{false, true, false}));
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Tab;
	key.modifiers = oa::UiModifierShift;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_EQ(ui.input().focusId, firstFocus);
	activated = drawControls();
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Space;
	key.modifiers = oa::UiModifierNone;
	EXPECT_TRUE(ui.routeEvent(key));
	activated = drawControls();
	EXPECT_EQ(activated, (oa::Array<bool, 3>{true, false, false}));
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Tab;
	key.modifiers = oa::UiModifierShift;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_EQ(ui.input().focusId, checkboxFocus);
	activated = drawControls();
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Return;
	key.modifiers = oa::UiModifierNone;
	EXPECT_TRUE(ui.routeEvent(key));
	activated = drawControls();
	EXPECT_TRUE(activated[2]);
	EXPECT_FALSE(checked);
	ui.endFrame();

	ui.beginFrame(16.0F);
	ui.beginRow("orphan");
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, PopupMenusAndDropdownsOwnDismissalFocusAndSelection)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	const oa::PixelRect viewport{0, 0, 320, 240};

	oa::Array<bool, 2> menuActivated{};
	const auto drawMenu = [&] {
		menuActivated = {};
		oa::UiLayout layout;
		layout.padding = oa::UiEdge{};
		layout.gap = 0.0F;
		ui.beginPanel("popup-owner", {10, 10, 100, 30}, layout);
		if (ui.button("Menu")) ui.openPopup("main-menu");
		const bool open = ui.isPopupOpen("main-menu");
		if (ui.beginPopup("main-menu", {
			.width = 100, .height = 64, .gap = 4,
			.padding = oa::UiEdge{4.0F}})) {
			menuActivated[0] = ui.menuItem("first");
			menuActivated[1] = ui.menuItem("second");
			ui.endPopup();
		}
		ui.endPanel();
		return open;
	};
	const auto pointer = [](oa::UiEventType inType, oa::F32 inX, oa::F32 inY) {
		oa::UiEvent event;
		event.type = inType;
		event.button = 1;
		event.mouseX = inX;
		event.mouseY = inY;
		return event;
	};

	ui.beginFrame(16.0F, viewport);
	EXPECT_FALSE(drawMenu());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_FALSE(ui.routeEvent(pointer(oa::UiEventType::MouseDown, 20.0F, 20.0F)));
	EXPECT_FALSE(drawMenu());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseUp, 20.0F, 20.0F)));
	EXPECT_TRUE(drawMenu());
	ui.endFrame();

	// Escape is owned by the popup and restores focus to its anchor.
	ui.beginFrame(16.0F, viewport);
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Escape;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawMenu());
	EXPECT_NE(ui.input().focusId, 0U);
	ui.endFrame();

	// Reopen, traverse only the popup focus order, and activate the second item.
	ui.beginFrame(16.0F, viewport);
	(void)ui.routeEvent(pointer(oa::UiEventType::MouseDown, 20.0F, 20.0F));
	EXPECT_FALSE(drawMenu());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseUp, 20.0F, 20.0F)));
	EXPECT_TRUE(drawMenu());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	key.key = oa::UiKey::Down;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawMenu());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	key.key = oa::UiKey::Return;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawMenu());
	EXPECT_FALSE(menuActivated[0]);
	EXPECT_TRUE(menuActivated[1]);
	ui.endFrame();

	// An outside press dismisses and is not offered to the base layer that frame.
	ui.beginFrame(16.0F, viewport);
	(void)ui.routeEvent(pointer(oa::UiEventType::MouseDown, 20.0F, 20.0F));
	(void)drawMenu();
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	(void)ui.routeEvent(pointer(oa::UiEventType::MouseUp, 20.0F, 20.0F));
	EXPECT_TRUE(drawMenu());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseDown, 250.0F, 180.0F)));
	EXPECT_FALSE(drawMenu());
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();

	const oa::Array<oa::StringView, 3> choices{
		"Auto", "One", "Two",
	};
	oa::I32 selected = 0;
	bool changed = false;
	const auto drawDropdown = [&] {
		oa::UiLayout layout;
		layout.padding = oa::UiEdge{};
		layout.gap = 0.0F;
		ui.beginPanel("dropdown-owner", {10, 10, 180, 30}, layout);
		changed = ui.dropdown(
			"layout", oa::Span<const oa::StringView>(choices), selected,
			{.maxVisibleItems = 3});
		ui.tooltip("Metric card layout", {.delayMs = 0.0F});
		ui.endPanel();
	};
	ui.beginFrame(16.0F, viewport);
	(void)ui.routeEvent(pointer(oa::UiEventType::MouseUp, 250.0F, 180.0F));
	drawDropdown();
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_FALSE(ui.routeEvent(pointer(oa::UiEventType::MouseDown, 20.0F, 20.0F)));
	drawDropdown();
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseUp, 20.0F, 20.0F)));
	drawDropdown();
	EXPECT_FALSE(changed);
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseDown, 20.0F, 94.0F)));
	drawDropdown();
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseUp, 20.0F, 94.0F)));
	drawDropdown();
	EXPECT_TRUE(changed);
	EXPECT_EQ(selected, 2);
	EXPECT_TRUE(ui.recordRender(VK_NULL_HANDLE, 0U).isOk());
	ui.endFrame();

	// Reopening focuses the current selection; Up then Return selects its
	// predecessor without escaping into the base focus order.
	ui.beginFrame(16.0F, viewport);
	(void)ui.routeEvent(pointer(oa::UiEventType::MouseDown, 20.0F, 20.0F));
	drawDropdown();
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseUp, 20.0F, 20.0F)));
	drawDropdown();
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	key.key = oa::UiKey::Up;
	EXPECT_TRUE(ui.routeEvent(key));
	drawDropdown();
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	key.key = oa::UiKey::Return;
	EXPECT_TRUE(ui.routeEvent(key));
	drawDropdown();
	EXPECT_TRUE(changed);
	EXPECT_EQ(selected, 1);
	ui.endFrame();

	// A zero-delay tooltip uses the same valid top layer after its anchor.
	ui.beginFrame(16.0F, viewport);
	oa::UiEvent move;
	move.type = oa::UiEventType::MouseMove;
	move.mouseX = 20.0F;
	move.mouseY = 20.0F;
	EXPECT_FALSE(ui.routeEvent(move));
	drawDropdown();
	EXPECT_TRUE(ui.recordRender(VK_NULL_HANDLE, 0U).isOk());
	ui.endFrame();

	// A selected item outside the first page is focused and virtualized into
	// view; keyboard traversal still reaches the preceding option.
	const oa::Array<oa::StringView, 12> longChoices{
		"00", "01", "02", "03", "04", "05",
		"06", "07", "08", "09", "10", "11",
	};
	oa::I32 longSelected = 11;
	const auto drawLongDropdown = [&] {
		oa::UiLayout layout;
		layout.padding = oa::UiEdge{};
		layout.gap = 0.0F;
		ui.beginPanel("long-dropdown-owner", {10, 10, 180, 30}, layout);
		const bool didChange = ui.dropdown(
			"Long", oa::Span<const oa::StringView>(longChoices), longSelected,
			{.maxVisibleItems = 3});
		ui.endPanel();
		return didChange;
	};
	ui.beginFrame(16.0F, viewport);
	(void)ui.routeEvent(pointer(oa::UiEventType::MouseDown, 20.0F, 20.0F));
	EXPECT_FALSE(drawLongDropdown());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	EXPECT_TRUE(ui.routeEvent(pointer(oa::UiEventType::MouseUp, 20.0F, 20.0F)));
	EXPECT_FALSE(drawLongDropdown());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	key.key = oa::UiKey::Up;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawLongDropdown());
	ui.endFrame();
	ui.beginFrame(16.0F, viewport);
	key.key = oa::UiKey::Return;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawLongDropdown());
	EXPECT_EQ(longSelected, 10);
	ui.endFrame();

	// An unbalanced popup is rejected before command recording.
	ui.beginFrame(16.0F, viewport);
	ui.openPopup("unbalanced", {10, 10, 40, 20});
	ASSERT_TRUE(ui.beginPopup("unbalanced", {
		.width = 80, .height = 40, .gap = 0}));
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endPopup();
	ui.closePopup();
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, PopupCommandsRenderAfterLaterBaseCommandsOnGpu)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 64U, 64U).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	ui.beginFrame(16.0F, {0, 0, 64, 64});
	ui.openPopup("overlay", {8, 4, 40, 20});
	ASSERT_TRUE(ui.beginPopup("overlay", {
		.width = 40, .height = 30, .gap = 0,
		.padding = oa::UiEdge{}}));
	ui.rect({8, 24, 40, 30}, {0.0F, 1.0F, 0.0F, 1.0F});
	ui.endPopup();
	// This base draw is submitted later by the application but must remain below
	// the explicit overlay layer.
	ui.rect({0, 0, 64, 64}, {1.0F, 0.0F, 0.0F, 1.0F});

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 64U * 64U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	const oa::Usize sample = (30U * 64U + 12U) * 4U;
	EXPECT_EQ(pixels[sample + 0U], 0U);
	EXPECT_EQ(pixels[sample + 1U], 255U);
	EXPECT_EQ(pixels[sample + 2U], 0U);
	EXPECT_EQ(pixels[sample + 3U], 255U);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
}

TEST_VK(TestUi, InputTextEditsUtf8SelectionClipboardAndNativeFocus)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	oa::String text("A\xC3\xA9" "B");
	oa::UiLayout layout;
	layout.padding = oa::UiEdge{};
	layout.gap = 0.0F;
	const auto drawInput = [&] {
		ui.beginPanel("text-panel", {0, 0, 320, 32}, layout);
		const bool changed = ui.inputText("Name", text);
		ui.endPanel();
		return changed;
	};

	// The first rendered frame establishes the stable focus order. No platform
	// text-input request exists until the field actually owns focus.
	ui.beginFrame(16.0F);
	EXPECT_FALSE(drawInput());
	EXPECT_FALSE(ui.wantsTextInput());
	ui.endFrame();

	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Tab;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	EXPECT_TRUE(ui.wantsTextInput());
	const oa::PixelRect nativeCaret = ui.textInputRect();
	EXPECT_EQ(nativeCaret.w, 1);
	EXPECT_GT(nativeCaret.h, 0);
	ui.endFrame();

	// Committed input accepts either a UTF-8 payload or one Unicode scalar.
	oa::UiEvent textEvent;
	textEvent.type = oa::UiEventType::KeyChar;
	textEvent.codepoint = 0x03A9U;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(textEvent));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "A\xC3\xA9" "B\xCE\xA9");
	ui.endFrame();

	// scalar navigation never splits the two-byte UTF-8 codepoints.
	key.key = oa::UiKey::Left;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	key.key = oa::UiKey::Backspace;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "A\xC3\xA9\xCE\xA9");
	ui.endFrame();

	// Selection is retained across frames for copy/cut. The UI emits a platform
	// clipboard request without acquiring an SDL dependency itself.
	key.key = oa::UiKey::A;
	key.modifiers = oa::UiModifierCtrl;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	key.key = oa::UiKey::C;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	oa::String clipboard;
	EXPECT_TRUE(ui.takeClipboardWrite(clipboard));
	EXPECT_EQ(testStdString(clipboard), "A\xC3\xA9\xCE\xA9");
	EXPECT_FALSE(ui.takeClipboardWrite(clipboard));
	ui.endFrame();
	key.key = oa::UiKey::X;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_TRUE(text.empty());
	EXPECT_TRUE(ui.takeClipboardWrite(clipboard));
	ui.endFrame();

	textEvent.codepoint = 0U;
	textEvent.text = oa::String("caf\xC3\xA9");
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(textEvent));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "caf\xC3\xA9");
	ui.endFrame();

	key.modifiers = oa::UiModifierNone;
	key.key = oa::UiKey::Home;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	key.key = oa::UiKey::Delete;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "af\xC3\xA9");
	ui.endFrame();

	// Per-field history restores scalar-safe text and caret state. Redo accepts
	// both conventional ctrl+Y and editor-style ctrl+shift+Z.
	key.key = oa::UiKey::Z;
	key.modifiers = oa::UiModifierCtrl;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "caf\xC3\xA9");
	ui.endFrame();
	key.modifiers = oa::UiModifierCtrl | oa::UiModifierShift;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "af\xC3\xA9");
	ui.endFrame();
	key.key = oa::UiKey::Z;
	key.modifiers = oa::UiModifierCtrl;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "caf\xC3\xA9");
	ui.endFrame();
	key.key = oa::UiKey::Y;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "af\xC3\xA9");
	ui.endFrame();

	// IME pre-edit is visible interaction state, not caller data. Editing keys
	// remain owned by the active composition; Escape cancels it without dropping
	// field focus, and committed UTF-8 enters the ordinary undo history.
	oa::UiEvent preedit;
	preedit.type = oa::UiEventType::TextEditing;
	preedit.text = oa::String("\xE4\xB8\x96\xE7\x95\x8C");
	preedit.textSelectionStart = 1;
	preedit.textSelectionLength = 1;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(preedit));
	EXPECT_FALSE(drawInput());
	EXPECT_EQ(testStdString(text), "af\xC3\xA9");
	EXPECT_TRUE(ui.wantsTextInput());
	ui.endFrame();
	key.key = oa::UiKey::Backspace;
	key.modifiers = oa::UiModifierNone;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	EXPECT_EQ(testStdString(text), "af\xC3\xA9");
	ui.endFrame();
	key.key = oa::UiKey::Escape;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	EXPECT_TRUE(ui.wantsTextInput());
	ui.endFrame();
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(preedit));
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	textEvent.text = oa::String("\xE4\xB8\x96");
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(textEvent));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "\xE4\xB8\x96" "af\xC3\xA9");
	ui.endFrame();
	key.key = oa::UiKey::Z;
	key.modifiers = oa::UiModifierCtrl;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInput());
	EXPECT_EQ(testStdString(text), "af\xC3\xA9");
	ui.endFrame();
	preedit.text = oa::String("\xC3", 1U);
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(preedit));
	EXPECT_FALSE(drawInput());
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();
	preedit.text = oa::String("\xE4\xB8\x96\xE7\x95\x8C");

	// An external replacement rebases field-local history. Undo cannot revive
	// the application's superseded value.
	text = oa::String("alpha beta");
	ui.beginFrame(16.0F);
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	EXPECT_EQ(testStdString(text), "alpha beta");
	ui.endFrame();

	// Double click selects one word and triple click selects the complete
	// single-line value. Selection is proved through the platform clipboard
	// request rather than private widget state.
	oa::UiEvent pointer;
	pointer.type = oa::UiEventType::MouseDown;
	pointer.button = 1;
	pointer.clickCount = 2;
	pointer.mouseX = 145.0F;
	pointer.mouseY = 16.0F;
	ui.beginFrame(16.0F);
	(void)ui.routeEvent(pointer);
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	key.key = oa::UiKey::C;
	key.modifiers = oa::UiModifierCtrl;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	ASSERT_TRUE(ui.takeClipboardWrite(clipboard));
	EXPECT_EQ(testStdString(clipboard), "alpha");
	ui.endFrame();
	pointer.clickCount = 3;
	ui.beginFrame(16.0F);
	(void)ui.routeEvent(pointer);
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	ASSERT_TRUE(ui.takeClipboardWrite(clipboard));
	EXPECT_EQ(testStdString(clipboard), "alpha beta");
	ui.endFrame();

	oa::UiEvent pointerUp = pointer;
	pointerUp.type = oa::UiEventType::MouseUp;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(pointerUp));
	EXPECT_FALSE(drawInput());
	ui.endFrame();

	key.key = oa::UiKey::Return;
	key.modifiers = oa::UiModifierNone;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	EXPECT_FALSE(ui.wantsTextInput());
	ui.endFrame();

	// Passive swatches are real layout/render commands, while malformed external
	// text is rejected at the public boundary instead of being edited by bytes.
	ui.beginFrame(16.0F);
	ui.beginPanel("swatch-panel", {0, 0, 32, 32}, layout);
	ui.colorSwatch({1.0F, 0.0F, 0.0F, 1.0F}, {16.0F, 16.0F});
	ui.endPanel();
	EXPECT_TRUE(ui.recordRender(VK_NULL_HANDLE, 0U).isOk());
	ui.endFrame();
	text = oa::String("\xC3", 1U);
	ui.beginFrame(16.0F);
	EXPECT_FALSE(drawInput());
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, InputTextRendersImeCompositionSelectionAndUnderlineOnGpu)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	UiStorageTarget target;
	ASSERT_TRUE(target.init(*engine, 128U, 32U).isOk());
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	oa::UiLayout layout;
	layout.padding = oa::UiEdge{};
	layout.gap = 0.0F;
	oa::UiStyle style = ui.currentStyle();
	style.background = {0.0F, 0.0F, 0.0F, 1.0F};
	style.surface = style.background;
	style.surfaceHover = style.background;
	style.surfaceActive = style.background;
	style.border = {0.0F, 0.0F, 1.0F, 1.0F};
	style.accent = {1.0F, 0.0F, 0.0F, 1.0F};
	style.accentHover = {0.0F, 1.0F, 0.0F, 1.0F};
	style.text = {1.0F, 1.0F, 1.0F, 1.0F};
	style.textSecondary = style.text;
	oa::String text("ab");
	const auto drawInput = [&] {
		ui.pushStyle(style);
		ui.beginPanel("ime-panel", {0, 0, 128, 32}, layout);
		const bool changed = ui.inputText("Name", text);
		ui.endPanel();
		ui.popStyle();
		return changed;
	};

	ui.beginFrame(16.0F, {0, 0, 128, 32});
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Tab;
	ui.beginFrame(16.0F, {0, 0, 128, 32});
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	ui.endFrame();
	key.key = oa::UiKey::A;
	key.modifiers = oa::UiModifierCtrl;
	ui.beginFrame(16.0F, {0, 0, 128, 32});
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_FALSE(drawInput());
	ui.endFrame();

	oa::UiEvent preedit;
	preedit.type = oa::UiEventType::TextEditing;
	preedit.text = oa::String("XY");
	preedit.textSelectionStart = 0;
	preedit.textSelectionLength = 1;
	ui.beginFrame(16.0F, {0, 0, 128, 32});
	EXPECT_TRUE(ui.routeEvent(preedit));
	EXPECT_FALSE(drawInput());
	EXPECT_EQ(testStdString(text), "ab");
	const oa::PixelRect nativeCaret = ui.textInputRect();
	EXPECT_GT(nativeCaret.x, 0);
	EXPECT_GT(nativeCaret.h, 0);

	auto streamResult = oavk::Stream::createCompute(
		oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(streamResult.isOk()) << streamResult.getStatus().toString();
	oavk::Stream stream = oa::move(*streamResult);
	ASSERT_TRUE(stream.begin(oa::EngineDeviceAccess::get(*engine)).isOk());
	const VkCommandBuffer commandBuffer =
		static_cast<VkCommandBuffer>(stream.commandBuffer);
	target.recordInitialize(commandBuffer);
	ASSERT_TRUE(ui.recordRender(commandBuffer, target.bindlessIndex).isOk());
	target.recordReadback(stream);
	ASSERT_TRUE(stream.submitAndWait(*engine).isOk());
	const oa::Event completion = oa::EventAccess::create(
		oa::EngineDeviceAccess::get(*engine),
		stream.timelineSem,
		stream.timelineValue);
	ASSERT_TRUE(ui.markFrameSubmitted(completion).isOk());
	ui.endFrame();

	oa::Array<oa::U8, 128U * 32U * 4U> pixels{};
	ASSERT_TRUE(oa::EngineResourceAccess::readbackBuffer(
		*engine,
		target.readback,
		0U,
		pixels.data(),
		pixels.size()).isOk());
	const auto pixel = [&](oa::U32 inX, oa::U32 inY, oa::U32 inChannel) {
		return pixels[(inY * 128U + inX) * 4U + inChannel];
	};
	bool selectedPreedit = false;
	bool compositionUnderline = false;
	const oa::U32 underlineY = static_cast<oa::U32>(
		nativeCaret.y + nativeCaret.h - 3);
	for (oa::U32 x = 1U; x + 1U < 128U; ++x) {
		selectedPreedit = selectedPreedit
			|| (pixel(x, 3U, 0U) > 32U
				&& pixel(x, 3U, 1U) < 16U
				&& pixel(x, 3U, 2U) < 16U);
		compositionUnderline = compositionUnderline
			|| (pixel(x, underlineY, 1U) > 200U
				&& pixel(x, underlineY, 0U) < 16U
				&& pixel(x, underlineY, 2U) < 16U);
	}
	EXPECT_TRUE(selectedPreedit);
	EXPECT_TRUE(compositionUnderline);

	stream.destroy(oa::EngineDeviceAccess::get(*engine));
	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, SlidersSeparatorsAndProgressBarsHonorInteractionContracts)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::TextAtlas atlas;
	ASSERT_TRUE(atlas.init(*engine).isOk());
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.bindTextAtlas(atlas).isOk());
	oa::UiLayout layout;
	layout.padding = oa::UiEdge{};
	layout.gap = 0.0F;
	oa::F32 floating = -2.0F;
	oa::I32 integer = 0;

	const auto drawFloat = [&] {
		ui.beginPanel("float-panel", {0, 0, 200, 32}, layout);
		const bool changed = ui.sliderF32(
			"Exposure", &floating, 0.0F, 1.0F, "%+.2f");
		ui.endPanel();
		return changed;
	};
	const auto drawInteger = [&] {
		ui.beginPanel("integer-panel", {0, 0, 200, 32}, layout);
		const bool changed = ui.sliderI32("samples", &integer, 0, 1000);
		ui.endPanel();
		return changed;
	};

	// external values are normalized at the public boundary and report a real
	// mutation instead of allowing the visual and stored state to diverge.
	ui.beginFrame(16.0F);
	EXPECT_TRUE(drawFloat());
	EXPECT_FLOAT_EQ(floating, 0.0F);
	ui.endFrame();

	// Pressing maps absolute pointer position to the range and establishes both
	// pointer capture and keyboard focus.
	ui.beginFrame(16.0F);
	oa::UiEvent pointer;
	pointer.type = oa::UiEventType::MouseDown;
	pointer.button = 1;
	pointer.mouseX = 50.0F;
	pointer.mouseY = 10.0F;
	EXPECT_FALSE(ui.routeEvent(pointer));
	EXPECT_TRUE(drawFloat());
	EXPECT_NEAR(floating, 50.0F / 199.0F, 0.0001F);
	ASSERT_NE(ui.input().activeId, 0U);
	EXPECT_EQ(ui.input().focusId, ui.input().activeId);
	ui.endFrame();

	// capture persists outside the item, and the mapped value clamps at max.
	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseMove;
	pointer.mouseX = 250.0F;
	EXPECT_TRUE(ui.routeEvent(pointer));
	EXPECT_TRUE(drawFloat());
	EXPECT_FLOAT_EQ(floating, 1.0F);
	ui.endFrame();

	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseUp;
	pointer.button = 1;
	EXPECT_TRUE(ui.routeEvent(pointer));
	EXPECT_FALSE(drawFloat());
	EXPECT_EQ(ui.input().activeId, 0U);
	ui.endFrame();

	// Arrow repeat is admitted for adjustable controls. shift is coarse and
	// ctrl is fine; the focused slider consumes these keys before app shortcuts.
	oa::UiEvent key;
	key.type = oa::UiEventType::KeyDown;
	key.key = oa::UiKey::Left;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawFloat());
	EXPECT_NEAR(floating, 0.99F, 0.0001F);
	ui.endFrame();

	key.modifiers = oa::UiModifierShift;
	key.keyRepeat = true;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawFloat());
	EXPECT_NEAR(floating, 0.89F, 0.0001F);
	ui.endFrame();

	key.modifiers = oa::UiModifierCtrl;
	key.key = oa::UiKey::Right;
	ui.beginFrame(16.0F);
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawFloat());
	EXPECT_NEAR(floating, 0.891F, 0.0001F);
	ui.endFrame();

	// Integer sliders round pointer mapping and retain a minimum one-unit
	// keyboard step, including the fine-adjust path.
	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseDown;
	pointer.button = 1;
	pointer.mouseX = 100.0F;
	pointer.mouseY = 10.0F;
	EXPECT_FALSE(ui.routeEvent(pointer));
	EXPECT_TRUE(drawInteger());
	EXPECT_EQ(integer, 503);
	ui.endFrame();

	ui.beginFrame(16.0F);
	pointer.type = oa::UiEventType::MouseUp;
	EXPECT_TRUE(ui.routeEvent(pointer));
	EXPECT_FALSE(drawInteger());
	ui.endFrame();

	ui.beginFrame(16.0F);
	key.key = oa::UiKey::Right;
	key.modifiers = oa::UiModifierCtrl;
	key.keyRepeat = false;
	EXPECT_TRUE(ui.routeEvent(key));
	EXPECT_TRUE(drawInteger());
	EXPECT_EQ(integer, 504);
	ui.endFrame();

	// The full signed 32-bit range is evaluated through 64-bit arithmetic.
	integer = 0;
	ui.beginFrame(16.0F);
	ui.beginPanel("wide-integer", {0, 0, 240, 32}, layout);
	EXPECT_FALSE(ui.sliderI32(
		"Wide", &integer,
		std::numeric_limits<oa::I32>::lowest(),
		std::numeric_limits<oa::I32>::max()));
	ui.endPanel();
	ui.endFrame();

	// valid passive controls participate in row/column layout and text batching.
	ui.beginFrame(16.0F);
	ui.beginPanel("passive", {0, 0, 240, 80}, layout);
	ui.separator();
	ui.progressBar(1.5F);
	ui.beginRow("passive-row");
	ui.separator();
	ui.progressBar(-0.5F, "Queued");
	ui.endRow();
	ui.endPanel();
	EXPECT_TRUE(ui.recordRender(VK_NULL_HANDLE, 0U).isOk());
	ui.endFrame();

	// format strings are admitted before reaching snprintf, preventing a caller
	// mismatch such as %s from becoming undefined variadic behavior.
	ui.beginFrame(16.0F);
	ui.beginPanel("invalid-format", {0, 0, 200, 32}, layout);
	EXPECT_FALSE(ui.sliderF32("bad", &floating, 0.0F, 1.0F, "%s"));
	ui.endPanel();
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ui.beginFrame(16.0F);
	ui.progressBar(std::numeric_limits<oa::F32>::quiet_NaN());
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::InvalidArgument);
	ui.endFrame();

	ui.beginFrame(16.0F);
	ui.separator();
	EXPECT_EQ(
		ui.recordRender(VK_NULL_HANDLE, 0U).getCode(),
		oa::StatusCode::FailedPrecondition);
	ui.endFrame();

	ASSERT_TRUE(ui.close().isOk());
	atlas = {};
}

TEST_VK(TestUi, RenderRejectsDirectGroupsBeyondLiveDeviceLimitBeforeRecording)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	oa::Ui ui;
	ASSERT_TRUE(ui.init(*engine).isOk());
	ASSERT_TRUE(ui.initBlit(nullptr).isOk());
	ui.beginFrame(16.0F);
	ui.rect({0, 0, 16, 8}, oa::Color{});

	auto& maximum =
		oa::EngineDeviceAccess::get(*engine).info.hardware.maxComputeWorkGroupCountX;
	const oa::U32 savedMaximum = maximum;
	ASSERT_GT(savedMaximum, 1U);
	struct RestoreLimit {
		oa::U32& Value;
		oa::U32 Saved;
		~RestoreLimit() { Value = Saved; }
	} restoreMaximum{maximum, savedMaximum};
	maximum = 1U;

	const auto rejected = ui.recordRender(VK_NULL_HANDLE, 0U);
	EXPECT_EQ(rejected.getCode(), oa::StatusCode::OutOfRange);
	maximum = savedMaximum;
	ui.endFrame();
}

TEST_F(TestUi, KeyRepeatIsExplicitPerAction)
{
	oa::InputSystem input;
	oa::U32 discreteCount = 0;
	oa::U32 repeatCount = 0;
	input.registerAction({.name = "discrete", .binding = {.key = oa::UiKey::Left},
		.callback = [&] { ++discreteCount; }});
	input.registerAction({.name = "repeat", .binding = {.key = oa::UiKey::Right},
		.allowRepeat = true, .callback = [&] { ++repeatCount; }});

	oa::UiEvent event;
	event.type = oa::UiEventType::KeyDown;
	event.key = oa::UiKey::Left;
	EXPECT_TRUE(input.dispatch(event));
	event.keyRepeat = true;
	EXPECT_TRUE(input.dispatch(event));
	EXPECT_EQ(discreteCount, 1U);

	event.key = oa::UiKey::Right;
	event.keyRepeat = false;
	EXPECT_TRUE(input.dispatch(event));
	event.keyRepeat = true;
	EXPECT_TRUE(input.dispatch(event));
	EXPECT_EQ(repeatCount, 2U);
}

TEST_F(TestUi, InputDefaultsAndYamlBindingsAreRealAndTransactional)
{
	oa::InputSystem defaults;
	defaults.registerDefaults();
	defaults.registerDefaults();
	oa::U32 screenshotCount = 0U;
	oa::U32 undoCount = 0U;
	defaults.setCallback("screenshot", [&] { ++screenshotCount; });
	defaults.setCallback("undo", [&] { ++undoCount; });
	oa::UiEvent event;
	event.type = oa::UiEventType::KeyDown;
	event.key = oa::UiKey::F12;
	EXPECT_TRUE(defaults.dispatch(event));
	EXPECT_EQ(screenshotCount, 1U);
	event.key = oa::UiKey::Z;
	event.modifiers = oa::UiModifierCtrl;
	EXPECT_FALSE(defaults.dispatch(event));
	defaults.setContext(oa::InputContext::NodeCanvas);
	EXPECT_TRUE(defaults.dispatch(event));
	EXPECT_EQ(undoCount, 1U);

	const oa::Path bindingsPath("/tmp/oa_ui_bindings.yaml");
	const oa::Path invalidPath("/tmp/oa_ui_bindings_invalid.yaml");
	(void)oa::Filesystem::removeFile(bindingsPath);
	(void)oa::Filesystem::removeFile(invalidPath);
	defaults.rebind("undo", {
		.key = oa::UiKey::U,
		.modifiers = oa::UiModifierCtrl | oa::UiModifierShift,
	});
	ASSERT_TRUE(defaults.saveBindingsYaml(bindingsPath.string()).isOk());
	auto encoded = oa::Filesystem::readText(bindingsPath);
	ASSERT_TRUE(encoded.isOk());
	EXPECT_NE(encoded->find("version: 1"), oa::String::Npos);
	EXPECT_NE(encoded->find("action: undo"), oa::String::Npos);
	EXPECT_NE(encoded->find("- ctrl"), oa::String::Npos);
	EXPECT_NE(encoded->find("- shift"), oa::String::Npos);

	oa::InputSystem loaded;
	loaded.registerDefaults();
	oa::U32 loadedUndoCount = 0U;
	loaded.setCallback("undo", [&] { ++loadedUndoCount; });
	ASSERT_TRUE(loaded.loadBindingsYaml(bindingsPath.string()).isOk());
	loaded.setContext(oa::InputContext::NodeCanvas);
	event.key = oa::UiKey::U;
	event.modifiers = oa::UiModifierCtrl | oa::UiModifierShift;
	EXPECT_TRUE(loaded.dispatch(event));
	EXPECT_EQ(loadedUndoCount, 1U);

	ASSERT_TRUE(oa::Filesystem::writeText(invalidPath,
		"version: 1\n"
		"bindings:\n"
		"  - action: undo\n"
		"    key: NotAKey\n"
		"    modifiers: []\n"
		"    context: NodeCanvas\n"
		"    allow_repeat: false\n").isOk());
	const oa::Status invalid = loaded.loadBindingsYaml(invalidPath.string());
	EXPECT_EQ(invalid.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_TRUE(loaded.dispatch(event));
	EXPECT_EQ(loadedUndoCount, 2U);
	EXPECT_EQ(loaded.loadBindingsYaml("/tmp/oa_missing_bindings.yaml").getCode(),
		oa::StatusCode::FileNotFound);
	EXPECT_EQ(loaded.saveBindingsYaml("").getCode(),
		oa::StatusCode::InvalidArgument);

	EXPECT_TRUE(oa::Filesystem::removeFile(bindingsPath).isOk());
	EXPECT_TRUE(oa::Filesystem::removeFile(invalidPath).isOk());
}

TEST_F(TestUi, PlatformInputReportsDeclaredKeyboardEdges)
{
	oa::input::shutdown();
	oa::input::initialize();
	SDL_Event event{};
	event.type = SDL_EVENT_KEY_DOWN;
	event.key.scancode = SDL_SCANCODE_UP;
	oa::input::processEvent(&event);
	oa::input::update();
	EXPECT_TRUE(oa::input::down(oa::input::Button::KeyUp));
	EXPECT_TRUE(oa::input::press(oa::input::Button::KeyUp));
	EXPECT_FALSE(oa::input::release(oa::input::Button::KeyUp));
	oa::input::clearForNextFrame();
	oa::input::update();
	EXPECT_TRUE(oa::input::down(oa::input::Button::KeyUp));
	EXPECT_FALSE(oa::input::press(oa::input::Button::KeyUp));

	event.type = SDL_EVENT_KEY_UP;
	oa::input::processEvent(&event);
	oa::input::update();
	EXPECT_FALSE(oa::input::down(oa::input::Button::KeyUp));
	EXPECT_TRUE(oa::input::release(oa::input::Button::KeyUp));
	oa::input::shutdown();
}

TEST_F(TestUi, WindowDecorationUsesCompositorHitRegions)
{
	constexpr oa::I32 width = 1280;
	constexpr oa::I32 height = 720;

	EXPECT_EQ(
		oa::windowDecorationHitTest(3, 3, width, height, true, false),
		oa::WindowDecorationHit::ResizeTopLeft);
	EXPECT_EQ(
		oa::windowDecorationHitTest(width - 3, 3, width, height, true, false),
		oa::WindowDecorationHit::ResizeTopRight);
	EXPECT_EQ(
		oa::windowDecorationHitTest(3, height - 3, width, height, true, false),
		oa::WindowDecorationHit::ResizeBottomLeft);
	EXPECT_EQ(
		oa::windowDecorationHitTest(
			width - 3, height - 3, width, height, true, false),
		oa::WindowDecorationHit::ResizeBottomRight);
	EXPECT_EQ(
		oa::windowDecorationHitTest(640, 3, width, height, true, false),
		oa::WindowDecorationHit::ResizeTop);
	EXPECT_EQ(
		oa::windowDecorationHitTest(640, height - 3, width, height, true, false),
		oa::WindowDecorationHit::ResizeBottom);
	EXPECT_EQ(
		oa::windowDecorationHitTest(200, 20, width, height, true, false),
		oa::WindowDecorationHit::Draggable);

	EXPECT_EQ(
		oa::windowDecorationControlAt(width - 120, 20, width),
		oa::WindowDecorationControl::Minimize);
	EXPECT_EQ(
		oa::windowDecorationControlAt(width - 70, 20, width),
		oa::WindowDecorationControl::Maximize);
	EXPECT_EQ(
		oa::windowDecorationControlAt(width - 20, 20, width),
		oa::WindowDecorationControl::Close);
	EXPECT_EQ(
		oa::windowDecorationHitTest(width - 20, 20, width, height, true, false),
		oa::WindowDecorationHit::Normal);

	EXPECT_EQ(
		oa::windowDecorationHitTest(3, 20, width, height, true, true),
		oa::WindowDecorationHit::Draggable);
	EXPECT_EQ(
		oa::windowDecorationHitTest(3, 20, width, height, false, false),
		oa::WindowDecorationHit::Draggable);
	EXPECT_EQ(
		oa::windowDecorationHitTest(-1, 20, width, height, true, false),
		oa::WindowDecorationHit::Normal);
	EXPECT_EQ(oa::windowLogicalSizeForPixels(1920U, 2.0F), 960);
	EXPECT_EQ(oa::windowLogicalSizeForPixels(841U, 1.5F), 561);
	EXPECT_EQ(oa::windowLogicalSizeForPixels(0U, 2.0F), 1);
	EXPECT_EQ(oa::windowLogicalSizeForPixels(640U, 0.0F), 640);
}

TEST_VK(TestUi, TextureBlitAndClearAreByteExact)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const oa::Array<oa::U8, 16> sourcePixels{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	};
	const oa::Array<oa::U8, 16> zeroPixels{};
	auto sourceResult = oa::FnTexture::fromPixels(
		*engine,
		oa::Span<const oa::U8>(sourcePixels.data(), sourcePixels.size()),
		2,
		2);
	auto targetResult = oa::FnTexture::fromPixels(
		*engine,
		oa::Span<const oa::U8>(zeroPixels.data(), zeroPixels.size()),
		2,
		2);
	ASSERT_TRUE(sourceResult.isOk()) << sourceResult.getStatus().toString();
	ASSERT_TRUE(targetResult.isOk()) << targetResult.getStatus().toString();
	oa::Texture source = oa::move(*sourceResult);
	oa::Texture target = oa::move(*targetResult);

	oa::BlitDesc blit;
	blit.src = &source;
	blit.dst = &target;
	ASSERT_TRUE(oa::FnTexture::blit(blit).isOk());
	// The recorded node owns both leases. Releasing the only public source
	// value before submission must not invalidate its bindless descriptor.
	oa::Texture targetReadback = target;
	source = {};
	target = {};
	auto blitEvent = engine->submit();
	ASSERT_TRUE(blitEvent.isOk()) << blitEvent.getStatus().toString();
	ASSERT_TRUE(engine->wait(*blitEvent).isOk());
	oa::Array<oa::U8, 16> readback{};
	ASSERT_TRUE(oa::FnTexture::copyToHost(
		*engine, targetReadback, readback.data(), readback.size()).isOk());
	EXPECT_EQ(readback, sourcePixels);

	ASSERT_TRUE(oa::FnTexture::clear(
		targetReadback, oa::ClearColor{0.95F, 0.10F, 0.10F, 1.0F}).isOk());
	auto clearEvent = engine->submit();
	ASSERT_TRUE(clearEvent.isOk()) << clearEvent.getStatus().toString();
	ASSERT_TRUE(engine->wait(*clearEvent).isOk());
	ASSERT_TRUE(oa::FnTexture::copyToHost(
		*engine, targetReadback, readback.data(), readback.size()).isOk());
	for (oa::Usize pixel = 0; pixel < 4; ++pixel) {
		EXPECT_EQ(readback[pixel * 4 + 0], 242U);
		EXPECT_EQ(readback[pixel * 4 + 1], 26U);
		EXPECT_EQ(readback[pixel * 4 + 2], 26U);
		EXPECT_EQ(readback[pixel * 4 + 3], 255U);
	}

}

TEST_VK(TestUi, SemanticImageLayoutsAndFormatsConvertByteExactly)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const auto check = [&](const auto& inValues,
		oa::MatrixShape inShape,
		oa::ImageLayout inLayout,
		oa::ImageFormat inFormat,
		const oa::Array<oa::U8, 8>& inExpected) {
		auto matrix = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(
				reinterpret_cast<const oa::U8*>(inValues.data()),
				inValues.size() * sizeof(oa::F32)),
			inShape,
			oa::ScalarType::Float32);
		ASSERT_TRUE(matrix.hasStorage());
		oa::Image image(oa::move(matrix), inLayout, inFormat);
		auto textureResult = oa::FnTexture::fromImage(*engine, image);
		ASSERT_TRUE(textureResult.isOk())
			<< textureResult.getStatus().toString();
		oa::Array<oa::U8, 8> actual{};
		ASSERT_TRUE(oa::FnTexture::copyToHost(
			*engine, *textureResult, actual.data(), actual.size()).isOk());
		EXPECT_EQ(actual, inExpected);
	};

	const oa::Array<oa::F32, 6> nchwRgb{
		1.0F, 0.0F, 0.0F, 1.0F, 0.5F, 0.25F};
	check(nchwRgb, {1, 3, 1, 2}, oa::ImageLayout::Nchw, oa::ImageFormat::Rgb,
		{255U, 0U, 128U, 255U, 0U, 255U, 64U, 255U});

	const oa::Array<oa::F32, 6> nhwcBgr{
		0.5F, 0.0F, 1.0F, 0.25F, 1.0F, 0.0F};
	check(nhwcBgr, {1, 1, 2, 3}, oa::ImageLayout::Nhwc, oa::ImageFormat::Bgr,
		{255U, 0U, 128U, 255U, 0U, 255U, 64U, 255U});

	const oa::Array<oa::F32, 4> chwGrayAlpha{0.25F, 0.75F, 0.5F, 1.0F};
	check(chwGrayAlpha, {2, 1, 2}, oa::ImageLayout::Chw,
		oa::ImageFormat::GrayAlpha,
		{64U, 64U, 64U, 128U, 191U, 191U, 191U, 255U});

	const oa::Array<oa::F32, 8> hwcBgra{
		0.5F, 0.0F, 1.0F, 0.25F, 0.25F, 1.0F, 0.0F, 0.75F};
	check(hwcBgra, {1, 2, 4}, oa::ImageLayout::Hwc, oa::ImageFormat::Bgra,
		{255U, 0U, 128U, 64U, 0U, 255U, 64U, 191U});

	const oa::Array<oa::F32, 2> hwGray{-1.0F, 2.0F};
	check(hwGray, {1, 2}, oa::ImageLayout::Hw, oa::ImageFormat::Gray,
		{0U, 0U, 0U, 255U, 255U, 255U, 255U, 255U});

	const oa::Array<oa::F32, 8> nchwRgba{
		1.0F, 0.0F, 0.0F, 1.0F, 0.5F, 0.25F, 0.25F, 0.75F};
	check(nchwRgba, {1, 4, 1, 2}, oa::ImageLayout::Nchw, oa::ImageFormat::Rgba,
		{255U, 0U, 128U, 64U, 0U, 255U, 64U, 191U});
}

TEST_VK(TestUi, TextureCopiesShareOneLease)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const oa::Array<oa::U8, 8> pixels{
		1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
	auto result = oa::FnTexture::fromPixels(
		*engine, oa::Span<const oa::U8>(pixels.data(), pixels.size()), 2, 1);
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	oa::Texture retained = *result;
	*result = {};
	oa::Array<oa::U8, 8> actual{};
	ASSERT_TRUE(oa::FnTexture::copyToHost(
		*engine, retained, actual.data(), actual.size()).isOk());
	EXPECT_EQ(actual, pixels);
}

TEST_VK(TestUi, MetricFigureSavesCurvesAndConfusionHeatmap)
{
	const oa::Array<oa::F32, 8> loss{
		1.0F, 0.78F, 0.61F, 0.49F, 0.40F, 0.34F, 0.30F, 0.27F};
	const oa::Array<oa::F32, 9> confusion{
		12.0F, 1.0F, 0.0F,
		2.0F, 10.0F, 1.0F,
		0.0F, 1.0F, 11.0F};
	oa::plot::Figure figure({
		.title = "training evaluation",
		.rows = 1,
		.cols = 2,
		.width = 480,
		.height = 240,
		.hSpacing = 12,
		.padding = 12,
		.background = {0.04F, 0.04F, 0.05F, 1.0F},
	});
	figure.ax(0, 0).title("loss");
	figure.ax(0, 0).plot(loss);
	figure.ax(0, 1).title("confusion matrix");
	figure.ax(0, 1).heatmap(confusion, 3, 3);

	const char* path = "/tmp/oa_metric_figure.png";
	ASSERT_TRUE(figure.saveTo(path).isOk());
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(file.good());
	EXPECT_GT(file.tellg(), std::streampos(512));
	file.close();
	std::remove(path);
}

TEST_VK(TestUi, MetricFigureRendersSemanticImage)
{
	const oa::Array<oa::F32, 8> loss{
		1.0F, 0.78F, 0.61F, 0.49F, 0.40F, 0.34F, 0.30F, 0.27F};
	oa::plot::Figure figure({
		.title = "training metrics",
		.rows = 1,
		.cols = 1,
		.width = 96,
		.height = 64,
		.padding = 8,
		.background = {0.04F, 0.04F, 0.05F, 1.0F},
	});
	figure.ax(0, 0).plot(loss);

	auto render = figure.render();
	ASSERT_TRUE(render.isOk()) << render.getStatus().toString();
	const oa::Image& image = *render;
	EXPECT_TRUE(image.validate());
	EXPECT_EQ(image.width(), 96);
	EXPECT_EQ(image.height(), 64);
	EXPECT_EQ(image.channels(), 4);
	EXPECT_EQ(image.layout(), oa::ImageLayout::Nchw);
	EXPECT_EQ(image.format(), oa::ImageFormat::Rgba);
	EXPECT_EQ(image.getDtype(), oa::ScalarType::Float32);

	std::vector<oa::F32> pixels(
		static_cast<oa::Usize>(image.asMatrix().numElements()));
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		image.asMatrix(),
		pixels.data(),
		pixels.size() * sizeof(oa::F32)).isOk());
	const oa::Usize planeSize = 96U * 64U;
	const auto planeEnd =
		pixels.begin() + static_cast<std::ptrdiff_t>(planeSize);
	const auto [minimum, maximum] = std::minmax_element(
		pixels.begin(),
		planeEnd);
	ASSERT_NE(minimum, planeEnd);
	EXPECT_GE(*minimum, 0.0F);
	EXPECT_LE(*maximum, 1.0F);
	EXPECT_GT(*maximum - *minimum, 0.25F);
}

TEST_VK(TestUi, PlotThemesResolveDarkByDefaultAndLightExplicitly)
{
	const auto renderTheme = [](oa::plot::Theme theme) {
		oa::plot::Figure figure({
			.rows = 1,
			.cols = 1,
			.width = 48,
			.height = 32,
			.theme = theme,
		});
		auto render = figure.render();
		EXPECT_TRUE(render.isOk()) << render.getStatus().toString();
		std::vector<oa::F32> pixels;
		if (not render.isOk()) return pixels;
		pixels.resize(static_cast<oa::Usize>(render->asMatrix().numElements()));
		EXPECT_TRUE(oa::FnMatrix::copyToHost(
			render->asMatrix(), pixels.data(), pixels.size() * sizeof(oa::F32)).isOk());
		return pixels;
	};
	const auto dark = renderTheme(oa::plot::Theme::Dark);
	const auto light = renderTheme(oa::plot::Theme::Light);
	ASSERT_EQ(dark.size(), 48U * 32U * 4U);
	ASSERT_EQ(light.size(), dark.size());
	EXPECT_LT(dark[0], 0.15F);
	EXPECT_GT(light[0], 0.95F);
}

TEST_VK(TestUi, OrderedPlotArtistsComposeExplicitLinesScatterBarsAndLegend)
{
	const oa::Array<oa::F32, 6> x{0.0F, 0.12F, 0.27F, 0.48F, 0.74F, 1.0F};
	const oa::Array<oa::F32, 6> roc{0.0F, 0.48F, 0.71F, 0.86F, 0.95F, 1.0F};
	const oa::Array<oa::F32, 6> pr{1.0F, 0.94F, 0.88F, 0.81F, 0.72F, 0.61F};
	const oa::Array<oa::F32, 5> sx{0.1F, 0.3F, 0.5F, 0.7F, 0.9F};
	const oa::Array<oa::F32, 5> sy{0.2F, 0.6F, 0.45F, 0.82F, 0.68F};
	const oa::Array<oa::F32, 8> samples{
		0.08F, 0.12F, 0.18F, 0.42F, 0.46F, 0.51F, 0.76F, 0.88F};
	oa::plot::Figure figure({
		.rows = 1,
		.cols = 1,
		.width = 320,
		.height = 220,
		.padding = 12,
	});
	auto& axes = figure.ax(0, 0);
	axes.title("Evaluation artists");
	axes.xLabel("rate");
	axes.yLabel("score");
	axes.limits(0.0F, 1.0F, 0.0F, 1.0F);
	axes.plot(x, roc, {.color = oa::Color::accent(), .label = "ROC"});
	axes.plot(x, pr, {.color = oa::Color::success(), .label = "PR"});
	axes.scatter(sx, sy, {.color = oa::Color::cyan(), .label = "samples", .radius = 3.0F});
	axes.histogram(samples, 8,
		{.color = oa::Color::warning(), .label = "score bins", .gap = 0.28F});

	auto render = figure.render();
	ASSERT_TRUE(render.isOk()) << render.getStatus().toString();
	std::vector<oa::F32> pixels(
		static_cast<oa::Usize>(render->asMatrix().numElements()));
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		render->asMatrix(), pixels.data(), pixels.size() * sizeof(oa::F32)).isOk());
	const oa::Usize plane = 320U * 220U;
	oa::U32 indigo = 0U;
	oa::U32 green = 0U;
	oa::U32 cyan = 0U;
	oa::U32 amber = 0U;
	oa::U32 opaqueAmber = 0U;
	const oa::Color encodedWarning = oa::Color::fromU32(oa::Color::warning().toU32());
	for (oa::Usize pixel = 0U; pixel < plane; ++pixel) {
		const oa::F32 r = pixels[pixel];
		const oa::F32 g = pixels[plane + pixel];
		const oa::F32 b = pixels[2U * plane + pixel];
		const oa::F32 a = pixels[3U * plane + pixel];
		indigo += b > 0.55F and r > 0.20F and g > 0.20F ? 1U : 0U;
		green += g > 0.55F and g > r * 1.5F and g > b * 1.2F ? 1U : 0U;
		cyan += g > 0.60F and b > 0.60F and r < 0.35F ? 1U : 0U;
		amber += r > 0.60F and g > 0.30F and b < 0.25F ? 1U : 0U;
		opaqueAmber += std::abs(r - encodedWarning.r) < 1.0e-4F
			and std::abs(g - encodedWarning.g) < 1.0e-4F
			and std::abs(b - encodedWarning.b) < 1.0e-4F
			and std::abs(a - 1.0F) < 1.0e-6F ? 1U : 0U;
	}
	EXPECT_GT(indigo, 40U);
	EXPECT_GT(green, 40U);
	EXPECT_GT(cyan, 30U);
	EXPECT_GT(amber, 100U);
	EXPECT_GT(opaqueAmber, 100U);
}

TEST_VK(TestUi, HeadlessFigureComposesGeneratedCoverageText)
{
	oa::plot::Figure figure({
		.rows = 1,
		.cols = 1,
		.width = 192,
		.height = 96,
		.padding = 4,
		.background = {0.0F, 0.0F, 0.0F, 1.0F},
	});
	figure.ax(0, 0).title("M\xC3\xA9trique", {1.0F, 0.0F, 0.0F, 1.0F});
	figure.ax(0, 0).caption("\xC3\xA9tape", {0.0F, 1.0F, 0.0F, 1.0F});

	auto render = figure.render();
	ASSERT_TRUE(render.isOk()) << render.getStatus().toString();
	const oa::Image& image = *render;
	std::vector<oa::F32> pixels(
		static_cast<oa::Usize>(image.asMatrix().numElements()));
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		image.asMatrix(), pixels.data(), pixels.size() * sizeof(oa::F32)).isOk());

	const oa::Usize width = 192U;
	const oa::Usize height = 96U;
	const oa::Usize plane = width * height;
	oa::U32 titleRed = 0U;
	oa::U32 titleGreen = 0U;
	oa::U32 captionRed = 0U;
	oa::U32 captionGreen = 0U;
	for (oa::Usize y = 0U; y < height; ++y) {
		for (oa::Usize x = 0U; x < width; ++x) {
			const oa::Usize pixel = y * width + x;
			const oa::F32 red = pixels[pixel];
			const oa::F32 green = pixels[plane + pixel];
			if (y < 28U) {
				titleRed += red > 0.20F and red > green * 2.0F ? 1U : 0U;
				titleGreen += green > 0.20F and green > red * 2.0F ? 1U : 0U;
			}
			if (y >= height - 24U) {
				captionRed += red > 0.20F and red > green * 2.0F ? 1U : 0U;
				captionGreen += green > 0.20F and green > red * 2.0F ? 1U : 0U;
			}
		}
	}
	EXPECT_GT(titleRed, 24U);
	EXPECT_EQ(titleGreen, 0U);
	EXPECT_EQ(captionRed, 0U);
	EXPECT_GT(captionGreen, 16U);
}

TEST_VK(TestUi, FigureAndAxesLabelsReserveAndComposeCoverageBands)
{
	const oa::plot::FigureConfig config{
		.title = "Window title only",
		.rows = 1,
		.cols = 1,
		.width = 320,
		.height = 220,
		.padding = 8,
		.background = {0.0F, 0.0F, 0.0F, 1.0F},
	};
	oa::plot::Figure unlabeled(config);
	const oa::plot::Figure::Rect fullCell = unlabeled.cellRect(0, 0, 320, 220);
	EXPECT_EQ(fullCell.x, 8);
	EXPECT_EQ(fullCell.y, 8);
	EXPECT_EQ(fullCell.w, 304);
	EXPECT_EQ(fullCell.h, 204);

	oa::plot::Figure figure(config);
	figure.title("Figure heading");
	figure.xLabel("Global horizontal");
	figure.yLabel("Global vertical");
	figure.ax(0, 0).title("Axes heading");
	figure.ax(0, 0).xLabel("Axes horizontal");
	figure.ax(0, 0).yLabel("Axes vertical");
	const oa::plot::Figure::Rect labeledCell = figure.cellRect(0, 0, 320, 220);
	EXPECT_EQ(labeledCell.x, 38);
	EXPECT_EQ(labeledCell.y, 46);
	EXPECT_EQ(labeledCell.w, 244);
	EXPECT_EQ(labeledCell.h, 136);

	auto render = figure.render();
	ASSERT_TRUE(render.isOk()) << render.getStatus().toString();
	const oa::Image& image = *render;
	std::vector<oa::F32> pixels(
		static_cast<oa::Usize>(image.asMatrix().numElements()));
	ASSERT_TRUE(oa::FnMatrix::copyToHost(
		image.asMatrix(), pixels.data(), pixels.size() * sizeof(oa::F32)).isOk());

	constexpr oa::Usize width = 320U;
	constexpr oa::Usize height = 220U;
	constexpr oa::Usize plane = width * height;
	const auto lit = [&](oa::Usize x, oa::Usize y) {
		const oa::Usize pixel = y * width + x;
		return std::max({pixels[pixel], pixels[plane + pixel],
			pixels[2U * plane + pixel]}) > 0.08F;
	};
	const auto countLit = [&](oa::Usize x, oa::Usize y, oa::Usize w, oa::Usize h) {
		oa::U32 count = 0U;
		for (oa::Usize py = y; py < y + h; ++py) {
			for (oa::Usize px = x; px < x + w; ++px) {
				count += lit(px, py) ? 1U : 0U;
			}
		}
		return count;
	};
	EXPECT_GT(countLit(38U, 8U, 244U, 38U), 80U);   // figure title
	EXPECT_GT(countLit(38U, 182U, 244U, 30U), 60U); // figure x label
	EXPECT_GT(countLit(66U, 46U, 188U, 22U), 50U);  // axes title
	EXPECT_GT(countLit(66U, 154U, 188U, 28U), 50U); // axes x label

	const auto verticalInkExtent = [&](oa::Usize x, oa::Usize y,
		oa::Usize w, oa::Usize h) {
		oa::Usize minX = x + w;
		oa::Usize minY = y + h;
		oa::Usize maxX = x;
		oa::Usize maxY = y;
		oa::U32 count = 0U;
		for (oa::Usize py = y; py < y + h; ++py) {
			for (oa::Usize px = x; px < x + w; ++px) {
				if (not lit(px, py)) continue;
				minX = std::min(minX, px);
				minY = std::min(minY, py);
				maxX = std::max(maxX, px);
				maxY = std::max(maxY, py);
				++count;
			}
		}
		return oa::Array<oa::Usize, 3>{
			count,
			count > 0U ? maxX - minX + 1U : 0U,
			count > 0U ? maxY - minY + 1U : 0U};
	};
	const auto figureY = verticalInkExtent(8U, 46U, 30U, 136U);
	EXPECT_GT(figureY[0], 60U);
	EXPECT_GT(figureY[2], figureY[1] * 4U);
	const auto axesY = verticalInkExtent(38U, 68U, 28U, 86U);
	EXPECT_GT(axesY[0], 40U);
	EXPECT_GT(axesY[2], axesY[1] * 3U);

	// Render is the terminal readback of the same GPU command replay consumed
	// directly by show; there is no second CPU-rasterized canvas path.
}

TEST_VK(TestUi, TextureReadbackFeedsCvFigureAndImageFileSinks)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	const oa::Array<oa::U8, 16> sourcePixels{
		255, 0, 0, 255,
		0, 255, 0, 255,
		0, 0, 255, 255,
		255, 255, 255, 255,
	};
	auto sourceResult = oa::FnTexture::fromPixels(
		*engine,
		oa::Span<const oa::U8>(sourcePixels.data(), sourcePixels.size()),
		2,
		2);
	ASSERT_TRUE(sourceResult.isOk()) << sourceResult.getStatus().toString();
	oa::Texture source = oa::move(*sourceResult);

	oa::CvFrame frame;
	frame.base = oa::TextureAccess::buffer(source);
	frame.w = 2;
	frame.h = 2;
	auto compositeResult = frame.render(*engine);
	ASSERT_TRUE(compositeResult.isOk())
		<< compositeResult.getStatus().toString();
	oa::Texture composite = oa::move(*compositeResult);
	oa::Array<oa::U8, 16> compositePixels{};
	ASSERT_TRUE(oa::FnTexture::copyToHost(*engine, composite,
		compositePixels.data(), compositePixels.size()).isOk());
	EXPECT_EQ(compositePixels, sourcePixels);
	oavk::Buffer undersizedBase = *oa::TextureAccess::buffer(source);
	undersizedBase.size = 1U;
	oa::CvFrame invalidFrame;
	invalidFrame.base = &undersizedBase;
	invalidFrame.w = 2;
	invalidFrame.h = 2;
	auto invalidComposite = invalidFrame.render(*engine);
	EXPECT_FALSE(invalidComposite.isOk());
	EXPECT_EQ(invalidComposite.getStatus().getCode(),
		oa::StatusCode::InvalidArgument);

	const char* imagePath = "/tmp/oa_texture_readback.png";
	ASSERT_TRUE(oa::FnImage::saveTextureFile(*engine, source, imagePath).isOk());
	std::ifstream imageFile(imagePath, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(imageFile.good());
	EXPECT_GT(imageFile.tellg(), std::streampos(32));
	imageFile.close();
	std::remove(imagePath);

	const char* figurePath = "/tmp/oa_texture_figure.png";
	{
		oa::plot::Figure figure({
			.rows = 1,
			.cols = 1,
			.width = 96,
			.height = 96,
			.padding = 4,
		});
		figure.ax(0, 0).imshow(source);
		ASSERT_TRUE(figure.saveTo(figurePath).isOk());
	}
	std::ifstream figureFile(figurePath, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(figureFile.good());
	EXPECT_GT(figureFile.tellg(), std::streampos(128));
	figureFile.close();
	std::remove(figurePath);

}

TEST_VK(TestUi, CvFrameCompositesOnlyValidatedCpuBoundingBoxes)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);

	constexpr oa::I32 width = 12;
	constexpr oa::I32 height = 10;
	oa::Array<oa::U8, width * height * 4> source{};
	for (oa::Usize i = 3U; i < source.size(); i += 4U) source[i] = 255U;

	oa::CvFrame frame;
	frame.w = width;
	frame.h = height;
	frame.addBboxes(
		{{.x = 2.0F, .y = 3.0F, .w = 6.0F, .h = 5.0F,
			.score = 1.0F, .classId = 0, .label = {}}},
		{.color = {1.0F, 0.0F, 0.0F, 1.0F},
			.lineWidth = 1.0F,
			.alpha = 1.0F,
			.showLabels = false,
			.showScores = false});
	auto result = frame.render(
		*engine, oa::Span<const oa::U8>(source.data(), source.size()));
	ASSERT_TRUE(result.isOk()) << result.getStatus().toString();
	oa::Array<oa::U8, width * height * 4> actual{};
	ASSERT_TRUE(oa::FnTexture::copyToHost(
		*engine, *result, actual.data(), actual.size()).isOk());

	const auto expectPixel = [&](oa::I32 x, oa::I32 y,
		oa::U8 r, oa::U8 g, oa::U8 b, oa::U8 a) {
		const oa::Usize offset =
			(static_cast<oa::Usize>(y) * width + static_cast<oa::Usize>(x)) * 4U;
		EXPECT_EQ(actual[offset + 0U], r);
		EXPECT_EQ(actual[offset + 1U], g);
		EXPECT_EQ(actual[offset + 2U], b);
		EXPECT_EQ(actual[offset + 3U], a);
	};
	expectPixel(2, 3, 255U, 0U, 0U, 255U);
	expectPixel(7, 3, 255U, 0U, 0U, 255U);
	expectPixel(2, 7, 255U, 0U, 0U, 255U);
	expectPixel(7, 7, 255U, 0U, 0U, 255U);
	expectPixel(5, 3, 255U, 0U, 0U, 255U);
	expectPixel(2, 5, 255U, 0U, 0U, 255U);
	expectPixel(3, 4, 0U, 0U, 0U, 255U);
	expectPixel(1, 3, 0U, 0U, 0U, 255U);

	auto undersized = frame.render(
		*engine, oa::Span<const oa::U8>(source.data(), source.size() - 1U));
	ASSERT_FALSE(undersized.isOk());
	EXPECT_EQ(undersized.getStatus().getCode(), oa::StatusCode::InvalidArgument);

	oa::CvFrame invalidConfig;
	invalidConfig.w = width;
	invalidConfig.h = height;
	invalidConfig.addBboxes(
		{{.x = 2.0F, .y = 3.0F, .w = 6.0F, .h = 5.0F,
			.score = 1.0F, .classId = 0, .label = {}}},
		{.alpha = std::numeric_limits<oa::F32>::quiet_NaN()});
	auto invalidConfigResult = invalidConfig.render(
		*engine, oa::Span<const oa::U8>(source.data(), source.size()));
	ASSERT_FALSE(invalidConfigResult.isOk());
	EXPECT_EQ(invalidConfigResult.getStatus().getCode(),
		oa::StatusCode::InvalidArgument);

	oa::CvFrame invalidBox;
	invalidBox.w = width;
	invalidBox.h = height;
	invalidBox.addBboxes(
		{{.x = 2.0F, .y = 3.0F, .w = 0.5F, .h = 5.0F,
			.score = 1.0F, .classId = 0, .label = {}}},
		{.showLabels = false, .showScores = false});
	auto invalidBoxResult = invalidBox.render(
		*engine, oa::Span<const oa::U8>(source.data(), source.size()));
	ASSERT_FALSE(invalidBoxResult.isOk());
	EXPECT_EQ(invalidBoxResult.getStatus().getCode(),
		oa::StatusCode::InvalidArgument);
}

TEST_VK(TestUi, DeferredSemanticTextureCompletesAtSinks)
{
	auto* engine = testEnginePtr();
	ASSERT_NE(engine, nullptr);
	std::unique_ptr<oa::ExecutionSession> context(new oa::ExecutionSession(engine));
	ASSERT_NE(context, nullptr);
	oa::ExecutionSession::RecordingScope recording(*context);

	const oa::Array<oa::F32, 16> nchw{
		-1.0F, 0.0F, 0.5F, 2.0F,
		1.0F, 0.25F, 0.75F, 0.0F,
		0.0F, 0.5F, 1.0F, 0.25F,
		0.0F, 0.5F, 1.0F, 2.0F,
	};
	const oa::Array<oa::U8, 16> expected{
		0U, 255U, 0U, 0U,
		0U, 64U, 128U, 128U,
		128U, 191U, 255U, 255U,
		255U, 0U, 64U, 255U,
	};
	const oa::Array<oa::U8, 16> poison{
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
	};
	struct DeferredTexture {
		oa::Image image;
		oa::Texture texture;
	};
	auto makeDeferred = [&]() {
		DeferredTexture deferred;
		auto matrix = oa::FnMatrix::fromBytes(
			oa::Span<const oa::U8>(
				reinterpret_cast<const oa::U8*>(nchw.data()), sizeof(nchw)),
			oa::MatrixShape{1, 4, 2, 2}, oa::ScalarType::Float32);
		EXPECT_TRUE(matrix.hasStorage());
		deferred.image = oa::Image(
			oa::move(matrix), oa::ImageLayout::Nchw, oa::ImageFormat::Rgba);
		auto textureResult = oa::FnTexture::fromImage(*engine, deferred.image);
		EXPECT_TRUE(textureResult.isOk())
			<< textureResult.getStatus().toString();
		if (not textureResult.isOk()) return deferred;
		deferred.texture = oa::move(*textureResult);
		EXPECT_GT(context->nodeCount(), 0U);
		const oa::Status poisonStatus = oa::EngineResourceAccess::uploadBuffer(
			*engine, *oa::TextureAccess::buffer(deferred.texture), 0U,
			poison.data(), poison.size());
		EXPECT_TRUE(poisonStatus.isOk()) << poisonStatus.toString();
		return deferred;
	};

	auto cvSource = makeDeferred();
	ASSERT_TRUE(cvSource.texture.isValid());
	oa::CvFrame frame;
	frame.base = oa::TextureAccess::buffer(cvSource.texture);
	frame.w = 2;
	frame.h = 2;
	auto compositeResult = frame.render(*engine);
	ASSERT_TRUE(compositeResult.isOk())
		<< compositeResult.getStatus().toString();
	oa::Texture composite = oa::move(*compositeResult);
	oa::Array<oa::U8, 16> compositePixels{};
	ASSERT_TRUE(oa::FnTexture::copyToHost(*engine, composite,
		compositePixels.data(), compositePixels.size()).isOk());
	EXPECT_EQ(compositePixels, expected);

	auto fileSource = makeDeferred();
	ASSERT_TRUE(fileSource.texture.isValid());
	const char* imagePath = "/tmp/oa_deferred_texture.png";
	ASSERT_TRUE(oa::FnImage::saveTextureFile(
		*engine, fileSource.texture, imagePath).isOk());
	auto decodedImage = oa::FnImage::decodeFile(imagePath, oa::ImageFormat::Rgba);
	ASSERT_TRUE(decodedImage.isOk()) << decodedImage.getStatus().toString();
	auto loadedImageResult = oa::FnTexture::fromImage(*engine, *decodedImage);
	ASSERT_TRUE(loadedImageResult.isOk()) << loadedImageResult.getStatus().toString();
	oa::Texture loadedImage = oa::move(*loadedImageResult);
	oa::Array<oa::U8, 16> loadedPixels{};
	ASSERT_TRUE(oa::FnTexture::copyToHost(*engine, loadedImage,
		loadedPixels.data(), loadedPixels.size()).isOk());
	EXPECT_EQ(loadedPixels, expected);
	std::remove(imagePath);

	auto figureSource = makeDeferred();
	ASSERT_TRUE(figureSource.texture.isValid());
	const char* figurePath = "/tmp/oa_deferred_figure.png";
	{
		oa::plot::Figure figure({
			.rows = 1,
			.cols = 1,
			.width = 32,
			.height = 32,
			.hSpacing = 0,
			.vSpacing = 0,
			.padding = 0,
		});
		figure.ax(0, 0).imshow(figureSource.texture);
		ASSERT_TRUE(figure.saveTo(figurePath).isOk());
	}
	auto decodedFigure = oa::FnImage::decodeFile(figurePath, oa::ImageFormat::Rgba);
	ASSERT_TRUE(decodedFigure.isOk()) << decodedFigure.getStatus().toString();
	auto loadedFigureResult = oa::FnTexture::fromImage(*engine, *decodedFigure);
	ASSERT_TRUE(loadedFigureResult.isOk()) << loadedFigureResult.getStatus().toString();
	oa::Texture loadedFigure = oa::move(*loadedFigureResult);
	oa::Array<oa::U8, 32U * 32U * 4U> figurePixels{};
	ASSERT_TRUE(oa::FnTexture::copyToHost(*engine, loadedFigure,
		figurePixels.data(), figurePixels.size()).isOk());
	const auto expectPixel = [&](oa::U32 inX, oa::U32 inY, oa::U32 inSourcePixel) {
		const oa::Usize actualOffset = (inY * 32U + inX) * 4U;
		const oa::Usize expectedOffset = inSourcePixel * 4U;
		for (oa::Usize channel = 0U; channel < 4U; ++channel) {
			EXPECT_EQ(figurePixels[actualOffset + channel],
				expected[expectedOffset + channel]);
		}
	};
	expectPixel(8U, 8U, 0U);
	expectPixel(24U, 8U, 1U);
	expectPixel(8U, 24U, 2U);
	expectPixel(24U, 24U, 3U);
	std::remove(figurePath);
}
