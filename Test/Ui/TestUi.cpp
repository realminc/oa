#include "../OaTest.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Context.h>
#include <Oa/Core/FnMatrix.h>
#include <Oa/Ui/Image.h>
#include <Oa/Ui/Cv.h>
#include <Oa/Ui/Input.h>
#include <Oa/Ui/Plot/Plot.h>
#include <Oa/Ui/Ui.h>
#include <Oa/Vision/FnImage.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <memory>

namespace {

class TestUi : public ::testing::Test {};

} // namespace

TEST_VK(TestUi, TimelinePointerDragAndRelease)
{
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	OaUi ui;
	ASSERT_TRUE(ui.Init(*engine).IsOk());
	const OaPixelRect rect{10, 10, 100, 12};
	OaF32 fraction = 0.25F;

	ui.BeginFrame(16.0F);
	OaUiEvent down;
	down.Type = OuiEventType::MouseDown;
	down.Button = 1;
	down.MouseX = 85.0F;
	down.MouseY = 16.0F;
	EXPECT_FALSE(ui.RouteEvent(down));
	EXPECT_TRUE(ui.Timeline("transport", rect, fraction));
	EXPECT_FLOAT_EQ(fraction, 0.75F);
	EXPECT_NE(ui.Input().ActiveId, 0U);
	ui.EndFrame();

	ui.BeginFrame(16.0F);
	OaUiEvent move;
	move.Type = OuiEventType::MouseMove;
	move.MouseX = 35.0F;
	move.MouseY = 16.0F;
	EXPECT_TRUE(ui.RouteEvent(move));
	EXPECT_TRUE(ui.Timeline("transport", rect, fraction));
	EXPECT_FLOAT_EQ(fraction, 0.25F);
	ui.EndFrame();

	ui.BeginFrame(16.0F);
	OaUiEvent up;
	up.Type = OuiEventType::MouseUp;
	up.Button = 1;
	up.MouseX = 60.0F;
	up.MouseY = 16.0F;
	EXPECT_TRUE(ui.RouteEvent(up));
	EXPECT_TRUE(ui.Timeline("transport", rect, fraction));
	EXPECT_FLOAT_EQ(fraction, 0.5F);
	EXPECT_EQ(ui.Input().ActiveId, 0U);
	ui.EndFrame();
}

TEST_VK(TestUi, TimelineClampsFraction)
{
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	OaUi ui;
	ASSERT_TRUE(ui.Init(*engine).IsOk());
	ui.BeginFrame(16.0F);
	OaF32 fraction = 2.0F;
	EXPECT_FALSE(ui.Timeline("transport", {10, 10, 100, 12}, fraction));
	EXPECT_FLOAT_EQ(fraction, 1.0F);
	ui.EndFrame();
}

TEST_F(TestUi, KeyRepeatIsExplicitPerAction)
{
	OaInputSystem input;
	OaU32 discreteCount = 0;
	OaU32 repeatCount = 0;
	input.RegisterAction({.Name = "discrete", .Binding = {.Key = OuiKey::Left},
		.Callback = [&] { ++discreteCount; }});
	input.RegisterAction({.Name = "repeat", .Binding = {.Key = OuiKey::Right},
		.AllowRepeat = true, .Callback = [&] { ++repeatCount; }});

	OaUiEvent event;
	event.Type = OuiEventType::KeyDown;
	event.Key = OuiKey::Left;
	EXPECT_TRUE(input.Dispatch(event));
	event.KeyRepeat = true;
	EXPECT_TRUE(input.Dispatch(event));
	EXPECT_EQ(discreteCount, 1U);

	event.Key = OuiKey::Right;
	event.KeyRepeat = false;
	EXPECT_TRUE(input.Dispatch(event));
	event.KeyRepeat = true;
	EXPECT_TRUE(input.Dispatch(event));
	EXPECT_EQ(repeatCount, 2U);
}

TEST_VK(TestUi, TextureBlitAndClearAreByteExact)
{
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	auto& context = OaContext::GetDefault();
	const std::array<OaU8, 16> sourcePixels{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	};
	const std::array<OaU8, 16> zeroPixels{};
	auto sourceResult = OaTexture::FromPixels(
		*engine,
		OaSpan<const OaU8>(sourcePixels.data(), sourcePixels.size()),
		2,
		2);
	auto targetResult = OaTexture::FromPixels(
		*engine,
		OaSpan<const OaU8>(zeroPixels.data(), zeroPixels.size()),
		2,
		2);
	ASSERT_TRUE(sourceResult.IsOk()) << sourceResult.GetStatus().ToString();
	ASSERT_TRUE(targetResult.IsOk()) << targetResult.GetStatus().ToString();
	OaTexture source = OaStdMove(*sourceResult);
	OaTexture target = OaStdMove(*targetResult);

	OaBlitDesc blit;
	blit.Src = &source;
	blit.Dst = &target;
	ASSERT_TRUE(OaFnTexture::Blit(context, blit).IsOk());
	auto blitEvent = context.Submit();
	ASSERT_TRUE(blitEvent.IsOk()) << blitEvent.GetStatus().ToString();
	ASSERT_TRUE(context.Wait(*blitEvent).IsOk());
	std::array<OaU8, 16> readback{};
	ASSERT_TRUE(engine->ReadbackBuffer(
		target.DeviceBuf, 0, readback.data(), readback.size()).IsOk());
	EXPECT_EQ(readback, sourcePixels);

	ASSERT_TRUE(OaFnTexture::Clear(
		context, target, OaClearColor{0.95F, 0.10F, 0.10F, 1.0F}).IsOk());
	auto clearEvent = context.Submit();
	ASSERT_TRUE(clearEvent.IsOk()) << clearEvent.GetStatus().ToString();
	ASSERT_TRUE(context.Wait(*clearEvent).IsOk());
	ASSERT_TRUE(engine->ReadbackBuffer(
		target.DeviceBuf, 0, readback.data(), readback.size()).IsOk());
	for (OaUsize pixel = 0; pixel < 4; ++pixel) {
		EXPECT_EQ(readback[pixel * 4 + 0], 242U);
		EXPECT_EQ(readback[pixel * 4 + 1], 26U);
		EXPECT_EQ(readback[pixel * 4 + 2], 26U);
		EXPECT_EQ(readback[pixel * 4 + 3], 255U);
	}

	target.Destroy(*engine);
	source.Destroy(*engine);
}

TEST_VK(TestUi, MetricFigureSavesCurvesAndConfusionHeatmap)
{
	const std::array<OaF32, 8> loss{
		1.0F, 0.78F, 0.61F, 0.49F, 0.40F, 0.34F, 0.30F, 0.27F};
	const std::array<OaF32, 9> confusion{
		12.0F, 1.0F, 0.0F,
		2.0F, 10.0F, 1.0F,
		0.0F, 1.0F, 11.0F};
	OaPlot::Figure figure({
		.Title = "Training evaluation",
		.Rows = 1,
		.Cols = 2,
		.Width = 480,
		.Height = 240,
		.HSpacing = 12,
		.Padding = 12,
		.Background = {0.04F, 0.04F, 0.05F, 1.0F},
	});
	figure.Ax(0, 0).Title("Loss");
	figure.Ax(0, 0).Plot(loss);
	figure.Ax(0, 1).Title("Confusion matrix");
	figure.Ax(0, 1).Heatmap(confusion, 3, 3);

	const char* path = "/tmp/oa_metric_figure.png";
	ASSERT_TRUE(figure.SaveFig(path).IsOk());
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(file.good());
	EXPECT_GT(file.tellg(), std::streampos(512));
	file.close();
	std::remove(path);
}

TEST_VK(TestUi, TextureReadbackFeedsCvFigureAndImageFileSinks)
{
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	const std::array<OaU8, 16> sourcePixels{
		255, 0, 0, 255,
		0, 255, 0, 255,
		0, 0, 255, 255,
		255, 255, 255, 255,
	};
	auto sourceResult = OaTexture::FromPixels(
		*engine,
		OaSpan<const OaU8>(sourcePixels.data(), sourcePixels.size()),
		2,
		2);
	ASSERT_TRUE(sourceResult.IsOk()) << sourceResult.GetStatus().ToString();
	OaTexture source = OaStdMove(*sourceResult);

	OaCvFrame frame;
	frame.Base = &source.DeviceBuf;
	frame.W = 2;
	frame.H = 2;
	auto compositeResult = frame.Render(*engine);
	ASSERT_TRUE(compositeResult.IsOk())
		<< compositeResult.GetStatus().ToString();
	OaTexture composite = OaStdMove(*compositeResult);
	std::array<OaU8, 16> compositePixels{};
	ASSERT_TRUE(engine->ReadbackBuffer(
		composite.DeviceBuf,
		0U,
		compositePixels.data(),
		compositePixels.size()).IsOk());
	EXPECT_EQ(compositePixels, sourcePixels);
	OaVkBuffer undersizedBase = source.DeviceBuf;
	undersizedBase.Size = 1U;
	OaCvFrame invalidFrame;
	invalidFrame.Base = &undersizedBase;
	invalidFrame.W = 2;
	invalidFrame.H = 2;
	auto invalidComposite = invalidFrame.Render(*engine);
	EXPECT_FALSE(invalidComposite.IsOk());
	EXPECT_EQ(invalidComposite.GetStatus().GetCode(),
		OaStatusCode::InvalidArgument);

	const char* imagePath = "/tmp/oa_texture_readback.png";
	ASSERT_TRUE(OaFnImage::SaveFile(*engine, source, imagePath).IsOk());
	std::ifstream imageFile(imagePath, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(imageFile.good());
	EXPECT_GT(imageFile.tellg(), std::streampos(32));
	imageFile.close();
	std::remove(imagePath);

	const char* figurePath = "/tmp/oa_texture_figure.png";
	{
		OaPlot::Figure figure({
			.Rows = 1,
			.Cols = 1,
			.Width = 96,
			.Height = 96,
			.Padding = 4,
		});
		figure.Ax(0, 0).Imshow(source);
		ASSERT_TRUE(figure.SaveFig(figurePath).IsOk());
	}
	std::ifstream figureFile(figurePath, std::ios::binary | std::ios::ate);
	ASSERT_TRUE(figureFile.good());
	EXPECT_GT(figureFile.tellg(), std::streampos(128));
	figureFile.close();
	std::remove(figurePath);

	composite.Destroy(*engine);
	source.Destroy(*engine);
}

TEST_VK(TestUi, DeferredMatrixTextureCompletesAtCompatibilitySinks)
{
	auto* engine = OaEngine::GetGlobal();
	ASSERT_NE(engine, nullptr);
	std::unique_ptr<OaContext> context(OaContext::Create(engine));
	ASSERT_NE(context, nullptr);
	OaContext::RecordingScope recording(*context);

	const std::array<OaF32, 16> nchw{
		-1.0F, 0.0F, 0.5F, 2.0F,
		1.0F, 0.25F, 0.75F, 0.0F,
		0.0F, 0.5F, 1.0F, 0.25F,
		0.0F, 0.5F, 1.0F, 2.0F,
	};
	const std::array<OaU8, 16> expected{
		0U, 255U, 0U, 0U,
		0U, 64U, 128U, 128U,
		128U, 191U, 255U, 255U,
		255U, 0U, 64U, 255U,
	};
	const std::array<OaU8, 16> poison{
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
		0xA5U, 0xA5U, 0xA5U, 0xA5U,
	};
	struct DeferredTexture {
		OaMatrix Matrix;
		OaTexture Texture;
	};
	auto makeDeferred = [&]() {
		DeferredTexture deferred;
		deferred.Matrix = OaFnMatrix::FromBytes(
			OaSpan<const OaU8>(
				reinterpret_cast<const OaU8*>(nchw.data()), sizeof(nchw)),
			OaMatrixShape{1, 4, 2, 2}, OaScalarType::Float32);
		EXPECT_TRUE(deferred.Matrix.HasStorage());
		auto textureResult = OaTexture::FromMatrix(*engine, deferred.Matrix);
		EXPECT_TRUE(textureResult.IsOk())
			<< textureResult.GetStatus().ToString();
		if (not textureResult.IsOk()) return deferred;
		deferred.Texture = OaStdMove(*textureResult);
		EXPECT_GT(context->NodeCount(), 0U);
		const OaStatus poisonStatus = engine->UploadBuffer(
			deferred.Texture.DeviceBuf, 0U, poison.data(), poison.size());
		EXPECT_TRUE(poisonStatus.IsOk()) << poisonStatus.ToString();
		return deferred;
	};

	auto cvSource = makeDeferred();
	ASSERT_TRUE(cvSource.Texture.IsValid());
	OaCvFrame frame;
	frame.Base = &cvSource.Texture.DeviceBuf;
	frame.W = 2;
	frame.H = 2;
	auto compositeResult = frame.Render(*engine);
	ASSERT_TRUE(compositeResult.IsOk())
		<< compositeResult.GetStatus().ToString();
	OaTexture composite = OaStdMove(*compositeResult);
	std::array<OaU8, 16> compositePixels{};
	ASSERT_TRUE(engine->ReadbackBuffer(
		composite.DeviceBuf, 0U,
		compositePixels.data(), compositePixels.size()).IsOk());
	EXPECT_EQ(compositePixels, expected);
	composite.Destroy(*engine);
	cvSource.Texture.Destroy(*engine);

	auto fileSource = makeDeferred();
	ASSERT_TRUE(fileSource.Texture.IsValid());
	const char* imagePath = "/tmp/oa_deferred_texture.png";
	ASSERT_TRUE(OaFnImage::SaveFile(
		*engine, fileSource.Texture, imagePath).IsOk());
	auto loadedImageResult = OaTexture::LoadFile(*engine, imagePath);
	ASSERT_TRUE(loadedImageResult.IsOk())
		<< loadedImageResult.GetStatus().ToString();
	OaTexture loadedImage = OaStdMove(*loadedImageResult);
	std::array<OaU8, 16> loadedPixels{};
	ASSERT_TRUE(engine->ReadbackBuffer(
		loadedImage.DeviceBuf, 0U,
		loadedPixels.data(), loadedPixels.size()).IsOk());
	EXPECT_EQ(loadedPixels, expected);
	loadedImage.Destroy(*engine);
	fileSource.Texture.Destroy(*engine);
	std::remove(imagePath);

	auto figureSource = makeDeferred();
	ASSERT_TRUE(figureSource.Texture.IsValid());
	const char* figurePath = "/tmp/oa_deferred_figure.png";
	{
		OaPlot::Figure figure({
			.Rows = 1,
			.Cols = 1,
			.Width = 32,
			.Height = 32,
			.HSpacing = 0,
			.VSpacing = 0,
			.Padding = 0,
		});
		figure.Ax(0, 0).Imshow(figureSource.Texture);
		ASSERT_TRUE(figure.SaveFig(figurePath).IsOk());
	}
	auto loadedFigureResult = OaTexture::LoadFile(*engine, figurePath);
	ASSERT_TRUE(loadedFigureResult.IsOk())
		<< loadedFigureResult.GetStatus().ToString();
	OaTexture loadedFigure = OaStdMove(*loadedFigureResult);
	std::array<OaU8, 32U * 32U * 4U> figurePixels{};
	ASSERT_TRUE(engine->ReadbackBuffer(
		loadedFigure.DeviceBuf, 0U,
		figurePixels.data(), figurePixels.size()).IsOk());
	const auto expectPixel = [&](OaU32 InX, OaU32 InY, OaU32 InSourcePixel) {
		const OaUsize actualOffset = (InY * 32U + InX) * 4U;
		const OaUsize expectedOffset = InSourcePixel * 4U;
		for (OaUsize channel = 0U; channel < 4U; ++channel) {
			EXPECT_EQ(figurePixels[actualOffset + channel],
				expected[expectedOffset + channel]);
		}
	};
	expectPixel(8U, 8U, 0U);
	expectPixel(24U, 8U, 1U);
	expectPixel(8U, 24U, 2U);
	expectPixel(24U, 24U, 3U);
	loadedFigure.Destroy(*engine);
	figureSource.Texture.Destroy(*engine);
	std::remove(figurePath);
}
