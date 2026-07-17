#include "../OaTest.h"

#include <Oa/Runtime/Engine.h>
#include <Oa/Ui/Input.h>
#include <Oa/Ui/Plot/Plot.h>
#include <Oa/Ui/Ui.h>

#include <array>
#include <cstdio>
#include <fstream>

namespace {

class TestUi : public ::testing::Test {};

} // namespace

TEST_VK(TestUi, TimelinePointerDragAndRelease)
{
	auto* engine = OaComputeEngine::GetGlobal();
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
	auto* engine = OaComputeEngine::GetGlobal();
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
