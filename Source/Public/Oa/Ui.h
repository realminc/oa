// OaUi — GPU-accelerated UI framework.  Include this for the full surface.
//
// Sub-headers may be included individually for faster compilation:
//   <Oa/Ui/Style.h>   — OaUiColor, OaUiStyle, theme presets
//   <Oa/Ui/Event.h>   — OaUiEvent, OaUiInputState, OuiKey
//   <Oa/Ui/Canvas.h>  — VlmVec2, OaPixelRect, OaNodeCanvas
//   <Oa/Ui/Ui.h>     — OaUi widget API
//   <Oa/Ui/Viz.h>     — OaHeatmap, OaChart, OaBranchViz, OaTokenGrid
//   <Oa/Ui/Text.h>    — OaTextAtlas, OaTextLayout
//   <Oa/Ui/Node.h>    — OaUiNode, OaNodeGraph
//   <Oa/Ui/DeviceUi.h>    — OaDeviceUi, OaDeviceUiApp, OaUiComposeImage, OaUiConfig
//   <Oa/Ui/Plot.h>    — OaPlot, OaFigure, OaAxes, OaColor
//   <Oa/Ui/Cv.h>      — OaCvFrame, OaCvOps
//   <Oa/Ui/Capture.h> — legacy OaCapture alias, OaScreenshot
//   <Oa/Ui/Input.h>   — OaInputSystem, OaKeyAction
//   <Oa/Ui/Image.h>   — OaTexture (upload RGBA8 → GPU buffer for OaUi::Image())

#pragma once

#include <Oa/Ui/Style.h>
#include <Oa/Ui/Event.h>
#include <Oa/Ui/Canvas.h>
#include <Oa/Ui/Text.h>
#include <Oa/Ui/Ui.h>
#include <Oa/Ui/Viz.h>
#include <Oa/Ui/Node.h>
#include <Oa/Ui/Input.h>
#include <Oa/Ui/DeviceUi.h>
#include <Oa/Ui/Plot.h>
#include <Oa/Ui/Cv.h>
#include <Oa/Ui/Rl.h>
#include <Oa/Ui/Capture.h>
#include <Oa/Ui/Image.h>
