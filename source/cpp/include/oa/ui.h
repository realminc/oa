// OA Ui — GPU-accelerated UI framework. Include this for the full surface.
//
// Sub-headers may be included individually for faster compilation:
//   <oa/ui/style.h>   — oa::Color, oa::UiStyle, theme presets
//   <oa/ui/event.h>   — oa::UiEvent, oa::UiInputState, oa::UiKey
//   <oa/ui/canvas.h>  — oa::vlm::Vec2, oa::PixelRect, oa::NodeCanvas
//   <oa/ui/ui.h>      — oa::Ui widget API
//   <oa/ui/renderConfig.h> — oa::Renderer UI-composition configuration
//   <oa/render/renderer.h> — completion-safe render-to-texture session
//   <oa/ui/text.h>    — oa::TextAtlas, oa::TextLayout
//   <oa/ui/plot/plot.h> — oa::plot::Figure and oa::plot::Axes
//   <oa/ui/cv.h>      — diagnostic oa::CvFrame composition
//   <oa/ui/detectionOverlay.h> — GPU boxes and SDF labels
//   Camera capture is owned by <oa/vision/cameraCapture.h>.
//   <oa/ui/input.h>   — oa::InputSystem, oa::KeyAction
//   <oa/ui/platformInput.h> — low-level input snapshots and gesture classification
//   <oa/ui/navigation.h> — 2D viewer pan/zoom state
//   <oa/ui/motion.h> — shared widget/navigation animation-speed policy
//   <oa/runtime/texture.h> — oa::Texture consumed by oa::Ui::image()
//   <oa/ui/viewport.h> — passive camera/target/view description
//   <oa/ui/viewer.h>  — the single image/video inspection application
//   <oa/ui/trainingViewer.h> — immutable oa::TrainingSession dashboard adapter

#pragma once

#include <oa/ui/style.h>
#include <oa/ui/motion.h>
#include <oa/ui/event.h>
#include <oa/ui/canvas.h>
#include <oa/ui/text.h>
#include <oa/ui/ui.h>
#include <oa/ui/renderConfig.h>
#include <oa/render/renderer.h>
#include <oa/ui/input.h>
#include <oa/ui/platformInput.h>
#include <oa/ui/navigation.h>
#include <oa/ui/plot/plot.h>
#include <oa/ui/cv.h>
#include <oa/ui/detectionOverlay.h>
#include <oa/ui/capture.h>
#include <oa/ui/image.h>
#include <oa/ui/viewport.h>
#include <oa/ui/viewer.h>
#include <oa/ui/trainingViewer.h>
