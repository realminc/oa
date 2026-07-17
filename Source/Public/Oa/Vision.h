// OA Vision — Umbrella Header
// Hardware video decode/encode + image processing operators
// Portable vision-preprocess layer built on Vulkan compute shaders
//
// OaFnImage vision overloads (Resize, Normalize, GaussianBlur, etc.) are declared
// in <Oa/Core/MatrixFn.h> and do not need a separate header here.

#pragma once

#include <Oa/Vision/VideoDecoder.h>
#include <Oa/Vision/Video.h>
#include <Oa/Vision/VideoEncoder.h>
#include <Oa/Vision/VideoRecorder.h>
#include <Oa/Vision/ScreenCapture.h>
#include <Oa/Vision/CameraCapture.h>
#include <Oa/Vision/FnImage.h>
#include <Oa/Vision/FnVideo.h>
#include <Oa/Vision/JpegDecoder.h>
#include <Oa/Vision/Detection.h>
#include <Oa/Vision/FnDetection.h>
