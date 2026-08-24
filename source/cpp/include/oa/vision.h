// OA Vision — Umbrella header
// Images, video capture, codec/container sessions, playback, and recording.
// Portable vision-preprocess layer built on vulkan compute shaders
//
// oa::FnImage owns stateless image transforms and still-image codec boundaries.

#pragma once

#include <oa/vision/videoDecoder.h>
#include <oa/vision/videoEncoder.h>
#include <oa/vision/videoDemuxer.h>
#include <oa/vision/videoMuxer.h>
#include <oa/vision/videoPlayer.h>
#include <oa/vision/videoRecorder.h>
#include <oa/vision/screenCapture.h>
#include <oa/vision/cameraCapture.h>
#include <oa/vision/fnImage.h>
#include <oa/vision/fnVideo.h>
#include <oa/vision/detection.h>
#include <oa/vision/fnDetection.h>
