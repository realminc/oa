#pragma once

// Vulkan record graph: OaDeviceDgNode base + concrete nodes + OaDeviceVkRecord helpers.

#include <Oa/Core/Node/DeviceDgNode.h>
#include <Oa/Core/Node/DeviceAddNode.h>
#include <Oa/Core/Node/DeviceCrossEntropyMeanNode.h>
#include <Oa/Core/Node/DeviceGatherRowsNode.h>
#include <Oa/Core/Node/DeviceMatmulNode.h>
#include <Oa/Core/Node/DeviceMulNode.h>
#include <Oa/Core/Node/DeviceScaleNode.h>
#include <Oa/Core/Node/DeviceSubNode.h>
#include <Oa/Core/Node/DeviceVkRecord.h>
