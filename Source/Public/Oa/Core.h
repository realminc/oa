// OA CORE - All Foundation Types
//
// Convenience header that includes all core components.
// Like Vulkan's vulkan_core.h - one include for everything foundational.
//
// You can also include individual files if you want minimal dependencies:
//   #include <Oa/Core/Types.h>   - Scalars, strings, containers
//   #include <Oa/Core/Status.h>  - OaStatus, OaResult
//   #include <Oa/Core/Math.h>    - OaFixed, OaPrice, OaQty
//   #include <Oa/Core/Std/Chrono.h>  - OaSteadyNow, OaHighResolutionNow, duration helpers
//   #include <Oa/Core/Time.h>    - OaTimestamp, OaDuration
//   #include <Oa/Core/Device.h>  - OaDevice, GPU abstraction
//   #include <Oa/Core/Log.h>     - OaLog, logging macros
//   #include <Oa/Core/Validation.h> - OA_VALIDATE, debug counters

#pragma once

#include <Oa/Core/Types.h>      // Foundation: scalars, strings, containers (includes Type.gen.h)
#include <Oa/Core/Status.h>     // Error handling: OaStatus, OaResult
#include <Oa/Core/Math.h>       // Fixed-point: OaPrice, OaQty, OaBalance
#include <Oa/Core/Std/Chrono.h> // std::chrono aliases: OaSteadyNow, OaChronoToMilli, ...
#include <Oa/Core/Time.h>       // Timestamps: OaTimestamp, OaDuration
#include <Oa/Core/Device.h>     // GPU abstraction: OaDevice, OaDeviceInfo
#include <Oa/Core/Log.h>        // Logging: OaLog, OaLogMetrics, OA_LOG_* macros
#include <Oa/Core/Validation.h> // Validation: OA_VALIDATE, OA_WARN_PERF, debug counters
#include <Oa/Core/Cli.h>        // CLI: OaCli<T>, 3-way precedence (CLI11)
#include <Oa/Core/Memory.h>     // Fast memcpy: OaMemcpy, OaMemset (AVX/SSE)
#include <Oa/Core/Filesystem.h> // Host filesystem I/O: OaFilesystem
#include <Oa/Core/Paths.h>      // Named locations: OaPaths::Asset/Var/Temp
#include <Oa/Core/MappedFile.h> // Read-only whole-file mapping with checked spans
#include <Oa/Core/Thread.h>     // Threading: OaThreadPool, OaChannel, OaTask, OaRwLock, OaSpinlock
#include <Oa/Core/Yaml.h>       // OaYaml: Get, GetList, LoadFile (yaml-cpp)
#include <Oa/Core/Config.h>     // Checkpoint/log YAML: OaLoadCheckpointYaml, OaLoadLogYaml
#include <Oa/Core/Simd.h>       // SIMD: OaSimd::DotF32, OaSimd::ScaleF32 (Google Highway)
#include <Oa/Core/MatrixShape.h>      // OaMatrixShape, OA_MAX_TENSOR_DIMS
// Complex.h removed along with the SSM module.
#include <Oa/Core/Constant.h>   // Branding: ASCII banners, app subtitles, crypto info

#include <Oa/Core/Matrix.h>         // OaStride, OaMatrix, OaMemoryBlock
#include <Oa/Core/MatrixStorage.h>  // OaMatrixStorage
#include <Oa/Core/FnMatrix.h>       // OaFnMatrix
#include <Oa/Core/MatrixRef.h>      // OaMatrixRef
#include <Oa/Core/MatrixList.h>     // OaMatrixList
#include <Oa/Core/Image.h>          // OaImage, OaImageLayout, OaImageFormat
