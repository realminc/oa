// OA CORE - All Foundation Types
//
// Convenience header that includes all core components.
// Like vulkan's vulkan_core.h - one include for everything foundational.
//
// You can also include individual files if you want minimal dependencies:
//   #include <oa/core/types.h>   - Scalars, strings, containers
//   #include <oa/core/status.h>  - oa::Status, oa::Result
//   #include <oa/core/math.h>    - oa::Fixed, oa::Price, oa::Qty
//   #include <oa/core/std/chrono.h>  - oa::steadyNow, oa::highResolutionNow, duration helpers
//   #include <oa/core/time.h>    - oa::Timestamp, oa::Stopwatch
//   #include <oa/core/device.h>  - oa::Device, GPU abstraction
//   #include <oa/core/log.h>     - oa::Log, logging macros
//   #include <oa/core/validation.h> - OA_VALIDATE, debug counters

#pragma once

#include <oa/core/types.h>      // Foundation: scalars, strings, containers (includes type.gen.h)
#include <oa/core/status.h>     // Error handling: oa::Status, oa::Result
#include <oa/core/math.h>       // Fixed-point: oa::Price, oa::Qty, oa::Balance
#include <oa/core/std/chrono.h> // OA clock values: oa::steadyNow, oa::Duration, ...
#include <oa/core/time.h>       // Timestamps: oa::Timestamp, oa::Stopwatch
#include <oa/core/device.h>     // Device placement: oa::Device, oa::DeviceType
#include <oa/core/log.h>        // Logging: oa::Log, oa::LogMetrics, OaLog* macros
#include <oa/core/validation.h> // Validation: OA_VALIDATE, OA_WARN_PERF, debug counters
#include <oa/core/cli.h>        // CLI: oa::Cli<T>, native 3-way precedence
#include <oa/core/memory.h>     // Fast memcpy: oa::memcpy, oa::memset (AVX/SSE)
#include <oa/core/filesystem.h> // Host filesystem I/O: oa::Filesystem
#include <oa/core/paths.h>      // Named locations: oa::Paths::asset/data/var/temp
#include <oa/core/mappedFile.h> // Read-only whole-file mapping with checked spans
#include <oa/core/thread.h>     // Threading: oa::ThreadPool, oa::Channel, oa::Task, oa::RwLock, oa::Spinlock
#include <oa/core/yaml.h>       // oa::Yaml: get, getList, loadFile (yaml-cpp)
#include <oa/core/config.h>     // checkpoint/log YAML: oa::loadCheckpointYaml, oa::loadLogYaml
#include <oa/core/simd.h>       // SIMD: oa::FnSimd::dotF32, oa::FnSimd::scaleF32
#include <oa/core/matrixShape.h>      // oa::MatrixShape, OA_MAX_TENSOR_DIMS
// Complex.h removed along with the SSM module.
#include <oa/core/constant.h>   // Branding: ASCII banners, app subtitles, crypto info

#include <oa/core/matrix.h>         // oa::Stride, oa::Matrix, oa::MemoryBlock
#include <oa/core/fnMatrix.h>       // oa::FnMatrix
#include <oa/core/image.h>          // oa::Image, oa::ImageLayout, oa::ImageFormat
