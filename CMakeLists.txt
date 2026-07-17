# OA — REALM FOUNDATION LIBRARY
#
# Vulkan 1.4 compute. Slang shaders. One backend across NVIDIA, AMD, and Intel.
cmake_minimum_required(VERSION 3.20)

# ─── Version: single source of truth ───
# The top-level VERSION file is the ONE place the version lives — read here by CMake and
# by the Python package (Source/Python/oa/__init__.py). It may carry a pre-release suffix
# (e.g. "0.7.0-dev"); strip it to numeric X.Y.Z for project(), keep the full string for
# the OA_VERSION_STRING compile definition consumed by <Oa/Core/Version.h>.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" OA_VERSION_FULL LIMIT_COUNT 1)
string(STRIP "${OA_VERSION_FULL}" OA_VERSION_FULL)
if(OA_VERSION_FULL MATCHES "^([0-9]+[.][0-9]+[.][0-9]+)")
	set(OA_VERSION_NUMERIC "${CMAKE_MATCH_1}")
else()
	set(OA_VERSION_NUMERIC "0.0.0")
	set(OA_VERSION_FULL "0.0.0-unknown")
endif()

# ─── vcpkg manifest features ───
# Optional dependencies live behind vcpkg.json features (python → nanobind,
# database → duckdb). vcpkg resolves the manifest during project() below, so the
# feature selection MUST happen here — before project() — by reading the build
# options straight from the cache (they arrive via -D before option() formalizes
# them further down). A default configure enables neither, so no extra ports are
# fetched unless the matching option is explicitly turned on.
if(OA_BUILD_PYTHON)
	list(APPEND VCPKG_MANIFEST_FEATURES "python")
endif()
if(OA_WITH_DATABASE)
	list(APPEND VCPKG_MANIFEST_FEATURES "database")
endif()
if(OA_BUILD_CRYPTO)
	list(APPEND VCPKG_MANIFEST_FEATURES "crypto")
endif()

project(oa
	VERSION ${OA_VERSION_NUMERIC}
	DESCRIPTION "Realm Foundation Library — Vulkan compute, ML, Crypto"
	LANGUAGES C CXX
)

# Full version string (incl. any pre-release suffix) available to all C/C++ sources.
add_compile_definitions(OA_VERSION_STRING="${OA_VERSION_FULL}")

# ─── Submodule Detection ───
# When oa is included via add_subdirectory() from ml/ or chain/, we ALWAYS embed
# OA's core kernels into liboa.a. Parent projects use --whole-archive linker flag
# to force-link all OA symbols (including embedded shaders) into their executables.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	message(STATUS "oa: running as submodule — OA kernels will be embedded in liboa (parent uses --whole-archive)")
else()
	message(STATUS "oa: running as standalone project — OA kernels will be embedded in liboa")
endif()

# OPTIONS
option(OA_BUILD_TESTS "Build unit tests" ON)
option(OA_OASTD_EXTRA_TARGETS "Optional: oa_oastd_headers (IDE file list) + oastd -> TestOaStd" ON)
option(OA_VULKAN_VALIDATION "Enable Vulkan validation layers" OFF)
option(OA_BUILD_CRYPTO "Build crypto implementations (requires pqcrypto)" OFF)
option(OA_EMBED_SHADERS "Embed compute shaders (ML+crypto) into liboa (ON=release, OFF=file loading)" ON)
option(OA_WITH_DATABASE "Embedded OHLCV SQL store (DuckDB in Source/Private/Oa/Data/Database.cpp only; vcpkg feature database)" OFF)
option(OA_BUILD_PYTHON "Build Python bindings via nanobind (vcpkg feature python)" OFF)
option(OA_BUILD_SHARED "Build liboa as a shared library (required for distributable runtime packages)" OFF)
option(OA_BUILD_TUTORIALS "Build the tutorial executables (installable as oa-* CLI tools)" OFF)
option(OA_ANDROID_ML "Build the Android-safe Runtime + Core + ML source profile" ${ANDROID})

# COMPILER SETTINGS — CLANG ONLY
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	message(WARNING "OA requires Clang. Detected: ${CMAKE_CXX_COMPILER_ID}")
	if(OA_STRICT_COMPILER)
		message(FATAL_ERROR "GCC is not supported. Use Clang.")
	endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(Source)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=unused-result -Wno-error=unused-parameter -Wno-error=missing-field-initializers)
	if(CMAKE_BUILD_TYPE STREQUAL "Release")
		add_compile_options(-O3)
		if(NOT ANDROID)
			add_compile_options(-march=native)
		endif()
	endif()
else()
	# GCC configuration
	add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror=unused-result -Wno-error=unused-parameter -Wno-error=missing-field-initializers)
	if(CMAKE_BUILD_TYPE STREQUAL "Release")
		add_compile_options(-O3)
		if(NOT ANDROID)
			add_compile_options(-march=native)
		endif()
	endif()
endif()

# ─── Binary output: Bin/{preset}/ and Bin/{preset}/Test/ ────────────────────
cmake_path(GET CMAKE_BINARY_DIR FILENAME OA_PRESET_NAME)
set(OA_BIN_DIR "${CMAKE_SOURCE_DIR}/Bin/${OA_PRESET_NAME}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}")
if(OA_BUILD_TESTS)
	file(MAKE_DIRECTORY "${OA_BIN_DIR}/Test")
endif()

add_compile_definitions(OA_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_definitions(OA_SPIRV_DIR="${CMAKE_CURRENT_BINARY_DIR}/spirv")
if(WIN32)
	add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

# VULKAN SDK + SLANG COMPILER — Required. No CPU-only builds.
if(NOT ANDROID)
	find_package(VulkanHeaders CONFIG)
	if(NOT VulkanHeaders_FOUND)
		find_package(Vulkan)
		if(Vulkan_FOUND)
			set(VULKAN_INCLUDE_DIR "${Vulkan_INCLUDE_DIRS}")
		else()
			message(FATAL_ERROR "Neither VulkanHeaders (vcpkg) nor Vulkan SDK found")
		endif()
	endif()
else()
	# The NDK provides core Vulkan headers but omits the standard vk_video
	# include set referenced by OA's generated loader. Vulkan-Headers is
	# architecture-neutral; use the workstation package for declarations only.
	unset(OA_ANDROID_VULKAN_HEADERS CACHE)
	find_path(OA_ANDROID_VULKAN_HEADERS vk_video/vulkan_video_codec_av1std.h
		PATHS "$ENV{HOME}/.vcpkg/installed/x64-linux/include"
		NO_DEFAULT_PATH
		NO_CMAKE_FIND_ROOT_PATH)
	if(NOT OA_ANDROID_VULKAN_HEADERS)
		message(FATAL_ERROR "Android OA build requires Vulkan-Headers (vk_video includes)")
	endif()
endif()

# Prefer a standalone Slang install (bf16/fp8 fixes land in shader-slang releases, often ahead of Vulkan SDK).
# https://github.com/shader-slang/slang/releases/ — extract e.g. slang-2026.5.2-linux-x86_64.tar.gz and set SLANG_ROOT or PATH.
find_program(SLANGC slangc HINTS
	"$ENV{SLANG_ROOT}/bin"
	"$ENV{HOME}/tools/slang/bin"
	"$ENV{VCPKG_INSTALLED_DIR}/x64-linux/tools/shader-slang"
	"$ENV{HOME}/.vcpkg/vcpkg/installed/x64-linux/tools/shader-slang"
	"$ENV{VCPKG_ROOT}/installed/x64-windows/tools/shader-slang"
	"$ENV{VCPKG_ROOT}/installed/x64-windows-static/tools/shader-slang"
	"$ENV{VULKAN_SDK}/bin"
	"/usr/bin")
find_program(GLSLANG_VALIDATOR glslangValidator)
if(SLANGC)
	execute_process(COMMAND "${SLANGC}" -version
		OUTPUT_VARIABLE OA_SLANG_VERSION
		ERROR_VARIABLE OA_SLANG_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE OA_SLANG_VER_RC)
	string(STRIP "${OA_SLANG_VERSION}" OA_SLANG_VERSION)
	if(OA_SLANG_VER_RC EQUAL 0 AND OA_SLANG_VERSION)
		message(STATUS "slangc: ${SLANGC} (Slang ${OA_SLANG_VERSION})")
	else()
		message(STATUS "slangc: ${SLANGC}")
	endif()
elseif(OA_EMBED_SHADERS)
	message(FATAL_ERROR "OA_EMBED_SHADERS=ON but slangc not found. Install Slang or set -DSLANGC=<path>.")
endif()

if(OA_VULKAN_VALIDATION)
	add_compile_definitions(OA_VULKAN_VALIDATION)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Slang feature-test compiles (PR-1 item A from OaLlamaCppVulkanLessons.md §9).
# Each test compiles a minimal .slang file that requires exactly one feature.
# If slangc rejects it, the corresponding OA_HAVE_*_SLANG cache var is OFF and
# the dependent code paths (SPV embed list entries, runtime ext enable) drop
# out of the build cleanly. Same source tree builds correctly on a host with
# old slangc as a host with new slangc.
# Mirrors llama.cpp's test_shader_extension_support in
# ggml/src/ggml-vulkan/CMakeLists.txt:38-57.
# ─────────────────────────────────────────────────────────────────────────────
function(test_slang_extension_support FEATURE_NAME TEST_SHADER RESULT_VAR)
	if(NOT SLANGC)
		set(${RESULT_VAR} OFF PARENT_SCOPE)
		return()
	endif()
	set(_OA_SLANG_CAP_ARGS "")
	foreach(_CAP ${ARGN})
		list(APPEND _OA_SLANG_CAP_ARGS -capability ${_CAP})
	endforeach()
	# Need both include paths (umbrella + math modules) so basic imports resolve.
	set(_OA_FT_INC_UMBRELLA "${CMAKE_CURRENT_SOURCE_DIR}/Source/Public/Oa/Core")
	set(_OA_FT_INC_MODULES  "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Math")
	execute_process(
		COMMAND ${SLANGC} ${TEST_SHADER}
			-I ${_OA_FT_INC_UMBRELLA} -I ${_OA_FT_INC_MODULES}
			-target spirv -entry main ${_OA_SLANG_CAP_ARGS}
			-o ${CMAKE_BINARY_DIR}/_slang_feature_test.spv
		OUTPUT_VARIABLE _slang_out
		ERROR_VARIABLE  _slang_err
		RESULT_VARIABLE _slang_rc
	)
	if(_slang_rc EQUAL 0)
		message(STATUS "slang feature: ${FEATURE_NAME} supported (${RESULT_VAR}=ON)")
		set(${RESULT_VAR} ON PARENT_SCOPE)
		add_compile_definitions(${RESULT_VAR})
	else()
		message(STATUS "slang feature: ${FEATURE_NAME} NOT supported (${RESULT_VAR}=OFF)")
		set(${RESULT_VAR} OFF PARENT_SCOPE)
	endif()
endfunction()

function(test_glsl_extension_support FEATURE_NAME TEST_SHADER RESULT_VAR)
	if(NOT GLSLANG_VALIDATOR)
		message(STATUS "glsl feature: ${FEATURE_NAME} NOT supported (${RESULT_VAR}=OFF, glslangValidator not found)")
		set(${RESULT_VAR} OFF PARENT_SCOPE)
		return()
	endif()
	execute_process(
		COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.4 ${TEST_SHADER}
			-o ${CMAKE_BINARY_DIR}/_glsl_feature_test.spv
		OUTPUT_VARIABLE _glsl_out
		ERROR_VARIABLE  _glsl_err
		RESULT_VARIABLE _glsl_rc
	)
	if(_glsl_rc EQUAL 0)
		message(STATUS "glsl feature: ${FEATURE_NAME} supported (${RESULT_VAR}=ON)")
		set(${RESULT_VAR} ON PARENT_SCOPE)
		add_compile_definitions(${RESULT_VAR})
	else()
		message(STATUS "glsl feature: ${FEATURE_NAME} NOT supported (${RESULT_VAR}=OFF)")
		set(${RESULT_VAR} OFF PARENT_SCOPE)
	endif()
endfunction()

set(_OA_FEATURE_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/feature-tests")
test_slang_extension_support("VK_KHR_cooperative_matrix"
	"${_OA_FEATURE_TESTS_DIR}/CoopMatKhr.slang"   OA_HAVE_COOPMAT_KHR_SLANG
	spvCooperativeMatrixKHR)
test_slang_extension_support("VK_KHR_shader_bfloat16 (type only)"
	"${_OA_FEATURE_TESTS_DIR}/ShaderBfloat16.slang" OA_HAVE_BFLOAT16_SLANG)
test_slang_extension_support("VK_KHR_shader_bfloat16 + KHR_cooperative_matrix"
	"${_OA_FEATURE_TESTS_DIR}/CoopMatBf16.slang"  OA_HAVE_COOPMAT_BF16_SLANG
	spvCooperativeMatrixKHR spvCooperativeMatrixConversionsNV)
test_slang_extension_support("VK_KHR_shader_integer_dot_product"
	"${_OA_FEATURE_TESTS_DIR}/IntegerDotProduct.slang" OA_HAVE_INTEGER_DOT_PRODUCT_SLANG
	spvIntegerDotProduct)
test_slang_extension_support("VK_NV_cooperative_vector"
	"${_OA_FEATURE_TESTS_DIR}/CoopVecNv.slang"    OA_HAVE_COOPVEC_NV_SLANG
	spvCooperativeVectorNV)
test_slang_extension_support("VK_NV_cooperative_matrix2"
	"${_OA_FEATURE_TESTS_DIR}/CoopMat2Nv.slang"   OA_HAVE_COOPMAT2_NV_SLANG
	SPV_NV_cooperative_matrix2 spvCooperativeMatrixTensorAddressingNV)
test_glsl_extension_support("GL_NV_cooperative_matrix2"
	"${_OA_FEATURE_TESTS_DIR}/CoopMat2Nv.comp"    OA_HAVE_COOPMAT2_NV_GLSL)
# Pure-KHR bf16 CoopMat1 in GLSL — the portable SMEM GEMM's exact feature set
# (no NV extension). Available on NVIDIA and AMD RDNA3.5/Strix.
test_glsl_extension_support("GL_KHR_cooperative_matrix + GL_EXT_bfloat16"
	"${_OA_FEATURE_TESTS_DIR}/CoopMat1Bf16Khr.comp" OA_HAVE_COOPMAT1_BF16_KHR_GLSL)


# NVTX profiling support (CUDA toolkit)
find_path(NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
	HINTS
		/opt/cuda/targets/x86_64-linux/include
		/usr/local/cuda/targets/x86_64-linux/include
		$ENV{CUDA_PATH}/include
)
if(NVTX_INCLUDE_DIR)
	message(STATUS "NVTX found: ${NVTX_INCLUDE_DIR}")
	include_directories(${NVTX_INCLUDE_DIR})
else()
	message(STATUS "NVTX not found - profiling markers will be disabled")
endif()

# DEPENDENCIES
if(NOT ANDROID)
	find_package(yaml-cpp CONFIG)
	find_package(CLI11 CONFIG)
	find_package(fmt CONFIG)
	find_package(hwy CONFIG)
	find_package(utf8proc CONFIG REQUIRED)
endif()

# HEADER-ONLY INTERFACE LIBRARY (transitive oa_api + optional CLI/YAML/fmt)
add_library(oa_headers INTERFACE)
target_link_libraries(oa_headers INTERFACE oa_api)
target_compile_features(oa_headers INTERFACE cxx_std_20)
if(yaml-cpp_FOUND)
	target_link_libraries(oa_headers INTERFACE yaml-cpp::yaml-cpp)
	target_compile_definitions(oa_headers INTERFACE OA_HAS_YAML_CPP)
endif()
if(CLI11_FOUND)
	target_link_libraries(oa_headers INTERFACE CLI11::CLI11)
	target_compile_definitions(oa_headers INTERFACE OA_HAS_CLI11)
endif()
if(fmt_FOUND)
	target_link_libraries(oa_headers INTERFACE fmt::fmt)
	target_compile_definitions(oa_headers INTERFACE OA_HAS_FMT)
endif()

# Vendored GLM for Vlm.h (transitioning to native VLM implementation)
target_include_directories(oa_headers SYSTEM INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Source/ThirdParty/glm>)

include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/FnMatrix/FnMatrixSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/FnMatrix/FnMatrixSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/Nn/NnSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Audio/FnAudio/FnAudioSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Vision/FnImage/FnImageSources.gen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Vision/FnVideo/FnVideoSources.gen.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ui/Matrix/FnMatrixSources.gen.cmake OPTIONAL)
include(${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Crypto/Hash/FnSources.gen.cmake OPTIONAL)

# OA LIBRARY (implementation: Source/Private/Oa — PascalCase .cpp/.c)
set(OA_SOURCES
	# Core — extension and kernel registry
	Source/Private/Oa/Core/KernelRegistry.cpp

	# Core
	Source/Private/Oa/Core/EnvFlag.cpp
	Source/Private/Oa/Core/Validation.cpp
	Source/Private/Oa/Core/Memory.cpp
	Source/Private/Oa/Core/Image.cpp
	Source/Private/Oa/Core/Color.cpp
	Source/Private/Oa/Core/Matrix.cpp
	Source/Private/Oa/Core/MatrixStorage.cpp
	Source/Private/Oa/Core/MatrixRef.cpp
	Source/Private/Oa/Core/MatrixList.cpp
	Source/Private/Oa/Core/Autograd.cpp
	Source/Private/Oa/Core/FnMatrix/FnMatrix.cpp
	Source/Private/Oa/Core/FnMatrix/Alloc/FnMatrixAlloc.cpp
	Source/Private/Oa/Core/FnMatrix/View/FnMatrixView.cpp
	# FnMatrix — modularized by category (GPU-only, pure Vulkan)
	Source/Private/Oa/Core/FnMatrix/Elemwise/FnMatrixElemwise.cpp
	Source/Private/Oa/Core/FnMatrix/Blas/FnMatrixBlas.cpp
	Source/Private/Oa/Core/FnMatrix/Reduce/FnMatrixReduce.cpp
	Source/Private/Oa/Core/FnMatrix/Index/FnMatrixIndex.cpp
	Source/Private/Oa/Core/FnMatrix/Rng/FnMatrixRng.cpp
	# FnMatrixFused.cpp removed — fusion is a runtime/matcher concern (OaBlasGemm.md Phase 5)

	# FnMatrix — AUTOGENERATED from OaFnAutogen schemas (100% coverage)
	${OA_CORE_FN_MATRIX_GENERATED_SOURCES}
	Source/Private/Oa/Runtime/Gemm/Dispatch.cpp
	Source/Private/Oa/Runtime/Gemm/Registry.cpp
	Source/Private/Oa/Runtime/Gemm/Router.cpp
	Source/Private/Oa/Runtime/Gemm/Tuner.cpp
	Source/Private/Oa/Core/ByteLogitsHost.cpp
	Source/Private/Oa/Core/FileIo.cpp
	Source/Private/Oa/Core/MappedFile.cpp
	Source/Private/Oa/Core/PerfStat.cpp
	Source/Private/Oa/Core/PerfStore.cpp
	Source/Private/Oa/Data/Database.cpp
	Source/Private/Oa/Data/Dataframe.cpp
	Source/Private/Oa/Core/CpuTopology.cpp
	Source/Private/Oa/Core/ThreadPool.cpp
	Source/Private/Oa/Core/Simd.cpp
	Source/Private/Oa/Crypto/Hash.cpp
	Source/Private/Oa/Crypto/Keccak.cpp

	# ML — autograd (OaAutograd.md)
	Source/Private/Oa/Ml/Autograd/Core/AutogradCore.cpp

	# ML — utilities (data pipeline, encoding, checkpoints, .oam)
	# Source/Private/Oa/Ml/AdapterRegistry.cpp  # stripped — apps construct OaModule directly
	# Source/Private/Oa/Ml/ModelAdapter.cpp  # stripped — apps construct OaModule directly
	Source/Private/Oa/Ml/Checkpoint.cpp
	Source/Private/Oa/Ml/Budget.cpp
	Source/Private/Oa/Ml/Oam.cpp
	Source/Private/Oa/Ml/Optim/FnOptim.cpp
	Source/Private/Oa/Ml/Optim/MuonRef.cpp
	Source/Private/Oa/Ml/Byte/Encoder.cpp
	Source/Private/Oa/Data/Dataset.cpp
	Source/Private/Oa/Data/AsyncPipeline.cpp
	Source/Private/Oa/Data/IoUringReader.cpp
	Source/Private/Oa/Data/DsSharded.cpp
	Source/Private/Oa/Data/DsSubset.cpp
	Source/Private/Oa/Ml/Data/Oad.cpp
	Source/Private/Oa/Data/DsMnist.cpp
	Source/Private/Oa/Data/DsDailyDialog.cpp
	Source/Private/Oa/Data/DsHumanMl3d.cpp
	Source/Private/Oa/Data/FnDataset.cpp

	# ML — I/O formats (SafeTensors, ONNX)
	Source/Private/Oa/Ml/Io/TransferWeights.cpp
	Source/Private/Oa/Ml/Io/ShardedWeightSource.cpp
	Source/Private/Oa/Ml/Io/SafeTensorReader.cpp

	# Vulkan runtime
	Source/Private/Oa/Runtime/OaVk.c
	Source/Private/Oa/Runtime/OaVma.cpp
	Source/Private/Oa/Runtime/Device/Device.cpp
	Source/Private/Oa/Runtime/Instance.cpp
	Source/Private/Oa/Runtime/Device/DeviceUtils.cpp
	# Device modularization (PR-1)
	Source/Private/Oa/Runtime/Device/ComputeDevice.cpp
	Source/Private/Oa/Runtime/Device/RenderDevice.cpp
	Source/Private/Oa/Runtime/Device/DeviceBuilder.cpp
	Source/Private/Oa/Runtime/Device/Features/CoreFeatures.cpp
	Source/Private/Oa/Runtime/Device/Features/MlFeatures.cpp
	Source/Private/Oa/Runtime/Device/Features/VisionFeatures.cpp
	Source/Private/Oa/Runtime/Device/Features/AudioFeatures.cpp
	Source/Private/Oa/Runtime/Device/Features/RenderFeatures.cpp
	Source/Private/Oa/Runtime/Allocator.cpp
	Source/Private/Oa/Runtime/Pipeline.cpp
	Source/Private/Oa/Runtime/Bindless.cpp
	Source/Private/Oa/Runtime/Cmd.cpp
	Source/Private/Oa/Runtime/Sync.cpp
	Source/Private/Oa/Runtime/Stream.cpp
	Source/Private/Oa/Runtime/UploadRing.cpp
	Source/Private/Oa/Runtime/Prefetch.cpp
	Source/Private/Oa/Runtime/Timestamp.cpp
	Source/Private/Oa/Runtime/GpuTimer.cpp
	Source/Private/Oa/Runtime/Timer.cpp
	Source/Private/Oa/Runtime/Dispatch.cpp
	Source/Private/Oa/Runtime/ImageDispatch.cpp
	Source/Private/Oa/Runtime/ComputeGraph.cpp
	Source/Private/Oa/Runtime/Dnn.cpp
	Source/Private/Oa/Runtime/Context/ContextCore.cpp
	Source/Private/Oa/Runtime/Context/ContextLowering.cpp
	Source/Private/Oa/Runtime/Context/ContextExecute.cpp

	Source/Private/Oa/Runtime/Persistent.cpp
	Source/Private/Oa/Runtime/Spirv/Providers.cpp
	Source/Private/Oa/Runtime/Spirv/Reflect.cpp
	Source/Private/Oa/Runtime/ShaderProvider.cpp
	Source/Private/Oa/Runtime/Engine/Engine.cpp
	Source/Private/Oa/Runtime/Engine/ComputeEngine.cpp
	Source/Private/Oa/Runtime/Engine/GraphicsEngine.cpp
	Source/Private/Oa/Runtime/Distributed/Topology.cpp
	Source/Private/Oa/Runtime/Distributed/Scheduler.cpp
	Source/Private/Oa/Runtime/Distributed/Collective.cpp
	Source/Private/Oa/Runtime/Distributed/ExternalMemory.cpp
	Source/Private/Oa/Runtime/App.cpp
	Source/Private/Oa/Runtime/Window.cpp
	Source/Private/Oa/Runtime/Pool/InferencePool.cpp
	Source/Private/Oa/Runtime/Pool/PqcPool.cpp

	# ML — Matrix operations (OaFnMatrix namespace extensions)
	Source/Private/Oa/Ml/FnMatrix/Blas/FnMatrixBlas.cpp
	Source/Private/Oa/Ml/FnMatrix/Blas/FnMatrixPackedLinear.cpp
	Source/Private/Oa/Ml/FnMatrix/Norm/FnMatrixNorm.cpp
	Source/Private/Oa/Ml/FnMatrix/Modules/FnMatrixModules.cpp
	Source/Private/Oa/Ml/FnMatrix/Pool/FnMatrixPool.cpp
	Source/Private/Oa/Ml/FnMatrix/Quant/FnMatrixQuant.cpp
	Source/Private/Oa/Ml/FnMatrix/Ssm/FnMatrixMamba3.cpp
	Source/Private/Oa/Ml/FnMatrix/Ssm/FnMatrixEmpyrealm.cpp
	Source/Private/Oa/Ml/FnMatrix/Vq/FnMatrixVq.cpp
	Source/Private/Oa/Ml/FnMatrix/Bmm/FnMatrixBmm.cpp
	Source/Private/Oa/Ml/FnMatrix/Attention/FnMatrixAttention.cpp
	Source/Private/Oa/Ml/FnMatrix/Attention/FnMatrixFlashAttention.cpp
	Source/Private/Oa/Ml/FnMatrix/Moe/FnMatrixMoe.cpp
	Source/Private/Oa/Ml/FnMatrix/Nn/FnMatrixGru.cpp
	Source/Private/Oa/Ml/FnMatrix/Nn/FnMatrixRnn.cpp
	Source/Private/Oa/Ml/FnMatrix/Backward/FnMatrixBackward.cpp
	# FnMatrixOptim.cpp removed - optimizer functions now in Ml/Optim/ (OaFnOptim namespace)

	# ML — Loss functions (OaFnLoss namespace) — all hand-written with fused kernels
	Source/Private/Oa/Ml/FnLoss/FnLoss.cpp

	# ML — reinforcement learning
	Source/Private/Oa/Ml/Rl/Environment.cpp
	Source/Private/Oa/Ml/Rl/ActorCritic.cpp
	Source/Private/Oa/Ml/Rl/Collector.cpp
	Source/Private/Oa/Ml/Rl/DqnTrainer.cpp
	Source/Private/Oa/Ml/Rl/Gae.cpp
	Source/Private/Oa/Ml/Rl/Evaluation.cpp
	Source/Private/Oa/Ml/Rl/ItRlTraining.cpp
	Source/Private/Oa/Ml/Rl/Policy.cpp
	Source/Private/Oa/Ml/Rl/PpoTrainer.cpp
	Source/Private/Oa/Ml/Rl/Replay.cpp
	Source/Private/Oa/Ml/Rl/SacTrainer.cpp
	Source/Private/Oa/Ml/Rl/Transform.cpp
	Source/Private/Oa/Ml/Rl/FnLoss/Ppo.cpp
	Source/Private/Oa/Ml/Rl/FnLoss/Dqn.cpp
	Source/Private/Oa/Ml/Rl/FnLoss/Sac.cpp
	Source/Private/Oa/Ml/Rl/Rollout.cpp
	
	# ML — AUTOGENERATED from OaFnAutogen schemas
	${OA_ML_FN_MATRIX_GENERATED_SOURCES}

	# ML — nn modules (Vulkan dispatch)
	Source/Private/Oa/Ml/Nn/Module.cpp
	Source/Private/Oa/Ml/ItTraining.cpp
	Source/Private/Oa/Ml/TrainingSession.cpp
	Source/Private/Oa/Ml/TrainingProgram.cpp
	Source/Private/Oa/Ml/Generative/Flow.cpp
	Source/Private/Oa/Ml/Generative/FlowDenoiser.cpp
	Source/Private/Oa/Ml/Generative/FlowTransformer.cpp
	Source/Private/Oa/Ml/Metric.cpp
	Source/Private/Oa/Ml/Tokenizer.cpp
	Source/Private/Oa/Ml/NlpSuite.cpp
	${OA_NN_GENERATED_SOURCES}
	Source/Private/Oa/Ml/Nn/Activation.cpp
	Source/Private/Oa/Ml/Nn/Byte.cpp
	Source/Private/Oa/Ml/Nn/Rope/Rope.cpp
	# Source/Private/Oa/Ml/Nn/Ssm.cpp # Disabled: SSM not yet implemented (uses OaDeviceMatrix which doesn't exist)
	Source/Private/Oa/Ml/Nn/Mamba3/Mamba3.cpp
	# Hand-written modules (NOT schema-generated) — must live here, not in
	# NnSources.gen.cmake, or a `--live` NN regen drops them from the build.
	Source/Private/Oa/Ml/Nn/Ffn/Ffn.cpp
	Source/Private/Oa/Ml/Nn/Moe/Moe.cpp
	Source/Private/Oa/Ml/Nn/Gru/Gru.cpp
	Source/Private/Oa/Ml/Nn/Rnn/Rnn.cpp
	Source/Private/Oa/Ml/Nn/Transformer/Transformer.cpp
	Source/Private/Oa/Ml/Nn/Empyrealm/EmpyrealmCore.cpp
	Source/Private/Oa/Ml/Nn/Empyrealm/EmpyrealmModule.cpp
	Source/Private/Oa/Ml/Nn/Vq/Vq.cpp
	# SeqVqTokenizer and ConvVqTokenizer — moved to Extensions/
	# Note: Sequential, Attention, Vision (BatchNorm2d, AdaptiveAvgPool2d), Upsample are now generated
	
	# Ui — widget layer, present, image loading
	Source/Private/Oa/Ui/StbImageImpl.cpp
	Source/Private/Oa/Ui/Image.cpp
	Source/Private/Oa/Ui/Cv.cpp
	Source/Private/Oa/Ui/DetectionOverlay.cpp
	Source/Private/Oa/Ui/Canvas.cpp
	Source/Private/Oa/Ui/Input.cpp
	Source/Private/Oa/Ui/Text.cpp
	Source/Private/Oa/Ui/Node.cpp
	Source/Private/Oa/Ui/Ui.cpp
	Source/Private/Oa/Ui/DeviceUi.cpp
	Source/Private/Oa/Ui/DeviceUiApp.cpp
	Source/Private/Oa/Ui/Camera.cpp
	Source/Private/Oa/Ui/Viewport.cpp
	Source/Private/Oa/Ui/FnCamera.cpp

	# Core — shared interaction/navigation primitives
	Source/Private/Oa/Core/Input.cpp
	Source/Private/Oa/Core/Navigation.cpp
	Source/Private/Oa/Core/LiveTexture.cpp
	# Ui — the single windowed/headless media viewer application
	Source/Private/Oa/Ui/Viewer.cpp
	Source/Private/Oa/Ui/TrainingViewer.cpp

	# Plot — matplotlib-style plotting over OaContext sinks
	Source/Private/Oa/Ui/Plot/Axes.cpp
	Source/Private/Oa/Ui/Plot/Figure.cpp

	# Vision (OaCv) — plot-integrated CV overlays
	Source/Private/Oa/Vision/CvFrame.cpp

	# Audio
	Source/Private/Oa/Audio/AudioDecoder.cpp
	Source/Private/Oa/Audio/AudioEncoder.cpp
	Source/Private/Oa/Audio/AudioCapture.cpp
	Source/Private/Oa/Audio/AudioStream.cpp
	Source/Private/Oa/Audio/FnAudio/FnAudio.cpp
	Source/Private/Oa/Audio/FnAudio/Signal/FnAudioSignal.cpp
	Source/Private/Oa/Audio/FnAudio/Transform/FnAudioTransform.cpp
	
	# Audio — AUTOGENERATED from OaFnAutogen schemas
	${OA_AUDIO_FN_AUDIO_GENERATED_SOURCES}

	# Animation — time/timeline/curves for unified pan/zoom animation
	Source/Private/Oa/Animation/Timeline.cpp

	# Render — 2D/3D rendering infrastructure
	Source/Private/Oa/Render/Renderer.cpp
	Source/Private/Oa/Render/FnMaterial.cpp
	Source/Private/Oa/Render/FnMesh.cpp
	Source/Private/Oa/Render/Material.cpp
	Source/Private/Oa/Render/Mesh.cpp

	# Vision
	Source/Private/Oa/Vision/Detection.cpp
	Source/Private/Oa/Vision/FnDetection/FnDetection.cpp
	Source/Private/Oa/Vision/FnImage/Postprocess/FnImageSegmentation.cpp
	Source/Private/Oa/Vision/FnImageSaveFile.cpp
	Source/Private/Oa/Vision/Image/Decoder/JpegDecoder.cpp
	Source/Private/Oa/Vision/Video/Decoder/VideoDecoder.cpp
	Source/Private/Oa/Vision/Video/Decoder/VideoDecoderProfile.cpp
	Source/Private/Oa/Vision/Video/Decoder/VideoDecoderOps.cpp
	Source/Private/Oa/Vision/Video/Decoder/VideoDecoderInternal.cpp
	Source/Private/Oa/Vision/FnVideo/Color/FnVideoDecoderColor.cpp
	Source/Private/Oa/Vision/Video/Decoder/VideoDecoderDecode.cpp
	Source/Private/Oa/Vision/Video/Decoder/VideoDecoderFrame.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/FnVideoDecoderShared.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/FnVideoDecoderRecordShared.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/H264/FnVideoDecoderH264.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/H264/FnVideoDecoderH264Record.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/H265/FnVideoDecoderH265.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/H265/FnVideoDecoderH265Record.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/Av1/FnVideoDecoderAv1.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/Av1/FnVideoDecoderAv1Record.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/Vp9/FnVideoDecoderVp9.cpp
	Source/Private/Oa/Vision/FnVideo/Decoder/Vp9/FnVideoDecoderVp9Record.cpp
	Source/Private/Oa/Vision/Video/Encoder/VideoEncoder.cpp
	Source/Private/Oa/Vision/Video/Encoder/VideoRecorder.cpp
	Source/Private/Oa/Vision/Capture/ScreenCapture.cpp
	Source/Private/Oa/Vision/Capture/CameraCapture.cpp
	Source/Private/Oa/Vision/VideoStream.cpp
	Source/Private/Oa/Vision/Video.cpp
	Source/Private/Oa/Vision/VideoMuxer.cpp
	Source/Private/Oa/Vision/ItVideo.cpp
	
	# Vision — Video Codec Registry
	Source/Private/Oa/Vision/Video/Codec/CodecRegistry.cpp
	Source/Private/Oa/Vision/Video/Codec/VcpH264.cpp
	Source/Private/Oa/Vision/Video/Codec/VcpH265.cpp
	Source/Private/Oa/Vision/Video/Codec/VcpAv1.cpp
	Source/Private/Oa/Vision/Video/Codec/VcpVp9.cpp
	
	# Runtime — Vulkan Video Core (Layer 1)
	Source/Private/Oa/Runtime/OaVkVideo.cpp
	
	# Vision Image Functions (GPU-only, pure Vulkan)
	Source/Private/Oa/Vision/FnImage/FnImage.cpp
	Source/Private/Oa/Vision/FnImage/FnImageConversion.cpp
	Source/Private/Oa/Vision/FnImage/Color/FnImagePixel.cpp
	Source/Private/Oa/Vision/FnImage/Geometric/FnImageGeometric.cpp
	Source/Private/Oa/Vision/FnImage/Filter/FnImageFilter.cpp
	Source/Private/Oa/Vision/FnImage/FnImagePipeline.cpp

	# Vision Video Functions (OaFnVideo — Codec / Nal / Color / Preprocess)
	Source/Private/Oa/Vision/FnVideo/Codec/FnVideoCodec.cpp
	Source/Private/Oa/Vision/FnVideo/Nal/FnVideoNal.cpp
	Source/Private/Oa/Vision/FnVideo/Color/FnVideoColor.cpp
	Source/Private/Oa/Vision/FnVideo/Preprocess/FnVideoPreprocess.cpp
	
	# Vision — AUTOGENERATED from OaFnAutogen schemas
	${OA_VISION_FN_IMAGE_GENERATED_SOURCES}
	${OA_VISION_FN_VIDEO_GENERATED_SOURCES}
	
	Source/Private/Oa/Ml/Nn/Byte.cpp

	# ML — optimizers
	Source/Private/Oa/Ml/Optim/Sgd.cpp
	Source/Private/Oa/Ml/Optim/Adam.cpp
	Source/Private/Oa/Ml/Optim/AdamW.cpp
	Source/Private/Oa/Ml/Optim/Muon.cpp
	Source/Private/Oa/Ml/Optim/OptimComposite.cpp
	Source/Private/Oa/Ml/Optim/OptimUtil.cpp
	Source/Private/Oa/Ml/Callback/LrScheduler.cpp



	# Network
	Source/Private/Oa/Network/Tcp.cpp
	Source/Private/Oa/Network/TcpFramed.cpp
	Source/Private/Oa/Network/Http.cpp
)

if(OA_BUILD_CRYPTO)
	list(APPEND OA_SOURCES
		# Single-shot CPU keygen/sign/verify (liboqs)
		Source/Private/Oa/Crypto/Sign.cpp

		# OaFnHash — GPU batch hash (auto-context); CPU primitives live in
		# Crypto/Keccak.cpp + Crypto/Hash.cpp (compiled unconditionally above).
		${OA_CRYPTO_HASH_FN_GENERATED_SOURCES}
		Source/Private/Oa/Crypto/Hash/FnHash.cpp

	)
endif()

# UI — AUTOGENERATED from OaFnAutogen schemas
list(APPEND OA_SOURCES
	${OA_UI_FN_MATRIX_GENERATED_SOURCES}
)

# Android ships the actual OA compute/ML stack, not the desktop product shell.
# Keep this as a source boundary instead of scattering __ANDROID__ through UI,
# capture, codec, database, io_uring, and networking implementations.
if(OA_ANDROID_ML)
	list(FILTER OA_SOURCES EXCLUDE REGEX "/(Audio|Vision|Ui|Render|Animation|Network|Crypto)/")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Data/")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Core/(Input|Navigation|Viewport|LiveTexture|Simd|CpuTopology|PerfStat|PerfStore|MappedFile)[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Runtime/(App|Window|ImageDispatch|OaVkVideo)[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Runtime/Engine/GraphicsEngine[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Runtime/Distributed/ExternalMemory[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Runtime/Pool/PqcPool[.]cpp$")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Ml/(Tokenizer|Io|Data)/")
	list(FILTER OA_SOURCES EXCLUDE REGEX "/Ml/(Oad|Mamba3)[.]cpp$")
	target_compile_definitions(oa_api INTERFACE OA_ANDROID_ML=1)
	message(STATUS "OA source profile: Android Runtime + Core + ML")
endif()

# SHARED when building Python bindings (avoids TLS relocation issues) or when explicitly
# requested for a distributable runtime package (OA_BUILD_SHARED); STATIC otherwise.
if(OA_BUILD_PYTHON OR OA_BUILD_SHARED)
	add_library(oa SHARED ${OA_SOURCES})
else()
	add_library(oa STATIC ${OA_SOURCES})
endif()
target_link_libraries(oa PUBLIC oa_headers)
if(ANDROID)
	target_include_directories(oa_api SYSTEM INTERFACE ${OA_ANDROID_VULKAN_HEADERS})
endif()
if(WIN32)
	target_link_libraries(oa PUBLIC ws2_32)
endif()
target_include_directories(oa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source/ThirdParty/miniaudio)
target_include_directories(oa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Source/Private)

if(OA_WITH_DATABASE)
	find_package(DuckDB CONFIG REQUIRED)
	target_compile_definitions(oa PRIVATE OA_HAS_EMBEDDED_SQL_STORE)
	target_link_libraries(oa PRIVATE $<IF:$<TARGET_EXISTS:duckdb>,duckdb,duckdb_static>)
	message(STATUS "OA embedded SQL store: DuckDB (OA_HAS_EMBEDDED_SQL_STORE)")
else()
	message(STATUS "OA embedded SQL store: stub (enable OA_WITH_DATABASE + vcpkg feature database)")
endif()

# Highway SIMD
if(hwy_FOUND AND NOT ANDROID)
	target_link_libraries(oa PRIVATE hwy::hwy)
	message(STATUS "Highway SIMD: found")
elseif(NOT ANDROID)
	message(WARNING "Highway not found — OaSimd* functions will not compile")
endif()

if(NOT ANDROID)
	target_link_libraries(oa PRIVATE utf8proc::utf8proc)
endif()

# SDL3 — required by OaCameraCapture, OaWindow, OaDeviceUiApp
if(NOT ANDROID)
	find_package(SDL3 CONFIG QUIET)
endif()
if(TARGET SDL3::SDL3)
	target_link_libraries(oa PRIVATE SDL3::SDL3)
	message(STATUS "SDL3: found (OaCameraCapture enabled)")
elseif(NOT ANDROID)
	message(WARNING "SDL3 not found — OaCameraCapture and windowed apps will not work")
endif()

# Wayland screencast portal + PipeWire. Optional outside Linux; the public
# OaScreenCapture surface remains available and reports Unavailable at runtime.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	find_package(PkgConfig QUIET)
	if(PkgConfig_FOUND)
		pkg_check_modules(OA_PIPEWIRE QUIET IMPORTED_TARGET libpipewire-0.3)
		pkg_check_modules(OA_LIBPORTAL QUIET IMPORTED_TARGET libportal)
	endif()
	if(TARGET PkgConfig::OA_PIPEWIRE AND TARGET PkgConfig::OA_LIBPORTAL)
		target_link_libraries(oa PRIVATE PkgConfig::OA_PIPEWIRE PkgConfig::OA_LIBPORTAL)
		target_compile_definitions(oa PRIVATE OA_HAS_PIPEWIRE_SCREEN_CAPTURE=1)
		message(STATUS "Wayland screen capture: libportal + PipeWire")
	else()
		message(STATUS "Wayland screen capture: unavailable (libportal/PipeWire not found)")
	endif()
endif()

# liburing (Linux io_uring for async I/O)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
	find_library(URING_LIBRARY uring)
	if(URING_LIBRARY)
		target_link_libraries(oa PUBLIC ${URING_LIBRARY})
		message(STATUS "liburing found: ${URING_LIBRARY}")
	else()
		message(WARNING "liburing not found — OaIoUringReader will not build")
	endif()
endif()

# Vulkan: OaVk (forked volk) handles dynamic loading via dlopen — no static linking.
target_link_libraries(oa PRIVATE ${CMAKE_DL_LIBS})
if(TARGET Vulkan::Headers)
	target_link_libraries(oa PUBLIC Vulkan::Headers)
elseif(Vulkan_FOUND)
	target_include_directories(oa PUBLIC ${Vulkan_INCLUDE_DIRS})
endif()

# PQCrypto (liboqs — ML-DSA-65 / Dilithium-3)
if(OA_BUILD_CRYPTO)
	find_package(liboqs CONFIG REQUIRED)
	target_link_libraries(oa PRIVATE OQS::oqs)
	target_compile_definitions(oa PUBLIC OA_BUILD_CRYPTO)
	message(STATUS "Building with ML-DSA-65 signing (liboqs)")
endif()

# COMPUTE SHADERS — Slang → SPIR-V (ML primitives + crypto kernels)
if(SLANGC)
	set(OA_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/Shader/Compute")
	set(OA_SPIRV_DIR "${CMAKE_CURRENT_BINARY_DIR}/spirv")
	file(MAKE_DIRECTORY ${OA_SPIRV_DIR})

	# Slang importable math modules (no entry points) + Public/Core/Math.slang umbrella
	set(OA_SLANG_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Math")
	set(OA_KERNEL_FLAT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Flat")
	set(OA_KERNEL_MATRIX_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Matrix")
	set(OA_KERNEL_BLAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Blas")
	set(OA_KERNEL_ML_BLAS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/Shader/Compute/Blas")
	set(OA_KERNEL_RL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/Rl/Shader/Compute")
	set(OA_SLANG_UMBRELLA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Public/Oa/Core")
	set(OA_ML_SLANG_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/Shader/Compute/Nn/Math")
	set(OA_ML_SLANG_UMBRELLA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Public/Oa/Ml")
	file(GLOB OA_SLANG_MODULES "${OA_SLANG_MODULE_DIR}/*.slang")
	list(APPEND OA_SLANG_MODULES "${OA_SLANG_UMBRELLA_DIR}/Math.slang")
	file(GLOB OA_ML_SLANG_MODULES "${OA_ML_SLANG_MODULE_DIR}/*.slang")
	list(APPEND OA_SLANG_MODULES ${OA_ML_SLANG_MODULES})
	list(APPEND OA_SLANG_MODULES "${OA_ML_SLANG_UMBRELLA_DIR}/Math.slang")

	function(oa_compile_slang TARGET SOURCE)
		get_filename_component(NAME ${SOURCE} NAME_WE)
		set(SPV "${OA_SPIRV_DIR}/${NAME}.spv")
		add_custom_command(
			OUTPUT ${SPV}
			COMMAND ${SLANGC} ${SOURCE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${SPV}
			DEPENDS ${SOURCE} ${OA_SLANG_MODULES}
			COMMENT "slangc: ${NAME}.slang → ${NAME}.spv"
		)
		target_sources(${TARGET} PRIVATE ${SPV})
	endfunction()

	# ML compute kernels — PyTorch-standard primitives only (PascalCase paths on disk).
	# OA_ML_SHADER_REG = OaSpvFind / Dispatch name. OA_ML_SHADER_SRC = path under Ml/Shader/Compute/ (no .slang).
	# Flat tensor ops (linear-index maps + reductions) live under Core/Shader/Compute/Flat/ (same registry names).
	# Architecture-specific kernels (SSM, ISG, RDG, ACB, H-MSU, etc.) live in ml repo.
	set(OA_ML_SHADER_REG
		PackedLinear
		Add Sub Mul MulBwd Div Neg Abs Fill MultiMatrixFill MultiMatrixAdd Scale Log Exp Sin Cos Reciprocal ClampMax ClampMin Broadcast Copy CopyAtOffset MatrixCopyRegion CastU8ToU32 Equal GreaterEqual CausalMaskApply CausalMaskBwd Sum SumDim Max MaxBwd Argmax TopK TopKMask SampleSortedLogits SampleDenseLogits CategoricalAccuracyCount MaskedCategoricalAccuracyCount ReduceCols ReduceMean MeanBwd
		AddBcast SubBcast MulBcast DivBcast AddScalar SubScalar DivScalar
		Tanh TanhBwd Sigmoid SigmoidBwd LeakyRelu LeakyReluBwd Elu EluBwd Mish MishBwd Softplus SoftplusBwd
		Sqrt Pow
		BiasAdd Embedding Gather GatherLastDim GatherLastDimBwd VqAssign VqEmaUpdate Bmm SplitHeads MergeHeads FlashCausal FlashCausalBwdQ FlashCausalBwdKV GroupedGemmM GroupedGemmMDataBwd GroupedGemmMWeightBwd GroupedLinearM GroupedLinearMBiasBwd GroupedLinearMWeightBiasBwd RepeatInterleave RepeatInterleaveBwd CompactRows CompactRowsBwd CompactRowsBwdIndirect ScatterRows ScatterRowsBwd ScatterRowsIndirect ScatterRowsBwdIndirect MoeRoutingBiasUpdate MoeExpertPlan MoeRouteWeights MoeRouteWeightsBwd MoeCombine MoeCombineBwd ScatterAddRows MoeGatherBwd
		Silu SiluBwd Gelu GeluBwd Relu ReluBwd Swiglu SwigluBwd SiluMul SiluMulBwd Geglu GegluBwd GruCellPointwise GruCellPointwiseBwd GruCellLinear GruScan GruScanBwd RnnCellPointwise RnnCellPointwiseBwd RnnCellLinear RnnScan RnnScanBwd
		RmsNorm RmsNormBwd RmsNormGated RmsNormGatedBwd Softmax SoftmaxBwd LogSoftmax LogSoftmaxBwd SoftmaxScaledMasked SoftmaxScaledMaskedBwd
		CrossEntropy CrossEntropyBwd CrossEntropyLossGradBwd MaskedCrossEntropy MaskedCrossEntropyBwd
		SmoothL1 SmoothL1Bwd
		Mse MseBwd MaskedMse MaskedMseBwd L1 L1Bwd Bce BceBwd
		Conv1dBwdData Conv1dBwdWeight Conv2d Conv2dBwdData Conv2dBwdWeight ConvTranspose2dBwdWeight Im2Col Col2Im LayerNorm LayerNormBwd
		AvgPool2d AvgPool2dBwd MaxPool2d MaxPool2dBwd
		ByteEmbed ByteEmbedBwd GatherBwd
		LinearDataBwd LinearDataAccumBwd LinearDataReluBwd LinearReluBwdData LinearWeightBwd LinearWeightBiasBwd LinearWeightBiasBwdTiled
		ResidualRmsNorm
		Sgd SgdMomentum Adam Adamw AdamwGraphAdvance AdamwGraph AdamwMany4 AdamwMany4Graph
		MuonNesterov MuonNormalize MuonApply MuonVector
		ClipGradNormReduce ClipGradNormScale
		CollectiveReduce
		Quantize/QuantizeInt8 Quantize/DequantizeInt8 Quantize/GemmInt8 Quantize/ComputeScaleInt8
		Quantize/QuantizeQ8_0 Quantize/DequantizeQ8_0 Quantize/DequantizeQ4_K Quantize/QuantizeQ4_K
		Quantize/QuantizeQ4_0 Quantize/DequantizeQ4_0 Quantize/ComputeScaleQ4_0
		RopeApply RopeApplyBwd
		CastBf16ToF32 CastF32ToBf16
		RlGae RlPpoClip RlPpoClipBwd RlDqnTarget RlSacTarget RlRolloutAppend RlRolloutReset RlCartPoleReset RlCartPoleStep RlReplayAppend RlReplaySample
	)
	set(OA_ML_SHADER_SRC
		Blas/PackedLinear
		Flat/Add Flat/Sub Flat/Mul Nn/Reduction/MulBwd Flat/Div Flat/Neg Flat/Abs Flat/Fill Flat/MultiMatrixFill Flat/MultiMatrixAdd Flat/Scale Flat/Log Flat/Exp Flat/Sin Flat/Cos Flat/Reciprocal Flat/ClampMax Flat/ClampMin Flat/Broadcast Flat/Copy Flat/CopyAtOffset Flat/MatrixCopyRegion Flat/CastU8ToU32 Flat/Equal Flat/GreaterEqual Flat/CausalMaskApply Flat/CausalMaskBwd
		Flat/Sum Flat/SumDim Flat/Max Nn/Reduction/MaxBwd Flat/Argmax Flat/TopK Flat/TopKMask Flat/SampleSortedLogits Flat/SampleDenseLogits Flat/CategoricalAccuracyCount Flat/MaskedCategoricalAccuracyCount Flat/ReduceCols Flat/ReduceMean Nn/Reduction/MeanBwd
		Flat/AddBcast Flat/SubBcast Flat/MulBcast Flat/DivBcast
		Flat/AddScalar Flat/SubScalar Flat/DivScalar
		Nn/Activation/Tanh Nn/Activation/TanhBwd Nn/Activation/Sigmoid Nn/Activation/SigmoidBwd
		Nn/Activation/LeakyRelu Nn/Activation/LeakyReluBwd Nn/Activation/Elu Nn/Activation/EluBwd
		Nn/Activation/Mish Nn/Activation/MishBwd Nn/Activation/Softplus Nn/Activation/SoftplusBwd
		Nn/Math/Sqrt Nn/Math/Pow
		Ops/BiasAdd Ops/Embedding Ops/Gather Ops/GatherLastDim Ops/GatherLastDimBwd Ops/VqAssign Ops/VqEmaUpdate Ops/Bmm Nn/Attention/SplitHeads Nn/Attention/MergeHeads Nn/Attention/FlashCausal Nn/Attention/FlashCausalBwdQ Nn/Attention/FlashCausalBwdKV Blas/GroupedGemmM Blas/GroupedGemmMDataBwd Blas/GroupedGemmMWeightBwd Blas/GroupedLinearM Blas/GroupedLinearMBiasBwd Blas/GroupedLinearMWeightBiasBwd Ops/RepeatInterleave Ops/RepeatInterleaveBwd Ops/CompactRows Ops/CompactRowsBwd Ops/CompactRowsBwdIndirect Ops/ScatterRows Ops/ScatterRowsBwd Ops/ScatterRowsIndirect Ops/ScatterRowsBwdIndirect Nn/Moe/MoeRoutingBiasUpdate Nn/Moe/MoeExpertPlan Nn/Moe/MoeRouteWeights Nn/Moe/MoeRouteWeightsBwd Nn/Moe/MoeCombine Nn/Moe/MoeCombineBwd Nn/Moe/ScatterAddRows Nn/Moe/MoeGatherBwd
		Nn/Silu/Silu Nn/Silu/SiluBwd
		Nn/Gelu/Gelu Nn/Gelu/GeluBwd
		Nn/Relu/Relu Nn/Relu/ReluBwd
		Nn/Swiglu/Swiglu Nn/Swiglu/SwigluBwd Nn/Swiglu/SiluMul Nn/Swiglu/SiluMulBwd Nn/Geglu/Geglu Nn/Geglu/GegluBwd Nn/Gru/GruCellPointwise Nn/Gru/GruCellPointwiseBwd Nn/Gru/GruCellLinear Nn/Gru/GruScan Nn/Gru/GruScanBwd Nn/Rnn/RnnCellPointwise Nn/Rnn/RnnCellPointwiseBwd Nn/Rnn/RnnCellLinear Nn/Rnn/RnnScan Nn/Rnn/RnnScanBwd
		Nn/RmsNorm/RmsNorm Nn/RmsNorm/RmsNormBwd Nn/RmsNorm/RmsNormGated Nn/RmsNorm/RmsNormGatedBwd
		Nn/Softmax/Softmax Nn/Softmax/SoftmaxBwd Nn/Softmax/LogSoftmax Nn/Softmax/LogSoftmaxBwd Nn/Softmax/SoftmaxScaledMasked Nn/Softmax/SoftmaxScaledMaskedBwd
		Nn/CrossEntropy/CrossEntropy Nn/CrossEntropy/CrossEntropyBwd Nn/CrossEntropy/CrossEntropyLossGradBwd Nn/CrossEntropy/MaskedCrossEntropy Nn/CrossEntropy/MaskedCrossEntropyBwd
		Loss/SmoothL1 Loss/SmoothL1Bwd
		Loss/Mse Loss/MseBwd Loss/MaskedMse Loss/MaskedMseBwd Loss/L1 Loss/L1Bwd Loss/Bce Loss/BceBwd
		Nn/Conv/Conv1dBwdData Nn/Conv/Conv1dBwdWeight Nn/Conv/Conv2d Nn/Conv/Conv2dBwdData Nn/Conv/Conv2dBwdWeight Nn/Conv/ConvTranspose2dBwdWeight Nn/Conv/Im2Col Nn/Conv/Col2Im Nn/LayerNorm/LayerNorm Nn/LayerNorm/LayerNormBwd
		Nn/Pooling/AvgPool2d Nn/Pooling/AvgPool2dBwd Nn/Pooling/MaxPool2d Nn/Pooling/MaxPool2dBwd
		Nn/ByteEmbed/ByteEmbed Nn/ByteEmbed/ByteEmbedBwd Nn/GatherBwd
		Nn/Linear/LinearDataBwd Nn/Linear/LinearDataAccumBwd Nn/Linear/LinearDataReluBwd Nn/Linear/LinearReluBwdData Nn/Linear/LinearWeightBwd Nn/Linear/LinearWeightBiasBwd Nn/Linear/LinearWeightBiasBwdTiled
		Nn/ResidualRmsNorm/ResidualRmsNorm
		Optim/Sgd/Sgd Optim/Sgd/SgdMomentum
		Optim/Adam/Adam
		Optim/AdamW/Adamw Optim/AdamW/AdamwGraphAdvance Optim/AdamW/AdamwGraph Optim/AdamW/AdamwMany4 Optim/AdamW/AdamwMany4Graph
		Optim/Muon/MuonNesterov Optim/Muon/MuonNormalize Optim/Muon/MuonApply Optim/Muon/MuonVector
		Optim/GradClip/ClipGradNormReduce Optim/GradClip/ClipGradNormScale
		Flat/CollectiveReduce
		Quantize/QuantizeInt8 Quantize/DequantizeInt8 Quantize/GemmInt8 Quantize/ComputeScaleInt8
		Quantize/QuantizeQ8_0 Quantize/DequantizeQ8_0 Quantize/DequantizeQ4_K Quantize/QuantizeQ4_K
		Quantize/QuantizeQ4_0 Quantize/DequantizeQ4_0 Quantize/ComputeScaleQ4_0
		Nn/Rope/RopeApply Nn/Rope/RopeApplyBwd
		Cast/CastBf16ToF32 Cast/CastF32ToBf16
		Rl/Gae/RlGae Rl/Loss/RlPpoClip Rl/Loss/RlPpoClipBwd Rl/Loss/RlDqnTarget Rl/Loss/RlSacTarget Rl/Rollout/RlRolloutAppend Rl/Rollout/RlRolloutReset Rl/Environment/RlCartPoleReset Rl/Environment/RlCartPoleStep Rl/Replay/RlReplayAppend Rl/Replay/RlReplaySample
	)
	set(OA_KERNEL_FLAT_OPS
		Add Sub Mul Div Neg Abs Fill MultiMatrixFill MultiMatrixAdd Scale Log Exp Sin Cos Reciprocal ClampMax ClampMin Broadcast Copy CopyAtOffset MatrixCopyRegion CastU8ToU32 Equal GreaterEqual CausalMaskApply CausalMaskBwd CollectiveReduce
		Sum SumDim Max Argmax TopK TopKMask SampleSortedLogits SampleDenseLogits CategoricalAccuracyCount MaskedCategoricalAccuracyCount ReduceCols ReduceMean
		AddBcast SubBcast MulBcast DivBcast
		AddScalar SubScalar DivScalar)
	set(OA_ML_SPV_OUTPUTS "")
	list(LENGTH OA_ML_SHADER_REG _OA_ML_REG_N)
	list(LENGTH OA_ML_SHADER_SRC _OA_ML_SRC_N)
	if(NOT _OA_ML_REG_N EQUAL _OA_ML_SRC_N)
		message(FATAL_ERROR "OA_ML_SHADER_REG (${_OA_ML_REG_N}) and OA_ML_SHADER_SRC (${_OA_ML_SRC_N}) length mismatch! "
			"These lists must be kept in 1:1 order. "
			"Last REG entry: ${OA_ML_SHADER_REG}; "
			"Last SRC entry: ${OA_ML_SHADER_SRC}")
	endif()
	list(LENGTH OA_ML_SHADER_REG _OA_ML_N)
	math(EXPR _OA_ML_LAST "${_OA_ML_N} - 1")
	foreach(_I RANGE ${_OA_ML_LAST})
		list(GET OA_ML_SHADER_REG ${_I} _REG)
		list(GET OA_ML_SHADER_SRC ${_I} _SRCREL)
		# Extract basename from registry name (strip directory prefix like "Quantize/")
		get_filename_component(_REG_BASENAME "${_REG}" NAME)
		if(_REG IN_LIST OA_KERNEL_FLAT_OPS)
			# Prefer .gen.slang (auto-generated) over manual .slang
			if(EXISTS "${OA_KERNEL_FLAT_DIR}/${_REG}.gen.slang")
				set(_SRC "${OA_KERNEL_FLAT_DIR}/${_REG}.gen.slang")
				set(_SLANG_COMMENT "slangc: Core/Shader/Compute/Flat/${_REG}.gen.slang → ${_REG_BASENAME}.spv")
			else()
				set(_SRC "${OA_KERNEL_FLAT_DIR}/${_REG}.slang")
				set(_SLANG_COMMENT "slangc: Core/Shader/Compute/Flat/${_REG}.slang → ${_REG_BASENAME}.spv")
			endif()
		elseif(_SRCREL MATCHES "^Rl/")
			string(REGEX REPLACE "^Rl/" "" _RL_SRCREL "${_SRCREL}")
			set(_SRC "${OA_KERNEL_RL_DIR}/${_RL_SRCREL}.slang")
			set(_SLANG_COMMENT "slangc: Ml/Rl/Shader/Compute/${_RL_SRCREL}.slang → ${_REG_BASENAME}.spv")
		else()
			set(_SRC "${OA_SHADER_DIR}/${_SRCREL}.slang")
			set(_SLANG_COMMENT "slangc: ${_SRCREL}.slang → ${_REG_BASENAME}.spv")
		endif()
		set(_SPV "${OA_SPIRV_DIR}/${_REG_BASENAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "${_SLANG_COMMENT}"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Core matrix compute kernels (bindless GEMM / layout — shared by ML and non-ML)
	# (Naive Transpose.slang retired — OaMatrix::Transpose 2D now dispatches TransposeTiled,
	#  the coalesced SMEM-staged kernel the GEMM stack already uses; there is no fallback ladder.)

	# Batched transpose of the last two axes ([B,R,C] → [B,C,R]) — channels-first conv layouts
	set(_TRANSPOSE_BATCHED_SRC "${OA_KERNEL_MATRIX_DIR}/TransposeBatched.slang")
	set(_TRANSPOSE_BATCHED_SPV "${OA_SPIRV_DIR}/TransposeBatched.spv")
	add_custom_command(
		OUTPUT ${_TRANSPOSE_BATCHED_SPV}
		COMMAND ${SLANGC} ${_TRANSPOSE_BATCHED_SRC}
			-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
			-target spirv -entry main -o ${_TRANSPOSE_BATCHED_SPV}
		DEPENDS ${_TRANSPOSE_BATCHED_SRC} ${OA_SLANG_MODULES}
		COMMENT "slangc: Shader/Compute/Matrix/TransposeBatched.slang → TransposeBatched.spv"
	)
	list(APPEND OA_ML_SPV_OUTPUTS ${_TRANSPOSE_BATCHED_SPV})

	# Tiled transpose — coalesced reads + writes via SMEM (replaces strided Transpose.slang dispatch)
	set(_TRANSPOSE_TILED_SRC "${OA_KERNEL_MATRIX_DIR}/TransposeTiled.slang")
	set(_TRANSPOSE_TILED_SPV "${OA_SPIRV_DIR}/TransposeTiled.spv")
	add_custom_command(
		OUTPUT ${_TRANSPOSE_TILED_SPV}
		COMMAND ${SLANGC} ${_TRANSPOSE_TILED_SRC}
			-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
			-target spirv -entry main -o ${_TRANSPOSE_TILED_SPV}
		DEPENDS ${_TRANSPOSE_TILED_SRC} ${OA_SLANG_MODULES}
		COMMENT "slangc: Shader/Compute/Matrix/TransposeTiled.slang → TransposeTiled.spv"
	)
	list(APPEND OA_ML_SPV_OUTPUTS ${_TRANSPOSE_TILED_SPV})

	# ═══════════════════════════════════════════════════════════════════════
	# BLAS Kernels — Unified GEMM/BLAS directory structure
	# ═══════════════════════════════════════════════════════════════════════
	
	# Core GEMM kernels (hierarchy: Naive → Tiled → CmSg/CmWg BF16)
	set(OA_BLAS_GEMM_KERNELS
		Gemm/GemmNaive
		Gemm/GemmStrided
	)

	# Generated from Tools/OaTile/Schema/Fp32Tiled.toml. The schema is the single
	# source of truth for compile geometry, stable kernel IDs and router metadata.
	include("${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Blas/Tile/OaTileFp32Variants.gen.cmake")
	# BF16 subgroup/workgroup CoopMat families and their epilogues are generated
	# from one capability-gated schema alongside the FP32 lattice.
	include("${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Blas/Tile/OaTileBf16Variants.gen.cmake")

	set(OA_BLAS_GLSL_COMP_KERNELS "")
	set(OA_ML_BLAS_GLSL_COMP_KERNELS "")

	# Utility kernels
	set(OA_BLAS_UTIL_KERNELS
		Util/GemmCoopVec
	)
	
	# Compile core GEMM kernels (some need cooperative matrix capabilities)
	foreach(_KERNEL ${OA_BLAS_GEMM_KERNELS})
		get_filename_component(_NAME ${_KERNEL} NAME)
		set(_SRC "${OA_KERNEL_BLAS_DIR}/${_KERNEL}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		
		# Determine if cooperative matrix capability is needed
		if(_NAME MATCHES "CoopMat")
			add_custom_command(
				OUTPUT ${_SPV}
				COMMAND ${SLANGC} ${_SRC}
					-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
					-target spirv -entry main
					-capability spvCooperativeMatrixKHR
					-capability spvCooperativeMatrixConversionsNV
					-o ${_SPV}
				DEPENDS ${_SRC} ${OA_SLANG_MODULES}
				COMMENT "slangc: Blas/${_KERNEL}.slang → ${_NAME}.spv"
			)
		else()
			add_custom_command(
				OUTPUT ${_SPV}
				COMMAND ${SLANGC} ${_SRC}
					-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
					-target spirv -entry main -o ${_SPV}
				DEPENDS ${_SRC} ${OA_SLANG_MODULES}
				COMMENT "slangc: Blas/${_KERNEL}.slang → ${_NAME}.spv"
			)
		endif()
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	set(_GEMM_FP32_TEMPLATE "${OA_KERNEL_BLAS_DIR}/Gemm/GemmTiled.slang")
	list(LENGTH OA_TILE_FP32_NAMES _OA_FP32_LEN)
	set(_OA_FP32_I 0)
	while(_OA_FP32_I LESS _OA_FP32_LEN)
		list(GET OA_TILE_FP32_NAMES ${_OA_FP32_I} _FP32_NAME)
		list(GET OA_TILE_FP32_TILE_M ${_OA_FP32_I} _FP32_TILE_M)
		list(GET OA_TILE_FP32_TILE_N ${_OA_FP32_I} _FP32_TILE_N)
		list(GET OA_TILE_FP32_TILE_K ${_OA_FP32_I} _FP32_TILE_K)
		list(GET OA_TILE_FP32_THREAD_M ${_OA_FP32_I} _FP32_THREAD_M)
		list(GET OA_TILE_FP32_THREAD_N ${_OA_FP32_I} _FP32_THREAD_N)
		list(GET OA_TILE_FP32_WORKGROUP ${_OA_FP32_I} _FP32_WORKGROUP)
		list(GET OA_TILE_FP32_GROUP_M ${_OA_FP32_I} _FP32_GROUP_M)
		list(GET OA_TILE_FP32_EPILOGUE ${_OA_FP32_I} _FP32_EPILOGUE)
		set(_FP32_SPV "${OA_SPIRV_DIR}/${_FP32_NAME}.spv")
		add_custom_command(
			OUTPUT ${_FP32_SPV}
			COMMAND ${SLANGC} ${_GEMM_FP32_TEMPLATE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-D OA_TILE_M=${_FP32_TILE_M}
				-D OA_TILE_N=${_FP32_TILE_N}
				-D OA_TILE_K=${_FP32_TILE_K}
				-D OA_THREAD_TILE_M=${_FP32_THREAD_M}
				-D OA_THREAD_TILE_N=${_FP32_THREAD_N}
				-D OA_TILE_WORKGROUP=${_FP32_WORKGROUP}
				-D OA_TILE_GROUP_M=${_FP32_GROUP_M}
				-D OA_TILE_EPILOGUE=${_FP32_EPILOGUE}
				-target spirv -entry main -o ${_FP32_SPV}
			DEPENDS ${_GEMM_FP32_TEMPLATE} ${OA_SLANG_MODULES}
			COMMENT "slangc: GemmTiled.slang -> ${_FP32_NAME}.spv (${_FP32_TILE_M}x${_FP32_TILE_N}x${_FP32_TILE_K}, thread ${_FP32_THREAD_M}x${_FP32_THREAD_N}, group-M ${_FP32_GROUP_M}, epilogue ${_FP32_EPILOGUE})"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_FP32_SPV})
		math(EXPR _OA_FP32_I "${_OA_FP32_I} + 1")
	endwhile()

	# GemmCmSgBf16 variants from one template source.
	set(_GEMM_BF16_TEMPLATE "${OA_KERNEL_BLAS_DIR}/Gemm/GemmCmSgBf16.slang")
	list(LENGTH OA_GEMM_BF16_VARIANTS _OA_BF16_LEN)
	math(EXPR _OA_BF16_LAST "${_OA_BF16_LEN} - 1")
	set(_OA_BF16_I 0)
	while(_OA_BF16_I LESS _OA_BF16_LAST)
		list(GET OA_GEMM_BF16_VARIANTS ${_OA_BF16_I} _BF16_NAME)
		math(EXPR _OA_BF16_J "${_OA_BF16_I} + 1")
		list(GET OA_GEMM_BF16_VARIANTS ${_OA_BF16_J} _BF16_ACT)
		set(_BF16_SPV "${OA_SPIRV_DIR}/${_BF16_NAME}.spv")
		add_custom_command(
			OUTPUT ${_BF16_SPV}
			COMMAND ${SLANGC} ${_GEMM_BF16_TEMPLATE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-D OA_ACT=${_BF16_ACT}
				-target spirv -entry main
				-capability spvCooperativeMatrixKHR
				-capability spvCooperativeMatrixConversionsNV
				-o ${_BF16_SPV}
			DEPENDS ${_GEMM_BF16_TEMPLATE} ${OA_SLANG_MODULES}
			COMMENT "slangc: GemmCmSgBf16.slang → ${_BF16_NAME}.spv (OA_ACT=${_BF16_ACT})"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_BF16_SPV})
		math(EXPR _OA_BF16_I "${_OA_BF16_I} + 2")
	endwhile()

	# GemmCmWgBf16 variants from one template source.
	set(_GEMM_WG_BF16_TEMPLATE "${OA_KERNEL_BLAS_DIR}/Gemm/GemmCmWgBf16.slang")
	list(LENGTH OA_GEMM_WG_BF16_VARIANTS _OA_WG_BF16_LEN)
	math(EXPR _OA_WG_BF16_LAST "${_OA_WG_BF16_LEN} - 1")
	set(_OA_WG_BF16_I 0)
	while(_OA_WG_BF16_I LESS ${_OA_WG_BF16_LAST})
		list(GET OA_GEMM_WG_BF16_VARIANTS ${_OA_WG_BF16_I} _WG_BF16_NAME)
		math(EXPR _OA_WG_BF16_J "${_OA_WG_BF16_I} + 1")
		list(GET OA_GEMM_WG_BF16_VARIANTS ${_OA_WG_BF16_J} _WG_BF16_ACT)
		set(_WG_BF16_SPV "${OA_SPIRV_DIR}/${_WG_BF16_NAME}.spv")
		add_custom_command(
			OUTPUT ${_WG_BF16_SPV}
			COMMAND ${SLANGC} ${_GEMM_WG_BF16_TEMPLATE}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-D OA_ACT=${_WG_BF16_ACT}
				-target spirv -entry main
				-capability spvCooperativeMatrixKHR
				-capability spvCooperativeMatrixConversionsNV
				-o ${_WG_BF16_SPV}
			DEPENDS ${_GEMM_WG_BF16_TEMPLATE} ${OA_SLANG_MODULES}
			COMMENT "slangc: GemmCmWgBf16.slang → ${_WG_BF16_NAME}.spv (OA_ACT=${_WG_BF16_ACT})"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_WG_BF16_SPV})
		math(EXPR _OA_WG_BF16_I "${_OA_WG_BF16_I} + 2")
	endwhile()

	# Compile GLSL .comp kernels.
	#
	# Raw GEMMs live in Source/Private/Oa/Core/Shader/Compute/Blas; fused NN-epilogue
	# variants live in Source/Private/Oa/Ml/Shader/Compute/Blas to mirror the .slang
	# fused-kernel layout.
	foreach(_KERNEL ${OA_BLAS_GLSL_COMP_KERNELS})
		get_filename_component(_NAME ${_KERNEL} NAME)
		set(_SRC "${OA_KERNEL_BLAS_DIR}/${_KERNEL}.comp")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.4 ${_SRC}
				-o ${_SPV}
			DEPENDS ${_SRC}
			COMMENT "glslangValidator: Blas/${_KERNEL}.comp → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()
	foreach(_KERNEL ${OA_ML_BLAS_GLSL_COMP_KERNELS})
		get_filename_component(_NAME ${_KERNEL} NAME)
		set(_SRC "${OA_KERNEL_ML_BLAS_DIR}/${_KERNEL}.comp")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.4 ${_SRC}
				-o ${_SPV}
			DEPENDS ${_SRC}
			COMMENT "glslangValidator: Ml/Shader/Compute/Blas/${_KERNEL}.comp → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Compile utility kernels (GemmCoopVec needs CooperativeVector)
	foreach(_KERNEL ${OA_BLAS_UTIL_KERNELS})
		get_filename_component(_NAME ${_KERNEL} NAME)
		set(_SRC "${OA_KERNEL_BLAS_DIR}/${_KERNEL}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main
				-capability spvCooperativeVectorNV
				-o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Blas/${_KERNEL}.slang → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()
	
	# SSM scan kernels
	# Mamba3/ for untouched reference, Empyrealm/ for fused/branded custom variants (One Node Club).
	# Both sets are built and loaded the same way so pipelines are pre-loaded for ItTraining async etc.
	set(OA_KERNEL_SSM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/Shader/Compute/Nn/Ssm")
	set(OA_SSM_SHADERS
		Mamba3SisoFwd
		Mamba3SisoBwd
		Mamba3SisoBwdShort
		Mamba3SisoBwdMobile
		Mamba3SisoStep
		Mamba3Adt
		Mamba3AdtBwd
		Mamba3Dt
		Mamba3DtBwd
		Mamba3Preprocess
		Mamba3PreprocessBwd
		EmpyrealmDt
		EmpyrealmDtBwd
		EmpyrealmAdt
		EmpyrealmAdtBwd
		EmpyrealmDtAdt
		EmpyrealmPreprocess
		EmpyrealmPreprocessBwd
		EmpyrealmSisoFwd
		EmpyrealmSisoBwd
		EmpyrealmSisoStep
	)
	foreach(_SHADER ${OA_SSM_SHADERS})
		if(_SHADER MATCHES "^Empyrealm")
			set(_SUBDIR "Empyrealm")
		else()
			set(_SUBDIR "Mamba3")
		endif()
		set(_SRC "${OA_KERNEL_SSM_DIR}/${_SUBDIR}/${_SHADER}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_SHADER}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${SLANGC_FLAGS} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-I ${OA_ML_SLANG_UMBRELLA_DIR} -I ${OA_ML_SLANG_MODULE_DIR}
				-o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Nn/Ssm/${_SUBDIR}/${_SHADER}.slang → ${_SPV}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# RNG shaders (Philox counter-based PRNG)
	set(OA_KERNEL_RNG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Core/Shader/Compute/Rng")
	set(OA_RNG_SHADERS
		PhiloxUniform
		PhiloxNormal
		PhiloxUniformGraph
		PhiloxNormalGraph
		PhiloxGraphAdvance
	)
	foreach(_SHADER ${OA_RNG_SHADERS})
		set(_SRC "${OA_KERNEL_RNG_DIR}/${_SHADER}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_SHADER}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Shader/Compute/Rng/${_SHADER}.slang → ${_SHADER}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# SR shaders (Stochastic Rounding - ML-specific)
	set(OA_KERNEL_SR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ml/Shader/Compute/Sr")
	set(OA_SR_SHADERS
		SrRoundBf16
	)
	foreach(_SHADER ${OA_SR_SHADERS})
		set(_SRC "${OA_KERNEL_SR_DIR}/${_SHADER}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_SHADER}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Shader/Compute/Sr/${_SHADER}.slang → ${_SHADER}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Vision preprocessing shaders — organized by category (cvCUDA-style)
	set(OA_KERNEL_VISION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Vision/Shader/Compute")
	
	# Conversion: Color space & format conversions
	set(OA_VISION_CONVERSION_SHADERS
		Conversion/CvtNv12ToBf16
		Conversion/CvtNv12ToRgb
		Conversion/CvtNv12YcbcrToBf16
		Conversion/CvtNv12YcbcrToRgba
		Conversion/CvtRgbaImageToNv12
		Conversion/CvtRgbaToNv12
		Conversion/CvtRgbToGray
		Conversion/NormalizeImage
		Conversion/ResizeNormalizeNchw
		Conversion/Nv12ResizeNormalizeToBf16
	)

	# Color: Pointwise, channel and compositing primitives
	set(OA_VISION_COLOR_SHADERS
		Color/ImagePointwise
		Color/ImageChannelTransform
		Color/ImageComposite
		Color/ImageColorMatrix
		Color/ImageSaltPepper
	)
	
	# Geometric: Spatial transformations
	set(OA_VISION_GEOMETRIC_SHADERS
		Geometric/ResizeBilinear
		Geometric/ResizeBilinearBwd
		Geometric/ResizeNearest
		Geometric/ResizeNearestBwd
		Geometric/CropNchw
		Geometric/FlipNchw
		Geometric/RotateNchw
		Geometric/ImagePad
		Geometric/ImageWarp
	)
	
	# Filter: Image filtering
	set(OA_VISION_FILTER_SHADERS
		Filter/ImageConvolve2d
		Filter/ImageSeparableConvolve
		Filter/ImageMorphology
		Filter/ImageNeighborhood
	)
	
	# Feature: Feature detection & extraction
	set(OA_VISION_FEATURE_SHADERS
		Feature/Canny
	)
	
	# Nn: Neural network layers (vision-specific)
	set(OA_VISION_NN_SHADERS
		Nn/BatchNorm2dStats
		Nn/BatchNorm2dNorm
	)

	# Detection postprocess and evaluation.
	set(OA_VISION_POSTPROCESS_SHADERS
		Postprocess/DetectionBoxIou
		Postprocess/DetectionNms
		Postprocess/DetectionConfusion
		Postprocess/DetectionBinaryCounts
		Postprocess/DetectionMetricCurves
		Postprocess/DetectionAveragePrecision
		Postprocess/DetectionMeanAveragePrecision
		Postprocess/SegmentationMetrics
		Postprocess/ImageSegmentationOverlay
	)
	
	# Combine all vision shaders
	set(OA_VISION_SHADERS
		${OA_VISION_CONVERSION_SHADERS}
		${OA_VISION_COLOR_SHADERS}
		${OA_VISION_GEOMETRIC_SHADERS}
		${OA_VISION_FILTER_SHADERS}
		${OA_VISION_FEATURE_SHADERS}
		${OA_VISION_NN_SHADERS}
		${OA_VISION_POSTPROCESS_SHADERS}
	)
	
	foreach(_SHADER ${OA_VISION_SHADERS})
		get_filename_component(_NAME ${_SHADER} NAME)
		set(_SRC "${OA_KERNEL_VISION_DIR}/${_SHADER}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-I ${OA_KERNEL_VISION_DIR}/Conversion
				-I ${OA_KERNEL_VISION_DIR}/Math
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
				${OA_KERNEL_VISION_DIR}/Conversion/RgbToYuv.slang
				${OA_KERNEL_VISION_DIR}/Math/Image.slang
			COMMENT "slangc: Vision/Shader/Compute/${_SHADER}.slang → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Ui blit shaders (compose image → swapchain)
	set(OA_KERNEL_UI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Ui/Shader/Compute")
	set(OA_UI_SHADERS MatrixToRgba8 BlitRgba BlitPlanar BlitImageRgba DrawRect DrawRectOutline DrawRectOutlines DrawLine DrawGlyphs DrawWaveform DrawPlotLine DrawHeatmap)
	foreach(_SHADER ${OA_UI_SHADERS})
		set(_SRC "${OA_KERNEL_UI_DIR}/${_SHADER}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_SHADER}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Ui/Shader/Compute/${_SHADER}.slang → ${_SHADER}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Audio DSP shaders. Entries may include category subdirectories; runtime
	# dispatch and embedded SPV names use only the basename.
	set(OA_KERNEL_AUDIO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Audio/Shader/Compute")
	set(OA_AUDIO_SHADERS Transform/Stft Transform/MelFilterbank Signal/AudioFade Signal/AudioMix Signal/AudioResample Signal/AudioWaveformEnvelope)
	foreach(_SHADER ${OA_AUDIO_SHADERS})
		get_filename_component(_NAME ${_SHADER} NAME)
		set(_SRC "${OA_KERNEL_AUDIO_DIR}/${_SHADER}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Audio/Shader/Compute/${_SHADER}.slang → ${_NAME}.spv"
		)
		list(APPEND OA_ML_SPV_OUTPUTS ${_SPV})
	endforeach()

	# Render graphics shaders — vertex/fragment (rasterization pipeline)
	set(OA_RENDER_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Render/Shader")
	set(OA_RENDER_SPV_OUTPUTS "")
	# Vertex shader
	set(_RENDER_VS_SRC "${OA_RENDER_SHADER_DIR}/UnlitTextured.vert.slang")
	set(_RENDER_VS_SPV "${OA_SPIRV_DIR}/UnlitTextured.vert.spv")
	add_custom_command(
		OUTPUT ${_RENDER_VS_SPV}
		COMMAND ${SLANGC} ${_RENDER_VS_SRC}
			-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
			-target spirv -stage vertex -entry main -o ${_RENDER_VS_SPV}
		DEPENDS ${_RENDER_VS_SRC} ${OA_SLANG_MODULES}
		COMMENT "slangc: Render/Shader/UnlitTextured.vert.slang → UnlitTextured.vert.spv"
	)
	list(APPEND OA_RENDER_SPV_OUTPUTS ${_RENDER_VS_SPV})
	# Fragment shader
	set(_RENDER_FS_SRC "${OA_RENDER_SHADER_DIR}/UnlitTextured.frag.slang")
	set(_RENDER_FS_SPV "${OA_SPIRV_DIR}/UnlitTextured.frag.spv")
	add_custom_command(
		OUTPUT ${_RENDER_FS_SPV}
		COMMAND ${SLANGC} ${_RENDER_FS_SRC}
			-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
			-target spirv -stage fragment -entry main -o ${_RENDER_FS_SPV}
		DEPENDS ${_RENDER_FS_SRC} ${OA_SLANG_MODULES}
		COMMENT "slangc: Render/Shader/UnlitTextured.frag.slang → UnlitTextured.frag.spv"
	)
	list(APPEND OA_RENDER_SPV_OUTPUTS ${_RENDER_FS_SPV})
	foreach(_RENDER_SHADER
		DetectionOverlay.vert DetectionOverlay.frag
		CanvasGlyph.vert CanvasGlyph.frag)
		set(_RENDER_SRC "${OA_RENDER_SHADER_DIR}/${_RENDER_SHADER}.slang")
		set(_RENDER_SPV "${OA_SPIRV_DIR}/${_RENDER_SHADER}.spv")
		get_filename_component(_RENDER_STAGE "${_RENDER_SHADER}" LAST_EXT)
		if(_RENDER_STAGE STREQUAL ".vert")
			set(_SLANG_STAGE vertex)
		else()
			set(_SLANG_STAGE fragment)
		endif()
		add_custom_command(
			OUTPUT ${_RENDER_SPV}
			COMMAND ${SLANGC} ${_RENDER_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -stage ${_SLANG_STAGE} -entry main
				-o ${_RENDER_SPV}
			DEPENDS ${_RENDER_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Render/Shader/${_RENDER_SHADER}.slang → ${_RENDER_SHADER}.spv"
		)
		list(APPEND OA_RENDER_SPV_OUTPUTS ${_RENDER_SPV})
	endforeach()

	# Crypto compute shaders — hashing (wired: Keccak/SHAKE/Merkle) + PQC
	# lattice primitives (NTT/Poly/ML-DSA: GPU roadmap, see OaPqc.md).
	set(OA_CRYPTO_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Crypto/Shader/Compute")
	set(OA_CRYPTO_SHADERS
		KeccakF1600
		Shake/Shake128 Shake/Shake256
		Merkle/MerkleReduce Merkle/MerkleVerify
		Ntt/NttForward Ntt/NttInverse Ntt/NttPointwise
		Poly/PolyArith Poly/PolyRound Poly/PolyHint Poly/PolySample
		Mldsa/MldsaVerify Mldsa/MldsaSign Mldsa/MldsaKeygen
	)
	set(OA_CRYPTO_SPV_OUTPUTS "")
	foreach(SHADER ${OA_CRYPTO_SHADERS})
		set(SRC "${OA_CRYPTO_SHADER_DIR}/${SHADER}.slang")
		set(SPV "${OA_SPIRV_DIR}/${SHADER}.spv")
		get_filename_component(SPV_DIR "${SPV}" DIRECTORY)
		add_custom_command(
			OUTPUT ${SPV}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${SPV_DIR}
			COMMAND ${SLANGC} ${SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${SPV}
			DEPENDS ${SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: crypto/${SHADER}.slang → ${SHADER}.spv"
		)
		list(APPEND OA_CRYPTO_SPV_OUTPUTS ${SPV})
	endforeach()

	# SPIR-V EMBEDDING — bake compute shaders into liboa.a (release) or stub (debug)
	# Collect all shader names (flattened to basename — no subdirs in .spv)
	set(OA_ALL_SPV_NAMES "")
	foreach(_REG ${OA_ML_SHADER_REG})
		# Strip directory prefix (e.g., "Quantize/QuantizeInt8" → "QuantizeInt8")
		get_filename_component(_REG_BASENAME "${_REG}" NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_REG_BASENAME})
	endforeach()
	list(APPEND OA_ALL_SPV_NAMES "TransposeBatched")
	foreach(_REG ${OA_KERNEL_MATRIX_MATMUL_REG})
		list(APPEND OA_ALL_SPV_NAMES ${_REG})
	endforeach()
	# SSM scan kernels
	foreach(_SHADER ${OA_SSM_SHADERS})
		list(APPEND OA_ALL_SPV_NAMES ${_SHADER})
	endforeach()
	# BLAS kernels. Keep this derived from the compile lists so new kernels
	# cannot build successfully while being absent from the embedded registry.
	foreach(_REG ${OA_BLAS_GEMM_KERNELS})
		get_filename_component(_REG_BASENAME "${_REG}" NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_REG_BASENAME})
	endforeach()
	# Generated FP32 raw and fused-epilogue variants.
	foreach(_REG ${OA_TILE_FP32_NAMES})
		list(APPEND OA_ALL_SPV_NAMES ${_REG})
	endforeach()
	# GemmCmSgBf16 template variants (raw, bias, bias+relu, bias+gelu, silu).
	set(_OA_BF16_REG_I 0)
	while(_OA_BF16_REG_I LESS ${_OA_BF16_LAST})
		list(GET OA_GEMM_BF16_VARIANTS ${_OA_BF16_REG_I} _BF16_REG_NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_BF16_REG_NAME})
		math(EXPR _OA_BF16_REG_I "${_OA_BF16_REG_I} + 2")
	endwhile()
	# GemmCmWgBf16 template variants.
	set(_OA_WG_BF16_REG_I 0)
	while(_OA_WG_BF16_REG_I LESS ${_OA_WG_BF16_LAST})
		list(GET OA_GEMM_WG_BF16_VARIANTS ${_OA_WG_BF16_REG_I} _WG_BF16_REG_NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_WG_BF16_REG_NAME})
		math(EXPR _OA_WG_BF16_REG_I "${_OA_WG_BF16_REG_I} + 2")
	endwhile()
	foreach(_REG ${OA_BLAS_GLSL_COMP_KERNELS})
		get_filename_component(_REG_BASENAME "${_REG}" NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_REG_BASENAME})
	endforeach()
	foreach(_REG ${OA_ML_BLAS_GLSL_COMP_KERNELS})
		get_filename_component(_REG_BASENAME "${_REG}" NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_REG_BASENAME})
	endforeach()
	foreach(_REG ${OA_BLAS_UTIL_KERNELS})
		get_filename_component(_REG_BASENAME "${_REG}" NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_REG_BASENAME})
	endforeach()
	list(APPEND OA_ALL_SPV_NAMES "TransposeTiled")
	# ScanSsm* SPVs removed along with the SSM module.
	list(APPEND OA_ALL_SPV_NAMES "PhiloxUniform")
	list(APPEND OA_ALL_SPV_NAMES "PhiloxNormal")
	list(APPEND OA_ALL_SPV_NAMES "PhiloxUniformGraph")
	list(APPEND OA_ALL_SPV_NAMES "PhiloxNormalGraph")
	list(APPEND OA_ALL_SPV_NAMES "PhiloxGraphAdvance")
	list(APPEND OA_ALL_SPV_NAMES "SrRoundBf16")
	# Audio DSP shaders. Derive names from the compile list so embedding cannot drift.
	foreach(_SHADER ${OA_AUDIO_SHADERS})
		get_filename_component(_NAME ${_SHADER} NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_NAME})
	endforeach()
	# Vision preprocessing shaders. Keep this derived from OA_VISION_SHADERS so
	# runtime dispatch names and embedded registry names stay in lock-step.
	foreach(_SHADER ${OA_VISION_SHADERS})
		get_filename_component(_NAME ${_SHADER} NAME)
		list(APPEND OA_ALL_SPV_NAMES ${_NAME})
	endforeach()
	# UI shaders. Derive embedding names from the compile list so the direct
	# pipeline loader cannot drift from generated SPIR-V.
	foreach(_SHADER ${OA_UI_SHADERS})
		list(APPEND OA_ALL_SPV_NAMES ${_SHADER})
	endforeach()
	# Render graphics shaders
	list(APPEND OA_ALL_SPV_NAMES "UnlitTextured.vert")
	list(APPEND OA_ALL_SPV_NAMES "UnlitTextured.frag")
	list(APPEND OA_ALL_SPV_NAMES "DetectionOverlay.vert")
	list(APPEND OA_ALL_SPV_NAMES "DetectionOverlay.frag")
	list(APPEND OA_ALL_SPV_NAMES "CanvasGlyph.vert")
	list(APPEND OA_ALL_SPV_NAMES "CanvasGlyph.frag")
	if(OA_BUILD_CRYPTO)
		foreach(SHADER ${OA_CRYPTO_SHADERS})
			list(APPEND OA_ALL_SPV_NAMES ${SHADER})
		endforeach()
	endif()

	# Extension shaders (OaAlm fusion kernels) — .slang files live in
	# Extensions/Private/Ml/Shader/Compute/ but are compiled + embedded here.
	set(OA_EXT_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Extensions/Private/Ml/Shader/Compute")
	set(OA_EXT_SHADER_REG
		ChannelNorm ChannelNormBwd
		ChannelNormRelu ChannelNormReluBwd
		SmoothL1Mean SmoothL1MeanBwd VelSmoothL1 VelSmoothL1Bwd
		Im2Col1d Col2Im1d
	)
	set(OA_EXT_SHADER_SRC
		Nn/ChannelNorm/ChannelNorm Nn/ChannelNorm/ChannelNormBwd
		Nn/ChannelNorm/ChannelNormRelu Nn/ChannelNorm/ChannelNormReluBwd
		Loss/SmoothL1Mean Loss/SmoothL1MeanBwd Loss/VelSmoothL1 Loss/VelSmoothL1Bwd
		Nn/Conv/Im2Col1d Nn/Conv/Col2Im1d
	)
	set(OA_EXT_SPV_OUTPUTS "")
	list(LENGTH OA_EXT_SHADER_REG _OA_EXT_N)
	math(EXPR _OA_EXT_LAST "${_OA_EXT_N} - 1")
	foreach(_I RANGE ${_OA_EXT_LAST})
		list(GET OA_EXT_SHADER_REG ${_I} _REG)
		list(GET OA_EXT_SHADER_SRC ${_I} _SRCREL)
		set(_SRC "${OA_EXT_SHADER_DIR}/${_SRCREL}.slang")
		set(_SPV "${OA_SPIRV_DIR}/${_REG}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${SLANGC} ${_SRC}
				-I ${OA_SLANG_UMBRELLA_DIR} -I ${OA_SLANG_MODULE_DIR}
				-target spirv -entry main -o ${_SPV}
			DEPENDS ${_SRC} ${OA_SLANG_MODULES}
			COMMENT "slangc: Extensions/${_SRCREL}.slang → ${_REG}.spv"
		)
		list(APPEND OA_EXT_SPV_OUTPUTS ${_SPV})
		list(APPEND OA_ALL_SPV_NAMES ${_REG})
	endforeach()

	set(OA_ALL_SPV_OUTPUTS ${OA_ML_SPV_OUTPUTS})
	list(APPEND OA_ALL_SPV_OUTPUTS ${OA_BLAS_SPV_OUTPUTS})
	list(APPEND OA_ALL_SPV_OUTPUTS ${OA_RENDER_SPV_OUTPUTS})
	if(OA_BUILD_CRYPTO)
		list(APPEND OA_ALL_SPV_OUTPUTS ${OA_CRYPTO_SPV_OUTPUTS})
	endif()
	list(APPEND OA_ALL_SPV_OUTPUTS ${OA_EXT_SPV_OUTPUTS})

	# Unified shader target — all kernels (ML + BLAS + Render + crypto + extensions)
	add_custom_target(oa_shaders ALL DEPENDS ${OA_ALL_SPV_OUTPUTS})

	# Remove orphan .spv files from previous builds (renamed/removed shaders)
	if(EXISTS "${OA_SPIRV_DIR}")
		file(GLOB_RECURSE OA_EXISTING_SPVS "${OA_SPIRV_DIR}/*.spv")
		foreach(SPV ${OA_EXISTING_SPVS})
			if(NOT SPV IN_LIST OA_ALL_SPV_OUTPUTS)
				file(REMOVE ${SPV})
				message(STATUS "Removed stale SPIR-V: ${SPV}")
			endif()
		endforeach()
	endif()

	# ninja clean_shaders → wipe all SPIR-V, forces full recompile
	add_custom_target(clean_shaders
		COMMAND ${CMAKE_COMMAND} -E rm -rf "${OA_SPIRV_DIR}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${OA_SPIRV_DIR}"
		COMMENT "Cleaning all compute shaders (ML + crypto)"
	)
	
	# ninja clean_cache → wipe pipeline cache (use after shader changes)
	add_custom_target(clean_cache
		COMMAND ${CMAKE_COMMAND} -E rm -f "${CMAKE_SOURCE_DIR}/var/vk/pipeline.vcache"
		COMMENT "Cleaning Vulkan pipeline cache"
	)
	
	# Set pipeline cache as additional clean file so 'ninja clean' removes it
	set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
		"${CMAKE_SOURCE_DIR}/var/vk/pipeline.vcache"
	)

	# Shader list file — only rewrite when content changes (avoids spurious embed rebuilds)
	set(OA_SHADER_LIST_FILE "${CMAKE_CURRENT_BINARY_DIR}/spirv_shader_list.txt")
	list(JOIN OA_ALL_SPV_NAMES "\n" OA_SHADER_LIST_CONTENT)
	set(OA_SHADER_LIST_CHANGED FALSE)
	if(EXISTS "${OA_SHADER_LIST_FILE}")
		file(READ "${OA_SHADER_LIST_FILE}" OA_SHADER_LIST_OLD)
		if(NOT OA_SHADER_LIST_OLD STREQUAL OA_SHADER_LIST_CONTENT)
			set(OA_SHADER_LIST_CHANGED TRUE)
		endif()
	else()
		set(OA_SHADER_LIST_CHANGED TRUE)
	endif()
	if(OA_SHADER_LIST_CHANGED)
		file(WRITE ${OA_SHADER_LIST_FILE} "${OA_SHADER_LIST_CONTENT}")
	endif()

	if(OA_EMBED_SHADERS)
		set(OA_EMBED_CPP "${CMAKE_CURRENT_BINARY_DIR}/oa_spirv_embed.cpp")
		add_custom_command(
			OUTPUT ${OA_EMBED_CPP}
			COMMAND ${CMAKE_COMMAND}
				-DSPIRV_DIR=${OA_SPIRV_DIR}
				-DOUTPUT=${OA_EMBED_CPP}
				-DSHADER_LIST_FILE=${OA_SHADER_LIST_FILE}
				-P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
			DEPENDS ${OA_ALL_SPV_OUTPUTS} ${OA_SHADER_LIST_FILE}
			COMMENT "Embedding SPIR-V shaders into liboa"
		)
		target_sources(oa PRIVATE ${OA_EMBED_CPP})
		target_compile_definitions(oa PUBLIC OA_EMBED_SHADERS)
		message(STATUS "SPIR-V embedding: ON (shaders baked into liboa.a)")
	else()
		target_sources(oa PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Runtime/Spirv/Registry.cpp")
		message(STATUS "SPIR-V embedding: OFF (file-based shader loading)")
	endif()
else()
	target_sources(oa PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Source/Private/Oa/Runtime/Spirv/Registry.cpp")
	message(STATUS "SPIR-V: slangc not found — stub registry only")
endif()

# TESTS & TUTORIALS
# Tutorials use the GTest harness (MlTestMain) + add_test, so GTest/enable_testing are
# needed whenever tutorials OR tests are built.
if(OA_BUILD_TESTS OR OA_BUILD_TUTORIALS)
	enable_testing()
	find_package(GTest CONFIG REQUIRED)
endif()

if(OA_BUILD_TESTS)
	# Test suite (unit tests, benchmarks, kernel correctness)
	add_subdirectory(Test)
endif()

# Tutorial suite (end-to-end learning materials; the headless ones install as oa-* tools)
if(OA_BUILD_TESTS OR OA_BUILD_TUTORIALS)
	add_subdirectory(Tutorial)
endif()

if(OA_BUILD_TESTS)
	# Examples (quick tests and prototypes)
	add_subdirectory(Examples)
endif()

# Desktop extensions and command-line applications are consumers of the OA
# library, not part of the Android AAR/APK native graph.
if(NOT OA_ANDROID_ML)
	add_subdirectory(Extensions)
	add_subdirectory(Apps)
endif()

# INSTALL
include(GNUInstallDirs)

# Install components (consumed by the AUR split packages):
#   oa      → runtime: the shared library only. With OA_EMBED_SHADERS=ON the SPIR-V is
#             baked into the .so, so the runtime package needs nothing else.
#   oa_sdk  → dev: headers, Slang shader sources, compiled SPIR-V, CMake package config.
install(TARGETS oa oa_headers oa_api
	EXPORT oa-targets
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT oa
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT oa_sdk
	INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Public API headers (Source/Public/Oa/{Core,Runtime,…}/ → include/Oa/{Core,Runtime,…}/)
install(DIRECTORY Source/Public/Oa/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Oa
	COMPONENT oa_sdk
)

# Slang math modules + core matrix kernels (sources for consumer -I paths)
install(DIRECTORY Source/Private/Oa/Core/Shader/Compute/Math/
	DESTINATION lib/Oa/Core/Shader/Compute/Math
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY Source/Private/Oa/Core/Shader/Compute/Matrix/
	DESTINATION lib/Oa/Core/Shader/Compute/Matrix
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY Source/Private/Oa/Core/Shader/Compute/Flat/
	DESTINATION lib/Oa/Core/Shader/Compute/Flat
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)

# Install Slang shader sources + compiled SPIR-V
install(DIRECTORY Source/Private/Oa/Ml/Shaders/
	DESTINATION lib/Oa/Ml/Shaders
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY Source/Private/Oa/Crypto/Shader/Compute/
	DESTINATION lib/Oa/Crypto/Shader/Compute
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.slang"
)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/spirv/
	DESTINATION lib/Oa/Ml/Shaders
	COMPONENT oa_sdk
	FILES_MATCHING PATTERN "*.spv"
)

install(EXPORT oa-targets
	FILE oa-targets.cmake
	NAMESPACE oa::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oa
	COMPONENT oa_sdk
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config-version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/oa-config.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config.cmake"
	@ONLY
)

install(FILES
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config.cmake"
	"${CMAKE_CURRENT_BINARY_DIR}/oa-config-version.cmake"
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/oa
	COMPONENT oa_sdk
)

# ─────────────────────────────────────────────────────────────────────────────
# Python Bindings (nanobind)
# ─────────────────────────────────────────────────────────────────────────────
# nanobind is a C++ dependency: prefer the vcpkg 'python' feature (vcpkg.json →
# nanobind, auto-selected when OA_BUILD_PYTHON is ON via VCPKG_MANIFEST_FEATURES
# above). Fall back to a pip-installed nanobind for builds outside the vcpkg
# toolchain.
if(OA_BUILD_PYTHON)
	# Link the extension against the interpreter the user actually imports from
	# (the newest system Python), NOT the private CPython that the vcpkg nanobind
	# port builds purely to satisfy its own dependency. That private copy sits on
	# CMAKE_PREFIX_PATH under the vcpkg toolchain, so pin the VERSION strategy:
	# FindPython picks the HIGHEST version found (system 3.14) over vcpkg's older
	# 3.12, regardless of search-path priority. Callers can still override with
	# -DPython_EXECUTABLE=... for a specific venv/conda target.
	if(NOT DEFINED Python_EXECUTABLE AND NOT DEFINED Python_ROOT_DIR)
		set(Python_FIND_STRATEGY VERSION)
		set(Python_FIND_VIRTUALENV STANDARD)
	endif()

	# nanobind requires Python (not Python3) to be found first
	find_package(Python COMPONENTS Interpreter Development REQUIRED)
	message(STATUS "Python bindings: using interpreter ${Python_EXECUTABLE} (${Python_VERSION})")

	# vcpkg (or any CMAKE_PREFIX_PATH) copy first — this is the supported path.
	find_package(nanobind CONFIG QUIET)

	if(nanobind_FOUND)
		message(STATUS "Found nanobind via vcpkg / CMAKE_PREFIX_PATH")
	else()
		# Fallback: locate a pip-installed nanobind and point CMake at its config.
		execute_process(
			COMMAND ${Python_EXECUTABLE} -m pip show nanobind
			OUTPUT_VARIABLE NANOBIND_PIP_INFO
			ERROR_QUIET
			RESULT_VARIABLE NANOBIND_PIP_RC
		)

		if(NANOBIND_PIP_RC EQUAL 0)
			message(STATUS "Found nanobind via pip")
			string(REGEX REPLACE ".*Location: ([^\r\n]+).*" "\\1" NANOBIND_PIP_DIR "${NANOBIND_PIP_INFO}")
			set(nanobind_DIR "${NANOBIND_PIP_DIR}/nanobind/cmake")
			find_package(nanobind CONFIG REQUIRED)
		else()
			message(FATAL_ERROR "nanobind not found. Enable the vcpkg 'python' feature "
				"(configure with -DOA_BUILD_PYTHON=ON under the vcpkg toolchain) or install "
				"it for the active interpreter: pip install nanobind")
		endif()
	endif()

	# Python native module. Binding sources mirror the public OA modules; one
	# private extension preserves a single nanobind type registry across them.
	set(OA_PYTHON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Source/Python")

	set(OA_PYTHON_SOURCES
		${OA_PYTHON_DIR}/Module.cpp
		${OA_PYTHON_DIR}/Core/Core.cpp
		${OA_PYTHON_DIR}/Core/Type.cpp
		${OA_PYTHON_DIR}/Core/Factory.cpp
		${OA_PYTHON_DIR}/Core/FnMatrix.cpp
		${OA_PYTHON_DIR}/Core/Backward.cpp
		${OA_PYTHON_DIR}/Runtime/Runtime.cpp
		${OA_PYTHON_DIR}/Ml/Ml.cpp
		${OA_PYTHON_DIR}/Ml/Module.cpp
		${OA_PYTHON_DIR}/Ml/Nn.cpp
		${OA_PYTHON_DIR}/Ml/Loss.cpp
		${OA_PYTHON_DIR}/Ml/Autograd.cpp
		${OA_PYTHON_DIR}/Ml/Optim.cpp
		${OA_PYTHON_DIR}/Ml/Training.cpp
		${OA_PYTHON_DIR}/Ml/Rl.cpp
		${OA_PYTHON_DIR}/Audio/Audio.cpp
		${OA_PYTHON_DIR}/Audio/Type.cpp
		${OA_PYTHON_DIR}/Audio/Codec.cpp
		${OA_PYTHON_DIR}/Audio/FnAudio.cpp
		${OA_PYTHON_DIR}/Vision/Vision.cpp
		${OA_PYTHON_DIR}/Vision/Type.cpp
		${OA_PYTHON_DIR}/Vision/Image.cpp
		${OA_PYTHON_DIR}/Vision/Detection.cpp
		${OA_PYTHON_DIR}/Vision/Codec.cpp
		${OA_PYTHON_DIR}/Vision/Video.cpp
		${OA_PYTHON_DIR}/Ui/Plot.cpp
	)
	# Crypto binding only when the crypto backend is built (OaFnHash/OaSign +
	# liboqs). Module.cpp guards the BindCrypto() call with #ifdef OA_BUILD_CRYPTO.
	if(OA_BUILD_CRYPTO)
		list(APPEND OA_PYTHON_SOURCES
			${OA_PYTHON_DIR}/Crypto/Crypto.cpp
			${OA_PYTHON_DIR}/Crypto/Hash.cpp
			${OA_PYTHON_DIR}/Crypto/Sign.cpp
			${OA_PYTHON_DIR}/Crypto/FnHash.cpp
		)
	endif()

	# `_oa` is deliberately private. Users import the public `oa` package and its
	# `core`, `runtime`, `ml`, `audio`, `vision`, and optional `crypto` modules.
	nanobind_add_module(_oa
		${OA_PYTHON_SOURCES}
	)
	target_link_libraries(_oa PRIVATE oa)
	target_include_directories(_oa PRIVATE
		${OA_PYTHON_DIR}
		${CMAKE_CURRENT_SOURCE_DIR}/Source/Public
		${CMAKE_CURRENT_SOURCE_DIR}/Source/Private
	)
	# Fix TLS relocation error for shared library
	target_compile_options(_oa PRIVATE -ftls-model=initial-exec)

	if(SKBUILD)
		# Keep native implementation details inside the `oa` package. Both DSOs live
		# together, so the extension resolves liboa through $ORIGIN.
		set_target_properties(_oa PROPERTIES INSTALL_RPATH "$ORIGIN")
		install(TARGETS _oa LIBRARY DESTINATION oa COMPONENT PythonWheel)
		install(TARGETS oa LIBRARY DESTINATION oa COMPONENT PythonWheel)
	else()
		# Dev/system install uses the same package layout as the wheel.
		install(TARGETS _oa LIBRARY DESTINATION ${Python_SITEARCH}/oa)
		install(DIRECTORY ${OA_PYTHON_DIR}/oa/
			DESTINATION ${Python_SITEARCH}/oa
			FILES_MATCHING PATTERN "*.py")
	endif()

	message(STATUS "Python bindings: ENABLED (nanobind)")
else()
	message(STATUS "Python bindings: DISABLED (set -DOA_BUILD_PYTHON=ON to enable)")
endif()

# SUMMARY
message(STATUS "")
message(STATUS "╔═══════════════════════════════════════════════════════════════╗")
message(STATUS "║             OA — REALM FOUNDATION LIBRARY                ")
message(STATUS "╠═══════════════════════════════════════════════════════════════╣")
message(STATUS "║  Version:     ${PROJECT_VERSION}                                        ")
message(STATUS "║  Build type:  ${CMAKE_BUILD_TYPE}                                     ")
message(STATUS "║  Validation:  ${OA_VULKAN_VALIDATION}                                       ")
message(STATUS "║  Crypto:      ${OA_BUILD_CRYPTO}                                       ")
message(STATUS "║  Python:      ${OA_BUILD_PYTHON}                                       ")
message(STATUS "║  Tests:       ${OA_BUILD_TESTS}                                       ")
message(STATUS "╚═══════════════════════════════════════════════════════════════╝")
message(STATUS "")
