# OA Test Suite
# All unit tests, benchmarks, and kernel correctness tests

function(oa_set_test_paths TARGET)
	set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT
		"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset"
		"OA_DATA_DIR=${OA_TEST_DATA_DIR}")
endfunction()

# The in-tree test harness deliberately exercises private engine state. Private
# runtime headers can include pinned third-party implementation headers, so
# every harness target needs both roots without publishing either through oa.
function(oa_add_private_test_includes TARGET)
	target_include_directories(${TARGET} PRIVATE
		${oa_SOURCE_DIR}/source/cpp/lib
		${oa_SOURCE_DIR}/source/cpp/thirdparty)
endfunction()

# Test helper for host tests. OA owns the main so capability profiles can turn
# any selected GoogleTest skip into a process failure.
function(oa_add_test TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaGtestMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest)
	target_include_directories(${TARGET} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/cpp
		${oa_SOURCE_DIR}/source/cpp/lib)
	oa_add_private_test_includes(${TARGET})
	string(REGEX REPLACE "^cpp/" "" _oa_output_subdir "${SUBDIR}")
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/${_oa_output_subdir}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	oa_set_test_paths(${TARGET})
endfunction()

# ML test helper (uses custom MlTestMain.cpp for Vulkan environment setup)
function(oa_add_ml_test TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
	oa_add_private_test_includes(${TARGET})
	# Engine-init suites use TEST_VK (alias for gtest TEST_F). Force-include the
	# alias header so gtest-direct and generated .gen.cpp sources see it without
	# per-file include churn. gcc/clang only (OA has no MSVC path).
	target_compile_options(${TARGET} PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
	string(REGEX REPLACE "^cpp/" "" _oa_output_subdir "${SUBDIR}")
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/${_oa_output_subdir}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	oa_set_test_paths(${TARGET})
	if(SOURCE MATCHES [[cpp/ml/]])
		set_tests_properties(${TARGET} PROPERTIES LABELS "gpu_bindless;ml")
	else()
		set_tests_properties(${TARGET} PROPERTIES LABELS "gpu_bindless")
	endif()
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
	set_property(GLOBAL APPEND PROPERTY OA_NATIVE_GPU_TESTS ${TARGET})
endfunction()

function(oa_require_no_skips PROFILE)
	foreach(TARGET IN LISTS ARGN)
		if(TEST ${TARGET})
			set_property(TEST ${TARGET} APPEND PROPERTY LABELS ${PROFILE})
			set_property(TEST ${TARGET} APPEND PROPERTY ENVIRONMENT
				"OA_TEST_FORBID_SKIPS=1")
		endif()
	endforeach()
endfunction()

function(oa_add_native_profile_test NAME TARGET PROFILE FILTER)
	add_test(NAME ${NAME}
		COMMAND $<TARGET_FILE:${TARGET}> --gtest_filter=${FILTER})
	set_tests_properties(${NAME} PROPERTIES
		LABELS ${PROFILE}
		ENVIRONMENT
			"OA_TEST_FORBID_SKIPS=1;OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}")
endfunction()

# Crypto test helper (uses custom CryptoTestMain.cpp for Vulkan environment setup)
function(oa_add_crypto_test TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/cpp/crypto/cryptoTestMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
	oa_add_private_test_includes(${TARGET})
	string(REGEX REPLACE "^cpp/" "" _oa_output_subdir "${SUBDIR}")
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/${_oa_output_subdir}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	oa_set_test_paths(${TARGET})
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# CORE TESTS

# cpp/core/misc tests
oa_add_test(TestTypes            cpp/core/misc/testTypes.cpp            cpp/core/misc)
oa_add_test(TestCoreMisc         cpp/core/misc/testCoreMisc.cpp         cpp/core/misc)
oa_add_test(TestCli              cpp/core/misc/testCli.cpp              cpp/core/misc)
oa_add_test(TestValidation       cpp/core/misc/testValidation.cpp       cpp/core/misc)
oa_add_test(TestVlm              cpp/core/vlm/testVlm.cpp               cpp/core/vlm)

# Host-only VLM microbenchmark. It is deliberately not a CTest: release claims
# use fresh-process runs with pinned compiler, CPU, clocks, and thermal state.
add_executable(BenchVlm cpp/core/vlm/benchVlm.cpp)
target_link_libraries(BenchVlm PRIVATE oa)
target_include_directories(BenchVlm SYSTEM PRIVATE
	${PROJECT_SOURCE_DIR}/source/cpp/thirdparty/xsimd/include)
set_target_properties(BenchVlm PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/core/vlm")
set(OA_VLM_GLM_ROOT "" CACHE PATH
	"Optional GLM include root for the historical VLM peer benchmark")
if(OA_VLM_GLM_ROOT AND EXISTS "${OA_VLM_GLM_ROOT}/glm/glm.hpp")
	target_include_directories(BenchVlm SYSTEM PRIVATE "${OA_VLM_GLM_ROOT}")
	target_compile_definitions(BenchVlm PRIVATE OA_VLM_HAS_GLM=1)
endif()

# cpp/core/memory tests
oa_add_test(TestMemory           cpp/core/memory/testMemory.cpp         cpp/core/memory)

# cpp/core/container tests
oa_add_test(TestContainers       cpp/core/container/testContainers.cpp  cpp/core/container)

# cpp/core/image tests (Phase 1: metadata wrapper, requires Vulkan since OA is GPU-only)
oa_add_ml_test(TestImage         cpp/core/testImage.cpp                 cpp/core)
oa_add_ml_test(TestUi            cpp/ui/testUi.cpp                      cpp/ui)
target_include_directories(TestUi PRIVATE
	${oa_SOURCE_DIR}/source/cpp/thirdparty)
oa_add_test(TestCamera           cpp/render/testCamera.cpp              cpp/render)
set_tests_properties(TestCamera PROPERTIES LABELS "core;render")
oa_add_test(TestLunarLander3dRender cpp/render/testLunarLander3dRender.cpp cpp/render)
target_link_libraries(TestLunarLander3dRender PRIVATE oaSdkSupport)
target_sources(TestLunarLander3dRender PRIVATE
	${oa_SOURCE_DIR}/sdk/cpp/tutorials/ml/rl/lunarLander3d/lunarLander3dRender.cpp)
target_include_directories(TestLunarLander3dRender PRIVATE
	${oa_SOURCE_DIR}/sdk/cpp/tutorials/ml/rl/lunarLander3d)
set_tests_properties(TestLunarLander3dRender PROPERTIES
	LABELS "gpu_bindless;render;rl;headless_readback")
if(TARGET oa_shaders)
	add_dependencies(TestLunarLander3dRender oa_shaders)
endif()

# cpp/core/audio tests (Phase 6: audio wrapper, requires Vulkan since OA is GPU-only)
oa_add_ml_test(TestAudio         cpp/core/testAudio.cpp                 cpp/core)

# cpp/core/matrix operator overloads + shape+fill constructor (needs Vulkan).
oa_add_ml_test(TestMatrixOperators cpp/core/testMatrixOperators.cpp     cpp/core)

# cpp/core/threading tests
oa_add_test(TestThreading        cpp/core/threading/testThreading.cpp   cpp/core/threading)

# Network tests
oa_add_test(TestMcp              cpp/mcp/testMcp.cpp                     cpp/mcp)
set_tests_properties(TestMcp PROPERTIES LABELS "core;mcp")
oa_add_test(TestTcpFramed        cpp/network/tcp/testTcpFramed.cpp      cpp/network/tcp)
oa_add_test(TestSatelliteProtocol cpp/network/satellite/testSatelliteProtocol.cpp cpp/network/satellite)
oa_add_test(TestSatelliteSession cpp/network/satellite/testSatelliteSession.cpp cpp/network/satellite)
oa_add_ml_test(TestSatelliteVulkan cpp/network/satellite/testSatelliteVulkan.cpp cpp/network/satellite)
target_include_directories(TestSatelliteVulkan PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)
oa_add_ml_test(TestSatelliteNlp cpp/network/satellite/testSatelliteNlp.cpp cpp/network/satellite)
target_link_libraries(TestSatelliteNlp PRIVATE oaSdkSupport)
target_sources(TestSatelliteNlp PRIVATE
	${oa_SOURCE_DIR}/sdk/cpp/applications/network/satellite/nlpCoordinator.cpp
	${oa_SOURCE_DIR}/sdk/cpp/applications/network/satellite/nlpStep.cpp)
target_include_directories(TestSatelliteNlp PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)
set_tests_properties(TestSatelliteNlp PROPERTIES
	LABELS "gpu_bindless;ml;network;distributed")

oa_add_test(TestTransferWeights  cpp/ml/io/testSafeTensorReader.cpp     cpp/ml/io)
oa_add_test(TestModelFileIntegrity     cpp/ml/io/testModelFileIntegrity.cpp         cpp/ml/io)
oa_add_ml_test(TestCheckpoint    cpp/ml/io/testCheckpoint.cpp           cpp/ml/io)
oa_add_test(TestTokenizer        cpp/ml/data/testTokenizer.cpp          cpp/ml/data)
oa_add_test(TestGptOssConfig     cpp/ml/io/testGptOssConfig.cpp         cpp/ml/io)
target_link_libraries(TestGptOssConfig PRIVATE oaSdkSupport)
oa_add_ml_test(TestAttention     cpp/ml/nn/testAttention.cpp            cpp/ml/nn)
oa_add_ml_test(TestFlow          cpp/ml/generative/testFlow.cpp         cpp/ml/generative)
target_link_libraries(TestFlow PRIVATE oaSdkSupport)
if(OA_ENABLE_ALM)
	oa_add_ml_test(TestClipText cpp/ml/nn/testClipText.cpp cpp/ml/nn)
	target_link_libraries(TestClipText PRIVATE oaSdkSupport)
endif()

set_tests_properties(
	TestTypes TestMemory TestContainers TestThreading TestCoreMisc TestCli TestValidation TestVlm
	TestTransferWeights TestModelFileIntegrity TestTokenizer TestGptOssConfig
	PROPERTIES LABELS "core"
)

# ═══════════════════════════════════════════════════════════════════════════
# OASTD TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════

add_executable(TestOaStd
	cpp/oaGtestMain.cpp
	cpp/core/std/testChrono.cpp
	cpp/core/std/testAllocator.cpp
	cpp/core/std/testArray.cpp
	cpp/core/std/testAssert.cpp
	cpp/core/std/testOptional.cpp
	cpp/core/std/testStringView.cpp
	cpp/core/std/testString.cpp
	cpp/core/std/testSpan.cpp
	cpp/core/std/testHashMap.cpp
	cpp/core/std/testHashSet.cpp
	cpp/core/std/testUniquePtr.cpp
	cpp/core/std/testSharedPtr.cpp
	cpp/core/std/testVariant.cpp
	cpp/core/std/testFunction.cpp
	cpp/core/std/testPath.cpp
	cpp/core/std/testAlgo.cpp
	cpp/core/std/testVector.cpp
	cpp/core/std/testRandom.cpp
	cpp/core/std/testFormat.cpp
	cpp/core/std/testSync.cpp
	cpp/core/std/testCString.cpp
	cpp/core/std/testPairLimits.cpp
	cpp/core/std/testScalarMath.cpp
)
target_link_libraries(TestOaStd PRIVATE oa GTest::gtest)
target_include_directories(TestOaStd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
oa_add_private_test_includes(TestOaStd)
set_target_properties(TestOaStd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/core/std")
add_test(NAME TestOaStd COMMAND TestOaStd)
set_tests_properties(TestOaStd PROPERTIES LABELS "core;oastd")

# Correctness-gated OA foundation versus hosted standard-library benchmark.
# It is deliberately not a CTest: release comparisons use the canonical
# fresh-process benchmark runner with fixed CPU, compiler and thermal evidence.
add_executable(BenchStd cpp/core/std/benchStd.cpp)
target_link_libraries(BenchStd PRIVATE oa)
set_target_properties(BenchStd PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/core/std")

if(OA_OASTD_EXTRA_TARGETS)
	set(_OA_OASTD_STD_HDRS
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/chrono.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/allocator.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/array.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/optional.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/uniquePtr.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/sharedPtr.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/variant.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/function.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/hashMap.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/span.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/string.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/stringView.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/typeTraits.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/path.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/algo.h"
		"${oa_SOURCE_DIR}/source/cpp/include/oa/core/std/vector.h"
	)
	add_custom_target(oa_oastd_headers SOURCES ${_OA_OASTD_STD_HDRS})
	add_custom_target(oastd COMMENT "Build OA standard-library test suite (TestOaStd)" DEPENDS TestOaStd)
endif()

# ═══════════════════════════════════════════════════════════════════════════
# ML TESTS
# ═══════════════════════════════════════════════════════════════════════════

oa_add_ml_test(TestTimer cpp/runtime/profiling/testTimer.cpp cpp/runtime/profiling)
oa_add_ml_test(TestGemmRouter cpp/runtime/gemm/testGemmRouter.cpp cpp/runtime/gemm)
target_include_directories(TestGemmRouter PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)

# ═══════════════════════════════════════════════════════════════════════════
# CORE MATRIX TESTS (Generated from schemas)
# ═══════════════════════════════════════════════════════════════════════════

# Re-enabled: fixed generated tests (swiglu/siluMul/geglu marked skip_test)
oa_add_ml_test(TestFnMatrixBlas cpp/core/fnmatrix/blas/testFnMatrixBlas.cpp cpp/core/fnmatrix/blas)
oa_add_ml_test(TestFnMatrixBlasSimple cpp/core/fnmatrix/blas/testFnMatrixBlasSimple.cpp cpp/core/fnmatrix/blas)
oa_add_ml_test(TestFnMatrixElemwise cpp/core/fnmatrix/elemwise/testFnMatrixElemwise.gen.cpp cpp/core/fnmatrix/elemwise)
oa_add_ml_test(TestFnMatrixReduce cpp/core/fnmatrix/reduce/testFnMatrixReduce.gen.cpp cpp/core/fnmatrix/reduce)
oa_add_ml_test(TestFnMatrixRng cpp/core/fnmatrix/rng/testFnMatrixRng.gen.cpp cpp/core/fnmatrix/rng)
oa_add_ml_test(TestFnMatrixRngManual cpp/core/fnmatrix/rng/testFnMatrixRngManual.cpp cpp/core/fnmatrix/rng)
oa_add_ml_test(TestFnMatrixShape cpp/core/fnmatrix/shape/testFnMatrixShape.gen.cpp cpp/core/fnmatrix/shape)
oa_add_ml_test(TestFnMatrixShapeManual cpp/core/fnmatrix/shape/testFnMatrixShapeManual.cpp cpp/core/fnmatrix/shape)
oa_add_ml_test(TestFnMatrixIndex cpp/core/fnmatrix/index/testFnMatrixIndex.cpp cpp/core/fnmatrix/index)
oa_add_ml_test(TestFnMatrixUtil cpp/core/fnmatrix/util/testFnMatrixUtil.cpp cpp/core/fnmatrix/util)
oa_add_ml_test(TestTopKBug cpp/core/fnmatrix/util/testTopKBug.cpp cpp/core/fnmatrix/util)

# ═══════════════════════════════════════════════════════════════════════════
# ML MATRIX TESTS (Generated from schemas)
# ═══════════════════════════════════════════════════════════════════════════

# Re-enabled: fixed generated tests (swiglu/siluMul/geglu marked skip_test)
oa_add_ml_test(TestFnMatrixActivation cpp/ml/fnmatrix/activation/testFnMatrixActivation.gen.cpp cpp/ml/fnmatrix/activation)
oa_add_ml_test(TestFnMatrixEmbed cpp/ml/fnmatrix/embed/testFnMatrixEmbed.gen.cpp cpp/ml/fnmatrix/embed)
oa_add_ml_test(TestFnLoss cpp/ml/loss/testFnLoss.cpp cpp/ml/loss)
oa_add_ml_test(TestRl cpp/ml/rl/testRl.cpp cpp/ml/rl)
target_sources(TestRl PRIVATE
	cpp/ml/rl/testCartPole.cpp)
target_link_libraries(TestRl PRIVATE oaSdkSupport)
target_include_directories(TestRl PRIVATE
	${oa_SOURCE_DIR}/sdk/cpp/tutorials/ml/rl
	${oa_SOURCE_DIR}/sdk/cpp/lib)
oa_add_test(TestLunarLander3d cpp/ml/rl/testLunarLander3d.cpp cpp/ml/rl)
target_link_libraries(TestLunarLander3d PRIVATE oaSdkSupport)
set_tests_properties(TestLunarLander3d PROPERTIES LABELS "host_oracle;ml;rl")
oa_add_ml_test(
	TestLunarLander3dVector
	cpp/ml/rl/testLunarLander3dVector.cpp
	cpp/ml/rl)
target_link_libraries(TestLunarLander3dVector PRIVATE oaSdkSupport)
target_include_directories(TestLunarLander3dVector PRIVATE
	${oa_SOURCE_DIR}/sdk/cpp/lib)
set_tests_properties(TestLunarLander3dVector PROPERTIES
	LABELS "gpu_bindless;ml;rl;host_differential")
oa_add_ml_test(
	TestLunarLander3dPpo
	cpp/ml/rl/testLunarLander3dPpo.cpp
	cpp/ml/rl)
target_link_libraries(TestLunarLander3dPpo PRIVATE oaSdkSupport)
target_sources(TestLunarLander3dPpo PRIVATE
	cpp/ml/rl/lunarLander3dPpo.cpp)
set_tests_properties(TestLunarLander3dPpo PROPERTIES
	LABELS "gpu_bindless;ml;rl;training"
	TIMEOUT 120)
oa_add_ml_test(TestFnMatrixPool cpp/ml/fnmatrix/pool/testFnMatrixPool.cpp cpp/ml/fnmatrix/pool)
oa_add_ml_test(TestOptimizers cpp/ml/optim/testOptimizers.cpp cpp/ml/optim)
oa_add_ml_test(TestFnMatrixBackward cpp/ml/fnmatrix/backward/testFnMatrixBackward.cpp cpp/ml/fnmatrix/backward)
# TestFnMatrixConv (Core) references a phantom oa::FnMatrix::im2Col/col2Im/conv2d API that
# does not exist (real Conv2d is in cpp/ml/fnMatrix.h) — orphaned since 6c1c36e, left unregistered.
oa_add_ml_test(TestFnMatrixConvBwd cpp/ml/fnmatrix/conv/testFnMatrixConvBwd.cpp cpp/ml/fnmatrix/conv)
oa_add_ml_test(TestFnMatrixConvTranspose2d cpp/ml/fnmatrix/conv/testFnMatrixConvTranspose2d.cpp cpp/ml/fnmatrix/conv)
oa_add_ml_test(TestMoeGradcheck cpp/ml/nn/testMoeGradcheck.cpp cpp/ml/nn)
oa_add_ml_test(TestMoeSystems cpp/ml/nn/testMoeSystems.cpp cpp/ml/nn)
# Recovered orphans (added in 6c1c36e "100% coverage" but never wired into the build).
oa_add_ml_test(TestGradFnMatrix cpp/ml/grad/testFnMatrix.cpp cpp/ml/grad)
oa_add_ml_test(TestFnMatrixNormAdvanced cpp/ml/fnmatrix/norm/testFnMatrixNormAdvanced.cpp cpp/ml/fnmatrix/norm)
# TestFnMatrixActivationBwd surfaced a real MishBwd op bug (used sigmoid(softplus(x)) instead of
# sigmoid(x)) — fixed in MishBwd.slang; SoftplusBwd failures were test bugs (passed input, not output).
oa_add_ml_test(TestFnMatrixActivationBwd cpp/ml/fnmatrix/activation/testFnMatrixActivationBwd.cpp cpp/ml/fnmatrix/activation)
# Other 6c1c36e orphans NOT re-registered — dead API or wrong-contract tests. See OaSanitization.md §0c:
#   - TestFnMatrixElemwiseBwd / TestFnMatrixShapeBwd / TestFnMatrixAllocManual : dead API (~20 errs each)
#   - TestNnModules / TestByteEmbedding : dead API (3-6 errs)
#   - cpp/core/fnmatrix/conv/testFnMatrixConv : phantom im2Col/col2Im/conv2d API
oa_add_ml_test(TestFnMatrixQuant cpp/ml/quant/testFnMatrixQuant.cpp cpp/ml/quant)

# ═══════════════════════════════════════════════════════════════════════════
# ML INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

oa_add_ml_test(TestByte cpp/ml/precision/testByte.cpp cpp/ml/precision)
oa_add_ml_test(TestDataLoader cpp/ml/data/testDataLoader.cpp cpp/ml/data)
oa_add_ml_test(TestTrainingMetrics cpp/ml/training/testTrainingMetrics.cpp cpp/ml/training)
target_include_directories(TestTrainingMetrics PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)
oa_add_ml_test(TestAutogradLeak cpp/ml/training/testAutogradLeak.cpp cpp/ml/training)
oa_add_test(TestPoseClip cpp/ml/nn/testPoseClip.cpp cpp/ml/nn)
target_link_libraries(TestPoseClip PRIVATE oaSdkSupport)
# Gen3dAnim tests removed — will be redone after sanitization + MotionGpt work.

# 3D anim engine: Core (transform/joint) → Rig (skeleton) → Retarget.
oa_add_test(TestTransform cpp/core/transform/testTransform.cpp cpp/core/transform)
oa_add_test(TestSkeleton cpp/rig/testSkeleton.cpp cpp/rig)
oa_add_test(TestRetarget cpp/retarget/testRetarget.cpp cpp/retarget)
target_link_libraries(TestTransform PRIVATE oaSdkSupport)
target_link_libraries(TestSkeleton PRIVATE oaSdkSupport)
target_link_libraries(TestRetarget PRIVATE oaSdkSupport)

# ═══════════════════════════════════════════════════════════════════════════
# KERNEL CORRECTNESS TESTS (GPU with CPU reference validation)
# ═══════════════════════════════════════════════════════════════════════════

# DISABLED: Missing source files (deleted with test/cpp/core/matrix cleanup)
# oa_add_ml_test(TestFlatKernels cpp/core/matrix/testFlatKernels.cpp cpp/core/matrix)
# oa_add_ml_test(TestMatrixKernels cpp/core/matrix/testMatrixKernels.cpp cpp/core/matrix)
# oa_add_ml_test(TestQuantization cpp/core/matrix/testQuantization.cpp cpp/core/matrix)
oa_add_ml_test(TestNnKernels cpp/ml/nn/testNnKernels.cpp cpp/ml/nn)
oa_add_ml_test(TestNnLayers cpp/ml/nn/testNnLayers.cpp cpp/ml/nn)
oa_add_ml_test(TestMamba3 cpp/ml/nn/testMamba3.cpp cpp/ml/nn)
oa_add_ml_test(TestVqAssign cpp/ml/nn/testVqAssign.cpp cpp/ml/nn)
oa_add_ml_test(TestGru cpp/ml/nn/testGru.cpp cpp/ml/nn)
oa_add_ml_test(TestGruNumericalGrad cpp/ml/nn/testGruNumericalGrad.cpp cpp/ml/nn)
oa_add_ml_test(TestGruScan cpp/ml/nn/testGruScan.cpp cpp/ml/nn)
oa_add_ml_test(TestLinearGelu cpp/ml/nn/testLinearGelu.cpp cpp/ml/nn)
oa_add_ml_test(TestRnn cpp/ml/nn/testRnn.cpp cpp/ml/nn)
oa_add_ml_test(TestRnnScan cpp/ml/nn/testRnnScan.cpp cpp/ml/nn)
oa_add_ml_test(TestRope cpp/ml/nn/testRope.cpp cpp/ml/nn)
oa_add_ml_test(TestOptimKernels cpp/ml/optim/testOptimKernels.cpp cpp/ml/optim)
oa_add_ml_test(TestOptimMuon cpp/ml/optim/testOptimMuon.cpp cpp/ml/optim)

# TestGemmKernels and TestRngKernels have their own main() — use oa_add_test
# DISABLED: Missing source file (deleted with test/cpp/core/matrix cleanup)
# oa_add_test(TestGemmKernels cpp/core/matrix/testGemmKernels.cpp cpp/core/matrix)
# if(TARGET oa_shaders)
# 	add_dependencies(TestGemmKernels oa_shaders)
# endif()

# TestRngKernels disabled — uses legacy fillUniform/fillNormal API that was removed
# TODO: Rewrite to use oa::Matrix-level RNG operations (rand, randN, etc.)
# oa_add_test(TestRngKernels cpp/core/matrix/testRngKernels.cpp cpp/core/matrix)
# if(TARGET oa_shaders)
# 	add_dependencies(TestRngKernels oa_shaders)
# endif()

# DISABLED: Removed missing tests (TestFlatKernels, TestMatrixKernels, TestQuantization, TestGemmKernels)
set_tests_properties(
	TestNnKernels TestOptimKernels TestOptimMuon TestVqAssign
	PROPERTIES LABELS "gpu_bindless;kernel_correctness"
)

# ═══════════════════════════════════════════════════════════════════════════
# VISION TESTS
# ═══════════════════════════════════════════════════════════════════════════

# Phase 1 — Image Processing (COMPLETE)
oa_add_ml_test(TestImageCodec cpp/vision/codec/testImageCodec.cpp cpp/vision/codec)
oa_add_ml_test(TestImageProcessor cpp/vision/processor/testImageProcessor.cpp cpp/vision/processor)
# Phase 2 — Video Decode (IN PROGRESS)
oa_add_ml_test(TestVideoDecoder cpp/vision/codec/testVideoDecoder.cpp cpp/vision/codec)
# Phase 2a — End-to-end .mp4 decode validated vs ffmpeg reference frames
oa_add_ml_test(TestVideoDecodeReference cpp/vision/codec/testVideoDecodeReference.cpp cpp/vision/codec)
# Phase 2b — MP4 demuxer + OaFnVideo NAL ops
oa_add_ml_test(TestVideoDemuxer  cpp/vision/stream/testVideoDemuxer.cpp cpp/vision/stream)
oa_add_test(TestVideoSessionLifecycle cpp/vision/stream/testVideoSessionLifecycle.cpp cpp/vision/stream)
# Phase 2c — encode → mux → demux (→ decode) roundtrip
oa_add_ml_test(TestVideoRoundtrip cpp/vision/stream/testVideoRoundtrip.cpp cpp/vision/stream)
# Phase 3 — Unified OaFnImage API (GPU-only)
oa_add_ml_test(TestFnImage cpp/vision/image/testFnImage.cpp cpp/vision/image)
# Phase 4 — Kernel-level tests with CPU reference verification
oa_add_ml_test(TestGeometricKernels cpp/vision/image/kernel/testGeometricKernels.cpp cpp/vision/image/kernel)
oa_add_ml_test(TestFilterKernels cpp/vision/image/kernel/testFilterKernels.cpp cpp/vision/image/kernel)
oa_add_ml_test(TestVision50Ops cpp/vision/image/kernel/testVision50Ops.cpp cpp/vision/image/kernel)
oa_add_ml_test(TestFnDetection cpp/vision/detection/testFnDetection.cpp cpp/vision/detection)

set_tests_properties(TestImageCodec TestImageProcessor TestVideoDecoder TestVideoDecodeReference TestFnImage TestGeometricKernels TestFilterKernels TestVision50Ops TestFnDetection
	PROPERTIES LABELS "gpu_bindless;ml;vision")
set_tests_properties(TestVideoSessionLifecycle
	PROPERTIES LABELS "gpu_bindless;vision")

# ═══════════════════════════════════════════════════════════════════════════
# AUDIO TESTS
# ═══════════════════════════════════════════════════════════════════════════

# decoder/encoder round-trips + OaFnAudio ops vs in-test CPU oracles.
oa_add_ml_test(TestAudioDecoder cpp/audio/codec/testAudioDecoder.cpp cpp/audio/codec)
oa_add_test(TestAudioSessionLifecycle cpp/audio/codec/testAudioSessionLifecycle.cpp cpp/audio/codec)
oa_add_ml_test(TestFnAudio      cpp/audio/testFnAudio.cpp            cpp/audio)
set_tests_properties(TestAudioDecoder TestAudioSessionLifecycle TestFnAudio
	PROPERTIES LABELS "gpu_bindless;audio")

# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME TESTS
# ═══════════════════════════════════════════════════════════════════════════

oa_add_test(TestAllocator cpp/runtime/memory/testAllocator.cpp cpp/runtime/memory)
target_include_directories(TestAllocator PRIVATE
	${oa_SOURCE_DIR}/source/cpp/lib
	${oa_SOURCE_DIR}/source/cpp/thirdparty)
oa_add_test(TestVma cpp/runtime/memory/testVma.cpp cpp/runtime/memory)
target_include_directories(TestVma PRIVATE
	${oa_SOURCE_DIR}/source/cpp/thirdparty)
oa_add_test(TestVmaInternals cpp/runtime/memory/testVmaInternals.cpp cpp/runtime/memory)
target_include_directories(TestVmaInternals PRIVATE
	${oa_SOURCE_DIR}/source/cpp/thirdparty)

oa_add_ml_test(TestKernelRegistry cpp/runtime/engine/testKernelRegistry.cpp cpp/runtime/engine)
add_test(
	NAME TestSpirvEmbedGenerator
	COMMAND ${CMAKE_COMMAND}
		-DOA_EMBED_SCRIPT=${oa_SOURCE_DIR}/cmake/embed_spirv.cmake
		-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/spirvEmbedGeneratorTest
		-P ${oa_SOURCE_DIR}/cmake/test_embed_spirv.cmake)
set_tests_properties(TestSpirvEmbedGenerator PROPERTIES LABELS "core;generation")
find_program(OA_TEST_PYTHON_EXECUTABLE NAMES python3 python REQUIRED)
add_test(
	NAME TestKernelGeneration
	COMMAND ${OA_TEST_PYTHON_EXECUTABLE}
		${oa_SOURCE_DIR}/tools/gen/kernel/testGenerate.py)
set_tests_properties(TestKernelGeneration PROPERTIES LABELS "core;generation")
oa_add_ml_test(TestEngineInit cpp/runtime/engine/testEngineInit.cpp cpp/runtime/engine)
oa_add_test(TestVkDispatch cpp/runtime/engine/testVkDispatch.cpp cpp/runtime/engine)
target_include_directories(TestVkDispatch PRIVATE
	${oa_SOURCE_DIR}/source/cpp/thirdparty)
set_tests_properties(TestVkDispatch PROPERTIES LABELS "core;runtime")
oa_add_test(TestComputeApp cpp/runtime/engine/testComputeApp.cpp cpp/runtime/engine)
set_tests_properties(TestComputeApp PROPERTIES LABELS "gpu_bindless")
if(TARGET SDL3::SDL3)
	oa_add_test(TestPresenterWsi cpp/runtime/engine/testPresenterWsi.cpp cpp/runtime/engine)
	target_link_libraries(TestPresenterWsi PRIVATE SDL3::SDL3)
	set_tests_properties(TestPresenterWsi PROPERTIES LABELS "gpu_wsi")
	if(OA_SANITIZERS MATCHES "(^|,)address(,|$)")
		# SDL's Wayland libdecor GTK plug-in retains gTK/pango/fontconfig
		# process caches. Suppress only leak stacks rooted in that external
		# plug-in; OA and all other WSI allocations remain visible to LSan.
		set_tests_properties(TestPresenterWsi PROPERTIES
			ENVIRONMENT_MODIFICATION
				"LSAN_OPTIONS=string_append::suppressions=${CMAKE_CURRENT_SOURCE_DIR}/cpp/runtime/engine/lsan.supp")
	endif()
endif()
oa_add_ml_test(TestStream cpp/runtime/stream/testStream.cpp cpp/runtime/stream)
target_include_directories(TestStream PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)
oa_add_ml_test(TestExecutableGraph cpp/runtime/graph/testExecutableGraph.cpp cpp/runtime/graph)
target_include_directories(TestExecutableGraph PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)
oa_add_ml_test(TestCollective cpp/runtime/collective/testCollective.cpp cpp/runtime/collective)

# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO TESTS
# ═══════════════════════════════════════════════════════════════════════════

# CPU keccak/sHAKE — always available, no external deps
oa_add_test(TestKeccak cpp/crypto/hash/testKeccak.cpp cpp/crypto/hash)
oa_add_test(TestHash cpp/crypto/hash/testHash.cpp cpp/crypto/hash)
set_tests_properties(TestKeccak TestHash PROPERTIES LABELS "crypto;cpu")

if(OA_BUILD_CRYPTO)
	# oa::FnHash tests — unified crypto API (requires Vulkan for TEST_F fixture)
	oa_add_crypto_test(TestFnHash cpp/crypto/hash/testFnHash.cpp cpp/crypto/hash)
	set_tests_properties(TestFnHash PROPERTIES LABELS "crypto;gpu")

	# oa::FnHash extended tests — graph/executor modes (requires Vulkan)
	oa_add_crypto_test(TestFnHashExtended cpp/crypto/hash/testFnHashExtended.cpp cpp/crypto/hash)
	set_tests_properties(TestFnHashExtended PROPERTIES LABELS "crypto;gpu")

	# Crypto benchmarks — CPU vs GPU performance validation
	oa_add_crypto_test(BenchHash cpp/crypto/hash/benchHash.cpp cpp/crypto/hash)
	set_tests_properties(BenchHash PROPERTIES LABELS "crypto;benchmark;gpu")

	# ML-DSA-65 sign/verify — CPU (liboqs), no Vulkan needed
	oa_add_test(TestSign cpp/crypto/sign/testSign.cpp cpp/crypto/sign)
	set_tests_properties(TestSign PROPERTIES LABELS "crypto")
endif()

# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════

# Host memcpy/libc/non-temporal comparison across size and alignment regimes.
add_executable(BenchMemory cpp/core/memory/benchMemory.cpp)
target_link_libraries(BenchMemory PRIVATE oa)
target_include_directories(BenchMemory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
set_target_properties(BenchMemory PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/core/memory")

# Engine-owned allocator-policy benchmark. Raw VMA internals remain the
# upstream allocator's responsibility; OA measures its cache/reuse seam here.
add_executable(BenchAllocator cpp/runtime/memory/benchAllocator.cpp)
target_link_libraries(BenchAllocator PRIVATE oa)
target_include_directories(BenchAllocator PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/cpp
	${PROJECT_SOURCE_DIR}/source/cpp/lib
	${PROJECT_SOURCE_DIR}/source/cpp/thirdparty)
set_target_properties(BenchAllocator PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/runtime/memory")

# Direct CPU-side VMA algorithm benchmark. The same source also has a
# standalone OA_BENCH_VMA_STOCK mode for pinned-upstream comparison.
add_executable(BenchVma cpp/runtime/memory/benchVma.cpp)
target_link_libraries(BenchVma PRIVATE oa)
target_include_directories(BenchVma PRIVATE
	${PROJECT_SOURCE_DIR}/source/cpp/thirdparty)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	# Match the flags used by the product VMA translation unit and the pinned
	# stock comparison. Different alias/overflow semantics invalidate ratios.
	target_compile_options(BenchVma PRIVATE
		-fno-strict-aliasing
		-fno-strict-overflow)
endif()
set_target_properties(BenchVma PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/runtime/memory")

# Persistent mapped upload ring vs allocate/submit/wait/free per copy.
# Deliberately not registered with ctest: this is a controlled benchmark.
add_executable(BenchUploadRing cpp/runtime/stream/benchUploadRing.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
target_link_libraries(BenchUploadRing PRIVATE oa GTest::gtest)
target_include_directories(BenchUploadRing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(BenchUploadRing PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
set_target_properties(BenchUploadRing PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/runtime/stream")

# Tuned KHR CoopMat BF16 GEMM (GemmCmSgBf16) — correctness + perf.
oa_add_ml_test(TestGemmCmSgBf16 cpp/compute/testGemmCmSgBf16.cpp cpp/compute)
set_tests_properties(TestGemmCmSgBf16 PROPERTIES LABELS "gpu_bindless;kernel_correctness")

# Mamba-3 isolated GPU-timestamped kernel/preprocess/prefill/step breakdown.
# Deliberately not registered with ctest: this is a controlled benchmark, not a unit test.
add_executable(BenchMamba3 cpp/ml/nn/benchMamba3.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
target_link_libraries(BenchMamba3 PRIVATE oa GTest::gtest)
target_include_directories(BenchMamba3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(BenchMamba3 PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
set_target_properties(BenchMamba3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/ml/nn")
if(TARGET oa_shaders)
	add_dependencies(BenchMamba3 oa_shaders)
endif()

add_executable(BenchMoeComponents cpp/ml/nn/benchMoeComponents.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
target_link_libraries(BenchMoeComponents PRIVATE oa GTest::gtest)
target_include_directories(BenchMoeComponents PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(BenchMoeComponents PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
set_target_properties(BenchMoeComponents PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/ml/nn")
if(TARGET oa_shaders)
	add_dependencies(BenchMoeComponents oa_shaders)
endif()

# Dense-Transformer forward/backward component breakdown for the canonical NLP shape.
# Deliberately not registered with ctest: this is a controlled benchmark.
add_executable(BenchTransformerComponents cpp/ml/nn/benchTransformerComponents.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
target_link_libraries(BenchTransformerComponents PRIVATE oa GTest::gtest)
target_include_directories(BenchTransformerComponents PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(BenchTransformerComponents PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
set_target_properties(BenchTransformerComponents PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/ml/nn")
if(TARGET oa_shaders)
	add_dependencies(BenchTransformerComponents oa_shaders)
endif()

# The canonical GEMM benchmark is TutorialCoreMatMulIntro. It exercises the
# public router, validates every shape against CPU, reports p50/p95 wall time,
# and includes both throughput and real transformer/fFN shapes. The removed
# BenchCompute sources used retired Matmul* shader names and did not execute the
# current context path, so their timings were not valid.

# Exact raw-GEMM variant a/b through the same public production path, measured
# with GPU timestamps. Deliberately not registered with ctest.
add_executable(BenchGemmVariants
	cpp/runtime/gemm/benchGemmVariants.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
target_link_libraries(BenchGemmVariants PRIVATE oa GTest::gtest)
target_include_directories(BenchGemmVariants PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/cpp
	${oa_SOURCE_DIR}/source/cpp/lib)
target_compile_options(BenchGemmVariants PRIVATE
	-include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
set_target_properties(BenchGemmVariants PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/runtime/gemm")
if(TARGET oa_shaders)
	add_dependencies(BenchGemmVariants oa_shaders)
endif()

# Fused q4/q8 plane consumption versus explicit dequantization followed by the
# production MatMulNt router. Deliberately not registered with CTest.
add_executable(BenchFnMatrixQuant
	cpp/ml/quant/benchFnMatrixQuant.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
target_link_libraries(BenchFnMatrixQuant PRIVATE oa GTest::gtest)
target_include_directories(BenchFnMatrixQuant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(BenchFnMatrixQuant PRIVATE
	-include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
set_target_properties(BenchFnMatrixQuant PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/ml/quant")
if(TARGET oa_shaders)
	add_dependencies(BenchFnMatrixQuant oa_shaders)
endif()

# Current-device empirical cpp/compute/memory ceilings with correctness oracles,
# calibrated clocks, and structured execution counters. Deliberately not
# registered with ctest: canonical evidence uses seven fresh processes through
# tools/diagnostics/oaBenchSuite.py.
add_executable(BenchDeviceRoofline
	cpp/runtime/profiling/benchDeviceRoofline.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/cpp/ml/mlTestMain.cpp)
target_link_libraries(BenchDeviceRoofline PRIVATE oa GTest::gtest)
target_include_directories(BenchDeviceRoofline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_compile_options(BenchDeviceRoofline PRIVATE
	-include ${CMAKE_CURRENT_SOURCE_DIR}/cpp/oaTestVk.h)
set_target_properties(BenchDeviceRoofline PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/test/runtime/profiling")
if(TARGET oa_shaders)
	add_dependencies(BenchDeviceRoofline oa_shaders)
endif()

foreach(TARGET IN ITEMS
	BenchUploadRing
	BenchMamba3
	BenchMoeComponents
	BenchTransformerComponents
	BenchGemmVariants
	BenchFnMatrixQuant
	BenchDeviceRoofline)
	oa_add_private_test_includes(${TARGET})
endforeach()

# ─── Vulkan smoke subsets ──────────────────────────────────────────────────
# The complete executables remain the small real-GPU gate. Filtered profile tests
# below exercise the same ownership -> graph -> recurrent path on hosted Lavapipe
# without admitting optional-extension skips or running the full GPU catalog in a
# CPU JIT. Neither profile substitutes for hardware validation or performance gates.
foreach(_oa_smoke TestEngineInit TestExecutableGraph TestGru)
	if(TEST ${_oa_smoke})
		set_tests_properties(${_oa_smoke} PROPERTIES LABELS "gpu_bindless;ml;gpu_smoke")
	endif()
endforeach()

oa_add_native_profile_test(CpuVulkanEngineSmoke TestEngineInit cpu_vulkan_smoke
	"EngineInit.SuiteEngineFromEnvironment:VkEngineTestFixture.UsesSharedEngine:EngineInit.HeadlessGraphicsDoesNotEnableWsi")
oa_add_native_profile_test(CpuVulkanGraphSmoke TestExecutableGraph cpu_vulkan_smoke
	"ExecutableGraph.DeviceAdmissionCanaryUsesIndependentKnownAnswers:ExecutableGraph.BasicAddAndExecute:ExecutableGraph.CompileAndReplay")
oa_add_native_profile_test(CpuVulkanGruSmoke TestGru cpu_vulkan_smoke
	"Gru.GruCellForwardSingleStep:Gru.GruForwardSequence:Gru.GruCellGradientCheck")

# Native capability-owned profiles. Every selected GoogleTest executable fails
# when an internal case skips. Hardware, external-data, and physical-i/o packs
# are separate from the portable host/GPU contracts.
oa_require_no_skips(native_host
	TestTypes TestMemory TestContainers TestThreading TestCoreMisc TestValidation
	TestVlm
	TestCamera TestTransferWeights TestModelFileIntegrity TestTokenizer TestGptOssConfig
	TestOaStd TestSpirvEmbedGenerator TestKernelGeneration)

get_property(_oa_native_gpu_tests GLOBAL PROPERTY OA_NATIVE_GPU_TESTS)
list(REMOVE_ITEM _oa_native_gpu_tests
	TestEngineInit
	TestClipText
	TestVideoDecoder TestVideoDecodeReference TestVideoDemuxer TestVideoRoundtrip
	TestAudioDecoder
	TestGemmCmSgBf16)
oa_require_no_skips(native_gpu ${_oa_native_gpu_tests})
oa_require_no_skips(native_gpu TestComputeApp TestLunarLander3dRender)
oa_add_native_profile_test(
	NativeGpuEngineInit TestEngineInit native_gpu
	"*-EngineInit.CooperativeMatrixStrictEnv")
oa_add_native_profile_test(
	NativeGpuAudio TestAudioDecoder native_gpu
	"*-TestAudioDecoder.AudioPlayerPlayPauseSeek")

if(TEST TestPresenterWsi)
	oa_require_no_skips(native_wsi TestPresenterWsi)
endif()

find_program(OA_TEST_FFMPEG_EXECUTABLE ffmpeg)
set(_oa_video_dataset "${OA_TEST_DATA_DIR}/video")
if(OA_TEST_FFMPEG_EXECUTABLE
	AND EXISTS "${_oa_video_dataset}/shibuya_crossing_1080p30_h264.mp4"
	AND EXISTS "${_oa_video_dataset}/shibuya_crossing_1080p30_h265.mp4"
	AND EXISTS "${_oa_video_dataset}/shibuya_crossing_1080p30_av1.mp4"
	AND EXISTS "${_oa_video_dataset}/shibuya_crossing_1080p30_vp9.mp4")
	oa_add_native_profile_test(
		NativeIrisXeVideoDecoder TestVideoDecoder native_iris_xe_media "*")
	oa_add_native_profile_test(
		NativeIrisXeVideoDecodeReference TestVideoDecodeReference
		native_iris_xe_media "*")
	oa_add_native_profile_test(
		NativeIrisXeVideoDemuxer TestVideoDemuxer native_iris_xe_media
		"*")
	oa_add_native_profile_test(
		NativeIrisXeVideoRoundtrip TestVideoRoundtrip native_iris_xe_media
		"*-CaptureLifecycle.AbandonedCameraRetiresAtEngineClose:CaptureLifecycle.AbandonedScreenCaptureRetiresAtEngineClose:VideoRoundtrip.AdvertisedRateControlModesEncode:VideoRoundtrip.EncodeMuxDemuxDecodeH265")
	oa_add_native_profile_test(
		NativeIrisXeVideoSessionLifecycle TestVideoSessionLifecycle
		native_iris_xe_media "*")
	set_tests_properties(
		NativeIrisXeVideoDecoder NativeIrisXeVideoDecodeReference
		NativeIrisXeVideoDemuxer NativeIrisXeVideoRoundtrip
		NativeIrisXeVideoSessionLifecycle
		PROPERTIES
			ENVIRONMENT
				"OA_TEST_FORBID_SKIPS=1;OA_REQUIRE_DEVICE_PROFILE=iris-xe-tgl;OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}"
			WORKING_DIRECTORY "${oa_SOURCE_DIR}")
else()
	message(STATUS
		"Native Iris Xe media profile: UNAVAILABLE (ffmpeg or complete Shibuya dataset missing)")
endif()
