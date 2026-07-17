# OA Test Suite
# All unit tests, benchmarks, and kernel correctness tests

# Test helper function for basic tests (uses GTest::gtest_main)
function(oa_add_test TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE})
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest_main)
	target_include_directories(${TARGET} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
		${oa_SOURCE_DIR}/Source/Private)
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
endfunction()

# ML test helper (uses custom MlTestMain.cpp for Vulkan environment setup)
function(oa_add_ml_test TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/Ml/MlTestMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	# Engine-init suites use TEST_VK (alias for gtest TEST_F). Force-include the
	# alias header so gtest-direct and generated .gen.cpp sources see it without
	# per-file include churn. gcc/clang only (OA has no MSVC path).
	target_compile_options(${TARGET} PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/OaTestVk.h)
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	if(SOURCE MATCHES [[Ml/]])
		set_tests_properties(${TARGET} PROPERTIES LABELS "gpu_bindless;ml")
	else()
		set_tests_properties(${TARGET} PROPERTIES LABELS "gpu_bindless")
	endif()
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# Crypto test helper (uses custom CryptoTestMain.cpp for Vulkan environment setup)
function(oa_add_crypto_test TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/Crypto/CryptoTestMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest)
	target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# CORE TESTS

# Core/Misc tests
oa_add_test(TestTypes            Core/Misc/TestTypes.cpp            Core/Misc)
oa_add_test(TestCoreMisc         Core/Misc/TestCoreMisc.cpp         Core/Misc)
oa_add_test(TestValidation       Core/Misc/TestValidation.cpp       Core/Misc)

# Core/Memory tests
oa_add_test(TestMemory           Core/Memory/TestMemory.cpp         Core/Memory)

# Core/Container tests
oa_add_test(TestContainers       Core/Container/TestContainers.cpp  Core/Container)

# Core/Image tests (Phase 1: metadata wrapper, requires Vulkan since OA is GPU-only)
oa_add_ml_test(TestImage         Core/TestImage.cpp                 Core)
oa_add_ml_test(TestUi            Ui/TestUi.cpp                      Ui)

# Core/Audio tests (Phase 6: audio wrapper, requires Vulkan since OA is GPU-only)
oa_add_ml_test(TestAudio         Core/TestAudio.cpp                 Core)

# Core/Matrix operator overloads + shape+fill constructor (needs Vulkan).
oa_add_ml_test(TestMatrixOperators Core/TestMatrixOperators.cpp     Core)

# Core/Threading tests
oa_add_test(TestThreading        Core/Threading/TestThreading.cpp   Core/Threading)

# Network tests
oa_add_test(TestTcpFramed        Network/Tcp/TestTcpFramed.cpp      Network/Tcp)

# ML/Precision tests
oa_add_test(TestPrecisionDtype   Ml/Precision/TestPrecisionDtype.cpp Ml/Precision)
oa_add_test(TestTransferWeights  Ml/Io/TestSafeTensorReader.cpp     Ml/Io)
oa_add_test(TestTokenizer        Ml/Data/TestTokenizer.cpp          Ml/Data)
oa_add_test(TestGptOssConfig     Ml/Io/TestGptOssConfig.cpp         Ml/Io)
oa_add_ml_test(TestAttention     Ml/Nn/TestAttention.cpp            Ml/Nn)
oa_add_ml_test(TestFlow          Ml/Generative/TestFlow.cpp         Ml/Generative)
oa_add_ml_test(TestClipText      Ml/Nn/TestClipText.cpp             Ml/Nn)

set_tests_properties(
	TestTypes TestMemory TestContainers TestThreading TestCoreMisc TestPrecisionDtype TestTokenizer TestGptOssConfig
	PROPERTIES LABELS "core"
)

# ═══════════════════════════════════════════════════════════════════════════
# OASTD TEST SUITE
# ═══════════════════════════════════════════════════════════════════════════

add_executable(TestOaStd
	Core/Std/TestAllocator.cpp
	Core/Std/TestArray.cpp
	Core/Std/TestOptional.cpp
	Core/Std/TestStringView.cpp
	Core/Std/TestString.cpp
	Core/Std/TestSpan.cpp
	Core/Std/TestHashMap.cpp
	Core/Std/TestHashSet.cpp
	Core/Std/TestUniquePtr.cpp
	Core/Std/TestSharedPtr.cpp
	Core/Std/TestVariant.cpp
	Core/Std/TestFunction.cpp
	Core/Std/TestPath.cpp
	Core/Std/TestAlgo.cpp
	Core/Std/TestVec.cpp
	Core/Std/TestRandom.cpp
	Core/Std/TestFormat.cpp
	Core/Std/TestSync.cpp
	Core/Std/TestCString.cpp
	Core/Std/TestPairLimits.cpp
)
target_link_libraries(TestOaStd PRIVATE oa GTest::gtest_main)
target_include_directories(TestOaStd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(TestOaStd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/Core/Std")
add_test(NAME TestOaStd COMMAND TestOaStd)
set_tests_properties(TestOaStd PROPERTIES LABELS "core;oastd")

if(OA_OASTD_EXTRA_TARGETS)
	set(_OA_OASTD_STD_HDRS
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Chrono.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Allocator.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Array.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Optional.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/UniquePtr.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/SharedPtr.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Variant.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Function.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/HashMap.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Span.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/String.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/StringView.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/TypeTraits.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Path.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Filesystem.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Algo.h"
		"${oa_SOURCE_DIR}/Source/Public/Oa/Core/Std/Vec.h"
	)
	add_custom_target(oa_oastd_headers SOURCES ${_OA_OASTD_STD_HDRS})
	add_custom_target(oastd COMMENT "Build OaStd test suite (TestOaStd)" DEPENDS TestOaStd)
endif()

# ═══════════════════════════════════════════════════════════════════════════
# ML TESTS
# ═══════════════════════════════════════════════════════════════════════════

oa_add_ml_test(TestGpuTimer Runtime/Profiling/TestGpuTimer.cpp Runtime/Profiling)
oa_add_ml_test(TestGemmRouter Runtime/Gemm/TestGemmRouter.cpp Runtime/Gemm)
target_include_directories(TestGemmRouter PRIVATE ${oa_SOURCE_DIR}/Source/Private)

# ═══════════════════════════════════════════════════════════════════════════
# CORE MATRIX TESTS (Generated from schemas)
# ═══════════════════════════════════════════════════════════════════════════

# Re-enabled: Fixed autogen test generator (Swiglu/SiluMul/Geglu marked skip_test)
oa_add_ml_test(TestFnMatrixBlas Core/FnMatrix/Blas/TestFnMatrixBlas.cpp Core/FnMatrix/Blas)
oa_add_ml_test(TestFnMatrixBlasSimple Core/FnMatrix/Blas/TestFnMatrixBlasSimple.cpp Core/FnMatrix/Blas)
oa_add_ml_test(TestFnMatrixElemwise Core/FnMatrix/Elemwise/TestFnMatrixElemwise.gen.cpp Core/FnMatrix/Elemwise)
oa_add_ml_test(TestFnMatrixReduce Core/FnMatrix/Reduce/TestFnMatrixReduce.gen.cpp Core/FnMatrix/Reduce)
oa_add_ml_test(TestFnMatrixRng Core/FnMatrix/Rng/TestFnMatrixRng.gen.cpp Core/FnMatrix/Rng)
oa_add_ml_test(TestFnMatrixRngManual Core/FnMatrix/Rng/TestFnMatrixRngManual.cpp Core/FnMatrix/Rng)
oa_add_ml_test(TestFnMatrixShape Core/FnMatrix/Shape/TestFnMatrixShape.gen.cpp Core/FnMatrix/Shape)
oa_add_ml_test(TestFnMatrixShapeManual Core/FnMatrix/Shape/TestFnMatrixShapeManual.cpp Core/FnMatrix/Shape)
oa_add_ml_test(TestFnMatrixIndex Core/FnMatrix/Index/TestFnMatrixIndex.cpp Core/FnMatrix/Index)
oa_add_ml_test(TestFnMatrixUtil Core/FnMatrix/Util/TestFnMatrixUtil.cpp Core/FnMatrix/Util)
oa_add_ml_test(TestTopKBug Core/FnMatrix/Util/TestTopKBug.cpp Core/FnMatrix/Util)

# ═══════════════════════════════════════════════════════════════════════════
# ML MATRIX TESTS (Generated from schemas)
# ═══════════════════════════════════════════════════════════════════════════

# Re-enabled: Fixed autogen test generator (Swiglu/SiluMul/Geglu marked skip_test)
oa_add_ml_test(TestFnMatrixActivation Ml/FnMatrix/Activation/TestFnMatrixActivation.gen.cpp Ml/FnMatrix/Activation)
oa_add_ml_test(TestFnMatrixEmbed Ml/FnMatrix/Embed/TestFnMatrixEmbed.gen.cpp Ml/FnMatrix/Embed)
oa_add_ml_test(TestFnMatrixNorm Ml/FnMatrix/Norm/TestFnMatrixNorm.gen.cpp Ml/FnMatrix/Norm)
oa_add_ml_test(TestFnLoss Ml/Loss/TestFnLoss.cpp Ml/Loss)
oa_add_ml_test(TestRl Ml/Rl/TestRl.cpp Ml/Rl)
target_sources(TestRl PRIVATE
	${oa_SOURCE_DIR}/Tutorial/Ml/Rl/CartPole.cpp
	Ml/Rl/TestCartPole.cpp)
target_include_directories(TestRl PRIVATE ${oa_SOURCE_DIR}/Tutorial/Ml/Rl)
oa_add_ml_test(TestFnMatrixPool Ml/FnMatrix/Pool/TestFnMatrixPool.cpp Ml/FnMatrix/Pool)
oa_add_ml_test(TestOptimizers Ml/Optim/TestOptimizers.cpp Ml/Optim)
oa_add_ml_test(TestFnMatrixBackward Ml/FnMatrix/Backward/TestFnMatrixBackward.cpp Ml/FnMatrix/Backward)
# TestFnMatrixConv (Core) references a phantom OaFnMatrix::Im2Col/Col2Im/Conv2d API that
# does not exist (real Conv2d is in Ml/FnMatrix.h) — orphaned since 6c1c36e, left unregistered.
oa_add_ml_test(TestFnMatrixConvBwd Ml/FnMatrix/Conv/TestFnMatrixConvBwd.cpp Ml/FnMatrix/Conv)
oa_add_ml_test(TestFnMatrixConvTranspose2d Ml/FnMatrix/Conv/TestFnMatrixConvTranspose2d.cpp Ml/FnMatrix/Conv)
oa_add_ml_test(TestMoeGradcheck Ml/Nn/TestMoeGradcheck.cpp Ml/Nn)
oa_add_ml_test(TestMoeSystems Ml/Nn/TestMoeSystems.cpp Ml/Nn)
# Recovered orphans (added in 6c1c36e "100% coverage" but never wired into the build).
oa_add_ml_test(TestGradFnMatrix Ml/Grad/TestFnMatrix.cpp Ml/Grad)
oa_add_ml_test(TestFnMatrixNormAdvanced Ml/FnMatrix/Norm/TestFnMatrixNormAdvanced.cpp Ml/FnMatrix/Norm)
# TestFnMatrixActivationBwd surfaced a real MishBwd op bug (used sigmoid(softplus(x)) instead of
# sigmoid(x)) — fixed in MishBwd.slang; SoftplusBwd failures were test bugs (passed input, not output).
oa_add_ml_test(TestFnMatrixActivationBwd Ml/FnMatrix/Activation/TestFnMatrixActivationBwd.cpp Ml/FnMatrix/Activation)
# Other 6c1c36e orphans NOT re-registered — dead API or wrong-contract tests. See OaSanitization.md §0c:
#   - TestFnMatrixElemwiseBwd / TestFnMatrixShapeBwd / TestFnMatrixAllocManual : dead API (~20 errs each)
#   - TestNnModules / TestByteEmbedding : dead API (3-6 errs)
#   - Core/FnMatrix/Conv/TestFnMatrixConv : phantom Im2Col/Col2Im/Conv2d API
# TestFnMatrixQuant — kept out of ctest on purpose. The Q4_0 kernels (ComputeScaleQ4_0,
# QuantizeQ4_0, DequantizeQ4_0) ARE compiled but produce wrong results on this class of
# hardware: ComputeScaleQ4_0 assumes subgroupSize==32 with 8 hardcoded wave-leaders and no
# SV_GroupID offset (only ever writes scale[0]); roundtrip error is ~full-scale and sign is
# not preserved. 8/9 tests fail. This is the llama.cpp-transplant quant path that is currently
# routed by nothing (dead) and slated for an OA-native redesign — re-enable this test as the
# acceptance gate for that work. See project_open_kernel_bugs (findings #1b, #2).
# oa_add_ml_test(TestFnMatrixQuant Ml/Quant/TestFnMatrixQuant.cpp Ml/Quant)

# ═══════════════════════════════════════════════════════════════════════════
# ML INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

oa_add_ml_test(TestByte Ml/Precision/TestByte.cpp Ml/Precision)
oa_add_ml_test(TestDataLoader Ml/Data/TestDataLoader.cpp Ml/Data)
oa_add_ml_test(TestTrainingMetrics Ml/Training/TestTrainingMetrics.cpp Ml/Training)
oa_add_ml_test(TestAutogradLeak Ml/Training/TestAutogradLeak.cpp Ml/Training)
oa_add_test(TestPoseClip Ml/Nn/TestPoseClip.cpp Ml/Nn)
# Gen3dAnim tests removed — will be redone after sanitization + MotionGpt work.

# 3D anim engine: Core (transform/joint) → Rig (skeleton) → Retarget.
oa_add_test(TestTransform Core/Transform/TestTransform.cpp Core/Transform)
oa_add_test(TestSkeleton Rig/TestSkeleton.cpp Rig)
oa_add_test(TestRetarget Retarget/TestRetarget.cpp Retarget)

# ═══════════════════════════════════════════════════════════════════════════
# KERNEL CORRECTNESS TESTS (GPU with CPU reference validation)
# ═══════════════════════════════════════════════════════════════════════════

# DISABLED: Missing source files (deleted with Test/Core/Matrix cleanup)
# oa_add_ml_test(TestFlatKernels Core/Matrix/TestFlatKernels.cpp Core/Matrix)
# oa_add_ml_test(TestMatrixKernels Core/Matrix/TestMatrixKernels.cpp Core/Matrix)
# oa_add_ml_test(TestQuantization Core/Matrix/TestQuantization.cpp Core/Matrix)
oa_add_ml_test(TestNnKernels Ml/Nn/TestNnKernels.cpp Ml/Nn)
oa_add_ml_test(TestNnLayers Ml/Nn/TestNnLayers.cpp Ml/Nn)
oa_add_ml_test(TestMamba3 Ml/Nn/TestMamba3.cpp Ml/Nn)
oa_add_ml_test(TestVqAssign Ml/Nn/TestVqAssign.cpp Ml/Nn)
oa_add_ml_test(TestGru Ml/Nn/TestGru.cpp Ml/Nn)
oa_add_ml_test(TestGruNumericalGrad Ml/Nn/TestGruNumericalGrad.cpp Ml/Nn)
oa_add_ml_test(TestGruScan Ml/Nn/TestGruScan.cpp Ml/Nn)
oa_add_ml_test(TestLinearGelu Ml/Nn/TestLinearGelu.cpp Ml/Nn)
oa_add_ml_test(TestRnn Ml/Nn/TestRnn.cpp Ml/Nn)
oa_add_ml_test(TestRnnScan Ml/Nn/TestRnnScan.cpp Ml/Nn)
oa_add_ml_test(TestRope Ml/Nn/TestRope.cpp Ml/Nn)
oa_add_ml_test(TestOptimKernels Ml/Optim/TestOptimKernels.cpp Ml/Optim)

# TestGemmKernels and TestRngKernels have their own main() — use oa_add_test
# DISABLED: Missing source file (deleted with Test/Core/Matrix cleanup)
# oa_add_test(TestGemmKernels Core/Matrix/TestGemmKernels.cpp Core/Matrix)
# if(TARGET oa_shaders)
# 	add_dependencies(TestGemmKernels oa_shaders)
# endif()

# TestRngKernels disabled — uses legacy FillUniform/FillNormal API that was removed
# TODO: Rewrite to use OaMatrix-level RNG operations (Rand, RandN, etc.)
# oa_add_test(TestRngKernels Core/Matrix/TestRngKernels.cpp Core/Matrix)
# if(TARGET oa_shaders)
# 	add_dependencies(TestRngKernels oa_shaders)
# endif()

# DISABLED: Removed missing tests (TestFlatKernels, TestMatrixKernels, TestQuantization, TestGemmKernels)
set_tests_properties(
	TestNnKernels TestOptimKernels TestVqAssign
	PROPERTIES LABELS "gpu_bindless;kernel_correctness"
)

# ═══════════════════════════════════════════════════════════════════════════
# VISION TESTS
# ═══════════════════════════════════════════════════════════════════════════

# Phase 1 — Image Processing (COMPLETE)
oa_add_ml_test(TestJpegDecoder Vision/Codec/TestJpegDecoder.cpp Vision/Codec)
oa_add_ml_test(TestImageProcessor Vision/Processor/TestImageProcessor.cpp Vision/Processor)
# Phase 2 — Video Decode (IN PROGRESS)
oa_add_ml_test(TestVideoDecoder Vision/Codec/TestVideoDecoder.cpp Vision/Codec)
# Phase 2a — End-to-end .mp4 decode validated vs ffmpeg reference frames
oa_add_ml_test(TestVideoDecodeReference Vision/Codec/TestVideoDecodeReference.cpp Vision/Codec)
# Phase 2b — MP4 demuxer + OaFnVideo NAL ops
oa_add_ml_test(TestVideoStream  Vision/Stream/TestVideoStream.cpp Vision/Stream)
# Phase 2c — encode → mux → demux (→ decode) roundtrip
oa_add_ml_test(TestVideoRoundtrip Vision/Stream/TestVideoRoundtrip.cpp Vision/Stream)
# Phase 3 — Unified OaFnImage API (GPU-only)
oa_add_ml_test(TestFnImage Vision/Image/TestFnImage.cpp Vision/Image)
# Phase 4 — Kernel-level tests with CPU reference verification
oa_add_ml_test(TestGeometricKernels Vision/Image/Kernel/TestGeometricKernels.cpp Vision/Image/Kernel)
oa_add_ml_test(TestFilterKernels Vision/Image/Kernel/TestFilterKernels.cpp Vision/Image/Kernel)
oa_add_ml_test(TestVision50Ops Vision/Image/Kernel/TestVision50Ops.cpp Vision/Image/Kernel)
oa_add_ml_test(TestFnDetection Vision/Detection/TestFnDetection.cpp Vision/Detection)

set_tests_properties(TestJpegDecoder TestImageProcessor TestVideoDecoder TestVideoDecodeReference TestFnImage TestGeometricKernels TestFilterKernels TestVision50Ops TestFnDetection
	PROPERTIES LABELS "gpu_bindless;ml;vision")

# ═══════════════════════════════════════════════════════════════════════════
# AUDIO TESTS
# ═══════════════════════════════════════════════════════════════════════════

# Decoder/encoder round-trips + OaFnAudio ops vs in-test CPU oracles.
oa_add_ml_test(TestAudioDecoder Audio/Codec/TestAudioDecoder.cpp Audio/Codec)
oa_add_ml_test(TestFnAudio      Audio/TestFnAudio.cpp            Audio)
set_tests_properties(TestAudioDecoder TestFnAudio PROPERTIES LABELS "gpu_bindless;audio")

# ═══════════════════════════════════════════════════════════════════════════
# RUNTIME TESTS
# ═══════════════════════════════════════════════════════════════════════════

oa_add_test(TestAllocator Runtime/Memory/TestAllocator.cpp Runtime/Memory)
target_include_directories(TestAllocator PRIVATE ${oa_SOURCE_DIR}/Source/Private)

oa_add_ml_test(TestKernelRegistry Runtime/Engine/TestKernelRegistry.cpp Runtime/Engine)
oa_add_ml_test(TestEngineInit Runtime/Engine/TestEngineInit.cpp Runtime/Engine)
oa_add_ml_test(TestStream Runtime/Stream/TestStream.cpp Runtime/Stream)
oa_add_ml_test(TestComputeGraph Runtime/Graph/TestComputeGraph.cpp Runtime/Graph)
# oa_add_ml_test(TestCollective Runtime/Collective/TestCollective.cpp Runtime/Collective)

# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO TESTS
# ═══════════════════════════════════════════════════════════════════════════

# CPU Keccak/SHAKE — always available, no external deps
oa_add_test(TestKeccak Crypto/Hash/TestKeccak.cpp Crypto/Hash)
oa_add_test(TestHash Crypto/Hash/TestHash.cpp Crypto/Hash)
set_tests_properties(TestKeccak TestHash PROPERTIES LABELS "crypto;cpu")

if(OA_BUILD_CRYPTO)
	# OaFnHash tests — unified crypto API (requires Vulkan for TEST_F fixture)
	oa_add_crypto_test(TestFnHash Crypto/Hash/TestFnHash.cpp Crypto/Hash)
	set_tests_properties(TestFnHash PROPERTIES LABELS "crypto;gpu")

	# OaFnHash extended tests — graph/executor modes (requires Vulkan)
	oa_add_crypto_test(TestFnHashExtended Crypto/Hash/TestFnHashExtended.cpp Crypto/Hash)
	set_tests_properties(TestFnHashExtended PROPERTIES LABELS "crypto;gpu")

	# Crypto benchmarks — CPU vs GPU performance validation
	oa_add_crypto_test(BenchHash Crypto/Hash/BenchHash.cpp Crypto/Hash)
	set_tests_properties(BenchHash PROPERTIES LABELS "crypto;benchmark;gpu")

	# ML-DSA-65 sign/verify — CPU (liboqs), no Vulkan needed
	oa_add_test(TestSign Crypto/Sign/TestSign.cpp Crypto/Sign)
	set_tests_properties(TestSign PROPERTIES LABELS "crypto")
endif()

# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════

# Host memcpy/libc/non-temporal comparison across size and alignment regimes.
add_executable(BenchMemory Core/Memory/BenchMemory.cpp)
target_link_libraries(BenchMemory PRIVATE oa)
target_include_directories(BenchMemory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(BenchMemory PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/Core/Memory")

# Persistent mapped upload ring vs allocate/submit/wait/free per copy.
# Deliberately not registered with ctest: this is a controlled benchmark.
add_executable(BenchUploadRing Runtime/Stream/BenchUploadRing.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Ml/MlTestMain.cpp)
target_link_libraries(BenchUploadRing PRIVATE oa GTest::gtest)
target_include_directories(BenchUploadRing PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(BenchUploadRing PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/OaTestVk.h)
set_target_properties(BenchUploadRing PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/Runtime/Stream")

# Tuned KHR CoopMat BF16 GEMM (GemmCmSgBf16) — correctness + perf.
oa_add_ml_test(TestGemmCmSgBf16 Compute/TestGemmCmSgBf16.cpp Compute)
set_tests_properties(TestGemmCmSgBf16 PROPERTIES LABELS "gpu_bindless;kernel_correctness")

# Mamba-3 isolated GPU-timestamped kernel/preprocess/prefill/step breakdown.
# Deliberately not registered with ctest: this is a controlled benchmark, not a unit test.
add_executable(BenchMamba3 Ml/Nn/BenchMamba3.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Ml/MlTestMain.cpp)
target_link_libraries(BenchMamba3 PRIVATE oa GTest::gtest)
target_include_directories(BenchMamba3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(BenchMamba3 PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/OaTestVk.h)
set_target_properties(BenchMamba3 PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/Ml/Nn")
if(TARGET oa_shaders)
	add_dependencies(BenchMamba3 oa_shaders)
endif()

add_executable(BenchMoeComponents Ml/Nn/BenchMoeComponents.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Ml/MlTestMain.cpp)
target_link_libraries(BenchMoeComponents PRIVATE oa GTest::gtest)
target_include_directories(BenchMoeComponents PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(BenchMoeComponents PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/OaTestVk.h)
set_target_properties(BenchMoeComponents PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/Ml/Nn")
if(TARGET oa_shaders)
	add_dependencies(BenchMoeComponents oa_shaders)
endif()

# Dense-Transformer forward/backward component breakdown for the canonical NLP shape.
# Deliberately not registered with ctest: this is a controlled benchmark.
add_executable(BenchTransformerComponents Ml/Nn/BenchTransformerComponents.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Ml/MlTestMain.cpp)
target_link_libraries(BenchTransformerComponents PRIVATE oa GTest::gtest)
target_include_directories(BenchTransformerComponents PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(BenchTransformerComponents PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/OaTestVk.h)
set_target_properties(BenchTransformerComponents PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/Ml/Nn")
if(TARGET oa_shaders)
	add_dependencies(BenchTransformerComponents oa_shaders)
endif()

# The canonical GEMM benchmark is TutorialCoreMatMulIntro. It exercises the
# public router, validates every shape against CPU, reports p50/p95 wall time,
# and includes both throughput and real Transformer/FFN shapes. The removed
# BenchCompute sources used retired Matmul* shader names and did not execute the
# current context path, so their timings were not valid.

# Exact raw-GEMM variant A/B through the same public production path, measured
# with GPU timestamps. Deliberately not registered with ctest.
add_executable(BenchGemmVariants
	Runtime/Gemm/BenchGemmVariants.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Ml/MlTestMain.cpp)
target_link_libraries(BenchGemmVariants PRIVATE oa GTest::gtest)
target_include_directories(BenchGemmVariants PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${oa_SOURCE_DIR}/Source/Private)
target_compile_options(BenchGemmVariants PRIVATE
	-include ${CMAKE_CURRENT_SOURCE_DIR}/OaTestVk.h)
set_target_properties(BenchGemmVariants PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Test/Runtime/Gemm")
if(TARGET oa_shaders)
	add_dependencies(BenchGemmVariants oa_shaders)
endif()

# ─── GPU smoke subset ───────────────────────────────────────────────────────
# A fast, fp32-safe path (engine init -> compute-graph dispatch -> recurrent GRU
# kernel) for a self-hosted GPU runner in CI: `ctest -L gpu_smoke`. Runs on any
# Vulkan GPU including integrated (needs no coopmat/bf16), so it doubles as a quick
# local sanity check. Extends (does not replace) the gpu_bindless label so these
# still stay out of the lavapipe CI run.
foreach(_oa_smoke TestEngineInit TestComputeGraph TestGru)
	if(TEST ${_oa_smoke})
		set_tests_properties(${_oa_smoke} PROPERTIES LABELS "gpu_bindless;ml;gpu_smoke")
	endif()
endforeach()
