# OA Tutorial Suite
# End-to-end tutorial executables with presentation-quality output

# Tutorial helper for ML tutorials (uses MlTestMain.cpp for Vulkan environment)
# Use ${oa_SOURCE_DIR} (not ${CMAKE_SOURCE_DIR}) so paths resolve correctly when
# oa is added as a subdirectory from a parent project (e.g. ml/).
function(oa_add_tutorial TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${oa_SOURCE_DIR}/test/cpp/ml/mlTestMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oaSdkSupport GTest::gtest)
	target_include_directories(${TARGET} PRIVATE
		${oa_SOURCE_DIR}/test/cpp
		${oa_SOURCE_DIR}/source/cpp/lib
		${oa_SOURCE_DIR}/source/cpp/thirdparty)
	set_target_properties(${TARGET} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/tutorial/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	set_tests_properties(${TARGET} PROPERTIES
		LABELS "gpu_bindless;tutorial"
		ENVIRONMENT
			"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}")
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# App tutorial helper for windowed applications (standalone, no GTest)
function(oa_add_app_tutorial TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE})
	target_link_libraries(${TARGET} PRIVATE oaSdkSupport)
	if(TARGET SDL3::SDL3)
		target_link_libraries(${TARGET} PRIVATE SDL3::SDL3)
	endif()
	set_target_properties(${TARGET} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/tutorial/${SUBDIR}")
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# NLP tutorials share their corpus/model scaffolding and live under one source
# and binary namespace. Target names remain stable for CTest and packaging.
function(oa_add_nlp_tutorial TARGET SOURCE)
	oa_add_tutorial(${TARGET} "ml/nlp/${SOURCE}" "ml/nlp")
	target_include_directories(${TARGET} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/ml
		${CMAKE_CURRENT_SOURCE_DIR}/ml/nlp)
endfunction()

# ═══════════════════════════════════════════════════════════════════════════
# ML TUTORIALS
# ═══════════════════════════════════════════════════════════════════════════

# End-to-end ML tutorials: presentation-quality output, all 3 API levels, inference predictions
# Parallel to TF Keras tutorial examples. Run with: ctest -L tutorial
#
# API Level 1: oa::Module + oa::Linear + oa::Autograd (highest level, most automatic)
# API Level 2: oa::FnMatrix + oa::Engine (explicit operations and completion)

oa_add_tutorial(TutorialMnistClassifierApi1 ml/tutorialMnistClassifierApi1.cpp ml)
# REMOVED: oa_add_tutorial(TutorialMnistClassifierApi2 ml/tutorialMnistClassifierApi2.cpp ml)
oa_add_tutorial(TutorialMnistClassifierAg ml/tutorialMnistClassifierAg.cpp ml)
# NLP fair-comparison suite — all all-position next-token LMs on one shared corpus
# (TutorialNlpCommon.h). Recurrent + attention + SSM, byte-256 and tokenizer vocab.
# The old flatten-MLP "feed-forward RNN" tutorials (rnn/rnnAg/byteRnn/byteRnnAg),
# the ByteGru autograd duplicate, and the unbuilt EmpyrealmApi1 fused twin were
# removed 2026-06-26 — single-prediction flatten windows are not the LM use case.
# Byte-vocab (256) family — TutorialNlpByte*
oa_add_nlp_tutorial(TutorialNlpByteRnnAg tutorialNlpByteRnnAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteGruAg tutorialNlpByteGruAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteTransformerAg tutorialNlpByteTransformerAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteMoeAg tutorialNlpByteMoeAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteMamba3Ag tutorialNlpByteMamba3Ag.cpp)
oa_add_nlp_tutorial(TutorialNlpByteEmpyrealmAg tutorialNlpByteEmpyrealmAg.cpp)
# Byte-pair tokenizer family — same model shapes, vocab=320 and exact byte/s.
oa_add_nlp_tutorial(TutorialNlpBpeRnnAg tutorialNlpBpeRnnAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeGruAg tutorialNlpBpeGruAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeTransformerAg tutorialNlpBpeTransformerAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeMoeAg tutorialNlpBpeMoeAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeMamba3Ag tutorialNlpBpeMamba3Ag.cpp)
# Character-tokenizer family — vocab=27 (a-z + space).
oa_add_nlp_tutorial(TutorialNlpCharRnnAg tutorialNlpCharRnnAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharGruAg tutorialNlpCharGruAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharTransformerAg tutorialNlpCharTransformerAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharMoeAg tutorialNlpCharMoeAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharMamba3Ag tutorialNlpCharMamba3Ag.cpp)
# TutorialGen3dAnim (AR next-frame regressor) REMOVED 2026-06-20: mean-collapsed.
# TutorialNoMask (Action2Motion RNN generator) REMOVED 2026-06-23: superseded.
# TutorialMld (Motion Latent Diffusion port) REMOVED 2026-06-23: abandoned. The
# motion generator is now the oa::Alm VQ + token-prior pipeline.
# TutorialMotionGen (VQ + masked-token motion generation) REMOVED 2026-07-09:
# the motion-generation stack is devkit support consumed by
# sdk/cpp/applications/gen3dAnim, not an installed OA framework contract.
# vQ/rVQ modules remain unit-tested in TestVqAssign.
# Legacy single-model flow proof. The comparative transformer/moE tutorials below are
# the current public replacement and retain stable target names for release automation.
oa_add_tutorial(TutorialMambaDiffusionPixel ml/tutorialMambaDiffusionPixel.cpp ml)
oa_add_tutorial(TutorialGenFashionFlowTransformerAg
	ml/generative/tutorialGenFashionFlowTransformerAg.cpp ml/generative)
oa_add_tutorial(TutorialGenFashionFlowMoeAg
	ml/generative/tutorialGenFashionFlowMoeAg.cpp ml/generative)
target_include_directories(TutorialGenFashionFlowTransformerAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml ${CMAKE_CURRENT_SOURCE_DIR}/ml/generative)
target_include_directories(TutorialGenFashionFlowMoeAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml ${CMAKE_CURRENT_SOURCE_DIR}/ml/generative)
oa_add_tutorial(TutorialGenMotionFlowTransformerAg
	ml/generative/tutorialGenMotionFlowTransformerAg.cpp ml/generative)
oa_add_tutorial(TutorialGenMotionFlowMoeAg
	ml/generative/tutorialGenMotionFlowMoeAg.cpp ml/generative)
target_include_directories(TutorialGenMotionFlowTransformerAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml ${CMAKE_CURRENT_SOURCE_DIR}/ml/generative)
target_include_directories(TutorialGenMotionFlowMoeAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml ${CMAKE_CURRENT_SOURCE_DIR}/ml/generative)
oa_add_tutorial(TutorialRlCartPoleRollout ml/rl/tutorialRlCartPoleRollout.cpp ml/rl)
target_include_directories(TutorialRlCartPoleRollout PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml/rl)
oa_add_tutorial(TutorialRlCartPolePpo ml/rl/tutorialRlCartPolePpo.cpp ml/rl)
target_sources(TutorialRlCartPolePpo PRIVATE
	ml/rl/cartPolePpo.cpp)
target_include_directories(TutorialRlCartPolePpo PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml/rl)
oa_add_app_tutorial(TutorialRlCartPoleViewer ml/rl/tutorialRlCartPoleViewer.cpp ml/rl)
target_sources(TutorialRlCartPoleViewer PRIVATE
	ml/rl/cartPolePpo.cpp)
target_include_directories(TutorialRlCartPoleViewer PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml/rl)

# Experimental deterministic scalar + headless Vulkan render checkpoint. Keep
# this standalone: it needs neither the GTest ML harness nor SDL presentation.
add_executable(TutorialRlLunarLander3dHeadless
	ml/rl/lunarLander3d/tutorialRlLunarLander3dHeadless.cpp
	ml/rl/lunarLander3d/lunarLander3dRender.cpp)
target_link_libraries(TutorialRlLunarLander3dHeadless PRIVATE oaSdkSupport)
target_include_directories(TutorialRlLunarLander3dHeadless PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml/rl/lunarLander3d)
set_target_properties(TutorialRlLunarLander3dHeadless PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/tutorial/ml/rl")
if(TARGET oa_shaders)
	add_dependencies(TutorialRlLunarLander3dHeadless oa_shaders)
endif()

set(_oa_lunar_headless_artifacts
	"${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>/TutorialRlLunarLander3dHeadless")
add_test(
	NAME TutorialRlLunarLander3dHeadless
	COMMAND TutorialRlLunarLander3dHeadless
		"${_oa_lunar_headless_artifacts}")
set_tests_properties(TutorialRlLunarLander3dHeadless PROPERTIES
	LABELS "gpu_bindless;tutorial;ml;rl;render;headless_readback"
	ENVIRONMENT
		"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}"
	SKIP_RETURN_CODE 125
	TIMEOUT 120
	WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Interactive scalar + Vulkan render path through the one oa::Viewer lifecycle.
# The live source passes the renderer's sampled image directly to the Viewer;
# there is no host readback/upload bridge or tutorial-owned presenter.
oa_add_app_tutorial(TutorialRlLunarLander3dViewer
	ml/rl/lunarLander3d/tutorialRlLunarLander3dViewer.cpp ml/rl)
target_sources(TutorialRlLunarLander3dViewer PRIVATE
	ml/rl/lunarLander3d/lunarLander3dRender.cpp
	ml/rl/lunarLander3d/lunarLander3dViewerSource.cpp)
target_include_directories(TutorialRlLunarLander3dViewer PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/ml/rl/lunarLander3d)
# REMOVED: oa_add_nlp_tutorial(TutorialNlpMamba3TinyShakespeare tutorialNlpMamba3TinyShakespeare.cpp)
# MOVED to sdk/cpp/examples/ml/motionGpt.cpp (converted to TEST format with training stubs)

# ═══════════════════════════════════════════════════════════════════════════
# CORE COMPUTE TUTORIALS
# ═══════════════════════════════════════════════════════════════════════════

# Raw compute tutorials that validate the public oa::FnMatrix -> oa::Engine
# capture/submit path without involving oa::Module, optimizers, or autograd.

oa_add_app_tutorial(TutorialCoreMatMulIntro core/tutorialCoreMatMulIntro.cpp core)
oa_add_app_tutorial(TutorialCoreEngine core/tutorialCoreEngine.cpp core)
oa_add_app_tutorial(TutorialCoreDeviceCanary core/tutorialCoreDeviceCanary.cpp core)

# Linux-only defensive-security checkpoint. The executable uses an in-memory
# fixture and has no filesystem or network input.
oa_add_app_tutorial(TutorialCryptoCyberRangeIntegrity
	crypto/tutorialCryptoCyberRangeIntegrity.cpp crypto)
add_test(
	NAME TutorialCryptoCyberRangeIntegrity
	COMMAND TutorialCryptoCyberRangeIntegrity)
set_tests_properties(TutorialCryptoCyberRangeIntegrity PROPERTIES
	LABELS "gpu_bindless;tutorial;crypto;security"
	ENVIRONMENT
		"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}"
	SKIP_RETURN_CODE 125
	TIMEOUT 60)

# ═══════════════════════════════════════════════════════════════════════════
# INSTALL — headless, self-contained tutorials as `oa-*` CLI tools
# ═══════════════════════════════════════════════════════════════════════════
# These need no external data (the NLP trainers use the embedded corpus in
# TutorialNlpCommon.h) and no display, so they run standalone like `vulkaninfo`.
# Installed under the `oa_tools` component (bundled into the SDK package).
include(GNUInstallDirs)
set(_oa_cli_tools
	"TutorialNlpCharRnnAg=oa-nlp-char-rnn"
	"TutorialNlpCharGruAg=oa-nlp-char-gru"
	"TutorialNlpCharTransformerAg=oa-nlp-char-transformer"
	"TutorialNlpCharMoeAg=oa-nlp-char-moe"
	"TutorialNlpCharMamba3Ag=oa-nlp-char-mamba3"
	"TutorialNlpByteRnnAg=oa-nlp-byte-rnn"
	"TutorialNlpByteGruAg=oa-nlp-byte-gru"
	"TutorialNlpByteTransformerAg=oa-nlp-byte-transformer"
	"TutorialNlpByteMoeAg=oa-nlp-byte-moe"
	"TutorialNlpByteMamba3Ag=oa-nlp-byte-mamba3"
	"TutorialNlpByteEmpyrealmAg=oa-nlp-byte-empyrealm"
	"TutorialNlpBpeRnnAg=oa-nlp-bpe-rnn"
	"TutorialNlpBpeGruAg=oa-nlp-bpe-gru"
	"TutorialNlpBpeTransformerAg=oa-nlp-bpe-transformer"
	"TutorialNlpBpeMoeAg=oa-nlp-bpe-moe"
	"TutorialNlpBpeMamba3Ag=oa-nlp-bpe-mamba3"
	"TutorialCoreMatMulIntro=oa-matmul-intro"
	"TutorialCoreDeviceCanary=oa-device-canary"
)
foreach(_pair IN LISTS _oa_cli_tools)
	string(REPLACE "=" ";" _kv "${_pair}")
	list(GET _kv 0 _oa_tool_tgt)
	list(GET _kv 1 _oa_tool_name)
	install(PROGRAMS "$<TARGET_FILE:${_oa_tool_tgt}>"
		DESTINATION ${CMAKE_INSTALL_BINDIR}
		COMPONENT oa_tools
		RENAME ${_oa_tool_name})
endforeach()

# ═══════════════════════════════════════════════════════════════════════════
# VISION TUTORIALS (Windowed Applications)
# ═══════════════════════════════════════════════════════════════════════════

# Vision / windowed tutorials — standalone executables (not GTest), require display + SDL3

oa_add_app_tutorial(TutorialViewerVideoH264High8bit420
	vision/tutorialViewerVideoH264High8bit420.cpp vision)
oa_add_app_tutorial(TutorialViewerVideoH265Main8bit420
	vision/tutorialViewerVideoH265Main8bit420.cpp vision)
oa_add_app_tutorial(TutorialViewerVideoAv1Main8bit420
	vision/tutorialViewerVideoAv1Main8bit420.cpp vision)
oa_add_app_tutorial(TutorialViewerVideoVp9Profile0_8bit420
	vision/tutorialViewerVideoVp9Profile0_8bit420.cpp vision)
oa_add_app_tutorial(TutorialViewerAudio vision/tutorialViewerAudio.cpp vision)
# TutorialVideoDecodeDiagnostic (manual PNG-dump) REMOVED 2026-07-09: decode is
# covered headlessly by test/vision/codec/testVideoDecoder + TestVideoDecodeReference.
# TutorialVideoDemuxerCompare REMOVED 2026-07-09: dead (minimp4 dep disabled);
# demuxer covered by test/vision/stream/testVideoDemuxer.
oa_add_app_tutorial(TutorialViewerImage vision/tutorialViewerImage.cpp vision)
oa_add_app_tutorial(TutorialViewerImageHeadless vision/tutorialViewerImageHeadless.cpp vision)
oa_add_app_tutorial(TutorialRendererUi ui/tutorialRendererUi.cpp ui)
oa_add_app_tutorial(TutorialDetectionVideo vision/tutorialDetectionVideo.cpp vision)
oa_add_app_tutorial(TutorialDetectionImage vision/tutorialDetectionImage.cpp vision)
oa_add_app_tutorial(TutorialCvEvaluation vision/tutorialCvEvaluation.cpp vision)
oa_add_app_tutorial(TutorialVisionDataAugmentation
	vision/tutorialVisionDataAugmentation.cpp vision)
set(_oa_vision_augmentation_artifact
	"${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>/TutorialVisionDataAugmentation.png")
add_test(
	NAME TutorialVisionDataAugmentation
	COMMAND TutorialVisionDataAugmentation
		"${oa_SOURCE_DIR}/sdk/asset/image/visionTestPattern320x180.jpg"
		"${_oa_vision_augmentation_artifact}")
set_tests_properties(TutorialVisionDataAugmentation PROPERTIES
	LABELS "gpu_bindless;tutorial;vision;plot;ui;headless_readback"
	ENVIRONMENT
		"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}"
	TIMEOUT 120
	WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
oa_add_app_tutorial(TutorialScreenRecordH264 vision/tutorialScreenRecordH264.cpp vision)
oa_add_app_tutorial(TutorialCameraRecordH264 vision/tutorialCameraRecordH264.cpp vision)
# TutorialEncodeBasic REMOVED 2026-07-09: encode path covered by
# test/vision/stream/testVideoRoundtrip (encode→mux→demux→readback).

# ═══════════════════════════════════════════════════════════════════════════
# PLOT TUTORIALS (oa::plot module)
# ═══════════════════════════════════════════════════════════════════════════

oa_add_app_tutorial(TutorialImageGridClassify plot/tutorialImageGridClassify.cpp plot)
oa_add_app_tutorial(TutorialPlotGallery plot/tutorialPlotGallery.cpp plot)
set(_oa_plot_gallery_artifacts
	"${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>/TutorialPlotGallery")
add_test(
	NAME TutorialPlotGalleryHeadless
	COMMAND TutorialPlotGallery
		--headless
		--output "${_oa_plot_gallery_artifacts}")
set_tests_properties(TutorialPlotGalleryHeadless PROPERTIES
	LABELS "gpu_bindless;tutorial;plot;ui;headless_readback"
	ENVIRONMENT
		"OA_ASSET_DIR=${oa_SOURCE_DIR}/sdk/asset;OA_DATA_DIR=${OA_TEST_DATA_DIR}"
	TIMEOUT 120
	WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
