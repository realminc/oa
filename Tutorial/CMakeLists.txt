# OA Tutorial Suite
# End-to-end tutorial executables with presentation-quality output

# Tutorial helper for ML tutorials (uses MlTestMain.cpp for Vulkan environment)
# Use ${oa_SOURCE_DIR} (not ${CMAKE_SOURCE_DIR}) so paths resolve correctly when
# oa is added as a subdirectory from a parent project (e.g. ml/).
function(oa_add_tutorial TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE} ${oa_SOURCE_DIR}/Test/Ml/MlTestMain.cpp)
	target_link_libraries(${TARGET} PRIVATE oa GTest::gtest)
	target_include_directories(${TARGET} PRIVATE ${oa_SOURCE_DIR}/Test)
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Tutorial/${SUBDIR}")
	add_test(NAME ${TARGET} COMMAND ${TARGET})
	set_tests_properties(${TARGET} PROPERTIES LABELS "gpu_bindless;tutorial")
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# App tutorial helper for windowed applications (standalone, no GTest)
function(oa_add_app_tutorial TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE})
	target_link_libraries(${TARGET} PRIVATE oa)
	if(TARGET SDL3::SDL3)
		target_link_libraries(${TARGET} PRIVATE SDL3::SDL3)
	endif()
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/Tutorial/${SUBDIR}")
	if(TARGET oa_shaders)
		add_dependencies(${TARGET} oa_shaders)
	endif()
endfunction()

# NLP tutorials share their corpus/model scaffolding and live under one source
# and binary namespace. Target names remain stable for CTest and packaging.
function(oa_add_nlp_tutorial TARGET SOURCE)
	oa_add_tutorial(${TARGET} "Ml/Nlp/${SOURCE}" "Ml/Nlp")
	target_include_directories(${TARGET} PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/Ml
		${CMAKE_CURRENT_SOURCE_DIR}/Ml/Nlp)
endfunction()

# ═══════════════════════════════════════════════════════════════════════════
# ML TUTORIALS
# ═══════════════════════════════════════════════════════════════════════════

# End-to-end ML tutorials: presentation-quality output, all 3 API levels, inference predictions
# Parallel to TF Keras tutorial examples. Run with: ctest -L tutorial
#
# API Level 1: OaModule + OaLinear + OaAutograd (highest level, most automatic)
# API Level 2: OaFnMatrix + OaContext (mid-level, explicit ops, auto-caching)

oa_add_tutorial(TutorialMnistClassifierApi1 Ml/TutorialMnistClassifierApi1.cpp Ml)
# REMOVED: oa_add_tutorial(TutorialMnistClassifierApi2 Ml/TutorialMnistClassifierApi2.cpp Ml)
oa_add_tutorial(TutorialMnistClassifierAg Ml/TutorialMnistClassifierAg.cpp Ml)
# NLP fair-comparison suite — all all-position next-token LMs on one shared corpus
# (TutorialNlpCommon.h). Recurrent + attention + SSM, byte-256 and tokenizer vocab.
# The old flatten-MLP "feed-forward RNN" tutorials (Rnn/RnnAg/ByteRnn/ByteRnnAg),
# the ByteGru autograd duplicate, and the unbuilt EmpyrealmApi1 fused twin were
# removed 2026-06-26 — single-prediction flatten windows are not the LM use case.
# Byte-vocab (256) family — TutorialNlpByte*
oa_add_nlp_tutorial(TutorialNlpByteRnnAg TutorialNlpByteRnnAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteGruAg TutorialNlpByteGruAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteTransformerAg TutorialNlpByteTransformerAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteMoeAg TutorialNlpByteMoeAg.cpp)
oa_add_nlp_tutorial(TutorialNlpByteMamba3Ag TutorialNlpByteMamba3Ag.cpp)
oa_add_nlp_tutorial(TutorialNlpByteEmpyrealmAg TutorialNlpByteEmpyrealmAg.cpp)
# Byte-pair tokenizer family — same model shapes, vocab=320 and exact byte/s.
oa_add_nlp_tutorial(TutorialNlpBpeRnnAg TutorialNlpBpeRnnAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeGruAg TutorialNlpBpeGruAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeTransformerAg TutorialNlpBpeTransformerAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeMoeAg TutorialNlpBpeMoeAg.cpp)
oa_add_nlp_tutorial(TutorialNlpBpeMamba3Ag TutorialNlpBpeMamba3Ag.cpp)
# Character-tokenizer family — vocab=27 (a-z + space).
oa_add_nlp_tutorial(TutorialNlpCharRnnAg TutorialNlpCharRnnAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharGruAg TutorialNlpCharGruAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharTransformerAg TutorialNlpCharTransformerAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharMoeAg TutorialNlpCharMoeAg.cpp)
oa_add_nlp_tutorial(TutorialNlpCharMamba3Ag TutorialNlpCharMamba3Ag.cpp)
# TutorialGen3dAnim (AR next-frame regressor) REMOVED 2026-06-20: mean-collapsed.
# TutorialNoMask (Action2Motion RNN generator) REMOVED 2026-06-23: superseded.
# TutorialMld (Motion Latent Diffusion port) REMOVED 2026-06-23: abandoned. The
# motion generator is now the OaAlm VQ + token-prior pipeline.
# TutorialMotionGen (VQ + masked-token motion generation) REMOVED 2026-07-09:
# the motion-generation stack ships as OaAlm under Extensions/…/Alm and
# Apps/Ml/Gen3dAnim, not as an in-tree tutorial.
# VQ/RVQ modules remain unit-tested in TestVqAssign.
# Legacy single-model flow proof. The comparative Transformer/MoE tutorials below are
# the current public replacement and retain stable target names for release automation.
oa_add_tutorial(TutorialMambaDiffusionPixel Ml/TutorialMambaDiffusionPixel.cpp Ml)
oa_add_tutorial(TutorialGenFashionFlowTransformerAg
	Ml/Generative/TutorialGenFashionFlowTransformerAg.cpp Ml/Generative)
oa_add_tutorial(TutorialGenFashionFlowMoeAg
	Ml/Generative/TutorialGenFashionFlowMoeAg.cpp Ml/Generative)
target_include_directories(TutorialGenFashionFlowTransformerAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/Ml ${CMAKE_CURRENT_SOURCE_DIR}/Ml/Generative)
target_include_directories(TutorialGenFashionFlowMoeAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/Ml ${CMAKE_CURRENT_SOURCE_DIR}/Ml/Generative)
oa_add_tutorial(TutorialGenMotionFlowTransformerAg
	Ml/Generative/TutorialGenMotionFlowTransformerAg.cpp Ml/Generative)
oa_add_tutorial(TutorialGenMotionFlowMoeAg
	Ml/Generative/TutorialGenMotionFlowMoeAg.cpp Ml/Generative)
target_include_directories(TutorialGenMotionFlowTransformerAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/Ml ${CMAKE_CURRENT_SOURCE_DIR}/Ml/Generative)
target_include_directories(TutorialGenMotionFlowMoeAg PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/Ml ${CMAKE_CURRENT_SOURCE_DIR}/Ml/Generative)
oa_add_tutorial(TutorialRlCartPoleRollout Ml/Rl/TutorialRlCartPoleRollout.cpp Ml/Rl)
target_sources(TutorialRlCartPoleRollout PRIVATE Ml/Rl/CartPole.cpp)
target_include_directories(TutorialRlCartPoleRollout PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/Ml/Rl)
oa_add_tutorial(TutorialRlCartPolePpo Ml/Rl/TutorialRlCartPolePpo.cpp Ml/Rl)
target_sources(TutorialRlCartPolePpo PRIVATE
	Ml/Rl/CartPole.cpp Ml/Rl/CartPolePpo.cpp)
target_include_directories(TutorialRlCartPolePpo PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/Ml/Rl)
oa_add_app_tutorial(TutorialRlCartPoleViewer Ml/Rl/TutorialRlCartPoleViewer.cpp Ml/Rl)
target_sources(TutorialRlCartPoleViewer PRIVATE
	Ml/Rl/CartPole.cpp Ml/Rl/CartPolePpo.cpp)
target_include_directories(TutorialRlCartPoleViewer PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/Ml/Rl)
# REMOVED: oa_add_nlp_tutorial(TutorialNlpMamba3TinyShakespeare TutorialNlpMamba3TinyShakespeare.cpp)
# MOVED to Examples/Ml/MotionGpt.cpp (converted to TEST format with training stubs)

# ═══════════════════════════════════════════════════════════════════════════
# CORE COMPUTE TUTORIALS
# ═══════════════════════════════════════════════════════════════════════════

# Raw compute tutorials that validate the OaFnMatrix -> OaContext -> graph/GEMM
# path without involving OaModule, optimizers, or autograd.

oa_add_app_tutorial(TutorialCoreMatMulIntro Core/TutorialCoreMatMulIntro.cpp Core)
target_include_directories(TutorialCoreMatMulIntro PRIVATE
	${oa_SOURCE_DIR}/Source/Private)
oa_add_app_tutorial(TutorialCoreEngine Core/TutorialCoreEngine.cpp Core)
oa_add_app_tutorial(TutorialCoreDeviceCanary Core/TutorialCoreDeviceCanary.cpp Core)

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

oa_add_app_tutorial(TutorialViewerVideoH264 Vision/TutorialViewerVideoH264.cpp Vision)
oa_add_app_tutorial(TutorialViewerVideoH265 Vision/TutorialViewerVideoH265.cpp Vision)
oa_add_app_tutorial(TutorialViewerVideoAv1  Vision/TutorialViewerVideoAv1.cpp  Vision)
oa_add_app_tutorial(TutorialViewerVideoVp9  Vision/TutorialViewerVideoVp9.cpp  Vision)
oa_add_app_tutorial(TutorialViewerAudio Vision/TutorialViewerAudio.cpp Vision)
# TutorialVideoDecodeDiagnostic (manual PNG-dump) REMOVED 2026-07-09: decode is
# covered headlessly by Test/Vision/Codec/TestVideoDecoder + TestVideoDecodeReference.
# TutorialVideoStreamCompare REMOVED 2026-07-09: dead (minimp4 dep disabled);
# demuxer covered by Test/Vision/Stream/TestVideoStream.
oa_add_app_tutorial(TutorialViewerImage Vision/TutorialViewerImage.cpp Vision)
oa_add_app_tutorial(TutorialViewerImageHeadless Vision/TutorialViewerImageHeadless.cpp Vision)
oa_add_app_tutorial(TutorialDetectionVideo Vision/TutorialDetectionVideo.cpp Vision)
oa_add_app_tutorial(TutorialDetectionImage Vision/TutorialDetectionImage.cpp Vision)
oa_add_app_tutorial(TutorialCvEvaluation Vision/TutorialCvEvaluation.cpp Vision)
oa_add_app_tutorial(TutorialScreenRecordH264 Vision/TutorialScreenRecordH264.cpp Vision)
oa_add_app_tutorial(TutorialCameraRecordH264 Vision/TutorialCameraRecordH264.cpp Vision)
# TutorialEncodeBasic REMOVED 2026-07-09: encode path covered by
# Test/Vision/Stream/TestVideoRoundtrip (encode→mux→demux→readback).

# ═══════════════════════════════════════════════════════════════════════════
# PLOT TUTORIALS (OaPlot module — FinalGlue Step 3e)
# ═══════════════════════════════════════════════════════════════════════════

oa_add_app_tutorial(TutorialImageGridClassify Plot/TutorialImageGridClassify.cpp Plot)
