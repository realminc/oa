# OA ML Extensions — model architectures and training utilities
# Moved from standalone ml repo into oa/Extensions/
#
# Public headers:  Extensions/Public/Ml/  (mirrors old ml/Source/Public/Ml/)
# Private sources: Extensions/Private/Ml/ (mirrors old ml/Source/Private/Ml/)
# Include path:     Extensions/Public/     (so #include <Ml/...> works)

target_include_directories(oa PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/Public>
	$<INSTALL_INTERFACE:include>
)

target_include_directories(oa PRIVATE
	${CMAKE_CURRENT_LIST_DIR}/Private
)

# Extension sources that compile today. GptOss currently supplies the exact
# published configuration contract; model layers land behind parity gates.
set(OA_ML_EXTENSION_SOURCES
	Private/Data/DsGen3dAnim.cpp
	Private/Ml/FnLoss.cpp
	Private/Ml/FnMatrix/FnMatrixChannelNorm.cpp
	Private/Ml/FnMatrix/FnMatrixConv1dGemm.cpp
	Private/Ml/SpirvInit.cpp
	Private/Ml/SpirvRegistry.cpp
	Private/Ml/Nn/GptOss/GptOss.cpp
	# OaAlm — autograd motion pipeline (Conv1d VQ-VAE tokenizer + Transformer token LM).
	Private/Ml/Nn/Alm/AlmAg.cpp
	Private/Ml/Nn/Alm/ClipTextAg.cpp
	Private/Ml/Nn/Alm/ClipTextWeightAdapter.cpp
	Private/Ml/Nn/Alm/ClipTokenizer.cpp
	Private/Ml/Nn/Alm/AlmTokenizerAg.cpp
	Private/Ml/Nn/Alm/AlmPriorAg.cpp
	# 3D anim engine: Core (transform/joint) → Rig (skeleton) → Anim (IO) → Retarget.
	Private/Core/Transform.cpp
	Private/Rig/Skeleton.cpp
	Private/Anim/PoseClip.cpp
	Private/Anim/Usd.cpp
	Private/Anim/PosePack.cpp
	Private/Anim/FbxWriter.cpp
	Private/Retarget/HumanIk.cpp
	Private/Retarget/Retarget.cpp
)

target_sources(oa PRIVATE ${OA_ML_EXTENSION_SOURCES})

# ML shader compilation (stubbed — TODO: integrate when models need custom kernels)
# When enabled, uncomment below and add the generated embed file to OA_ML_EXTENSION_SOURCES.
# find_program(SLANGC slangc ...)
# if(SLANGC)
#     include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/validate_kernel_registry.cmake)
#     ...
# endif()
