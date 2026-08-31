# OA developer-kit support — model architectures, animation and training tools.
#
# Declarations: sdk/cpp/include/ (repository-local devkit API)
# Implementation: sdk/cpp/lib/  (sources, private headers, and shaders)
# This is a source-companion boundary. These declarations are not installed as
# part of the OA framework API.

# Sample-domain sources that compile today. GptOss currently supplies the exact
# published configuration contract; model layers land behind parity gates.
set(OA_SDK_SUPPORT_SOURCES
	lib/data/datasetArchive.cpp
	lib/data/dsGen3dAnim.cpp
	lib/data/dsMnist.cpp
	lib/data/dsDailyDialog.cpp
	lib/data/dsHumanMl3d.cpp
	lib/ml/nlpSuite.cpp
	lib/ml/rl/cartPole.cpp
	lib/ml/rl/lunarlander3d/lunarLander3dTypes.cpp
	lib/ml/rl/lunarlander3d/lunarTerrain.cpp
	lib/ml/rl/lunarlander3d/lunarLander3dPhysics.cpp
	lib/ml/rl/lunarlander3d/lunarLander3d.cpp
	lib/ml/rl/lunarlander3d/lunarLander3dVector.cpp
	lib/ml/fnLoss.cpp
	lib/ml/nn/gptOss/gptOss.cpp
	# 3D anim support: installed Core Transform → SDK Joint/Rig → Anim/Retarget.
	lib/rig/skeleton.cpp
	lib/anim/poseClip.cpp
	lib/anim/usd.cpp
	lib/anim/posePack.cpp
	lib/anim/fbxWriter.cpp
	lib/retarget/humanIk.cpp
	lib/retarget/retarget.cpp
)

if(OA_ENABLE_ALM)
	list(APPEND OA_SDK_SUPPORT_SOURCES
		# Experimental autograd motion pipeline: VQ-VAE tokenizer + token LM.
		lib/ml/nn/alm/almAg.cpp
		lib/ml/nn/alm/clipTextAg.cpp
		lib/ml/nn/alm/clipTextModelTranslator.cpp
		lib/ml/nn/alm/clipTokenizer.cpp
		lib/ml/nn/alm/almTokenizerAg.cpp
		lib/ml/nn/alm/almPriorAg.cpp)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/gen/sdkEnvironmentKernels.cmake)

if(SLANGC)
	set(OA_SDK_ENVIRONMENT_SPIRV_DIR
		"${CMAKE_CURRENT_BINARY_DIR}/spirv/environment")
	set(OA_SDK_ENVIRONMENT_SPV_OUTPUTS "")
	list(LENGTH OA_SDK_ENVIRONMENT_KERNEL_NAMES _OA_SDK_ENVIRONMENT_COUNT)
	math(EXPR _OA_SDK_ENVIRONMENT_LAST "${_OA_SDK_ENVIRONMENT_COUNT} - 1")
	foreach(_I RANGE ${_OA_SDK_ENVIRONMENT_LAST})
		list(GET OA_SDK_ENVIRONMENT_KERNEL_NAMES ${_I} _NAME)
		list(GET OA_SDK_ENVIRONMENT_KERNEL_SOURCES ${_I} _SOURCE)
		set(_SLANG
			"${CMAKE_CURRENT_LIST_DIR}/lib/ml/rl/shader/${_SOURCE}.slang")
		set(_SPV "${OA_SDK_ENVIRONMENT_SPIRV_DIR}/${_NAME}.spv")
		add_custom_command(
			OUTPUT ${_SPV}
			COMMAND ${CMAKE_COMMAND} -E make_directory
				${OA_SDK_ENVIRONMENT_SPIRV_DIR}
			COMMAND ${SLANGC} ${_SLANG} -target spirv -entry main -o ${_SPV}
			DEPENDS ${_SLANG}
			COMMENT "slangc: SDK ${_SOURCE}.slang → ${_NAME}.spv")
		list(APPEND OA_SDK_ENVIRONMENT_SPV_OUTPUTS ${_SPV})
	endforeach()

	set(OA_SDK_ENVIRONMENT_SHADER_LIST
		"${CMAKE_CURRENT_BINARY_DIR}/environment_shader_list.txt")
	list(JOIN OA_SDK_ENVIRONMENT_KERNEL_NAMES "\n"
		OA_SDK_ENVIRONMENT_SHADER_LIST_CONTENT)
	file(GENERATE OUTPUT ${OA_SDK_ENVIRONMENT_SHADER_LIST}
		CONTENT "${OA_SDK_ENVIRONMENT_SHADER_LIST_CONTENT}")
	set(OA_SDK_ENVIRONMENT_EMBED_CPP
		"${CMAKE_CURRENT_BINARY_DIR}/environmentKernelPack.gen.cpp")
	add_custom_command(
		OUTPUT ${OA_SDK_ENVIRONMENT_EMBED_CPP}
		COMMAND ${CMAKE_COMMAND}
			-DSPIRV_DIR=${OA_SDK_ENVIRONMENT_SPIRV_DIR}
			-DOUTPUT=${OA_SDK_ENVIRONMENT_EMBED_CPP}
			-DSHADER_LIST_FILE=${OA_SDK_ENVIRONMENT_SHADER_LIST}
			-P ${oa_SOURCE_DIR}/cmake/embed_spirv_pack.cmake
		DEPENDS ${OA_SDK_ENVIRONMENT_SPV_OUTPUTS}
			${OA_SDK_ENVIRONMENT_SHADER_LIST}
			${oa_SOURCE_DIR}/cmake/embed_spirv_pack.cmake)
	list(APPEND OA_SDK_SUPPORT_SOURCES ${OA_SDK_ENVIRONMENT_EMBED_CPP})
endif()

add_library(oaSdkSupport STATIC ${OA_SDK_SUPPORT_SOURCES})
add_library(oa::sdk ALIAS oaSdkSupport)
target_link_libraries(oaSdkSupport PUBLIC oa)
target_include_directories(oaSdkSupport
	PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
	PRIVATE
		${CMAKE_CURRENT_LIST_DIR}/lib
		${oa_SOURCE_DIR}/source/cpp/lib
		${oa_SOURCE_DIR}/source/cpp/thirdparty
		${oa_SOURCE_DIR}/source/cpp/thirdparty/utf8proc)
set_target_properties(oaSdkSupport PROPERTIES
	POSITION_INDEPENDENT_CODE ON)

if(TARGET oa_shaders)
	add_dependencies(oaSdkSupport oa_shaders)
endif()

# ML shader compilation (stubbed — TODO: integrate when models need custom kernels)
# When enabled, uncomment below and add the generated embed file to OA_ML_EXTENSION_SOURCES.
# find_program(SLANGC slangc ...)
# if(SLANGC)
#     include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/validate_kernel_registry.cmake)
#     ...
# endif()

# Maintained developer-kit applications are ordinary consumers of OA.
function(oa_add_app TARGET SOURCE SUBDIR)
	add_executable(${TARGET} ${SOURCE})
	target_link_libraries(${TARGET} PRIVATE oaSdkSupport)
	set_target_properties(${TARGET} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${OA_SDK_BIN_DIR}/apps/${SUBDIR}")
endfunction()

add_subdirectory(apps)
add_subdirectory(apps/network)
