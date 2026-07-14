# Unified public API: Source/Public/Oa/{Core,Runtime,Ml,Crypto,Network,Data}/... (DG under Core)
# BUILD_INTERFACE root is Source/Public → #include <Oa/Core/Types.h> → Public/Oa/Core/Types.h
add_library(oa_api INTERFACE)
target_include_directories(oa_api INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/Public>
	$<INSTALL_INTERFACE:include>
)
target_compile_features(oa_api INTERFACE cxx_std_20)
