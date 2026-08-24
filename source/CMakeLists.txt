# source — Unified public API
#
# Public headers live under source/cpp/include/oa/{core,runtime,ml,...}/
# Implementation sources live under source/cpp/lib/oa/{core,runtime,ml,...}/
# Python bindings and the importable package live under source/py/.
add_library(oa_api INTERFACE)
target_include_directories(oa_api INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cpp/include>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/cpp/lib>
	$<INSTALL_INTERFACE:include>
)
target_compile_features(oa_api INTERFACE cxx_std_20)
