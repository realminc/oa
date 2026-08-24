cmake_minimum_required(VERSION 3.20)

project(oaInstalledConsumer LANGUAGES CXX)

if(NOT DEFINED OA_EXPECTED_VERSION OR OA_EXPECTED_VERSION STREQUAL "")
	message(FATAL_ERROR "OA_EXPECTED_VERSION is required")
endif()

find_package(oa ${OA_EXPECTED_VERSION} EXACT CONFIG REQUIRED)

add_executable(oaInstalledConsumer main.cpp)
target_compile_features(oaInstalledConsumer PRIVATE cxx_std_20)
target_link_libraries(oaInstalledConsumer PRIVATE oa::oa)
