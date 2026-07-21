cmake_minimum_required(VERSION 3.22)

project(oa_mobile_lab LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(OA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../../../..")
set(OA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OA_OASTD_EXTRA_TARGETS OFF CACHE BOOL "" FORCE)
set(OA_BUILD_TUTORIALS OFF CACHE BOOL "" FORCE)
set(OA_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(OA_BUILD_CRYPTO OFF CACHE BOOL "" FORCE)
set(OA_WITH_DATABASE OFF CACHE BOOL "" FORCE)
set(OA_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(OA_ANDROID_ML ON CACHE BOOL "" FORCE)
set(OA_EMBED_SHADERS ON CACHE BOOL "" FORCE)
find_program(OA_HOST_SLANGC slangc
    HINTS
        "$ENV{SLANG_ROOT}/bin"
        "$ENV{HOME}/.vcpkg/oa/release/installed/x64-linux/tools/shader-slang"
        "$ENV{HOME}/.vcpkg/installed/x64-linux/tools/shader-slang"
        "$ENV{VCPKG_ROOT}/installed/x64-linux/tools/shader-slang")
if(NOT OA_HOST_SLANGC)
    message(FATAL_ERROR
        "OaMobileLab requires a host slangc; set SLANG_ROOT or install shader-slang")
endif()
set(SLANGC "${OA_HOST_SLANGC}" CACHE FILEPATH "Host Slang compiler" FORCE)

add_subdirectory(
    "${OA_ROOT}"
    "${CMAKE_CURRENT_BINARY_DIR}/oa")

set(ADRENOTOOLS_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/libadrenotools")
add_subdirectory(
    "${ADRENOTOOLS_ROOT}"
    "${CMAKE_CURRENT_BINARY_DIR}/adrenotools")

add_library(oa_mobile_lab SHARED
    native_probe.cpp
    native_training.cpp)

target_compile_definitions(oa_mobile_lab PRIVATE VK_NO_PROTOTYPES)
target_include_directories(oa_mobile_lab PRIVATE
	"${ADRENOTOOLS_ROOT}/include")
target_link_libraries(oa_mobile_lab PRIVATE
    oa
    adrenotools
    android
    dl
    log)

add_dependencies(oa_mobile_lab
    hook_impl
    main_hook
    file_redirect_hook
    gsl_alloc_hook)
