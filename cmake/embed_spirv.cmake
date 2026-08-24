# embed_spirv.cmake — Build-time script
# Reads .spv files, generates C++ source with embedded byte arrays + registry
#
# Usage: cmake -DSPIRV_DIR=... -DOUTPUT=... -DSHADERS=name1;name2;... -P embed_spirv.cmake

if(NOT SPIRV_DIR OR NOT OUTPUT OR NOT SHADER_LIST_FILE)
	message(FATAL_ERROR "embed_spirv.cmake requires -DSPIRV_DIR -DOUTPUT -DSHADER_LIST_FILE")
endif()

file(STRINGS "${SHADER_LIST_FILE}" SHADERS)

set(TMP_OUTPUT "${OUTPUT}.tmp")
# Write incrementally. Accumulating the complete generated translation unit in
# one CMake string makes every append copy an ever-growing ~20 MB value.
# Keeping only one shader array in memory makes regeneration linear and leaves
# the previous output intact until the replacement is complete.
file(WRITE "${TMP_OUTPUT}"
	"// Auto-generated SPIR-V registry — do not edit\n"
	"// Built from: ${SPIRV_DIR}\n\n"
	"#include <cstring>\n"
	"#include <oa/runtime/spirv.h>\n\n")

set(REGISTRY_ENTRIES "")
set(SHADER_COUNT 0)

foreach(SHADER ${SHADERS})
	set(SPV_FILE "${SPIRV_DIR}/${SHADER}.spv")
	if(NOT EXISTS "${SPV_FILE}")
		message(FATAL_ERROR "Missing SPIR-V: ${SPV_FILE} — shader '${SHADER}' failed to compile")
	endif()

	file(READ "${SPV_FILE}" SPV_HEX HEX)
	file(SIZE "${SPV_FILE}" SPV_SIZE)
	file(SHA256 "${SPV_FILE}" SPV_SHA256)
	# The complete digest remains the build oracle. The registry carries its
	# first 64 bits as a compact executable/cache identity; changing any compiled
	# byte is therefore tracked without runtime hashing or mutable state.
	string(SUBSTRING "${SPV_SHA256}" 0 16 SPV_CONTENT_ID)
	if(SPV_CONTENT_ID STREQUAL "0000000000000000")
		message(FATAL_ERROR "Reserved zero content identity for SPIR-V: ${SPV_FILE}")
	endif()
	string(LENGTH "${SPV_HEX}" HEX_LEN)

	# Convert the complete hex string in one pass. A trailing comma is valid in
	# an initializer and avoids hundreds of thousands of substring/appends for
	# larger kernels.
	string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," C_BYTES "${SPV_HEX}")

	# Sanitize name for C identifier (replace /, - and . with _)
	string(REPLACE "/" "_" C_NAME "${SHADER}")
	string(REPLACE "-" "_" C_NAME "${C_NAME}")
	string(REPLACE "." "_" C_NAME "${C_NAME}")

	file(APPEND "${TMP_OUTPUT}"
		"static const oa::U8 kSpv_${C_NAME}[] = {\n\t${C_BYTES}\n};\n\n")
	string(APPEND REGISTRY_ENTRIES
		"\t{\"${SHADER}\", kSpv_${C_NAME}, ${SPV_SIZE}, 0x${SPV_CONTENT_ID}ULL},\n")
	math(EXPR SHADER_COUNT "${SHADER_COUNT} + 1")
endforeach()

file(APPEND "${TMP_OUTPUT}"
	"static const oavk::SpirvEntry kRegistry[] = {\n"
	"${REGISTRY_ENTRIES}"
	"\t{nullptr, nullptr, 0, 0}\n};\n\n"
	"const oavk::SpirvEntry* oavk::findSpirv(const char* inName) {\n"
	"\tif (inName == nullptr) return nullptr;\n"
	"\tfor (const auto* e = kRegistry; e->name; ++e) {\n"
	"\t\tif (std::strcmp(e->name, inName) == 0) return e;\n"
	"\t}\n\treturn nullptr;\n}\n\n"
	"const oavk::SpirvEntry* oavk::findSpirvByIndex(oa::U32 inIndex) {\n"
	"\tif (inIndex >= ${SHADER_COUNT}) return nullptr;\n"
	"\treturn &kRegistry[inIndex];\n}\n\n"
	"oa::U32 oavk::spirvCount() { return ${SHADER_COUNT}; }\n")

file(RENAME "${TMP_OUTPUT}" "${OUTPUT}")
message(STATUS "Embedded ${SHADER_COUNT} SPIR-V shaders → ${OUTPUT}")
