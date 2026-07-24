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
	"#include <Oa/Runtime/Spirv.h>\n\n")

set(REGISTRY_ENTRIES "")
set(SHADER_COUNT 0)

foreach(SHADER ${SHADERS})
	set(SPV_FILE "${SPIRV_DIR}/${SHADER}.spv")
	if(NOT EXISTS "${SPV_FILE}")
		message(FATAL_ERROR "Missing SPIR-V: ${SPV_FILE} — shader '${SHADER}' failed to compile")
	endif()

	file(READ "${SPV_FILE}" SPV_HEX HEX)
	file(SIZE "${SPV_FILE}" SPV_SIZE)
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
		"static const OaU8 kSpv_${C_NAME}[] = {\n\t${C_BYTES}\n};\n\n")
	string(APPEND REGISTRY_ENTRIES "\t{\"${SHADER}\", kSpv_${C_NAME}, ${SPV_SIZE}},\n")
	math(EXPR SHADER_COUNT "${SHADER_COUNT} + 1")
endforeach()

file(APPEND "${TMP_OUTPUT}"
	"static const OaSpvEntry kRegistry[] = {\n"
	"${REGISTRY_ENTRIES}"
	"\t{nullptr, nullptr, 0}\n};\n\n"
	"const OaSpvEntry* OaSpvFind(const char* InName) {\n"
	"\tfor (const auto* e = kRegistry; e->Name; ++e) {\n"
	"\t\tif (std::strcmp(e->Name, InName) == 0) return e;\n"
	"\t}\n\treturn nullptr;\n}\n\n"
	"const OaSpvEntry* OaSpvFindByIndex(OaU32 InIndex) {\n"
	"\tif (InIndex >= ${SHADER_COUNT}) return nullptr;\n"
	"\treturn &kRegistry[InIndex];\n}\n\n"
	"OaU32 OaSpvCount() { return ${SHADER_COUNT}; }\n")

file(RENAME "${TMP_OUTPUT}" "${OUTPUT}")
message(STATUS "Embedded ${SHADER_COUNT} SPIR-V shaders → ${OUTPUT}")
