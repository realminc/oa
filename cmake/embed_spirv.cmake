# embed_spirv.cmake — Build-time script
# Reads .spv files, generates C++ source with embedded byte arrays + registry
#
# Usage: cmake -DSPIRV_DIR=... -DOUTPUT=... -DSHADERS=name1;name2;... -P embed_spirv.cmake

if(NOT SPIRV_DIR OR NOT OUTPUT OR NOT SHADER_LIST_FILE)
	message(FATAL_ERROR "embed_spirv.cmake requires -DSPIRV_DIR -DOUTPUT -DSHADER_LIST_FILE")
endif()

file(STRINGS "${SHADER_LIST_FILE}" SHADERS)

set(CONTENT "")
string(APPEND CONTENT "// Auto-generated SPIR-V registry — do not edit\n")
string(APPEND CONTENT "// Built from: ${SPIRV_DIR}\n\n")
string(APPEND CONTENT "#include <cstring>\n")
string(APPEND CONTENT "#include <Oa/Runtime/Spirv.h>\n\n")

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

	# Convert hex string to 0xNN,0xNN,... format
	set(C_BYTES "")
	set(POS 0)
	set(COL 0)
	while(POS LESS HEX_LEN)
		string(SUBSTRING "${SPV_HEX}" ${POS} 2 BYTE)
		if(COL GREATER 0)
			string(APPEND C_BYTES ",")
		endif()
		if(COL EQUAL 16)
			string(APPEND C_BYTES "\n\t")
			set(COL 0)
		endif()
		string(APPEND C_BYTES "0x${BYTE}")
		math(EXPR POS "${POS} + 2")
		math(EXPR COL "${COL} + 1")
	endwhile()

	# Sanitize name for C identifier (replace /, - and . with _)
	string(REPLACE "/" "_" C_NAME "${SHADER}")
	string(REPLACE "-" "_" C_NAME "${C_NAME}")
	string(REPLACE "." "_" C_NAME "${C_NAME}")

	string(APPEND CONTENT "static const OaU8 kSpv_${C_NAME}[] = {\n\t${C_BYTES}\n};\n\n")
	string(APPEND REGISTRY_ENTRIES "\t{\"${SHADER}\", kSpv_${C_NAME}, ${SPV_SIZE}},\n")
	math(EXPR SHADER_COUNT "${SHADER_COUNT} + 1")
endforeach()

string(APPEND CONTENT "static const OaSpvEntry kRegistry[] = {\n")
string(APPEND CONTENT "${REGISTRY_ENTRIES}")
string(APPEND CONTENT "\t{nullptr, nullptr, 0}\n};\n\n")

string(APPEND CONTENT "const OaSpvEntry* OaSpvFind(const char* InName) {\n")
string(APPEND CONTENT "\tfor (const auto* e = kRegistry; e->Name; ++e) {\n")
string(APPEND CONTENT "\t\tif (std::strcmp(e->Name, InName) == 0) return e;\n")
string(APPEND CONTENT "\t}\n\treturn nullptr;\n}\n\n")

string(APPEND CONTENT "const OaSpvEntry* OaSpvFindByIndex(OaU32 InIndex) {\n")
string(APPEND CONTENT "\tif (InIndex >= ${SHADER_COUNT}) return nullptr;\n")
string(APPEND CONTENT "\treturn &kRegistry[InIndex];\n}\n\n")

string(APPEND CONTENT "OaU32 OaSpvCount() { return ${SHADER_COUNT}; }\n")

file(WRITE "${OUTPUT}" "${CONTENT}")
message(STATUS "Embedded ${SHADER_COUNT} SPIR-V shaders → ${OUTPUT}")
