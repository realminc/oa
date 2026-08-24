# Embeds one explicitly owned SPIR-V pack without extending liboa's global
# registry. The generated implementation installs every binary into the engine
# passed to ensureEnvironmentKernelPack().

if(NOT SPIRV_DIR OR NOT OUTPUT OR NOT SHADER_LIST_FILE)
	message(FATAL_ERROR
		"embed_spirv_pack.cmake requires SPIRV_DIR, OUTPUT and SHADER_LIST_FILE")
endif()

file(STRINGS "${SHADER_LIST_FILE}" SHADERS)
set(TMP_OUTPUT "${OUTPUT}.tmp")
file(WRITE "${TMP_OUTPUT}"
	"// Build-generated SDK environment kernel pack — do not edit\n"
	"#include <ml/rl/environmentKernelPack.h>\n"
	"#include <oa/runtime/engine/engineAccess.h>\n\n")

set(ENTRIES "")
foreach(SHADER ${SHADERS})
	set(SPV_FILE "${SPIRV_DIR}/${SHADER}.spv")
	if(NOT EXISTS "${SPV_FILE}")
		message(FATAL_ERROR "Missing SDK SPIR-V: ${SPV_FILE}")
	endif()
	file(READ "${SPV_FILE}" SPV_HEX HEX)
	file(SIZE "${SPV_FILE}" SPV_SIZE)
	string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," C_BYTES "${SPV_HEX}")
	string(REGEX REPLACE "[/.-]" "_" C_NAME "${SHADER}")
	file(APPEND "${TMP_OUTPUT}"
		"static const oa::U8 kSpv_${C_NAME}[] = {\n\t${C_BYTES}\n};\n\n")
	string(APPEND ENTRIES
		"\t{\"${SHADER}\", kSpv_${C_NAME}, ${SPV_SIZE}},\n")
endforeach()

file(APPEND "${TMP_OUTPUT}"
	"namespace {\n"
	"struct KernelBinary {\n"
	"\tconst char* name;\n"
	"\tconst oa::U8* bytes;\n"
	"\toa::Usize size;\n"
	"};\n"
	"constexpr KernelBinary kEnvironmentKernels[] = {\n"
	"${ENTRIES}"
	"};\n"
	"} // namespace\n\n"
	"oa::Status oa::ensureEnvironmentKernelPack(oa::Engine& inEngine) {\n"
	"\toa::EngineAccess access(inEngine);\n"
	"\tconst oa::PipelineSpec spec{\n"
	"\t\t.numBindings = 16,\n"
	"\t\t.pushConstantBytes = 128,\n"
	"\t\t.specConstants = {{.id = 0, .value = 0}},\n"
	"\t};\n"
	"\tfor (const auto& kernel : kEnvironmentKernels) {\n"
	"\t\tconst oa::Status status = access.ensurePipeline(\n"
	"\t\t\tkernel.name, {kernel.bytes, kernel.size}, spec);\n"
	"\t\tif (status.isError()) return status;\n"
	"\t}\n"
	"\treturn oa::Status::ok();\n"
	"}\n")

file(RENAME "${TMP_OUTPUT}" "${OUTPUT}")
