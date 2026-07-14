# OA Apps — Model control tools and utilities

# Helper function for apps (standalone, no GTest, links to oa)
function(oa_add_app TARGET SOURCE)
	add_executable(${TARGET} ${SOURCE})
	target_link_libraries(${TARGET} PRIVATE oa)
	set_target_properties(${TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}")
endfunction()

# ML apps (modelctl, datasetctl, OaAlm tools)
add_subdirectory(Ml)
