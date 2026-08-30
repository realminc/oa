oa_add_app(oa-satellite satellite/main.cpp network)
target_sources(oa-satellite PRIVATE
	satellite/nlpCoordinator.cpp
	satellite/nlpStep.cpp)
target_include_directories(oa-satellite PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)

set(OA_WEBMCP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/webmcp")
set(OA_WEBMCP_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/webmcp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
	"${OA_WEBMCP_SOURCE_DIR}/web/index.html"
	"${OA_WEBMCP_SOURCE_DIR}/web/generated/webmcp.js"
	"${OA_WEBMCP_SOURCE_DIR}/web/style.css")
file(READ "${OA_WEBMCP_SOURCE_DIR}/web/index.html" OA_WEBMCP_INDEX_HTML)
file(READ "${OA_WEBMCP_SOURCE_DIR}/web/generated/webmcp.js" OA_WEBMCP_SCRIPT_JS)
file(READ "${OA_WEBMCP_SOURCE_DIR}/web/style.css" OA_WEBMCP_STYLE_CSS)
file(MAKE_DIRECTORY "${OA_WEBMCP_GENERATED_DIR}")
configure_file(
	"${OA_WEBMCP_SOURCE_DIR}/webAssets.gen.h.in"
	"${OA_WEBMCP_GENERATED_DIR}/webAssets.gen.h"
	@ONLY)

oa_add_app(oa-webmcp webmcp/main.cpp network)
target_sources(oa-webmcp PRIVATE webmcp/webmcp.cpp)
target_include_directories(oa-webmcp PRIVATE "${OA_WEBMCP_GENERATED_DIR}")
if(WIN32)
	target_link_libraries(oa-webmcp PRIVATE bcrypt)
endif()
