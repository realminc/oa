# OA developer kit: source consumers and reusable sample-domain support.

add_subdirectory(cpp)

if(OA_BUILD_TESTS OR OA_BUILD_TUTORIALS)
	add_subdirectory(cpp/tutorials)
endif()

if(OA_BUILD_TESTS)
	add_subdirectory(cpp/examples)
endif()
