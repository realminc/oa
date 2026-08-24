oa_add_app(oa-satellite satellite/main.cpp)
target_sources(oa-satellite PRIVATE
	satellite/nlpCoordinator.cpp
	satellite/nlpStep.cpp)
target_include_directories(oa-satellite PRIVATE ${oa_SOURCE_DIR}/source/cpp/lib)
