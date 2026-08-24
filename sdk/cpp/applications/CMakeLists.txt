# OA ML Apps — train, chat, detect, serve, utilities

function(oa_add_ml_app TARGET SOURCE)
	add_executable(${TARGET} ${SOURCE})
	target_link_libraries(${TARGET} PRIVATE oaSdkSupport)
	set_target_properties(${TARGET} PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${OA_BIN_DIR}/app/ml"
	)
endfunction()

oa_add_ml_app(datasetctl util/datasetctl.cpp)
oa_add_ml_app(modelctl modelctl/main.cpp)

# OaAlm — skeletal-motion train + generate (GPU runtime). MotionGPT-lineage
# two-stage pipeline: Conv1d VQ-VAE tokenizer + autoregressive token LM.
oa_add_ml_app(trainalm gen3dAnim/trainalm.cpp)
oa_add_ml_app(genalm gen3dAnim/genalm.cpp)
# dsgen3danimctl — unified CPU data pipeline: clean / pack / info / bake
# (model-agnostic; builds the CMP .usd dataset OaAlm trains on)
oa_add_ml_app(dsgen3danimctl gen3dAnim/dsgen3danimctl.cpp)
