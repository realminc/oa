# Unified public API: Source/Public/Oa/{Core,Runtime,Ml,Crypto,Network,Data}/... (DG under Core)
# Public headers include generated declaration fragments by stable <Oa/...>
# paths. Those fragments remain physically under Source/Private and are mapped
# into the SDK include tree at install time.
add_library(oa_api INTERFACE)
target_include_directories(oa_api INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/Public>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/Private>
	$<INSTALL_INTERFACE:include>
)
target_compile_features(oa_api INTERFACE cxx_std_20)
