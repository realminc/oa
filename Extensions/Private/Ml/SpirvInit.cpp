#include <Ml/SpirvRegistry.h>

#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Spirv.h>
#include <Oa/Core/Types.h>
#include <Oa/Core/Std/Filesystem.h>

// ML shaders are embedded in libml.a via ml_spirv_embed.cpp
// OA shaders are embedded in liboa.a and force-linked via --whole-archive
// No manual initialization needed - ShaderProvider handles everything automatically

void MlSpvInit() {
	// Register ML's embedded shader provider with OA's shader lookup chain.
	// If ml_spirv_embed.cpp is not yet linked (stub registry), this is a
	// no-op — the provider simply returns nullptr and OA falls through to
	// its own embedded shaders or file-based loading.
	OaSpvRegisterProvider(MlSpvFind);
}

void MlAddShaderSearchPaths(OaEngine& InRt) {
	// CMake can inject the exact build-directory path at configure time.
#ifdef ML_OA_SPIRV_BUILD_DIR
	InRt.AddShaderSearchPath(ML_OA_SPIRV_BUILD_DIR);
#endif
#ifdef ML_TOP_SPIRV_BUILD_DIR
	InRt.AddShaderSearchPath(ML_TOP_SPIRV_BUILD_DIR);
#endif

	// Try to discover the repo root from the current working directory and
	// add the standard OA build output directories (PascalCase presets).
	OaStdPath cwd = OaStdFilesystem::CurrentPath();
	if (!cwd.Empty()) {
		// Running from repo root → Build/{preset}/spirv
		InRt.AddShaderSearchPath(OaString((cwd / "Build" / "Release" / "spirv").string()));
		InRt.AddShaderSearchPath(OaString((cwd / "Build" / "Debug"   / "spirv").string()));

		// Running from Bin/{preset}/App/Ml/ → four levels up to repo root
		OaStdPath up4 = cwd;
		for (int i = 0; i < 4 && !up4.Empty(); ++i) {
			up4 = up4.ParentPath();
		}
		if (!up4.Empty()) {
			InRt.AddShaderSearchPath(OaString((up4 / "Build" / "Release" / "spirv").string()));
			InRt.AddShaderSearchPath(OaString((up4 / "Build" / "Debug"   / "spirv").string()));
		}
	}

	// Final fallback: plain "spirv" next to CWD (dev convenience)
	InRt.AddShaderSearchPath("spirv");
}
