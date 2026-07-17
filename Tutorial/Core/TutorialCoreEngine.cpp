// OA Tutorial — one engine, one authoritative context.

#include <Oa/Oa.h>

#include <cmath>
#include <cstdio>
#include <utility>

int main() {
	OaEngineConfig config;
	config.AppName = "TutorialCoreEngine";
	config.Precision = OaPrecision::FP32;

	auto result = OaComputeEngine::Create(config);
	if (!result.IsOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			result.GetStatus().GetMessage().c_str());
		return 1;
	}
	auto engine = std::move(result).GetValue();
	auto& context = engine->GetContext();

	std::printf("OA engine\n");
	std::printf("  device: %.*s\n",
		static_cast<int>(engine->DeviceName().Size()), engine->DeviceName().Data());
	std::printf("  precision: %s\n",
		engine->GetPrecision() == OaPrecision::BF16 ? "BF16" : "FP32");

	// OaFnMatrix records into the thread's selected context. Creating the engine
	// selected its owned context as the default; no second runtime or context is
	// required.
	auto a = OaFnMatrix::Ones(OaMatrixShape{64, 128});
	auto b = OaFnMatrix::Full(OaMatrixShape{64, 128}, 2.0F);
	auto output = OaFnMatrix::Add(a, b);

	if (auto status = context.Execute(); !status.IsOk()) {
		std::fprintf(stderr, "Execute failed: %s\n", status.GetMessage().c_str());
		return 1;
	}
	if (auto status = context.Sync(); !status.IsOk()) {
		std::fprintf(stderr, "Sync failed: %s\n", status.GetMessage().c_str());
		return 1;
	}

	const OaF32 value = output.At(0);
	std::printf("  Add(Ones, Full(2)): %.1f\n", value);
	return std::abs(value - 3.0F) <= 1e-5F ? 0 : 1;
}
