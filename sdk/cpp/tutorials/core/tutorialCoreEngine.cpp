// OA Tutorial — one engine, explicit submission.

#include <oa/oa.h>

#include <cmath>
#include <cstdio>
#include <utility>

[[nodiscard]] static oa::Status submitAndWait(oa::Engine& inEngine) {
	auto submitted = inEngine.submit();
	if (not submitted.isOk()) return submitted.getStatus();
	return inEngine.wait(submitted.getValue());
}

int main() {
	oa::EngineConfig config;
	config.appName = "TutorialCoreEngine";
	config.precision = oa::Precision::FP32;

	auto result = oa::Engine::create(config);
	if (!result.isOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			result.getStatus().getMessage().cStr());
		return 1;
	}
	auto engine = std::move(result).getValue();

	std::printf("OA engine\n");
	std::printf("  device: %.*s\n",
		static_cast<int>(engine->deviceName().size()), engine->deviceName().data());
	std::printf("  precision: %s\n",
		engine->getPrecision() == oa::Precision::BF16 ? "BF16" : "FP32");

	// oa::FnMatrix records into the engine's private eager recorder. No public
	// graph or context object is required.
	auto a = oa::FnMatrix::ones(oa::MatrixShape{64, 128});
	auto b = oa::FnMatrix::full(oa::MatrixShape{64, 128}, 2.0F);
	auto output = oa::FnMatrix::add(a, b);

	if (auto status = submitAndWait(*engine); !status.isOk()) {
		std::fprintf(stderr, "Submission failed: %s\n", status.getMessage().cStr());
		return 1;
	}

	const oa::F32 value = output.at(0);
	std::printf("  add(Ones, full(2)): %.1f\n", value);
	return std::abs(value - 3.0F) <= 1e-5F ? 0 : 1;
}
