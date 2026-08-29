// OA Tutorial — one engine, explicit submission.

#include <oa/oa.h>

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
		oa::print(oa::PrintStream::Error, "Engine creation failed: {}",
			result.getStatus().getMessage().cStr());
		return 1;
	}
	auto engine = oa::move(result).getValue();

	oa::print("OA engine");
	oa::print("  device: {}", engine->deviceName());
	oa::print("  precision: {}",
		engine->getPrecision() == oa::Precision::BF16 ? "BF16" : "FP32");

	// oa::FnMatrix records into the engine's private eager recorder. No public
	// graph or context object is required.
	auto a = oa::FnMatrix::ones(oa::MatrixShape{64, 128});
	auto b = oa::FnMatrix::full(oa::MatrixShape{64, 128}, 2.0F);
	auto output = oa::FnMatrix::add(a, b);

	if (auto status = submitAndWait(*engine); !status.isOk()) {
		oa::print(oa::PrintStream::Error, "Submission failed: {}", status.getMessage().cStr());
		return 1;
	}

	const oa::F32 value = output.at(0);
	oa::print("  add(Ones, full(2)): {:.1f}", value);
	return oa::abs(value - 3.0F) <= 1e-5F ? 0 : 1;
}
