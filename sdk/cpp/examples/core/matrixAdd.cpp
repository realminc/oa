// OA_DOC_BEGIN: core-matrix-add
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

int main() {
	oa::EngineConfig config;
	config.appName = "ExampleCoreMatrixAdd";
	config.presentationMode = oa::PresentationMode::None;

	auto created = oa::Engine::create(config);
	if (not created.isOk()) {
		std::fprintf(stderr, "Engine creation failed: %s\n",
			created.getStatus().getMessage().cStr());
		return 1;
	}
	auto engine = std::move(created).getValue();

	auto one = oa::FnMatrix::ones({2, 3});
	auto two = oa::FnMatrix::full({2, 3}, 2.0F);
	auto sum = oa::FnMatrix::add(one, two);

	auto submitted = engine->submit();
	if (not submitted.isOk()) return 1;
	if (not engine->wait(submitted.getValue()).isOk()) return 1;

	std::array<oa::F32, 6> values{};
	if (not oa::FnMatrix::copyToHost(
		sum, values.data(), sizeof(values)).isOk()) return 1;
	for (const oa::F32 value : values) {
		if (std::abs(value - 3.0F) > 1.0e-6F) return 1;
	}

	std::puts("[3, 3, 3, 3, 3, 3]");
	return 0;
}
// OA_DOC_END: core-matrix-add
