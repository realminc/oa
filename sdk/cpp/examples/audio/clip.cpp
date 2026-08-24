// OA_DOC_BEGIN: audio-clip
#include <oa/audio/fnAudio.h>
#include <oa/core/fnMatrix.h>
#include <oa/runtime/engine.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

int main() {
	oa::EngineConfig config;
	config.appName = "ExampleAudioClip";
	config.presentationMode = oa::PresentationMode::None;

	auto created = oa::Engine::create(config);
	if (not created.isOk()) return 1;
	auto engine = std::move(created).getValue();

	oa::Audio audio(
		oa::FnMatrix::full({1, 8}, 0.75F),
		48'000U,
		oa::AudioChannelLayout::Mono
	);
	auto clipped = oa::FnAudio::clip(audio, -0.5F, 0.5F);

	auto submitted = engine->submit();
	if (not submitted.isOk()) return 1;
	if (not engine->wait(submitted.getValue()).isOk()) return 1;
	if (not clipped.validate() || clipped.sampleRate() != 48'000U) return 1;

	std::array<oa::F32, 8> values{};
	if (not oa::FnMatrix::copyToHost(
		clipped.asMatrix(), values.data(), sizeof(values)).isOk()) return 1;
	for (const oa::F32 value : values) {
		if (std::abs(value - 0.5F) > 1.0e-6F) return 1;
	}

	std::puts("8 mono samples clipped to 0.5");
	return 0;
}
// OA_DOC_END: audio-clip
