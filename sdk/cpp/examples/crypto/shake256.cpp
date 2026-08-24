// OA_DOC_BEGIN: crypto-shake256
#include <oa/core/fnMatrix.h>
#include <oa/crypto/fnHash.h>
#include <oa/crypto/keccak.h>
#include <oa/runtime/engine.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <utility>

int main() {
	oa::EngineConfig config;
	config.appName = "ExampleCryptoShake256";
	config.presentationMode = oa::PresentationMode::None;

	auto created = oa::Engine::create(config);
	if (not created.isOk()) return 1;
	auto engine = std::move(created).getValue();

	constexpr std::array<oa::U8, 15> messages{
		'a', 'l', 'p', 'h', 'a',
		'b', 'r', 'a', 'v', 'o',
		'c', 'r', 'y', 'p', 't',
	};
	auto input = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(messages.data(), messages.size()),
		{3, 5},
		oa::ScalarType::UInt8
	);
	auto digests = oa::FnHash::shake256(input, 32);

	auto submitted = engine->submit();
	if (not submitted.isOk()) return 1;
	if (not engine->wait(submitted.getValue()).isOk()) return 1;

	std::array<oa::U8, 96> gpu{};
	std::array<oa::U8, 96> cpu{};
	if (not oa::FnMatrix::copyToHost(
		digests, gpu.data(), gpu.size()).isOk()) return 1;
	for (oa::Usize row = 0; row < 3; ++row) {
		oa::shake256(
			messages.data() + row * 5,
			5,
			cpu.data() + row * 32,
			32
		);
	}
	if (std::memcmp(gpu.data(), cpu.data(), gpu.size()) != 0) return 1;

	std::puts("3 GPU SHAKE-256 digests match the CPU oracle");
	return 0;
}
// OA_DOC_END: crypto-shake256
