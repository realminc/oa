// OA_DOC_BEGIN: crypto-shake256
#include <oa/oa.h>

OA_MAIN("ExampleCryptoShake256") {
	constexpr oa::Array<oa::U8, 15> messagesBytes{
		97U, 108U, 112U, 104U, 97U, 98U, 114U, 97U, 118U, 111U, 99U, 114U, 121U, 112U, 116U
	};
	auto messages = oa::FnMatrix::fromBytes(
		oa::Span<const oa::U8>(messagesBytes.data(), messagesBytes.size()),
		{3, 5},
		oa::ScalarType::UInt8
	);

	auto digests = oa::FnHash::shake256(messages, 32U);

	oa::Array<oa::U8, 96> gpu{};
	oa::Array<oa::U8, 96> cpu{};
	if (not oa::FnMatrix::copyToHost(digests, gpu.data(), gpu.size()).isOk()) return 1;
	for (oa::Usize row = 0; row < 3U; ++row) {
		oa::shake256(
			messagesBytes.data() + row * 5U,
			5U,
			cpu.data() + row * 32U,
			32U
		);
	}
	if (oa::memcmp(gpu.data(), cpu.data(), gpu.size()) != 0) {
		return 1;
	}

	oa::print("3 GPU SHAKE-256 digests match the CPU oracle");
	return 0;
}
// OA_DOC_END: crypto-shake256
