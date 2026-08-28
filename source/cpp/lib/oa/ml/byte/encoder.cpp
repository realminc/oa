// oa::ByteEncoder — byte transfer plus GPU logit decoding.

#include <oa/ml/byte.h>

#include <oa/runtime/executionSession.h>

oa::Matrix oa::ByteEncoder::encode(oa::Span<const oa::U8> inBytes) {
	return oa::FnMatrix::fromBytes(inBytes, oa::MatrixShape{static_cast<oa::I64>(inBytes.size())}, oa::ScalarType::UInt8);
}

oa::Matrix oa::ByteEncoder::encodeBatched(oa::Span<const oa::U8> inBytes) {
	return oa::FnMatrix::fromBytes(inBytes, oa::MatrixShape{1, static_cast<oa::I64>(inBytes.size())}, oa::ScalarType::UInt8);
}

oa::Vector<oa::U8> oa::ByteEncoder::decode(const oa::Matrix& inLogits) {
	auto ids = oa::FnMatrix::sampleLogits(inLogits, 0.0F);
	if (ids.isEmpty()) return {};
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
	oa::Vector<oa::U8> result(static_cast<oa::Usize>(ids.numElements()));
	oa::Vector<oa::I32> host(static_cast<oa::Usize>(ids.numElements()));
	if (not oa::FnMatrix::copyToHost(ids, host.data(),
		static_cast<oa::U64>(ids.byteSize())).isOk()) return {};
	for (oa::I64 i = 0; i < ids.numElements(); ++i)
		result[static_cast<oa::Usize>(i)] = static_cast<oa::U8>(host[static_cast<oa::Usize>(i)]);
	return result;
}

oa::Vector<oa::U8> oa::ByteEncoder::sample(const oa::Matrix& inLogits, oa::F32 inTemperature, oa::F32 inTopP) {
	auto ids = oa::FnMatrix::sampleLogits(inLogits, inTemperature, 0, inTopP);
	if (ids.isEmpty()) return {};
	auto& ctx = oa::ExecutionSession::getActive();
	(void)ctx.submitAndWait();
	oa::Vector<oa::U8> result(static_cast<oa::Usize>(ids.numElements()));
	oa::Vector<oa::I32> host(static_cast<oa::Usize>(ids.numElements()));
	if (not oa::FnMatrix::copyToHost(ids, host.data(),
		static_cast<oa::U64>(ids.byteSize())).isOk()) return {};
	for (oa::I64 i = 0; i < ids.numElements(); ++i)
		result[static_cast<oa::Usize>(i)] = static_cast<oa::U8>(host[static_cast<oa::Usize>(i)]);
	return result;
}

oa::Matrix oa::ByteEncoder::encodeImage(oa::Span<const oa::U8> inPixels, oa::I32 inWidth, oa::I32 inHeight, oa::I32 inChannels) {
	(void)inWidth; (void)inHeight; (void)inChannels;
	return encode(inPixels);
}

oa::Matrix oa::ByteEncoder::encodeAudio(oa::Span<const oa::U8> inSamples, oa::I32 inSampleRate, oa::I32 inChannels) {
	(void)inSampleRate; (void)inChannels;
	return encode(inSamples);
}
