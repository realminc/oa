// OaByteEncoder — byte transfer plus GPU logit decoding.

#include <Oa/Ml/Byte.h>

#include <Oa/Runtime/Context.h>

OaMatrix OaByteEncoder::Encode(OaSpan<const OaU8> InBytes) {
	return OaFnMatrix::FromBytes(InBytes, OaMatrixShape{static_cast<OaI64>(InBytes.size())}, OaScalarType::UInt8);
}

OaMatrix OaByteEncoder::EncodeBatched(OaSpan<const OaU8> InBytes) {
	return OaFnMatrix::FromBytes(InBytes, OaMatrixShape{1, static_cast<OaI64>(InBytes.size())}, OaScalarType::UInt8);
}

OaVec<OaU8> OaByteEncoder::Decode(const OaMatrix& InLogits) {
	auto ids = OaFnMatrix::SampleLogits(InLogits, 0.0F);
	if (ids.IsEmpty()) return {};
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
	OaVec<OaU8> result(static_cast<OaUsize>(ids.NumElements()));
	const OaI32* host = ids.DataAs<const OaI32>();
	for (OaI64 i = 0; i < ids.NumElements(); ++i) result[static_cast<OaUsize>(i)] = static_cast<OaU8>(host[i]);
	return result;
}

OaVec<OaU8> OaByteEncoder::Sample(const OaMatrix& InLogits, OaF32 InTemperature, OaF32 InTopP) {
	auto ids = OaFnMatrix::SampleLogits(InLogits, InTemperature, 0, InTopP);
	if (ids.IsEmpty()) return {};
	auto& ctx = OaContext::GetDefault();
	(void)ctx.Execute(); (void)ctx.Sync();
	OaVec<OaU8> result(static_cast<OaUsize>(ids.NumElements()));
	const OaI32* host = ids.DataAs<const OaI32>();
	for (OaI64 i = 0; i < ids.NumElements(); ++i) result[static_cast<OaUsize>(i)] = static_cast<OaU8>(host[i]);
	return result;
}

OaMatrix OaByteEncoder::EncodeImage(OaSpan<const OaU8> InPixels, OaI32 InWidth, OaI32 InHeight, OaI32 InChannels) {
	(void)InWidth; (void)InHeight; (void)InChannels;
	return Encode(InPixels);
}

OaMatrix OaByteEncoder::EncodeAudio(OaSpan<const OaU8> InSamples, OaI32 InSampleRate, OaI32 InChannels) {
	(void)InSampleRate; (void)InChannels;
	return Encode(InSamples);
}
