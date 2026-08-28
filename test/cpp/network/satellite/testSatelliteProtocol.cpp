#include <gtest/gtest.h>

#include <oa/network/satelliteProtocol.h>

#include <cstring>

namespace {

oa::Array<oa::Byte, 32> bytes32(oa::Byte inSeed) {
	oa::Array<oa::Byte, 32> bytes{};
	for (oa::Usize i = 0; i < bytes.size(); ++i) {
		bytes[i] = static_cast<oa::Byte>(inSeed + static_cast<oa::Byte>(i));
	}
	return bytes;
}

void addBytes32(
	oa::SatelliteMessage& inOutMessage, oa::SatelliteFieldId inId, oa::Byte inSeed)
{
	const auto bytes = bytes32(inSeed);
	inOutMessage.fields.pushBack(oa::SatelliteField::bytes(inId, bytes));
}

oa::SatelliteMessage makeMessage(oa::SatelliteMessageType inType) {
	oa::SatelliteMessage message;
	message.type = inType;
	message.requestId = 17U;
	message.sessionEpoch = 23U;
	switch (inType) {
		case oa::SatelliteMessageType::Hello:
			message.sessionEpoch = 0U;
			message.fields.pushBack(oa::SatelliteField::u16(
				oa::SatelliteFieldId::HelloProtocolMin, 1U));
			message.fields.pushBack(oa::SatelliteField::u16(
				oa::SatelliteFieldId::HelloProtocolMax, 1U));
			addBytes32(message, oa::SatelliteFieldId::HelloClientNonce, 1U);
			addBytes32(message, oa::SatelliteFieldId::HelloAuthProof, 2U);
			addBytes32(message, oa::SatelliteFieldId::HelloBuildHash, 3U);
			addBytes32(message, oa::SatelliteFieldId::HelloSchemaHash, 4U);
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::HelloMaxPayloadBytes, 65536U));
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::HelloMaxResidentBytes, 1048576U));
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::HelloMaxObjects, 8U));
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::HelloMaxInflight, 1U));
			break;
		case oa::SatelliteMessageType::HelloReply:
			message.flags = oa::SatelliteProtocol::kResponseFlag;
			message.fields.pushBack(oa::SatelliteField::u16(
				oa::SatelliteFieldId::HelloReplyProtocol, 1U));
			addBytes32(message, oa::SatelliteFieldId::HelloReplyServerNonce, 5U);
			addBytes32(message, oa::SatelliteFieldId::HelloReplyAuthProof, 6U);
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::HelloReplySessionEpoch, 23U));
			message.fields.pushBack(oa::SatelliteField::string(
				oa::SatelliteFieldId::HelloReplyDeviceName, "Iris Xe"));
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::HelloReplyMaxPayloadBytes, 65536U));
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::HelloReplyMaxResidentBytes, 1048576U));
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::HelloReplyMaxObjects, 8U));
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::HelloReplyMaxInflight, 1U));
			break;
		case oa::SatelliteMessageType::PutObject: {
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::PutObjectId, 7U));
			message.fields.pushBack(oa::SatelliteField::u8(
				oa::SatelliteFieldId::PutObjectDtype, 0U));
			const oa::I64 shape[] = {2, 3};
			message.fields.pushBack(oa::SatelliteField::i64Array(
				oa::SatelliteFieldId::PutObjectShape, shape));
			const oa::Byte data[] = {1, 2, 3, 4};
			message.fields.pushBack(oa::SatelliteField::bytes(
				oa::SatelliteFieldId::PutObjectData, data));
			addBytes32(message, oa::SatelliteFieldId::PutObjectContentHash, 7U);
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::PutObjectVersion, 11U));
			break;
		}
		case oa::SatelliteMessageType::dropObject:
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::DropObjectId, 7U));
			break;
		case oa::SatelliteMessageType::ExecuteNamed: {
			message.fields.pushBack(oa::SatelliteField::string(
				oa::SatelliteFieldId::ExecuteOperation, "matrix-add-f32-v1"));
			const oa::U64 inputs[] = {1U, 2U};
			message.fields.pushBack(oa::SatelliteField::u64Array(
				oa::SatelliteFieldId::ExecuteInputObjectIds, inputs));
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::ExecuteOutputObjectId, 3U));
			const oa::Byte arguments[] = {9, 8};
			message.fields.pushBack(oa::SatelliteField::bytes(
				oa::SatelliteFieldId::ExecuteArguments, arguments));
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::ExecuteExpectedVersion, 4U));
			addBytes32(message, oa::SatelliteFieldId::ExecuteExpectedHash, 8U);
			break;
		}
		case oa::SatelliteMessageType::wait:
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::WaitRequestId, 11U));
			break;
		case oa::SatelliteMessageType::poll:
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::PollRequestId, 11U));
			break;
		case oa::SatelliteMessageType::cancel:
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::CancelRequestId, 11U));
			break;
		case oa::SatelliteMessageType::getResult:
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::GetResultRequestId, 11U));
			break;
		case oa::SatelliteMessageType::Error:
			message.flags = oa::SatelliteProtocol::kResponseFlag;
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::ErrorStatusCode,
				static_cast<oa::U32>(oa::StatusCode::InvalidArgument)));
			message.fields.pushBack(oa::SatelliteField::string(
				oa::SatelliteFieldId::ErrorMessage, "bad request"));
			message.fields.pushBack(oa::SatelliteField::u8(
				oa::SatelliteFieldId::ErrorPoisoned, 0U));
			break;
		case oa::SatelliteMessageType::abort:
			message.fields.pushBack(oa::SatelliteField::string(
				oa::SatelliteFieldId::AbortReason, "caller abort"));
			break;
		case oa::SatelliteMessageType::Close:
			break;
		case oa::SatelliteMessageType::result: {
			message.flags = oa::SatelliteProtocol::kResponseFlag;
			message.fields.pushBack(oa::SatelliteField::u64(
				oa::SatelliteFieldId::ResultRequestId, 11U));
			message.fields.pushBack(oa::SatelliteField::u8(
				oa::SatelliteFieldId::ResultComplete, 1U));
			message.fields.pushBack(oa::SatelliteField::u32(
				oa::SatelliteFieldId::ResultStatusCode,
				static_cast<oa::U32>(oa::StatusCode::Ok)));
			const oa::Byte result[] = {4, 5, 6};
			message.fields.pushBack(oa::SatelliteField::bytes(
				oa::SatelliteFieldId::ResultBytes, result));
			const oa::Byte profile[] = {7, 8};
			message.fields.pushBack(oa::SatelliteField::bytes(
				oa::SatelliteFieldId::ResultProfile, profile));
			break;
		}
	}
	return message;
}

void refreshChecksum(oa::Vector<oa::Byte>& inOutBytes) {
	const auto checksum = oa::SatelliteProtocol::stableDigest(
		oa::Span<const oa::Byte>(inOutBytes.data(), 32U),
		oa::Span<const oa::Byte>(
			inOutBytes.data() + oa::SatelliteProtocol::kHeaderBytes,
			inOutBytes.size() - oa::SatelliteProtocol::kHeaderBytes));
	std::memcpy(inOutBytes.data() + 32U, checksum.data(), checksum.size());
}

void writeU32Le(oa::Byte* outData, oa::U32 inValue) {
	for (oa::U32 i = 0; i < 4U; ++i) {
		outData[i] = static_cast<oa::Byte>((inValue >> (i * 8U)) & 0xffU);
	}
}

oa::U64 nextRandom(oa::U64& inOutState) {
	inOutState ^= inOutState << 13U;
	inOutState ^= inOutState >> 7U;
	inOutState ^= inOutState << 17U;
	return inOutState;
}

} // namespace

TEST(SatelliteProtocol, EverySchemaMessageRoundTripsCanonically) {
	const oa::SatelliteMessageType types[] = {
		oa::SatelliteMessageType::Hello,
		oa::SatelliteMessageType::HelloReply,
		oa::SatelliteMessageType::PutObject,
		oa::SatelliteMessageType::dropObject,
		oa::SatelliteMessageType::ExecuteNamed,
		oa::SatelliteMessageType::wait,
		oa::SatelliteMessageType::poll,
		oa::SatelliteMessageType::cancel,
		oa::SatelliteMessageType::getResult,
		oa::SatelliteMessageType::Error,
		oa::SatelliteMessageType::abort,
		oa::SatelliteMessageType::Close,
		oa::SatelliteMessageType::result,
	};
	for (const auto type : types) {
		const auto message = makeMessage(type);
		auto encoded = oa::SatelliteProtocol::encode(message);
		ASSERT_TRUE(encoded.isOk()) << encoded.getStatus().getMessage().cStr();
		auto decoded = oa::SatelliteProtocol::decode(oa::Span<const oa::Byte>(
			encoded.getValue().data(), encoded.getValue().size()));
		ASSERT_TRUE(decoded.isOk()) << decoded.getStatus().getMessage().cStr();
		auto reencoded = oa::SatelliteProtocol::encode(decoded.getValue());
		ASSERT_TRUE(reencoded.isOk()) << reencoded.getStatus().getMessage().cStr();
		ASSERT_EQ(encoded.getValue().size(), reencoded.getValue().size());
		EXPECT_EQ(std::memcmp(encoded.getValue().data(), reencoded.getValue().data(),
			encoded.getValue().size()), 0);
	}
}

TEST(SatelliteProtocol, TypedFieldsUseCanonicalLittleEndianEncoding) {
	const auto u16Field = oa::SatelliteField::u16(
		oa::SatelliteFieldId::HelloProtocolMin, 0x1234U);
	ASSERT_EQ(u16Field.data.size(), 2U);
	EXPECT_EQ(u16Field.data[0], 0x34U);
	EXPECT_EQ(u16Field.data[1], 0x12U);
	ASSERT_EQ(oa::SatelliteProtocol::readU16(u16Field).getValue(), 0x1234U);

	const oa::U64 source[] = {0x0102030405060708ULL, 9U};
	const auto arrayField = oa::SatelliteField::u64Array(
		oa::SatelliteFieldId::ExecuteInputObjectIds, source);
	auto values = oa::SatelliteProtocol::readU64Array(arrayField);
	ASSERT_TRUE(values.isOk());
	ASSERT_EQ(values.getValue().size(), 2U);
	EXPECT_EQ(values.getValue()[0], source[0]);
	EXPECT_EQ(values.getValue()[1], source[1]);
}

TEST(SatelliteProtocol, StableDigestPinsBytesAndSpanBoundary) {
	const oa::Byte first[] = {'O', 'A'};
	const oa::Byte second[] = {0U, 1U, 2U};
	const oa::Byte expected[] = {
		0x5bU, 0xd5U, 0xd6U, 0x1dU, 0x3dU, 0xf9U, 0x30U, 0xf6U,
		0x66U, 0x82U, 0xbdU, 0xe7U, 0xbcU, 0x22U, 0x7eU, 0xaeU,
		0xc7U, 0x56U, 0x82U, 0xb9U, 0xf5U, 0x10U, 0x51U, 0x7aU,
		0x9aU, 0x26U, 0x25U, 0xd1U, 0xb5U, 0x3fU, 0xe5U, 0x4cU,
	};
	const auto digest = oa::SatelliteProtocol::stableDigest(first, second);
	EXPECT_EQ(std::memcmp(digest.data(), expected, sizeof(expected)), 0);

	const oa::Byte differentFirst[] = {'O', 'A', 0U};
	const oa::Byte differentSecond[] = {1U, 2U};
	const auto different = oa::SatelliteProtocol::stableDigest(
		differentFirst, differentSecond);
	EXPECT_NE(digest, different);
}

TEST(SatelliteProtocol, ValidationRejectsMissingDuplicateUnsortedAndUnknownFields) {
	auto missing = makeMessage(oa::SatelliteMessageType::dropObject);
	missing.fields.clear();
	EXPECT_TRUE(oa::SatelliteProtocol::validate(missing).isError());

	auto duplicate = makeMessage(oa::SatelliteMessageType::Hello);
	duplicate.fields.pushBack(duplicate.fields.back());
	EXPECT_TRUE(oa::SatelliteProtocol::validate(duplicate).isError());

	auto unsorted = makeMessage(oa::SatelliteMessageType::PutObject);
	auto first = oa::move(unsorted.fields[0]);
	unsorted.fields[0] = oa::move(unsorted.fields[1]);
	unsorted.fields[1] = oa::move(first);
	EXPECT_TRUE(oa::SatelliteProtocol::validate(unsorted).isError());

	auto unknown = makeMessage(oa::SatelliteMessageType::dropObject);
	unknown.fields[0].id = static_cast<oa::SatelliteFieldId>(99U);
	EXPECT_TRUE(oa::SatelliteProtocol::validate(unknown).isError());
}

TEST(SatelliteProtocol, ValidationRejectsBoundsKindsAndMalformedUtf8) {
	auto wrongKind = makeMessage(oa::SatelliteMessageType::dropObject);
	wrongKind.fields[0].kind = oa::SatelliteFieldKind::Bytes;
	EXPECT_TRUE(oa::SatelliteProtocol::validate(wrongKind).isError());

	auto oversized = makeMessage(oa::SatelliteMessageType::abort);
	oversized.fields[0].data.resize(1025U);
	EXPECT_TRUE(oa::SatelliteProtocol::validate(oversized).isError());

	auto utf8 = makeMessage(oa::SatelliteMessageType::abort);
	utf8.fields[0].data = {0xc0U, 0x80U};
	EXPECT_TRUE(oa::SatelliteProtocol::validate(utf8).isError());

	auto partialArray = makeMessage(oa::SatelliteMessageType::ExecuteNamed);
	partialArray.fields[1].data.pushBack(0U);
	EXPECT_TRUE(oa::SatelliteProtocol::validate(partialArray).isError());
}

TEST(SatelliteProtocol, ParserRejectsEveryTruncationAndChecksumDamage) {
	auto encoded = oa::SatelliteProtocol::encode(
		makeMessage(oa::SatelliteMessageType::Hello));
	ASSERT_TRUE(encoded.isOk());
	for (oa::Usize bytes = 0; bytes < encoded.getValue().size(); ++bytes) {
		EXPECT_TRUE(oa::SatelliteProtocol::decode(oa::Span<const oa::Byte>(
			encoded.getValue().data(), bytes)).isError()) << bytes;
	}
	auto corrupt = encoded.getValue();
	corrupt.back() ^= 0x80U;
	EXPECT_EQ(oa::SatelliteProtocol::decode(oa::Span<const oa::Byte>(
		corrupt.data(), corrupt.size())).getStatus().getCode(), oa::StatusCode::DataLoss);
}

TEST(SatelliteProtocol, ParserRejectsUnknownVersionTypeFlagsAndTrailingBytes) {
	auto encodedResult = oa::SatelliteProtocol::encode(
		makeMessage(oa::SatelliteMessageType::Close));
	ASSERT_TRUE(encodedResult.isOk());

	auto version = encodedResult.getValue();
	version[4] = 2U;
	refreshChecksum(version);
	EXPECT_EQ(oa::SatelliteProtocol::decode(oa::Span<const oa::Byte>(
		version.data(), version.size())).getStatus().getCode(),
		oa::StatusCode::FailedPrecondition);

	auto type = encodedResult.getValue();
	type[6] = 99U;
	refreshChecksum(type);
	EXPECT_EQ(oa::SatelliteProtocol::decode(oa::Span<const oa::Byte>(
		type.data(), type.size())).getStatus().getCode(), oa::StatusCode::InvalidArgument);

	auto flags = encodedResult.getValue();
	flags[8] = 0x80U;
	refreshChecksum(flags);
	EXPECT_EQ(oa::SatelliteProtocol::decode(oa::Span<const oa::Byte>(
		flags.data(), flags.size())).getStatus().getCode(), oa::StatusCode::InvalidArgument);

	auto trailing = encodedResult.getValue();
	trailing.pushBack(0U);
	writeU32Le(trailing.data() + 28U, 3U);
	refreshChecksum(trailing);
	EXPECT_EQ(oa::SatelliteProtocol::decode(oa::Span<const oa::Byte>(
		trailing.data(), trailing.size())).getStatus().getCode(), oa::StatusCode::DataLoss);
}

TEST(SatelliteProtocol, ParserRejectsPayloadAllocationLimitBeforeFieldDecode) {
	oa::Vector<oa::Byte> header(oa::SatelliteProtocol::kHeaderBytes);
	header[0] = 'O';
	header[1] = 'A';
	header[2] = 'S';
	header[3] = 'P';
	writeU32Le(header.data() + 28U, oa::SatelliteProtocol::kMaxPayloadBytes + 1U);
	auto decoded = oa::SatelliteProtocol::decode(
		oa::Span<const oa::Byte>(header.data(), header.size()));
	EXPECT_EQ(decoded.getStatus().getCode(), oa::StatusCode::ResourceExhausted);
}

TEST(SatelliteProtocol, DeterministicMutationsEitherRejectOrRemainCanonical) {
	const oa::SatelliteMessageType types[] = {
		oa::SatelliteMessageType::Hello,
		oa::SatelliteMessageType::HelloReply,
		oa::SatelliteMessageType::PutObject,
		oa::SatelliteMessageType::ExecuteNamed,
		oa::SatelliteMessageType::Error,
		oa::SatelliteMessageType::result,
	};
	oa::U64 random = 0x4f41535046555a5aULL;
	for (oa::U32 iteration = 0; iteration < 10000U; ++iteration) {
		auto encoded = oa::SatelliteProtocol::encode(
			makeMessage(types[nextRandom(random) % (sizeof(types) / sizeof(types[0]))]));
		ASSERT_TRUE(encoded.isOk());
		auto mutated = encoded.getValue();
		const oa::Bool penetrateChecksum = (nextRandom(random) % 3U) == 0U;
		const oa::U32 mutationCount = 1U + (nextRandom(random) % 6U);
		for (oa::U32 mutation = 0; mutation < mutationCount; ++mutation) {
			const oa::Usize firstByte = penetrateChecksum
				? oa::SatelliteProtocol::kHeaderBytes
				: 0U;
			const oa::Usize index = firstByte
				+ (nextRandom(random) % (mutated.size() - firstByte));
			mutated[index] ^= static_cast<oa::Byte>(1U << (nextRandom(random) % 8U));
		}
		if (penetrateChecksum) refreshChecksum(mutated);

		auto decoded = oa::SatelliteProtocol::decode(
			oa::Span<const oa::Byte>(mutated.data(), mutated.size()));
		if (decoded.isError()) continue;
		auto reencoded = oa::SatelliteProtocol::encode(decoded.getValue());
		ASSERT_TRUE(reencoded.isOk());
		ASSERT_EQ(mutated.size(), reencoded.getValue().size());
		EXPECT_EQ(std::memcmp(mutated.data(), reencoded.getValue().data(), mutated.size()), 0)
			<< "iteration " << iteration;
	}
}
