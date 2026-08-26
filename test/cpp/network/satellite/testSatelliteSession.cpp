#include <gtest/gtest.h>

#include <oa/crypto/keccak.h>
#include <oa/network/satelliteSession.h>

#include <chrono>
#include <thread>

namespace {

oa::Status satelliteMac(
	oa::Span<const oa::Byte> inKey,
	oa::Span<const oa::Byte> inData,
	oa::StringView inCustom,
	oa::Array<oa::Byte, 32>& outMac)
{
	return oa::kmac256(
		inKey.data(), inKey.size(), inData.data(), inData.size(),
		reinterpret_cast<const oa::Byte*>(inCustom.data()), inCustom.size(),
		outMac.data(), outMac.size());
}

oa::SatelliteSecret makeSecret(oa::Byte inSeed) {
	oa::Array<oa::Byte, 32> bytes{};
	for (oa::Usize i = 0; i < bytes.size(); ++i) {
		bytes[i] = static_cast<oa::Byte>(inSeed + static_cast<oa::Byte>(i));
	}
	auto secret = oa::SatelliteSecret::fromBytes(bytes);
	EXPECT_TRUE(secret.isOk());
	return oa::move(*secret);
}

oa::SatelliteSessionConfig makeConfig(oa::Byte inSeed, oa::U32 inTimeoutMs = 500U) {
	oa::SatelliteSessionConfig config(makeSecret(inSeed), satelliteMac);
	config.ioTimeoutMs = inTimeoutMs;
	config.limits.maxPayloadBytes = 64U * 1024U;
	config.limits.maxResidentBytes = 1024U * 1024U;
	config.limits.maxObjects = 8U;
	config.limits.maxInflight = 1U;
	config.deviceName = "loopback-oracle";
	return config;
}

struct ListenerAndPort {
	oa::TcpListener listener;
	oa::U16 port = 0;
};

ListenerAndPort bindLoopback() {
	auto result = oa::TcpListener::bind("127.0.0.1", 0U, 8);
	EXPECT_TRUE(result.isOk()) << result.getStatus().getMessage().cStr();
	ListenerAndPort bound;
	bound.listener = oa::move(*result);
	bound.port = bound.listener.port();
	return bound;
}

oa::U32 resultStatus(const oa::SatelliteMessage& inMessage) {
	const auto* field = oa::SatelliteProtocol::findField(
		inMessage, oa::SatelliteFieldId::ResultStatusCode);
	EXPECT_NE(field, nullptr);
	return *oa::SatelliteProtocol::readU32(*field);
}

oa::U8 resultComplete(const oa::SatelliteMessage& inMessage) {
	const auto* field = oa::SatelliteProtocol::findField(
		inMessage, oa::SatelliteFieldId::ResultComplete);
	EXPECT_NE(field, nullptr);
	return *oa::SatelliteProtocol::readU8(*field);
}

} // namespace

TEST(SatelliteSession, AuthenticatedProbeNegotiatesLimitsAndRunsCpuOracle) {
	auto bound = bindLoopback();
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = bound.listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::SatelliteServerSession session(makeConfig(11U));
		serverStatus = session.serve(oa::move(*accepted));
	});

	auto clientConfig = makeConfig(11U);
	clientConfig.limits.maxPayloadBytes = 128U * 1024U;
	clientConfig.limits.maxResidentBytes = 2U * 1024U * 1024U;
	clientConfig.limits.maxObjects = 16U;
	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", bound.port, oa::move(clientConfig));
	ASSERT_TRUE(connected.isOk()) << connected.getStatus().getMessage().cStr();
	auto client = oa::move(*connected);
	EXPECT_NE(client.probe().sessionEpoch, 0U);
	EXPECT_EQ(client.probe().deviceName, "loopback-oracle");
	EXPECT_EQ(client.probe().limits.maxPayloadBytes, 64U * 1024U);
	EXPECT_EQ(client.probe().limits.maxResidentBytes, 1024U * 1024U);
	EXPECT_EQ(client.probe().limits.maxObjects, 8U);

	auto started = client.startCpuAdd(19U, 23U);
	ASSERT_TRUE(started.isOk());
	auto polled = client.poll(*started);
	ASSERT_TRUE(polled.isOk());
	EXPECT_EQ(resultComplete(*polled), 0U);
	auto waited = client.wait(*started);
	ASSERT_TRUE(waited.isOk());
	ASSERT_TRUE(oa::SatelliteClientSession::readCpuAddResult(*waited).isOk());
	EXPECT_EQ(*oa::SatelliteClientSession::readCpuAddResult(*waited), 42U);
	auto retained = client.getResult(*started);
	ASSERT_TRUE(retained.isOk());
	EXPECT_EQ(*oa::SatelliteClientSession::readCpuAddResult(*retained), 42U);
	EXPECT_TRUE(client.close().isOk());

	server.join();
	EXPECT_TRUE(serverStatus.isOk()) << serverStatus.toString().cStr();
}

TEST(SatelliteSession, CancellationBeforeExecutionLeavesNoResult) {
	auto bound = bindLoopback();
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = bound.listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::SatelliteServerSession session(makeConfig(21U));
		serverStatus = session.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", bound.port, makeConfig(21U));
	ASSERT_TRUE(connected.isOk());
	auto client = oa::move(*connected);
	auto started = client.startCpuAdd(0xffffffffU, 1U);
	ASSERT_TRUE(started.isOk());
	auto cancelled = client.cancel(*started);
	ASSERT_TRUE(cancelled.isOk());
	EXPECT_EQ(resultComplete(*cancelled), 1U);
	EXPECT_EQ(resultStatus(*cancelled), static_cast<oa::U32>(oa::StatusCode::Cancelled));
	EXPECT_EQ(oa::SatelliteClientSession::readCpuAddResult(*cancelled)
		.getStatus().getCode(), oa::StatusCode::Cancelled);
	auto waited = client.wait(*started);
	ASSERT_TRUE(waited.isOk());
	EXPECT_EQ(resultStatus(*waited), static_cast<oa::U32>(oa::StatusCode::Cancelled));
	EXPECT_TRUE(client.close().isOk());
	server.join();
	EXPECT_TRUE(serverStatus.isOk());
}

TEST(SatelliteSession, BoundedInflightAdmissionRejectsASecondPendingRequest) {
	auto bound = bindLoopback();
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = bound.listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::SatelliteServerSession session(makeConfig(26U));
		serverStatus = session.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", bound.port, makeConfig(26U));
	ASSERT_TRUE(connected.isOk());
	auto client = oa::move(*connected);
	auto first = client.startCpuAdd(1U, 2U);
	ASSERT_TRUE(first.isOk());
	auto second = client.startCpuAdd(3U, 4U);
	EXPECT_TRUE(second.isError());
	EXPECT_EQ(second.getStatus().getCode(), oa::StatusCode::ResourceExhausted);
	ASSERT_TRUE(client.cancel(*first).isOk());
	EXPECT_TRUE(client.close().isOk());
	server.join();
	EXPECT_TRUE(serverStatus.isOk());
}

TEST(SatelliteSession, ObjectAndResidentQuotasRejectBeforeMutation) {
	auto bound = bindLoopback();
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = bound.listener.accept();
		ASSERT_TRUE(accepted.isOk());
		auto config = makeConfig(28U);
		config.limits.maxObjects = 1U;
		config.limits.maxResidentBytes = sizeof(oa::F32);
		oa::SatelliteServerSession session(oa::move(config));
		serverStatus = session.serve(oa::move(*accepted));
	});

	auto clientConfig = makeConfig(28U);
	clientConfig.limits.maxObjects = 1U;
	clientConfig.limits.maxResidentBytes = sizeof(oa::F32);
	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", bound.port, oa::move(clientConfig));
	ASSERT_TRUE(connected.isOk());
	auto client = oa::move(*connected);
	const oa::I64 oneShape[] = {1};
	const oa::F32 one[] = {2.0F};
	ASSERT_TRUE(client.putF32(1U, oneShape, one).isOk());
	EXPECT_EQ(client.putF32(2U, oneShape, one).getCode(),
		oa::StatusCode::ResourceExhausted);
	ASSERT_TRUE(client.dropObject(1U).isOk());
	const oa::I64 twoShape[] = {2};
	const oa::F32 two[] = {2.0F, 3.0F};
	EXPECT_EQ(client.putF32(2U, twoShape, two).getCode(),
		oa::StatusCode::ResourceExhausted);
	EXPECT_EQ(client.putF32(3U, oa::Span<const oa::I64>(), oa::Span<const oa::F32>())
		.getCode(), oa::StatusCode::InvalidArgument);
	EXPECT_TRUE(client.close().isOk());
	server.join();
	EXPECT_TRUE(serverStatus.isOk());
}

TEST(SatelliteSession, EnvelopeRejectsDuplicateAndStaleEpochBeforeMutation) {
	oa::SatelliteMessage request;
	request.type = oa::SatelliteMessageType::Close;
	request.requestId = 9U;
	request.sessionEpoch = 17U;
	EXPECT_TRUE(oa::satelliteValidateRequestEnvelope(request, 17U, 8U).isOk());
	EXPECT_EQ(oa::satelliteValidateRequestEnvelope(request, 17U, 9U).getCode(),
		oa::StatusCode::Aborted);
	request.sessionEpoch = 16U;
	EXPECT_EQ(oa::satelliteValidateRequestEnvelope(request, 17U, 8U).getCode(),
		oa::StatusCode::Aborted);
}

TEST(SatelliteSession, WrongSecretIsRejectedBeforeApplicationMutation) {
	auto bound = bindLoopback();
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = bound.listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::SatelliteServerSession session(makeConfig(31U));
		serverStatus = session.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", bound.port, makeConfig(32U));
	EXPECT_TRUE(connected.isError());
	EXPECT_EQ(connected.getStatus().getCode(), oa::StatusCode::Unauthenticated);
	server.join();
	EXPECT_EQ(serverStatus.getCode(), oa::StatusCode::Unauthenticated);
}

TEST(SatelliteSession, IdleHandshakeTimesOut) {
	auto bound = bindLoopback();
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = bound.listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::SatelliteServerSession session(makeConfig(41U, 25U));
		serverStatus = session.serve(oa::move(*accepted));
	});

	auto rawClient = oa::TcpStream::connect("127.0.0.1", bound.port);
	ASSERT_TRUE(rawClient.isOk());
	server.join();
	EXPECT_EQ(serverStatus.getCode(), oa::StatusCode::Unavailable);
}

TEST(SatelliteSession, DisconnectAbandonsPendingWorkAndReconnectGetsFreshEpoch) {
	auto bound = bindLoopback();
	oa::Status firstStatus;
	oa::Status secondStatus;
	oa::U64 firstEpoch = 0U;
	oa::U64 secondEpoch = 0U;
	std::thread server([&] {
		oa::SatelliteServerSession session(makeConfig(51U));
		auto first = bound.listener.accept();
		ASSERT_TRUE(first.isOk());
		firstStatus = session.serve(oa::move(*first));
		firstEpoch = session.lastSessionEpoch();
		auto second = bound.listener.accept();
		ASSERT_TRUE(second.isOk());
		secondStatus = session.serve(oa::move(*second));
		secondEpoch = session.lastSessionEpoch();
	});

	{
		auto connected = oa::SatelliteClientSession::connect(
			"127.0.0.1", bound.port, makeConfig(51U));
		ASSERT_TRUE(connected.isOk());
		auto client = oa::move(*connected);
		ASSERT_TRUE(client.startCpuAdd(1U, 2U).isOk());
		// No explicit Close: destruction closes the socket only. It does not
		// execute, submit, wait, or drain the accepted pending operation.
	}
	auto reconnected = oa::SatelliteClientSession::connect(
		"127.0.0.1", bound.port, makeConfig(51U));
	ASSERT_TRUE(reconnected.isOk());
	auto secondClient = oa::move(*reconnected);
	EXPECT_TRUE(secondClient.close().isOk());
	server.join();

	EXPECT_EQ(firstStatus.getCode(), oa::StatusCode::Unavailable);
	EXPECT_TRUE(secondStatus.isOk());
	EXPECT_NE(firstEpoch, 0U);
	EXPECT_NE(secondEpoch, 0U);
	EXPECT_NE(firstEpoch, secondEpoch);
}

TEST(SatelliteSession, ExplicitAbortTerminatesTheEpoch) {
	auto bound = bindLoopback();
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = bound.listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::SatelliteServerSession session(makeConfig(61U));
		serverStatus = session.serve(oa::move(*accepted));
	});
	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", bound.port, makeConfig(61U));
	ASSERT_TRUE(connected.isOk());
	auto client = oa::move(*connected);
	EXPECT_TRUE(client.abort("test abort").isOk());
	server.join();
	EXPECT_EQ(serverStatus.getCode(), oa::StatusCode::Aborted);
}
