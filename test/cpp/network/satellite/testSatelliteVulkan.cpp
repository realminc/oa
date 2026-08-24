#include <gtest/gtest.h>

#include "../../oaTest.h"
#include <oa/crypto/keccak.h>
#include <oa/network/satelliteSession.h>

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

oa::SatelliteSecret makeVulkanSecret() {
	oa::Array<oa::Byte, 32> bytes{};
	for (oa::Usize i = 0; i < bytes.size(); ++i) {
		bytes[i] = static_cast<oa::Byte>(0xa0U + i);
	}
	auto secret = oa::SatelliteSecret::fromBytes(bytes);
	EXPECT_TRUE(secret.isOk());
	return oa::move(*secret);
}

oa::SatelliteSessionConfig makeVulkanConfig() {
	oa::SatelliteSessionConfig config(makeVulkanSecret(), satelliteMac);
	config.deviceName = oa::String(testEngine().deviceName());
	config.ioTimeoutMs = 1000U;
	config.limits.maxPayloadBytes = 64U * 1024U;
	config.limits.maxResidentBytes = 1024U * 1024U;
	config.limits.maxObjects = 8U;
	config.limits.maxInflight = 1U;
	return config;
}

} // namespace

TEST_VK(VkEngineTestFixture, SatelliteMatrixAddUsesExactOwnedEvent) {
	auto listenerResult = oa::TcpListener::bind("127.0.0.1", 0U, 8);
	ASSERT_TRUE(listenerResult.isOk());
	auto listener = oa::move(*listenerResult);
	oa::SatelliteServerSession serverSession(testEngine(), makeVulkanConfig());
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		serverStatus = serverSession.serve(oa::move(*accepted));
	});

	auto connected = oa::SatelliteClientSession::connect(
		"127.0.0.1", listener.port(), makeVulkanConfig());
	ASSERT_TRUE(connected.isOk()) << connected.getStatus().toString();
	auto client = oa::move(*connected);
	const oa::I64 shape[] = {5};
	const oa::F32 a[] = {1.0F, -2.0F, 3.5F, 4.0F, -6.0F};
	const oa::F32 b[] = {0.5F, 2.0F, -1.5F, 8.0F, 7.0F};
	ASSERT_TRUE(client.putF32(1U, shape, a).isOk());
	ASSERT_TRUE(client.putF32(2U, shape, b).isOk());

	// Cancellation is admitted before submission, and the destination object is
	// still absent. Reusing the same destination then proves no hidden mutation.
	auto cancelledRequest = client.startMatrixAddF32(1U, 2U, 3U);
	ASSERT_TRUE(cancelledRequest.isOk());
	auto cancelled = client.cancel(*cancelledRequest);
	ASSERT_TRUE(cancelled.isOk());
	EXPECT_EQ(client.dropObject(3U).getCode(), oa::StatusCode::NotFound);

	auto request = client.startMatrixAddF32(1U, 2U, 3U);
	ASSERT_TRUE(request.isOk());
	auto waited = client.wait(*request);
	ASSERT_TRUE(waited.isOk()) << waited.getStatus().toString();
	auto values = oa::SatelliteClientSession::readF32Result(*waited);
	ASSERT_TRUE(values.isOk()) << values.getStatus().toString();
	const oa::F32 expected[] = {1.5F, 0.0F, 2.0F, 12.0F, 1.0F};
	ASSERT_EQ(values->size(), sizeof(expected) / sizeof(expected[0]));
	for (oa::Usize i = 0; i < values->size(); ++i) {
		EXPECT_FLOAT_EQ((*values)[i], expected[i]);
	}
	auto retained = client.getResult(*request);
	ASSERT_TRUE(retained.isOk());
	auto retainedValues = oa::SatelliteClientSession::readF32Result(*retained);
	ASSERT_TRUE(retainedValues.isOk());
	EXPECT_EQ(retainedValues->size(), values->size());
	ASSERT_TRUE(client.dropObject(1U).isOk());
	ASSERT_TRUE(client.dropObject(2U).isOk());
	ASSERT_TRUE(client.dropObject(3U).isOk());
	ASSERT_TRUE(client.close().isOk());
	server.join();

	EXPECT_TRUE(serverStatus.isOk()) << serverStatus.toString();
	EXPECT_TRUE(serverSession.lastGpuEventWasOwned());
	EXPECT_TRUE(serverSession.lastGpuEventCompleted());
	EXPECT_NE(serverSession.lastGpuEventValue(), 0U);
}

TEST_VK(VkEngineTestFixture, AbandonPendingMatrixRequestDoesNotSubmitOrWait) {
	auto listenerResult = oa::TcpListener::bind("127.0.0.1", 0U, 8);
	ASSERT_TRUE(listenerResult.isOk());
	auto listener = oa::move(*listenerResult);
	oa::SatelliteServerSession serverSession(testEngine(), makeVulkanConfig());
	oa::Status serverStatus;
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		serverStatus = serverSession.serve(oa::move(*accepted));
	});

	{
		auto connected = oa::SatelliteClientSession::connect(
			"127.0.0.1", listener.port(), makeVulkanConfig());
		ASSERT_TRUE(connected.isOk());
		auto client = oa::move(*connected);
		const oa::I64 shape[] = {3};
		const oa::F32 a[] = {1.0F, 2.0F, 3.0F};
		const oa::F32 b[] = {4.0F, 5.0F, 6.0F};
		ASSERT_TRUE(client.putF32(11U, shape, a).isOk());
		ASSERT_TRUE(client.putF32(12U, shape, b).isOk());
		ASSERT_TRUE(client.startMatrixAddF32(11U, 12U, 13U).isOk());
		// Socket-only destruction abandons the admitted-but-unsubmitted request.
	}
	server.join();
	EXPECT_EQ(serverStatus.getCode(), oa::StatusCode::Unavailable);
	EXPECT_FALSE(serverSession.lastGpuEventWasOwned());
	EXPECT_FALSE(serverSession.lastGpuEventCompleted());
	EXPECT_EQ(serverSession.lastGpuEventValue(), 0U);
}
