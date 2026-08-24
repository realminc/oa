#include <gtest/gtest.h>

#include <oa/network/tcp.h>
#include <oa/network/tcpFramed.h>

#include <thread>

TEST(TcpFramed, RoundTripOverLoopback) {
	auto listenerRes = oa::TcpListener::bind("127.0.0.1", 0, 8);
	ASSERT_TRUE(listenerRes.isOk());
	oa::TcpListener listener = std::move(listenerRes.getValue());
	const oa::U16 port = listener.port();

	oa::Vec<oa::Byte> received;
	std::thread server([&] {
		auto acc = listener.accept();
		ASSERT_TRUE(acc.isOk());
		oa::TcpStream stream = std::move(acc.getValue());
		ASSERT_TRUE(oa::TcpFramed::readMessage(stream, received).isOk());
		const oa::Byte ack[] = {0x01, 0x02, 0x03};
		ASSERT_TRUE(oa::TcpFramed::writeMessage(stream, oa::Span<const oa::Byte>(ack, 3)).isOk());
		stream.close();
		listener.close();
	});

	auto conn = oa::TcpStream::connect(oa::String("127.0.0.1"), port);
	ASSERT_TRUE(conn.isOk());
	oa::TcpStream client = std::move(conn.getValue());
	const oa::Byte payload[] = {'h', 'i'};
	ASSERT_TRUE(oa::TcpFramed::writeMessage(client, oa::Span<const oa::Byte>(payload, 2)).isOk());
	oa::Vec<oa::Byte> back;
	ASSERT_TRUE(oa::TcpFramed::readMessage(client, back).isOk());
	ASSERT_EQ(back.size(), 3u);
	EXPECT_EQ(back[0], 0x01);
	EXPECT_EQ(back[1], 0x02);
	EXPECT_EQ(back[2], 0x03);
	client.close();
	server.join();

	ASSERT_EQ(received.size(), 2u);
	EXPECT_EQ(received[0], 'h');
	EXPECT_EQ(received[1], 'i');
}

TEST(TcpFramed, ConsumerLimitRejectsBeforePayloadAllocation) {
	auto listenerRes = oa::TcpListener::bind("127.0.0.1", 0, 8);
	ASSERT_TRUE(listenerRes.isOk());
	oa::TcpListener listener = std::move(listenerRes.getValue());
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::TcpStream stream = std::move(accepted.getValue());
		oa::Vec<oa::Byte> payload;
		const auto status = oa::TcpFramed::readMessage(stream, payload, 4U);
		EXPECT_TRUE(status.isError());
		EXPECT_TRUE(payload.empty());
	});

	auto connected = oa::TcpStream::connect("127.0.0.1", listener.port());
	ASSERT_TRUE(connected.isOk());
	oa::TcpStream client = std::move(connected.getValue());
	const oa::Byte length[] = {5U, 0U, 0U, 0U};
	ASSERT_EQ(client.writeAll(length, sizeof(length)), 4);
	server.join();
}

TEST(TcpFramed, IoTimeoutUnblocksAnIdlePeer) {
	auto listenerRes = oa::TcpListener::bind("127.0.0.1", 0, 8);
	ASSERT_TRUE(listenerRes.isOk());
	oa::TcpListener listener = std::move(listenerRes.getValue());
	std::thread server([&] {
		auto accepted = listener.accept();
		ASSERT_TRUE(accepted.isOk());
		oa::TcpStream stream = std::move(accepted.getValue());
		ASSERT_TRUE(stream.setIoTimeout(25U).isOk());
		oa::Vec<oa::Byte> payload;
		EXPECT_TRUE(oa::TcpFramed::readMessage(stream, payload).isError());
	});

	auto connected = oa::TcpStream::connect("127.0.0.1", listener.port());
	ASSERT_TRUE(connected.isOk());
	oa::TcpStream client = std::move(connected.getValue());
	server.join();
}
