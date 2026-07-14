#include <gtest/gtest.h>

#include <Oa/Network/Tcp.h>
#include <Oa/Network/TcpFramed.h>

#include <thread>

TEST(TcpFramed, RoundTripOverLoopback) {
	auto listenerRes = OaTcpListener::Bind("127.0.0.1", 0, 8);
	ASSERT_TRUE(listenerRes.IsOk());
	OaTcpListener listener = std::move(listenerRes.GetValue());
	const OaU16 port = listener.Port();

	OaVec<OaByte> received;
	std::thread server([&] {
		auto acc = listener.Accept();
		ASSERT_TRUE(acc.IsOk());
		OaTcpStream stream = std::move(acc.GetValue());
		ASSERT_TRUE(OaTcpFramed::ReadMessage(stream, received).IsOk());
		const OaByte ack[] = {0x01, 0x02, 0x03};
		ASSERT_TRUE(OaTcpFramed::WriteMessage(stream, OaSpan<const OaByte>(ack, 3)).IsOk());
		stream.Close();
		listener.Close();
	});

	auto conn = OaTcpStream::Connect(OaString("127.0.0.1"), port);
	ASSERT_TRUE(conn.IsOk());
	OaTcpStream client = std::move(conn.GetValue());
	const OaByte payload[] = {'h', 'i'};
	ASSERT_TRUE(OaTcpFramed::WriteMessage(client, OaSpan<const OaByte>(payload, 2)).IsOk());
	OaVec<OaByte> back;
	ASSERT_TRUE(OaTcpFramed::ReadMessage(client, back).IsOk());
	ASSERT_EQ(back.size(), 3u);
	EXPECT_EQ(back[0], 0x01);
	EXPECT_EQ(back[1], 0x02);
	EXPECT_EQ(back[2], 0x03);
	client.Close();
	server.join();

	ASSERT_EQ(received.size(), 2u);
	EXPECT_EQ(received[0], 'h');
	EXPECT_EQ(received[1], 'i');
}
