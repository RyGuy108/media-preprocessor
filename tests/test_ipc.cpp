#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <thread>

#include "mhprep/ipc.h"

using namespace mhprep;
using nlohmann::json;

TEST(Framing, DecodesAcrossArbitrarySplits) {
  const json a{{"type", "job"}, {"id", 1}}, b{{"type", "result"}, {"blob", std::string(1000, 'x')}};
  const std::string wire = FramedConn::encode_frame(a) + FramedConn::encode_frame(b);
  for (size_t split = 0; split <= wire.size(); ++split) {
    std::string buf;
    std::vector<json> out;
    buf.append(wire, 0, split);
    ASSERT_TRUE(FramedConn::decode_frames(buf, out));
    buf.append(wire, split);
    ASSERT_TRUE(FramedConn::decode_frames(buf, out));
    ASSERT_EQ(out.size(), 2u) << "split at " << split;
    EXPECT_EQ(out[0], a);
    EXPECT_EQ(out[1], b);
    EXPECT_TRUE(buf.empty());
  }
}

TEST(Framing, RejectsOversizedAndMalformedFrames) {
  std::string big = "\xff\xff\xff\x7f";
  std::vector<json> out;
  EXPECT_FALSE(FramedConn::decode_frames(big, out));
  std::string bad = std::string("\x03\x00\x00\x00", 4) + "{x}";
  EXPECT_FALSE(FramedConn::decode_frames(bad, out));
  EXPECT_TRUE(out.empty());
}

TEST(FramedConn, NonblockingLargeMessageRoundTrip) {
  int sv[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sv), 0);
  fcntl(sv[0], F_SETFL, O_NONBLOCK);
  FramedConn sup(sv[0]), worker(sv[1]);
  const json msg{{"type", "job"}, {"payload", std::string(4 << 20, 'm')}};  // >> socket buffer
  sup.queue(msg);
  std::thread t([&] {
    auto got = worker.recv_blocking();
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ((*got)["payload"].get<std::string>().size(), size_t{4} << 20);
    worker.send_blocking({{"type", "result"}, {"ok", true}});
  });
  int flushes = 0;
  while (sup.want_write()) {
    ASSERT_TRUE(sup.flush());
    ++flushes;
    if (sup.want_write()) std::this_thread::yield();
  }
  EXPECT_GT(flushes, 1);  // the nonblocking path really did write in pieces
  std::vector<json> got;
  while (got.empty()) ASSERT_TRUE(sup.read_available(got));
  t.join();
  EXPECT_EQ(got[0]["ok"], true);
  worker.close();
  std::vector<json> none;
  // Peer closed: read reports EOF eventually.
  bool open = true;
  for (int i = 0; i < 1000 && open; ++i) open = sup.read_available(none);
  EXPECT_FALSE(open);
}
