// The HTTP front end must serve many concurrent keep-alive clients without making any
// of them wait for another to disconnect.
#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <string>
#include <vector>

#include "../src/server.h"
#include "mhprep/supervisor.h"

using namespace mhprep;
using namespace std::chrono_literals;

namespace {

int connect_to(int port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(static_cast<uint16_t>(port));
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) {
    ::close(fd);
    return -1;
  }
  return fd;
}

// Reads until a full response header has arrived or the deadline passes.
bool got_response(int fd, std::chrono::milliseconds within) {
  std::string buf;
  const auto deadline = std::chrono::steady_clock::now() + within;
  while (buf.find("\r\n\r\n") == std::string::npos) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (left <= 0ms) return false;
    pollfd p{fd, POLLIN, 0};
    if (::poll(&p, 1, static_cast<int>(left.count())) <= 0) return false;
    char tmp[4096];
    const ssize_t n = ::read(fd, tmp, sizeof tmp);
    if (n <= 0) return false;
    buf.append(tmp, static_cast<size_t>(n));
  }
  return buf.rfind("HTTP/1.1 200", 0) == 0;
}

}  // namespace

TEST(HttpServer, ServesMoreKeepAliveClientsThanTheThreadFloor) {
  PoolConfig c;
  c.workers = 0;
  c.handle_term_signals = false;
  Supervisor sup(c);
  HttpServer http(sup);
  const int port = 20000 + ::getpid() % 20000;
  ASSERT_TRUE(http.start("127.0.0.1", port));

  // Far more open connections than the old fixed pool of 64 threads. Each stays open
  // (keep-alive), pinning a server thread.
  constexpr int kClients = 200;
  static_assert(kClients > 64 && kClients > static_cast<int>(HttpServer::kMinHttpThreads));
  std::vector<int> fds;
  for (int i = 0; i < kClients; ++i) {
    const int fd = connect_to(port);
    ASSERT_GE(fd, 0) << "connect " << i;
    fds.push_back(fd);
  }
  // With a fixed pool of 64, clients 65+ would wait for the first 64 to time out
  // (keep-alive timeout, 5 s). Every one must be answered promptly.
  const std::string req = "GET /metrics HTTP/1.1\r\nHost: t\r\n\r\n";
  for (int fd : fds)
    ASSERT_EQ(::write(fd, req.data(), req.size()), static_cast<ssize_t>(req.size()));
  int served = 0;
  for (int fd : fds) served += got_response(fd, 2s);
  EXPECT_EQ(served, kClients);
  for (int fd : fds) ::close(fd);
  http.stop();
}
