#pragma once
// Length-prefixed JSON frames over a Unix stream socket.
//   frame = uint32 little-endian payload length | payload (UTF-8 JSON)
// The supervisor side is nonblocking and driven by epoll: partial reads and writes are
// buffered. The worker side uses the blocking helpers.
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace mhprep {

constexpr uint32_t kMaxFrameBytes = 64u << 20;

class FramedConn {
 public:
  explicit FramedConn(int fd) : fd_(fd) {}
  ~FramedConn();
  FramedConn(const FramedConn&) = delete;
  FramedConn& operator=(const FramedConn&) = delete;

  int fd() const { return fd_; }
  void close();

  // --- nonblocking (supervisor) ---
  void queue(const nlohmann::json& msg);
  // Write as much buffered output as the socket accepts. False on a fatal error.
  bool flush();
  bool want_write() const { return out_off_ < out_.size(); }
  // Read everything currently available and decode complete frames into `msgs`.
  // Returns false on EOF, error, or a malformed/oversized frame.
  bool read_available(std::vector<nlohmann::json>& msgs);

  // --- blocking (worker) ---
  bool send_blocking(const nlohmann::json& msg);
  std::optional<nlohmann::json> recv_blocking();

  // Exposed for tests: decode complete frames from an arbitrary byte buffer.
  static bool decode_frames(std::string& buf, std::vector<nlohmann::json>& msgs);
  static std::string encode_frame(const nlohmann::json& msg);

 private:
  int fd_;
  std::string in_, out_;
  size_t out_off_ = 0;
};

}  // namespace mhprep
