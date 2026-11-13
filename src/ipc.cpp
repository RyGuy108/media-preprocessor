#include "mhprep/ipc.h"

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

namespace mhprep {

FramedConn::~FramedConn() { close(); }

void FramedConn::close() {
  if (fd_ >= 0) ::close(fd_);
  fd_ = -1;
}

std::string FramedConn::encode_frame(const nlohmann::json& msg) {
  const std::string body = msg.dump();
  const auto n = static_cast<uint32_t>(body.size());
  std::string frame(4 + body.size(), '\0');
  for (int i = 0; i < 4; ++i) frame[i] = static_cast<char>((n >> (8 * i)) & 0xff);
  std::memcpy(frame.data() + 4, body.data(), body.size());
  return frame;
}

bool FramedConn::decode_frames(std::string& buf, std::vector<nlohmann::json>& msgs) {
  size_t off = 0;
  bool ok = true;
  while (buf.size() - off >= 4) {
    uint32_t n = 0;
    for (int i = 0; i < 4; ++i) n |= uint32_t(static_cast<uint8_t>(buf[off + i])) << (8 * i);
    if (n > kMaxFrameBytes) {
      ok = false;
      break;
    }
    if (buf.size() - off - 4 < n) break;
    auto j = nlohmann::json::parse(buf.begin() + off + 4, buf.begin() + off + 4 + n, nullptr,
                                   /*allow_exceptions=*/false);
    if (j.is_discarded()) {
      ok = false;
      break;
    }
    msgs.push_back(std::move(j));
    off += 4 + n;
  }
  buf.erase(0, off);
  return ok;
}

void FramedConn::queue(const nlohmann::json& msg) {
  if (out_off_ == out_.size()) {
    out_.clear();
    out_off_ = 0;
  }
  out_ += encode_frame(msg);
}

bool FramedConn::flush() {
  while (out_off_ < out_.size()) {
    ssize_t n = ::send(fd_, out_.data() + out_off_, out_.size() - out_off_, MSG_NOSIGNAL);
    if (n > 0) {
      out_off_ += static_cast<size_t>(n);
    } else if (n < 0 && errno == EINTR) {
      continue;
    } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return true;
    } else {
      return false;
    }
  }
  return true;
}

bool FramedConn::read_available(std::vector<nlohmann::json>& msgs) {
  char buf[65536];
  for (;;) {
    ssize_t n = ::read(fd_, buf, sizeof buf);
    if (n > 0) {
      in_.append(buf, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) {
      decode_frames(in_, msgs);
      return false;  // peer closed
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
    return false;
  }
  return decode_frames(in_, msgs);
}

bool FramedConn::send_blocking(const nlohmann::json& msg) {
  const std::string frame = encode_frame(msg);
  size_t off = 0;
  while (off < frame.size()) {
    ssize_t n = ::send(fd_, frame.data() + off, frame.size() - off, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    off += static_cast<size_t>(n);
  }
  return true;
}

std::optional<nlohmann::json> FramedConn::recv_blocking() {
  std::vector<nlohmann::json> msgs;
  for (;;) {
    if (!decode_frames(in_, msgs)) return std::nullopt;
    if (!msgs.empty()) return std::move(msgs.front());  // one request in flight at a time
    char buf[65536];
    ssize_t n = ::read(fd_, buf, sizeof buf);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return std::nullopt;
    in_.append(buf, static_cast<size_t>(n));
  }
}

}  // namespace mhprep
