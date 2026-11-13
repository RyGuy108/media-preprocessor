#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "mhprep/supervisor.h"

namespace mhprep {

// REST API:
//   POST /v1/jobs            {"input": "...", "tenant": "...", "options": {...}} -> 202 {"id"}
//                            {"inputs": [...], ...}                              -> 202 {"ids"}
//   GET  /v1/jobs/{id}[?wait_ms=N][&manifest=0]                                  -> job record
//   GET  /healthz, GET /metrics (Prometheus)
class HttpServer {
 public:
  static constexpr int kListenBacklog = 4096;  // capped by net.core.somaxconn
  static constexpr size_t kMinHttpThreads = 16;
  static constexpr size_t kMaxHttpThreads = 4096;  // ~ max concurrent client connections
  explicit HttpServer(Supervisor& sup);
  ~HttpServer();
  bool start(const std::string& host, int port);  // binds, then serves on a background thread
  void shutdown();  // ask the listener to stop; returns immediately
  void stop();      // shutdown() and join the server thread

 private:
  struct Impl;
  Supervisor& sup_;
  std::unique_ptr<Impl> impl_;
  std::thread thread_;
  std::mutex join_mu_;
};

}  // namespace mhprep
