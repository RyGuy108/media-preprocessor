#pragma once
// Prefork-style supervisor for a pool of single-job worker processes.
//
//  * Workers are separate processes (posix_spawn + exec), each owning one job at a time
//    and talking to the supervisor over a socketpair with length-prefixed JSON.
//  * One epoll loop multiplexes worker sockets, a signalfd (SIGCHLD/SIGTERM/SIGINT),
//    a timerfd armed at the earliest job deadline, and an eventfd that HTTP threads use
//    to wake the loop. No polling, no sleeping.
//  * A job that exceeds its deadline gets its worker SIGKILLed. A worker that segfaults
//    or is OOM-limited takes only its own job down. Failed attempts are retried on a
//    fresh worker; a job that fails `max_attempts` times is quarantined as poison.
//  * Each worker runs under RLIMIT_AS, so a decompression bomb hits bad_alloc inside
//    that worker instead of the kernel OOM killer picking a victim.
//  * Cross-upload dedup: every unique keyframe/image is checked against a per-tenant
//    BK-tree of perceptual hashes, so re-uploads aren't sent to the API twice.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <nlohmann/json.hpp>

#include "mhprep/hash_index.h"
#include "mhprep/metrics.h"
#include "mhprep/phash.h"

namespace mhprep {

// AddressSanitizer reserves terabytes of virtual address space for its shadow memory, so
// it can't run under RLIMIT_AS. In sanitizer builds the worker memory cap is not applied.
#if defined(__SANITIZE_ADDRESS__)
inline constexpr bool kAddressSanitizer = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
inline constexpr bool kAddressSanitizer = true;
#else
inline constexpr bool kAddressSanitizer = false;
#endif
#else
inline constexpr bool kAddressSanitizer = false;
#endif

struct PoolConfig {
  int workers = 4;
  std::string worker_bin;  // path to the mhprep binary (exec'd as `mhprep worker`)
  std::string models_dir;
  std::string out_root = "out";
  std::chrono::milliseconds job_timeout{120'000};
  int max_attempts = 3;
  size_t worker_mem_mb = 3072;  // RLIMIT_AS per worker, 0 = unlimited
  nlohmann::json default_options = nlohmann::json::object();
  bool cross_job_dedup = true;
  int dup_radius = kDupHammingRadius;
  int color_tolerance = kDupColorTolerance;
  size_t max_queue = 100'000;          // beyond this, submit() refuses (HTTP 429)
  size_t max_finished_jobs = 200'000;  // older finished records are evicted
  std::chrono::milliseconds drain_timeout{30'000};
  bool handle_term_signals = true;  // SIGTERM/SIGINT start a graceful drain
};

enum class JobState { kQueued, kRunning, kSucceeded, kFailed };
const char* to_string(JobState s);

struct JobSpec {
  std::string input;
  std::string tenant = "default";
  nlohmann::json options = nlohmann::json::object();
};

struct JobRecord {
  uint64_t id = 0;
  JobSpec spec;
  JobState state = JobState::kQueued;
  int attempts = 0;
  std::string error;       // last error message
  std::string error_kind;  // input | timeout | crash | oom | internal | cancelled
  std::vector<std::string> attempt_log;
  nlohmann::json manifest;  // set on success
  int cross_job_duplicates = 0;
  std::chrono::steady_clock::time_point submitted, started, finished;
  double latency_ms() const;
  nlohmann::json to_json(bool include_manifest = true) const;
};

struct PoolStats {
  uint64_t submitted = 0, succeeded = 0, failed = 0, retries = 0;
  uint64_t timeouts = 0, crashes = 0, oom = 0, input_errors = 0;
  uint64_t worker_spawns = 0, worker_deaths = 0;
  uint64_t cross_job_duplicates = 0;
  size_t queued = 0, running = 0, workers_alive = 0;
};

class Supervisor {
 public:
  explicit Supervisor(PoolConfig cfg);
  ~Supervisor();
  Supervisor(const Supervisor&) = delete;
  Supervisor& operator=(const Supervisor&) = delete;

  // Block the signals the loop consumes via signalfd. Call in main() before starting
  // any thread so every thread inherits the mask.
  static void block_signals();

  // Thread-safe.
  std::optional<uint64_t> submit(JobSpec spec);
  std::optional<JobRecord> get(uint64_t id) const;
  // Serialised record, built under the lock without copying the manifest when it is
  // not wanted (the hot path for status polling).
  std::optional<nlohmann::json> describe(uint64_t id, bool include_manifest) const;
  // Wait until the job is finished or `timeout` passes.
  std::optional<JobRecord> wait(uint64_t id, std::chrono::milliseconds timeout) const;
  // Same, without copying the record out. Returns true if the job is finished.
  bool wait_done(uint64_t id, std::chrono::milliseconds timeout) const;
  PoolStats stats() const;
  const LatencyRecorder& latency() const { return latency_; }
  void request_stop();

  // Run the event loop on the calling thread. With exit_when_idle, returns once every
  // submitted job has finished; otherwise runs until request_stop() or SIGTERM/SIGINT.
  void run(bool exit_when_idle);

  // Called on the loop thread, with the job table locked, whenever a job reaches a
  // terminal state. Must not call back into the Supervisor.
  std::function<void(const JobRecord&)> on_finished;
  // Called on the loop thread when a stop is requested (e.g. to stop an HTTP server).
  std::function<void()> on_stop;

 private:
  struct Worker;
  void spawn_worker();
  void dispatch();
  void handle_worker_io(Worker& w, uint32_t events);
  void handle_message(Worker& w, const nlohmann::json& msg);
  void reap_children();
  void on_worker_exit(Worker& w, int status);
  void fail_or_retry(uint64_t job_id, const std::string& kind, const std::string& why);
  void finish(JobRecord& job, JobState st);
  void apply_cross_dedup(JobRecord& job);
  void check_deadlines();
  void arm_timer();
  void begin_stop();
  bool all_done() const;
  void epoll_mod(int fd, uint32_t events, bool add);

  PoolConfig cfg_;
  int epfd_ = -1, sigfd_ = -1, timerfd_ = -1, evfd_ = -1;
  std::map<int, std::unique_ptr<Worker>> workers_;  // keyed by socket fd
  std::unordered_map<pid_t, int> pid_to_fd_;
  std::deque<std::chrono::steady_clock::time_point> pending_spawns_;
  int startup_failures_ = 0;
  bool stopping_ = false;
  std::chrono::steady_clock::time_point stop_deadline_;

  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  std::unordered_map<uint64_t, JobRecord> jobs_;
  std::deque<uint64_t> queue_;
  std::deque<uint64_t> finished_order_;
  uint64_t next_id_ = 1;
  size_t unfinished_ = 0;
  bool stop_requested_ = false;
  PoolStats stats_;
  LatencyRecorder latency_;

  struct TenantIndex {
    TenantIndex(int r, int t) : index(r, t) {}
    DupIndex index;
    std::vector<std::pair<uint64_t, int>> owners;  // (job id, asset index)
  };
  std::unordered_map<std::string, TenantIndex> tenant_hashes_;
};

// Entry point for `mhprep worker`.
int worker_main(int fd, const std::string& models_dir, size_t mem_mb);

}  // namespace mhprep
