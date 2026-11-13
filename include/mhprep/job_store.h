#pragma once
// Durable job state in Postgres.
//
// The Supervisor keeps its job table in memory for speed; this store is the write-ahead
// record that lets a restarted (or crashed) supervisor pick up where it left off:
//
//  * submit() returns only after the job's row is committed, so a 202 from the API
//    means the job survives a crash. Concurrent submits share one transaction
//    (group commit) on a dedicated writer thread, so durability costs one fsync per
//    batch rather than one per request.
//  * State changes (started, retried, finished) are written behind on the same thread,
//    in order. If the process dies before a write lands, the row still says queued or
//    running and the job runs again on restart: delivery is at-least-once, and a job's
//    output directory is keyed by its id, so a rerun overwrites rather than duplicates.
//  * Each unique keyframe/image hash is stored with its job, so the per-tenant
//    cross-upload dedup index is rebuilt on startup instead of starting empty.
//  * Finished records the in-memory table has evicted are still served from Postgres.
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>

#include "mhprep/phash.h"

namespace mhprep {

struct JobRecord;

// A job recovered from the store at startup.
struct RecoveredJob {
  uint64_t id = 0;
  std::string input, tenant;
  nlohmann::json options;
  int attempts = 0;  // attempts that actually finished (an interrupted one is not counted)
  std::vector<std::string> attempt_log;
  std::chrono::system_clock::time_point submitted;
  bool was_running = false;
};

struct StoredHash {
  std::string tenant;
  uint64_t job_id = 0;
  int asset = 0;
  uint64_t phash = 0;
  ColorSig sig{};
};

class JobStore {
 public:
  // Connects, creates the schema if needed and starts the writer thread. Throws
  // std::runtime_error if the database is unreachable.
  explicit JobStore(std::string conninfo);
  ~JobStore();  // flushes pending writes, then stops the writer
  JobStore(const JobStore&) = delete;
  JobStore& operator=(const JobStore&) = delete;

  // Startup reads (call before the supervisor starts serving).
  uint64_t max_id();
  std::vector<RecoveredJob> load_unfinished();
  std::vector<StoredHash> load_hashes();

  // Inserts the job and blocks until it is committed or `timeout` passes. Returns
  // false if it could not be made durable; the row is then guaranteed not to exist.
  bool insert_durable(const JobRecord& job, std::chrono::milliseconds timeout);

  // Write-behind state changes, applied in call order after any earlier insert.
  void mark_running(uint64_t id, int attempts);
  void mark_queued(uint64_t id, int attempts, const std::vector<std::string>& attempt_log);
  void mark_finished(const JobRecord& job, std::vector<StoredHash> new_hashes);

  // A finished job's record as the API returns it, for jobs no longer held in memory.
  std::optional<nlohmann::json> describe(uint64_t id, bool include_manifest);

  // Blocks until every write queued so far is committed (tests, graceful shutdown).
  void flush();

  struct Stats {
    uint64_t transactions = 0, writes = 0, failed_transactions = 0;
  };
  Stats stats() const;

 private:
  struct Op;
  struct Conn;
  void writer_loop();
  bool apply_batch(Conn& c, std::vector<Op>& batch, bool include_inserts);

  std::string conninfo_;
  std::unique_ptr<Conn> read_conn_;  // startup reads + describe(), guarded by read_mu_
  std::mutex read_mu_;
  std::unique_ptr<Conn> lock_conn_;  // holds the one-supervisor-per-database lock

  mutable std::mutex mu_;
  std::condition_variable work_cv_, done_cv_;
  std::vector<Op> ops_;  // FIFO, in seq order
  uint64_t next_seq_ = 1;
  uint64_t done_seq_ = 0;  // every op with seq <= this has been applied or rejected
  std::vector<uint64_t> rejected_;  // insert seqs whose transaction failed
  bool stop_ = false;
  Stats stats_;
  std::thread writer_;
};

}  // namespace mhprep
