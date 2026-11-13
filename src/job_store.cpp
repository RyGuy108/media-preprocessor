#include "mhprep/job_store.h"

#include <libpq-fe.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "mhprep/supervisor.h"

namespace mhprep {

using nlohmann::json;
using namespace std::chrono_literals;

namespace {

const char* kSchema = R"SQL(
CREATE TABLE IF NOT EXISTS mhprep_jobs (
  id                   bigint PRIMARY KEY,
  tenant               text NOT NULL,
  input                text NOT NULL,
  options              jsonb NOT NULL DEFAULT '{}',
  state                text NOT NULL
                       CHECK (state IN ('queued', 'running', 'succeeded', 'failed')),
  attempts             integer NOT NULL DEFAULT 0,
  attempt_log          jsonb NOT NULL DEFAULT '[]',
  error                text,
  error_kind           text,
  manifest             jsonb,
  cross_job_duplicates integer NOT NULL DEFAULT 0,
  latency_ms           double precision,
  queue_ms             double precision,
  submitted_at         timestamptz NOT NULL DEFAULT now(),
  started_at           timestamptz,
  finished_at          timestamptz
);
-- Recovery scans only unfinished jobs; keep that cheap as the table grows.
CREATE INDEX IF NOT EXISTS mhprep_jobs_unfinished
  ON mhprep_jobs (id) WHERE state IN ('queued', 'running');
-- Retention deletes the oldest finished jobs first.
CREATE INDEX IF NOT EXISTS mhprep_jobs_finished_at
  ON mhprep_jobs (finished_at) WHERE state IN ('succeeded', 'failed');
CREATE TABLE IF NOT EXISTS mhprep_asset_hashes (
  job_id bigint NOT NULL REFERENCES mhprep_jobs (id) ON DELETE CASCADE,
  asset  integer NOT NULL,
  tenant text NOT NULL,
  phash  bigint NOT NULL,
  csig   bytea NOT NULL,
  PRIMARY KEY (job_id, asset)
);
)SQL";

using Params = std::vector<std::optional<std::string>>;
constexpr size_t kMaxBatch = 512;

struct ResultDeleter {
  void operator()(PGresult* r) const { PQclear(r); }
};
using Result = std::unique_ptr<PGresult, ResultDeleter>;

std::string log_json(const std::vector<std::string>& log) { return json(log).dump(); }

}  // namespace

// ---------------------------------------------------------------------------------------
// Connection wrapper
// ---------------------------------------------------------------------------------------
struct JobStore::Conn {
  PGconn* pg = nullptr;
  std::string conninfo;
  std::string last_error;

  explicit Conn(std::string ci) : conninfo(std::move(ci)) {}
  ~Conn() { close(); }

  void close() {
    if (pg) PQfinish(pg);
    pg = nullptr;
  }
  bool ok() const { return pg && PQstatus(pg) == CONNECTION_OK; }

  bool connect() {
    close();
    pg = PQconnectdb(conninfo.c_str());
    if (!ok()) {
      last_error = pg ? PQerrorMessage(pg) : "out of memory";
      close();
      return false;
    }
    // A write must never hang the writer thread (and with it every submit) forever.
    return exec_ok("SET statement_timeout = '10s'; SET client_min_messages = warning");
  }

  Result exec(const char* sql, const Params& params = {}) {
    std::vector<const char*> values;
    values.reserve(params.size());
    for (const auto& p : params) values.push_back(p ? p->c_str() : nullptr);
    // Without parameters, use the simple protocol: it allows multi-statement scripts.
    Result r(params.empty() ? PQexec(pg, sql)
                            : PQexecParams(pg, sql, static_cast<int>(values.size()), nullptr,
                                           values.data(), nullptr, nullptr, 0));
    const auto st = r ? PQresultStatus(r.get()) : PGRES_FATAL_ERROR;
    if (st != PGRES_COMMAND_OK && st != PGRES_TUPLES_OK) {
      last_error = PQerrorMessage(pg);
      return nullptr;
    }
    return r;
  }
  bool exec_ok(const char* sql, const Params& params = {}) { return exec(sql, params) != nullptr; }

  // Queue a statement in pipeline mode (no round trip).
  bool send(const char* sql, const Params& params) {
    std::vector<const char*> values;
    values.reserve(params.size());
    for (const auto& p : params) values.push_back(p ? p->c_str() : nullptr);
    if (PQsendQueryParams(pg, sql, static_cast<int>(values.size()), nullptr, values.data(),
                          nullptr, nullptr, 0))
      return true;
    last_error = PQerrorMessage(pg);
    return false;
  }

  // Sends every statement `queue` produces in one pipeline, then one Sync. The statements
  // run in a single implicit transaction: all commit at the Sync, or none do. One round
  // trip for the whole batch instead of one per statement.
  template <typename F>
  bool pipeline(F&& queue) {
    if (!PQenterPipelineMode(pg)) {
      last_error = PQerrorMessage(pg);
      return false;
    }
    bool ok = queue(*this);
    if (!PQpipelineSync(pg)) {
      last_error = PQerrorMessage(pg);
      ok = false;
    }
    // Drain: one result (then a null) per statement, ending with PGRES_PIPELINE_SYNC.
    int nulls = 0;
    for (;;) {
      PGresult* r = PQgetResult(pg);
      if (!r) {
        if (++nulls > 1 || PQstatus(pg) != CONNECTION_OK) {  // nothing more is coming
          ok = false;
          break;
        }
        continue;
      }
      nulls = 0;
      const auto st = PQresultStatus(r);
      if (st == PGRES_PIPELINE_SYNC) {
        PQclear(r);
        break;
      }
      if (st != PGRES_COMMAND_OK && st != PGRES_TUPLES_OK) {
        if (ok) last_error = st == PGRES_PIPELINE_ABORTED ? "pipeline aborted"
                                                          : PQresultErrorMessage(r);
        ok = false;
      }
      PQclear(r);
    }
    if (!PQexitPipelineMode(pg) && ok) {
      last_error = PQerrorMessage(pg);
      ok = false;
    }
    return ok;
  }
};

// ---------------------------------------------------------------------------------------
// Queued writes
// ---------------------------------------------------------------------------------------
struct JobStore::Op {
  enum class Kind { kInsert, kRunning, kQueued, kFinished } kind;
  uint64_t seq = 0;
  uint64_t id = 0;
  Params params;  // bound to the statement for `kind`
  std::vector<StoredHash> hashes;

  bool send(Conn& c) const {
    switch (kind) {
      case Kind::kInsert:
        return c.send(
            "INSERT INTO mhprep_jobs (id, tenant, input, options, state) "
            "VALUES ($1, $2, $3, $4::jsonb, 'queued')",
            params);
      case Kind::kRunning:
        return c.send(
            "UPDATE mhprep_jobs SET state = 'running', attempts = $2, "
            "started_at = coalesce(started_at, now()) WHERE id = $1",
            params);
      case Kind::kQueued:
        return c.send(
            "UPDATE mhprep_jobs SET state = 'queued', attempts = $2, attempt_log = $3::jsonb "
            "WHERE id = $1",
            params);
      case Kind::kFinished: {
        if (!c.send("UPDATE mhprep_jobs SET state = $2, attempts = $3, attempt_log = $4::jsonb, "
                       "error = $5, error_kind = $6, manifest = $7::jsonb, "
                       "cross_job_duplicates = $8, latency_ms = $9, queue_ms = $10, "
                       "started_at = coalesce(started_at, now()), finished_at = now() "
                       "WHERE id = $1",
                       params))
          return false;
        for (const auto& h : hashes) {
          if (!c.send("INSERT INTO mhprep_asset_hashes (job_id, asset, tenant, phash, csig) "
                         "VALUES ($1, $2, $3, $4, decode($5, 'hex')) ON CONFLICT DO NOTHING",
                         {std::to_string(h.job_id), std::to_string(h.asset), h.tenant,
                          std::to_string(static_cast<int64_t>(h.phash)), sig_hex(h.sig)}))
            return false;
        }
        return true;
      }
    }
    return false;
  }
};

JobStore::JobStore(std::string conninfo)
    : conninfo_(std::move(conninfo)), read_conn_(std::make_unique<Conn>(conninfo_)) {
  if (!read_conn_->connect())
    throw std::runtime_error("job store: cannot connect: " + read_conn_->last_error);
  // Two supervisors starting at once must not race on CREATE ... IF NOT EXISTS.
  Conn& c = *read_conn_;
  if (!c.exec_ok("BEGIN") || !c.exec_ok("SELECT pg_advisory_xact_lock(7745012)") ||
      !c.exec_ok(kSchema) || !c.exec_ok("COMMIT"))
    throw std::runtime_error("job store: cannot create schema: " + c.last_error);
  // One supervisor per database: two would both recover and run the same jobs. The lock
  // is held by this session and released when the connection closes, even on a crash.
  // It lives on its own connection, because the read connection may reconnect.
  lock_conn_ = std::make_unique<Conn>(conninfo_);
  if (!lock_conn_->connect())
    throw std::runtime_error("job store: cannot connect: " + lock_conn_->last_error);
  auto lock = lock_conn_->exec("SELECT pg_try_advisory_lock(7745013)");
  if (!lock || std::string(PQgetvalue(lock.get(), 0, 0)) != "t")
    throw std::runtime_error("job store: another mhprep supervisor is using this database");
  writer_ = std::thread([this] { writer_loop(); });
}

JobStore::~JobStore() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  maint_cv_.notify_all();
  if (maint_.joinable()) maint_.join();
  work_cv_.notify_all();
  if (writer_.joinable()) writer_.join();
}

uint64_t JobStore::prune_with(Conn& c, std::chrono::seconds older_than, size_t batch) {
  uint64_t total = 0;
  for (;;) {
    if ((!c.ok() && !c.connect())) break;
    // Oldest first, a bounded batch per statement; hashes go with their job (CASCADE).
    // The highest id is always kept: max_id() seeds the next id from it on restart, so
    // deleting it could hand out an id (and output directory) that was already used.
    auto r = c.exec(
        "DELETE FROM mhprep_jobs WHERE id IN ("
        "  SELECT id FROM mhprep_jobs"
        "  WHERE state IN ('succeeded', 'failed')"
        "    AND finished_at < now() - make_interval(secs => $1::float8)"
        "    AND id < (SELECT max(id) FROM mhprep_jobs)"
        "  ORDER BY finished_at LIMIT $2)",
        {std::to_string(older_than.count()), std::to_string(batch)});
    if (!r) {
      std::fprintf(stderr, "mhprep: job store retention: %s", c.last_error.c_str());
      break;
    }
    const uint64_t n = std::stoull(PQcmdTuples(r.get()));
    total += n;
    std::lock_guard lk(mu_);
    stats_.pruned += n;
    if (n < batch || stop_) break;
  }
  std::lock_guard lk(mu_);
  ++stats_.prune_passes;
  return total;
}

uint64_t JobStore::prune(std::chrono::seconds older_than, size_t batch) {
  Conn c(conninfo_);
  return prune_with(c, older_than, batch);
}

void JobStore::start_retention(Retention r) {
  if (r.keep.count() <= 0 || maint_.joinable()) return;
  maint_ = std::thread([this, r] {
    Conn c(conninfo_);
    std::unique_lock lk(mu_);
    while (!stop_) {
      lk.unlock();
      prune_with(c, r.keep, r.batch);
      lk.lock();
      maint_cv_.wait_for(lk, r.every, [&] { return stop_; });
    }
  });
}

uint64_t JobStore::max_id() {
  std::lock_guard lk(read_mu_);
  auto r = read_conn_->exec("SELECT coalesce(max(id), 0) FROM mhprep_jobs");
  if (!r) throw std::runtime_error("job store: " + read_conn_->last_error);
  return std::stoull(PQgetvalue(r.get(), 0, 0));
}

std::vector<RecoveredJob> JobStore::load_unfinished() {
  std::lock_guard lk(read_mu_);
  auto r = read_conn_->exec(
      "SELECT id, input, tenant, options::text, attempts, attempt_log::text, state, "
      "extract(epoch FROM submitted_at)::float8 "
      "FROM mhprep_jobs WHERE state IN ('queued', 'running') ORDER BY id");
  if (!r) throw std::runtime_error("job store: " + read_conn_->last_error);
  std::vector<RecoveredJob> out;
  for (int i = 0; i < PQntuples(r.get()); ++i) {
    RecoveredJob j;
    j.id = std::stoull(PQgetvalue(r.get(), i, 0));
    j.input = PQgetvalue(r.get(), i, 1);
    j.tenant = PQgetvalue(r.get(), i, 2);
    j.options = json::parse(PQgetvalue(r.get(), i, 3));
    j.attempts = std::stoi(PQgetvalue(r.get(), i, 4));
    j.attempt_log = json::parse(PQgetvalue(r.get(), i, 5)).get<std::vector<std::string>>();
    j.was_running = std::string(PQgetvalue(r.get(), i, 6)) == "running";
    const double epoch = std::stod(PQgetvalue(r.get(), i, 7));
    j.submitted = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::duration<double>(epoch)));
    if (j.was_running) {
      // The attempt in flight when the supervisor died never reported back. It says
      // nothing about the input, so it doesn't count toward quarantine.
      j.attempt_log.push_back("attempt " + std::to_string(j.attempts) +
                              ": interrupted: supervisor restarted");
      j.attempts = std::max(0, j.attempts - 1);
    }
    out.push_back(std::move(j));
  }
  return out;
}

std::vector<StoredHash> JobStore::load_hashes() {
  std::lock_guard lk(read_mu_);
  auto r = read_conn_->exec(
      "SELECT tenant, job_id, asset, phash, encode(csig, 'hex') FROM mhprep_asset_hashes "
      "ORDER BY job_id, asset");
  if (!r) throw std::runtime_error("job store: " + read_conn_->last_error);
  std::vector<StoredHash> out;
  out.reserve(static_cast<size_t>(PQntuples(r.get())));
  for (int i = 0; i < PQntuples(r.get()); ++i) {
    StoredHash h;
    h.tenant = PQgetvalue(r.get(), i, 0);
    h.job_id = std::stoull(PQgetvalue(r.get(), i, 1));
    h.asset = std::stoi(PQgetvalue(r.get(), i, 2));
    h.phash = static_cast<uint64_t>(std::stoll(PQgetvalue(r.get(), i, 3)));
    h.sig = parse_sig_hex(PQgetvalue(r.get(), i, 4));
    out.push_back(std::move(h));
  }
  return out;
}

std::optional<json> JobStore::describe(uint64_t id, bool include_manifest) {
  std::lock_guard lk(read_mu_);
  if (!read_conn_->ok() && !read_conn_->connect()) return std::nullopt;
  auto r = read_conn_->exec(
      include_manifest
          ? "SELECT input, tenant, state, attempts, attempt_log::text, error, error_kind, "
            "latency_ms, queue_ms, cross_job_duplicates, manifest::text FROM mhprep_jobs "
            "WHERE id = $1"
          : "SELECT input, tenant, state, attempts, attempt_log::text, error, error_kind, "
            "latency_ms, queue_ms, cross_job_duplicates FROM mhprep_jobs WHERE id = $1",
      {std::to_string(id)});
  if (!r || PQntuples(r.get()) == 0) return std::nullopt;
  auto* res = r.get();
  auto val = [&](int c) { return std::string(PQgetvalue(res, 0, c)); };
  auto null = [&](int c) { return PQgetisnull(res, 0, c) != 0; };
  const std::string state = val(2);
  json j{{"id", id}, {"input", val(0)}, {"tenant", val(1)}, {"status", state},
         {"attempts", std::stoi(val(3))}};
  if (json log = json::parse(val(4)); !log.empty()) j["attempt_log"] = log;
  if (state == "failed") {
    j["error"] = val(5);
    j["error_kind"] = val(6);
  }
  if (state == "succeeded" || state == "failed") {
    if (!null(7)) j["latency_ms"] = std::stod(val(7));
    if (!null(8)) j["queue_ms"] = std::stod(val(8));
  }
  if (state == "succeeded") {
    j["cross_job_duplicates"] = std::stoi(val(9));
    if (include_manifest && !null(10)) j["manifest"] = json::parse(val(10));
  }
  return j;
}

// ---------------------------------------------------------------------------------------
// Writes
// ---------------------------------------------------------------------------------------
bool JobStore::insert_durable(const JobRecord& job, std::chrono::milliseconds timeout) {
  std::unique_lock lk(mu_);
  if (stop_) return false;
  const uint64_t seq = next_seq_++;
  ops_.push_back({Op::Kind::kInsert, seq, job.id,
                  {std::to_string(job.id), job.spec.tenant, job.spec.input, job.spec.options.dump()},
                  {}});
  work_cv_.notify_one();
  if (!done_cv_.wait_for(lk, timeout, [&] { return done_seq_ >= seq; })) {
    // Still queued: withdraw it, so a late commit can't create a job the caller was
    // told failed. Already handed to the writer: its transaction is bounded by
    // statement_timeout, so wait for the verdict.
    auto it = std::find_if(ops_.begin(), ops_.end(), [&](const Op& o) { return o.seq == seq; });
    if (it != ops_.end()) {
      ops_.erase(it);
      return false;
    }
    done_cv_.wait(lk, [&] { return done_seq_ >= seq; });
  }
  auto rej = std::find(rejected_.begin(), rejected_.end(), seq);
  if (rej == rejected_.end()) return true;
  rejected_.erase(rej);
  return false;
}

void JobStore::mark_running(uint64_t id, int attempts) {
  std::lock_guard lk(mu_);
  ops_.push_back({Op::Kind::kRunning, next_seq_++, id,
                  {std::to_string(id), std::to_string(attempts)}, {}});
  work_cv_.notify_one();
}

void JobStore::mark_queued(uint64_t id, int attempts, const std::vector<std::string>& log) {
  std::lock_guard lk(mu_);
  ops_.push_back({Op::Kind::kQueued, next_seq_++, id,
                  {std::to_string(id), std::to_string(attempts), log_json(log)}, {}});
  work_cv_.notify_one();
}

void JobStore::mark_finished(const JobRecord& job, std::vector<StoredHash> new_hashes) {
  const bool ok = job.state == JobState::kSucceeded;
  const double queue_ms =
      std::chrono::duration<double, std::milli>(job.started - job.submitted).count();
  Params p{std::to_string(job.id),
           to_string(job.state),
           std::to_string(job.attempts),
           log_json(job.attempt_log),
           ok ? std::nullopt : std::optional<std::string>(job.error),
           ok ? std::nullopt : std::optional<std::string>(job.error_kind),
           ok ? std::optional<std::string>(job.manifest.dump()) : std::nullopt,
           std::to_string(job.cross_job_duplicates),
           std::to_string(job.latency_ms()),
           std::to_string(queue_ms)};
  std::lock_guard lk(mu_);
  ops_.push_back({Op::Kind::kFinished, next_seq_++, job.id, std::move(p), std::move(new_hashes)});
  work_cv_.notify_one();
}

void JobStore::flush() {
  std::unique_lock lk(mu_);
  const uint64_t target = next_seq_ - 1;
  done_cv_.wait(lk, [&] { return done_seq_ >= target; });
}

JobStore::Stats JobStore::stats() const {
  std::lock_guard lk(mu_);
  return stats_;
}

// Runs the whole batch as one pipelined transaction: it commits entirely or not at all.
bool JobStore::apply_batch(Conn& c, std::vector<Op>& batch, bool include_inserts) {
  return c.pipeline([&](Conn& pc) {
    for (const Op& op : batch) {
      if (!include_inserts && op.kind == Op::Kind::kInsert) continue;
      if (!op.send(pc)) return false;
    }
    return true;
  });
}

void JobStore::writer_loop() {
  Conn c(conninfo_);
  int backoff_ms = 50;
  for (;;) {
    std::vector<Op> batch;
    {
      std::unique_lock lk(mu_);
      work_cv_.wait(lk, [&] { return stop_ || !ops_.empty(); });
      if (ops_.empty()) return;  // stop_ and fully drained
      // Group commit: everything queued since the last transaction goes in this one, up
      // to a cap that keeps a blocking-mode pipeline well inside the socket buffers.
      const size_t n = std::min(ops_.size(), kMaxBatch);
      batch.assign(std::make_move_iterator(ops_.begin()),
                   std::make_move_iterator(ops_.begin() + static_cast<std::ptrdiff_t>(n)));
      ops_.erase(ops_.begin(), ops_.begin() + static_cast<std::ptrdiff_t>(n));
    }
    std::vector<uint64_t> rejected;
    bool committed = (c.ok() || c.connect()) && apply_batch(c, batch, true);
    if (!committed) {
      // Inserts fail fast: their callers are waiting and will answer 503, so they must
      // not be retried behind the callers' backs. State changes for jobs that already
      // exist are retried until they land.
      for (const Op& op : batch)
        if (op.kind == Op::Kind::kInsert) rejected.push_back(op.seq);
      int tries = 0;
      while (!committed) {
        if (c.ok()) {
          // The server is up, so something in the batch itself is bad. Apply ops one at a
          // time and drop any that the database refuses, rather than wedging the queue.
          committed = true;
          for (const Op& op : batch) {
            if (op.kind == Op::Kind::kInsert) continue;
            std::vector<Op> one;
            one.push_back(op);
            if (!apply_batch(c, one, false)) {
              if (!c.ok()) {
                committed = false;
                break;
              }
              std::fprintf(stderr, "mhprep: job store dropped a write for job %llu: %s",
                           static_cast<unsigned long long>(op.id), c.last_error.c_str());
            }
          }
          if (committed) break;
        }
        {
          std::lock_guard lk(mu_);
          ++stats_.failed_transactions;
          if (stop_ && ++tries > 3) break;  // shutting down with the database gone
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        backoff_ms = std::min(backoff_ms * 2, 2000);
        if (c.connect()) committed = apply_batch(c, batch, false);
      }
    }
    if (committed) backoff_ms = 50;
    std::lock_guard lk(mu_);
    ++stats_.transactions;
    stats_.writes += batch.size();
    for (const Op& op : batch) done_seq_ = std::max(done_seq_, op.seq);
    rejected_.insert(rejected_.end(), rejected.begin(), rejected.end());
    done_cv_.notify_all();
  }
}

}  // namespace mhprep
