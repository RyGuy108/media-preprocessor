#include "mhprep/supervisor.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <filesystem>
#include <spawn.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/timerfd.h>
#include <sys/wait.h>
#include <unistd.h>

#include "mhprep/ipc.h"
#include "mhprep/pipeline.h"

extern char** environ;

namespace mhprep {

using Clock = std::chrono::steady_clock;
using nlohmann::json;

namespace {

void check(bool ok, const char* what) {
  if (!ok) throw std::system_error(errno, std::generic_category(), what);
}

timespec to_timespec(Clock::time_point tp) {
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
  timespec ts{};
  ts.tv_sec = std::max<int64_t>(0, ns / 1'000'000'000);
  ts.tv_nsec = ns < 0 ? 1 : ns % 1'000'000'000;
  if (ts.tv_sec == 0 && ts.tv_nsec == 0) ts.tv_nsec = 1;  // 0 would disarm the timer
  return ts;
}

std::string describe_status(int status) {
  if (WIFSIGNALED(status)) {
    const int sig = WTERMSIG(status);
    const char* abbrev = sigabbrev_np(sig);
    return std::string("worker killed by SIG") + (abbrev ? abbrev : std::to_string(sig).c_str());
  }
  if (WIFEXITED(status)) return "worker exited with code " + std::to_string(WEXITSTATUS(status));
  return "worker ended (status " + std::to_string(status) + ")";
}

sigset_t loop_signals(bool term) {
  sigset_t s;
  sigemptyset(&s);
  sigaddset(&s, SIGCHLD);
  sigaddset(&s, SIGPIPE);
  if (term) {
    sigaddset(&s, SIGTERM);
    sigaddset(&s, SIGINT);
  }
  return s;
}

}  // namespace

const char* to_string(JobState s) {
  switch (s) {
    case JobState::kQueued: return "queued";
    case JobState::kRunning: return "running";
    case JobState::kSucceeded: return "succeeded";
    case JobState::kFailed: return "failed";
  }
  return "?";
}

double JobRecord::latency_ms() const {
  if (state != JobState::kSucceeded && state != JobState::kFailed) return 0;
  return std::chrono::duration<double, std::milli>(finished - submitted).count();
}

json JobRecord::to_json(bool include_manifest) const {
  json j{{"id", id},
         {"input", spec.input},
         {"tenant", spec.tenant},
         {"status", mhprep::to_string(state)},
         {"attempts", attempts}};
  if (!attempt_log.empty()) j["attempt_log"] = attempt_log;
  if (state == JobState::kFailed) {
    j["error"] = error;
    j["error_kind"] = error_kind;
  }
  if (state == JobState::kSucceeded || state == JobState::kFailed) {
    j["latency_ms"] = latency_ms();
    if (started != Clock::time_point{})
      j["queue_ms"] = std::chrono::duration<double, std::milli>(started - submitted).count();
  }
  if (state == JobState::kSucceeded) {
    j["cross_job_duplicates"] = cross_job_duplicates;
    if (include_manifest) j["manifest"] = manifest;
  }
  return j;
}

struct Supervisor::Worker {
  enum class State { kStarting, kIdle, kBusy };
  pid_t pid = -1;
  std::unique_ptr<FramedConn> conn;
  State state = State::kStarting;
  uint64_t job = 0;
  Clock::time_point deadline;
  int64_t timeout_ms = 0;
  bool killed_for_timeout = false;
  bool retiring = false;
  bool eof = false;
  bool in_epoll = false;
};

Supervisor::Supervisor(PoolConfig cfg)
    : cfg_(std::move(cfg)),
      latency_({50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000, 300000}) {
  sigset_t s = loop_signals(cfg_.handle_term_signals);
  pthread_sigmask(SIG_BLOCK, &s, nullptr);
  epfd_ = epoll_create1(EPOLL_CLOEXEC);
  check(epfd_ >= 0, "epoll_create1");
  sigset_t fdset;
  sigemptyset(&fdset);
  sigaddset(&fdset, SIGCHLD);
  if (cfg_.handle_term_signals) {
    sigaddset(&fdset, SIGTERM);
    sigaddset(&fdset, SIGINT);
  }
  sigfd_ = signalfd(-1, &fdset, SFD_NONBLOCK | SFD_CLOEXEC);
  check(sigfd_ >= 0, "signalfd");
  timerfd_ = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK | TFD_CLOEXEC);
  check(timerfd_ >= 0, "timerfd_create");
  evfd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  check(evfd_ >= 0, "eventfd");
  epoll_mod(sigfd_, EPOLLIN, true);
  epoll_mod(timerfd_, EPOLLIN, true);
  epoll_mod(evfd_, EPOLLIN, true);
}

Supervisor::~Supervisor() {
  for (auto& [fd, w] : workers_) {
    if (w->pid > 0) {
      ::kill(w->pid, SIGKILL);
      ::waitpid(w->pid, nullptr, 0);
    }
  }
  workers_.clear();
  for (int fd : {epfd_, sigfd_, timerfd_, evfd_})
    if (fd >= 0) ::close(fd);
}

void Supervisor::block_signals() {
  sigset_t s = loop_signals(true);
  pthread_sigmask(SIG_BLOCK, &s, nullptr);
}

void Supervisor::epoll_mod(int fd, uint32_t events, bool add) {
  epoll_event ev{};
  ev.events = events;
  ev.data.fd = fd;
  check(epoll_ctl(epfd_, add ? EPOLL_CTL_ADD : EPOLL_CTL_MOD, fd, &ev) == 0, "epoll_ctl");
}

// ---------------------------------------------------------------------------------------
// Public, thread-safe API
// ---------------------------------------------------------------------------------------
std::optional<uint64_t> Supervisor::submit(JobSpec spec) {
  uint64_t id;
  {
    std::lock_guard lk(mu_);
    if (stop_requested_ || queue_.size() >= cfg_.max_queue) return std::nullopt;
    id = next_id_++;
    JobRecord r;
    r.id = id;
    r.spec = std::move(spec);
    r.submitted = Clock::now();
    jobs_.emplace(id, std::move(r));
    queue_.push_back(id);
    ++unfinished_;
    ++stats_.submitted;
  }
  const uint64_t one = 1;
  [[maybe_unused]] auto n = ::write(evfd_, &one, sizeof one);
  return id;
}

std::optional<JobRecord> Supervisor::get(uint64_t id) const {
  std::lock_guard lk(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return std::nullopt;
  return it->second;
}

std::optional<json> Supervisor::describe(uint64_t id, bool include_manifest) const {
  std::lock_guard lk(mu_);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return std::nullopt;
  return it->second.to_json(include_manifest);
}

bool Supervisor::wait_done(uint64_t id, std::chrono::milliseconds timeout) const {
  std::unique_lock lk(mu_);
  return cv_.wait_for(lk, timeout, [&] {
    auto it = jobs_.find(id);
    return it == jobs_.end() || it->second.state == JobState::kSucceeded ||
           it->second.state == JobState::kFailed;
  });
}

std::optional<JobRecord> Supervisor::wait(uint64_t id, std::chrono::milliseconds timeout) const {
  std::unique_lock lk(mu_);
  auto done = [&] {
    auto it = jobs_.find(id);
    return it == jobs_.end() || it->second.state == JobState::kSucceeded ||
           it->second.state == JobState::kFailed;
  };
  cv_.wait_for(lk, timeout, done);
  auto it = jobs_.find(id);
  if (it == jobs_.end()) return std::nullopt;
  return it->second;
}

PoolStats Supervisor::stats() const {
  std::lock_guard lk(mu_);
  PoolStats s = stats_;
  s.queued = queue_.size();
  return s;
}

void Supervisor::request_stop() {
  {
    std::lock_guard lk(mu_);
    stop_requested_ = true;
  }
  const uint64_t one = 1;
  [[maybe_unused]] auto n = ::write(evfd_, &one, sizeof one);
}

// ---------------------------------------------------------------------------------------
// Workers
// ---------------------------------------------------------------------------------------
void Supervisor::spawn_worker() {
  int sv[2];
  check(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) == 0, "socketpair");
  int child_fd = sv[1];
  if (child_fd == 3) {  // dup2(3, 3) would not clear FD_CLOEXEC
    child_fd = fcntl(sv[1], F_DUPFD_CLOEXEC, 4);
    ::close(sv[1]);
  }
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, child_fd, 3);
  posix_spawnattr_t at;
  posix_spawnattr_init(&at);
  sigset_t empty, defaults;
  sigemptyset(&empty);
  sigemptyset(&defaults);
  for (int s : {SIGPIPE, SIGCHLD, SIGTERM, SIGINT}) sigaddset(&defaults, s);
  posix_spawnattr_setsigmask(&at, &empty);
  posix_spawnattr_setsigdefault(&at, &defaults);
  // Own process group: a terminal Ctrl-C reaches the supervisor, which drains, instead
  // of killing every worker mid-job.
  posix_spawnattr_setpgroup(&at, 0);
  posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF |
                                    POSIX_SPAWN_SETPGROUP);
  const std::string mem = std::to_string(cfg_.worker_mem_mb);
  std::vector<std::string> args = {cfg_.worker_bin, "worker",   "--fd", "3",
                                   "--models",      cfg_.models_dir, "--mem-mb", mem};
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = -1;
  const int rc = posix_spawn(&pid, cfg_.worker_bin.c_str(), &fa, &at, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  posix_spawnattr_destroy(&at);
  ::close(child_fd);
  if (rc != 0) {
    ::close(sv[0]);
    errno = rc;
    check(false, "posix_spawn");
  }
  fcntl(sv[0], F_SETFL, fcntl(sv[0], F_GETFL) | O_NONBLOCK);
  auto w = std::make_unique<Worker>();
  w->pid = pid;
  w->conn = std::make_unique<FramedConn>(sv[0]);
  epoll_mod(sv[0], EPOLLIN | EPOLLRDHUP, true);
  w->in_epoll = true;
  pid_to_fd_[pid] = sv[0];
  workers_[sv[0]] = std::move(w);
  std::lock_guard lk(mu_);
  ++stats_.worker_spawns;
  stats_.workers_alive = workers_.size();
}

void Supervisor::dispatch() {
  if (stopping_) return;
  for (auto& [fd, wp] : workers_) {
    Worker& w = *wp;
    if (w.state != Worker::State::kIdle || w.eof || w.retiring) continue;
    json msg;
    std::chrono::milliseconds timeout = cfg_.job_timeout;
    {
      std::lock_guard lk(mu_);
      if (queue_.empty()) return;
      const uint64_t id = queue_.front();
      queue_.pop_front();
      JobRecord& job = jobs_.at(id);
      job.state = JobState::kRunning;
      ++job.attempts;
      if (job.attempts == 1) job.started = Clock::now();
      ++stats_.running;
      json opts = cfg_.default_options;
      opts.merge_patch(job.spec.options);
      if (opts.contains("timeout_ms"))
        timeout = std::chrono::milliseconds(opts["timeout_ms"].get<int64_t>());
      const auto out = std::filesystem::path(cfg_.out_root) / job.spec.tenant / std::to_string(id);
      msg = {{"type", "job"},     {"id", id},           {"attempt", job.attempts},
             {"input", job.spec.input}, {"out", out.string()}, {"options", opts}};
      w.job = id;
    }
    w.state = Worker::State::kBusy;
    w.deadline = Clock::now() + timeout;
    w.timeout_ms = timeout.count();
    w.killed_for_timeout = false;
    w.conn->queue(msg);
    if (!w.conn->flush()) continue;  // dead socket; SIGCHLD will clean up
    if (w.conn->want_write()) epoll_mod(fd, EPOLLIN | EPOLLOUT | EPOLLRDHUP, false);
  }
}

void Supervisor::handle_worker_io(Worker& w, uint32_t events) {
  if (events & EPOLLOUT) {
    if (w.conn->flush() && !w.conn->want_write() && !w.eof)
      epoll_mod(w.conn->fd(), EPOLLIN | EPOLLRDHUP, false);
  }
  if (events & (EPOLLIN | EPOLLRDHUP | EPOLLHUP | EPOLLERR)) {
    std::vector<json> msgs;
    const bool open = w.conn->read_available(msgs);
    for (const auto& m : msgs) handle_message(w, m);
    if (!open && !w.eof) {
      // Peer closed. Stop watching the socket; the exit status arrives via SIGCHLD.
      w.eof = true;
      epoll_ctl(epfd_, EPOLL_CTL_DEL, w.conn->fd(), nullptr);
      w.in_epoll = false;
    }
  }
}

void Supervisor::handle_message(Worker& w, const json& msg) {
  const std::string type = msg.value("type", "");
  if (type == "hello") {
    if (w.state == Worker::State::kStarting) w.state = Worker::State::kIdle;
    startup_failures_ = 0;
    return;
  }
  if (type != "result" || w.state != Worker::State::kBusy) return;
  if (msg.value("id", uint64_t{0}) != w.job) return;  // stale (job already timed out)
  const uint64_t id = w.job;
  w.state = Worker::State::kIdle;
  w.job = 0;
  if (msg.value("worker_exiting", false)) w.retiring = true;  // about to exit; no more jobs
  std::lock_guard lk(mu_);
  --stats_.running;
  JobRecord& job = jobs_.at(id);
  if (msg.value("ok", false)) {
    job.manifest = msg["manifest"];
    if (cfg_.cross_job_dedup) apply_cross_dedup(job);
    finish(job, JobState::kSucceeded);
    return;
  }
  const std::string kind = msg.value("error_kind", "internal");
  const std::string why = msg.value("error", "unknown error");
  if (kind == "input") {
    ++stats_.input_errors;
    job.error = why;
    job.error_kind = kind;
    job.attempt_log.push_back("attempt " + std::to_string(job.attempts) + ": input: " + why);
    finish(job, JobState::kFailed);
  } else {
    mu_.unlock();
    fail_or_retry(id, kind, why);
    mu_.lock();
  }
}

void Supervisor::fail_or_retry(uint64_t job_id, const std::string& kind, const std::string& why) {
  std::lock_guard lk(mu_);
  JobRecord& job = jobs_.at(job_id);
  job.attempt_log.push_back("attempt " + std::to_string(job.attempts) + ": " + kind + ": " + why);
  if (kind == "timeout") ++stats_.timeouts;
  else if (kind == "crash") ++stats_.crashes;
  else if (kind == "oom") ++stats_.oom;
  if (job.attempts < cfg_.max_attempts && !stopping_) {
    job.state = JobState::kQueued;
    queue_.push_back(job_id);  // back of the line: a poison job can't monopolise workers
    ++stats_.retries;
    return;
  }
  job.error_kind = stopping_ ? "cancelled" : kind;
  job.error = why;
  if (job.attempts > 1 && !stopping_)
    job.error += " (quarantined after " + std::to_string(job.attempts) + " attempts)";
  finish(job, JobState::kFailed);
}

// Requires mu_ held.
void Supervisor::finish(JobRecord& job, JobState st) {
  job.state = st;
  job.finished = Clock::now();
  if (job.started == Clock::time_point{}) job.started = job.finished;
  --unfinished_;
  if (st == JobState::kSucceeded) ++stats_.succeeded;
  else ++stats_.failed;
  latency_.record(job.latency_ms());
  cv_.notify_all();
  if (on_finished) on_finished(job);
  finished_order_.push_back(job.id);
  while (finished_order_.size() > cfg_.max_finished_jobs) {  // bounded memory
    jobs_.erase(finished_order_.front());
    finished_order_.pop_front();
  }
}

// Requires mu_ held.
void Supervisor::apply_cross_dedup(JobRecord& job) {
  auto& assets = job.manifest["assets"];
  auto [tit, _] = tenant_hashes_.try_emplace(job.spec.tenant, cfg_.dup_radius, cfg_.color_tolerance);
  TenantIndex& t = tit->second;
  int dups = 0;
  for (size_t i = 0; i < assets.size(); ++i) {
    auto& a = assets[i];
    const std::string kind = a.value("kind", "");
    if ((kind != "keyframe" && kind != "image") || a.contains("duplicate_of")) continue;
    const uint64_t h = parse_hash_hex(a["phash"].get<std::string>());
    const ColorSig sig = parse_sig_hex(a.value("csig", std::string()));
    if (auto hit = t.index.find(h, sig)) {
      const auto& [owner_job, owner_asset] = t.owners[static_cast<size_t>(hit->id)];
      a["cross_duplicate_of"] = {{"job", owner_job}, {"asset", owner_asset},
                                 {"distance", hit->distance}};
      ++dups;
    } else {
      t.index.add(h, sig, static_cast<int64_t>(t.owners.size()));
      t.owners.emplace_back(job.id, static_cast<int>(i));
    }
  }
  job.cross_job_duplicates = dups;
  job.manifest["summary"]["assets_cross_job_duplicate"] = dups;
  stats_.cross_job_duplicates += static_cast<uint64_t>(dups);
}

void Supervisor::reap_children() {
  for (;;) {
    int status = 0;
    const pid_t pid = ::waitpid(-1, &status, WNOHANG);
    if (pid <= 0) return;
    auto it = pid_to_fd_.find(pid);
    if (it == pid_to_fd_.end()) continue;  // not one of ours
    const int fd = it->second;
    pid_to_fd_.erase(it);
    auto wit = workers_.find(fd);
    if (wit == workers_.end()) continue;
    on_worker_exit(*wit->second, status);
    workers_.erase(wit);
    std::lock_guard lk(mu_);
    stats_.workers_alive = workers_.size();
  }
}

void Supervisor::on_worker_exit(Worker& w, int status) {
  // A worker may send its result and exit right away (e.g. after an OOM): read it first.
  if (!w.eof) {
    std::vector<json> msgs;
    w.conn->read_available(msgs);
    for (const auto& m : msgs) handle_message(w, m);
  }
  if (w.in_epoll) epoll_ctl(epfd_, EPOLL_CTL_DEL, w.conn->fd(), nullptr);
  w.conn->close();
  const bool was_starting = w.state == Worker::State::kStarting;
  if (w.state == Worker::State::kBusy) {
    {
      std::lock_guard lk(mu_);
      --stats_.running;
      ++stats_.worker_deaths;
    }
    if (w.killed_for_timeout) {
      fail_or_retry(w.job, "timeout",
                    "no result within " + std::to_string(w.timeout_ms) +
                        " ms; worker SIGKILLed");
    } else {
      fail_or_retry(w.job, "crash", describe_status(status));
    }
  } else if (!stopping_) {
    std::lock_guard lk(mu_);
    ++stats_.worker_deaths;
  }
  if (stopping_) return;
  if (was_starting) ++startup_failures_;
  // Exponential backoff if workers die during startup (bad model path, broken binary).
  const int n = std::min(startup_failures_, 6);
  pending_spawns_.push_back(Clock::now() + std::chrono::milliseconds(n ? 100 << n : 0));
}

void Supervisor::check_deadlines() {
  const auto now = Clock::now();
  for (auto& [fd, w] : workers_) {
    if (w->state == Worker::State::kBusy && !w->killed_for_timeout && w->deadline <= now) {
      w->killed_for_timeout = true;
      ::kill(w->pid, SIGKILL);  // a wedged decoder can't be cancelled cooperatively
    }
  }
  if (stopping_ && now >= stop_deadline_) {
    for (auto& [fd, w] : workers_)
      if (w->state == Worker::State::kBusy) ::kill(w->pid, SIGKILL);
  }
}

void Supervisor::arm_timer() {
  std::optional<Clock::time_point> next;
  auto consider = [&](Clock::time_point t) {
    if (!next || t < *next) next = t;
  };
  for (auto& [fd, w] : workers_)
    if (w->state == Worker::State::kBusy && !w->killed_for_timeout) consider(w->deadline);
  for (auto t : pending_spawns_) consider(t);
  if (stopping_) consider(stop_deadline_);
  itimerspec its{};
  if (next) its.it_value = to_timespec(*next);
  timerfd_settime(timerfd_, TFD_TIMER_ABSTIME, &its, nullptr);
}

void Supervisor::begin_stop() {
  if (stopping_) return;
  stopping_ = true;
  stop_deadline_ = Clock::now() + cfg_.drain_timeout;
  pending_spawns_.clear();
  {
    std::lock_guard lk(mu_);
    stop_requested_ = true;
    for (uint64_t id : queue_) {
      JobRecord& job = jobs_.at(id);
      job.error = "service shutting down before the job started";
      job.error_kind = "cancelled";
      finish(job, JobState::kFailed);
    }
    queue_.clear();
  }
  if (on_stop) on_stop();
}

bool Supervisor::all_done() const {
  std::lock_guard lk(mu_);
  return unfinished_ == 0;
}

void Supervisor::run(bool exit_when_idle) {
  for (int i = static_cast<int>(workers_.size()); i < cfg_.workers; ++i) spawn_worker();
  std::vector<epoll_event> events(64);
  for (;;) {
    // Spawn replacements whose backoff has elapsed.
    const auto now = Clock::now();
    while (!pending_spawns_.empty() && pending_spawns_.front() <= now) {
      pending_spawns_.pop_front();
      spawn_worker();
    }
    dispatch();
    if (exit_when_idle && all_done()) break;
    if (stopping_) {
      bool busy = false;
      for (auto& [fd, w] : workers_) busy |= w->state == Worker::State::kBusy;
      if (!busy) break;
    }
    arm_timer();
    const int n = epoll_wait(epfd_, events.data(), static_cast<int>(events.size()), -1);
    if (n < 0) {
      if (errno == EINTR) continue;
      check(false, "epoll_wait");
    }
    for (int i = 0; i < n; ++i) {
      const int fd = events[static_cast<size_t>(i)].data.fd;
      const uint32_t ev = events[static_cast<size_t>(i)].events;
      if (fd == sigfd_) {
        signalfd_siginfo si;
        while (::read(sigfd_, &si, sizeof si) == sizeof si) {
          if (si.ssi_signo == SIGCHLD) reap_children();
          else if (si.ssi_signo == SIGTERM || si.ssi_signo == SIGINT) begin_stop();
        }
        reap_children();  // SIGCHLD coalesces; reaping is idempotent
      } else if (fd == timerfd_) {
        uint64_t expirations;
        [[maybe_unused]] auto r = ::read(timerfd_, &expirations, sizeof expirations);
        check_deadlines();
      } else if (fd == evfd_) {
        uint64_t v;
        [[maybe_unused]] auto r = ::read(evfd_, &v, sizeof v);
        bool stop;
        {
          std::lock_guard lk(mu_);
          stop = stop_requested_;
        }
        if (stop) begin_stop();
      } else if (auto it = workers_.find(fd); it != workers_.end()) {
        handle_worker_io(*it->second, ev);
      }
    }
  }

  // Shutdown: closing the socket tells idle workers to exit; give them a moment, then kill.
  for (auto& [fd, w] : workers_) {
    if (w->in_epoll) epoll_ctl(epfd_, EPOLL_CTL_DEL, fd, nullptr);
    w->in_epoll = false;
    w->conn->close();
  }
  const auto deadline = Clock::now() + std::chrono::seconds(2);
  while (!workers_.empty()) {
    for (auto it = workers_.begin(); it != workers_.end();) {
      if (::waitpid(it->second->pid, nullptr, WNOHANG) == it->second->pid) {
        pid_to_fd_.erase(it->second->pid);
        it = workers_.erase(it);
      } else {
        ++it;
      }
    }
    if (workers_.empty()) break;
    if (Clock::now() > deadline) {
      for (auto& [fd, w] : workers_) {
        ::kill(w->pid, SIGKILL);
        ::waitpid(w->pid, nullptr, 0);
      }
      workers_.clear();
      pid_to_fd_.clear();
      break;
    }
    // Wait for the next SIGCHLD (or 50 ms) rather than spinning.
    pollfd p{sigfd_, POLLIN, 0};
    ::poll(&p, 1, 50);
    signalfd_siginfo si;
    while (::read(sigfd_, &si, sizeof si) == sizeof si) {
    }
  }
  pending_spawns_.clear();
  std::lock_guard lk(mu_);
  stats_.workers_alive = 0;
}

}  // namespace mhprep
