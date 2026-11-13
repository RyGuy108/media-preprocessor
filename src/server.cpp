// HTTP front end. cpp-httplib's thread pool parses requests; all job state lives in the
// Supervisor, whose event loop runs on the main thread.
#include "server.h"

#include <httplib.h>

#include <arpa/inet.h>
#include <dirent.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <sstream>
#include <thread>

namespace mhprep {

using nlohmann::json;

namespace {

void reply_json(httplib::Response& res, int status, const json& body) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

std::optional<JobSpec> parse_spec(const json& j, std::string& err) {
  if (!j.is_object() || !j.contains("input") || !j["input"].is_string()) {
    err = "body must be a JSON object with a string \"input\"";
    return std::nullopt;
  }
  JobSpec s;
  s.input = j["input"].get<std::string>();
  s.tenant = j.value("tenant", std::string("default"));
  if (s.tenant.empty() || s.tenant.find('/') != std::string::npos || s.tenant == "..") {
    err = "invalid tenant";
    return std::nullopt;
  }
  if (j.contains("options")) s.options = j["options"];
  return s;
}

// cpp-httplib serves a keep-alive connection on one pool thread until the connection
// closes, so a fixed pool of N threads serves at most N connections; connection N+1 sits
// in the queue until some other client disconnects (or hits keep_alive_max_count). That
// was the old p99.9 tail: with 72 clients on 64 threads, p99.9 was 0.6 s and some requests
// timed out. This pool never makes a connection wait for a thread: when none is idle it
// starts one (up to a hard cap), and threads above the floor exit after idling.
class ElasticPool : public httplib::TaskQueue {
 public:
  ElasticPool(size_t min_threads, size_t max_threads, std::chrono::seconds idle_exit)
      : min_(min_threads), max_(max_threads), idle_exit_(idle_exit) {
    std::lock_guard lk(mu_);
    for (size_t i = 0; i < min_; ++i) spawn_locked();
  }
  ~ElasticPool() override { shutdown(); }

  void enqueue(std::function<void()> fn) override {
    std::lock_guard lk(mu_);
    jobs_.push_back(std::move(fn));
    // Spawn when queued work outnumbers idle threads, so nothing waits behind a
    // long-lived connection. Beyond max_, work queues (bounded memory under a flood).
    if (jobs_.size() > idle_ && threads_ < max_) spawn_locked();
    cv_.notify_one();
  }

  void shutdown() override {
    std::vector<std::thread> joinable;
    {
      std::lock_guard lk(mu_);
      if (stop_) return;
      stop_ = true;
      joinable.swap(all_);
    }
    cv_.notify_all();
    for (auto& t : joinable) t.join();
  }

 private:
  void spawn_locked() {
    // Join threads that retired since the last spawn. Holding mu_ here means each has
    // already released it for the last time and is just returning.
    for (auto it = all_.begin(); it != all_.end();) {
      if (std::find(retired_.begin(), retired_.end(), it->get_id()) != retired_.end()) {
        it->join();
        it = all_.erase(it);
      } else {
        ++it;
      }
    }
    retired_.clear();
    ++threads_;
    all_.emplace_back([this] { run(); });
  }

  void run() {
    std::unique_lock lk(mu_);
    for (;;) {
      ++idle_;
      const bool woke = cv_.wait_for(lk, idle_exit_, [&] { return stop_ || !jobs_.empty(); });
      --idle_;
      if (stop_ && jobs_.empty()) break;
      if (!woke) {
        if (threads_ > min_) break;  // idle too long and above the floor: retire
        continue;
      }
      auto fn = std::move(jobs_.front());
      jobs_.pop_front();
      lk.unlock();
      fn();
      lk.lock();
    }
    --threads_;
    retired_.push_back(std::this_thread::get_id());  // joined by the next spawn or shutdown
  }

  const size_t min_, max_;
  const std::chrono::seconds idle_exit_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> jobs_;
  std::vector<std::thread> all_;
  std::vector<std::thread::id> retired_;
  size_t threads_ = 0, idle_ = 0;
  bool stop_ = false;
};

// cpp-httplib (as packaged, a compiled library) listens with a backlog of 5. When many
// clients connect at once, the accept queue overflows, the kernel drops SYNs, and each
// dropped client waits out the 1 s SYN retransmit timeout. httplib has no option for it
// and doesn't expose the socket, so find our listening socket and call listen() again,
// which on Linux updates the backlog of a socket that is already listening.
bool raise_listen_backlog(int port, int backlog) {
  DIR* d = ::opendir("/proc/self/fd");
  if (!d) return false;
  bool done = false;
  while (dirent* e = ::readdir(d)) {
    const int fd = std::atoi(e->d_name);
    struct stat st{};
    if (fd <= 2 || ::fstat(fd, &st) != 0 || !S_ISSOCK(st.st_mode)) continue;
    int listening = 0;
    socklen_t len = sizeof listening;
    if (::getsockopt(fd, SOL_SOCKET, SO_ACCEPTCONN, &listening, &len) != 0 || !listening) continue;
    sockaddr_storage addr{};
    socklen_t alen = sizeof addr;
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &alen) != 0) continue;
    int p = -1;
    if (addr.ss_family == AF_INET) p = ntohs(reinterpret_cast<sockaddr_in*>(&addr)->sin_port);
    if (addr.ss_family == AF_INET6) p = ntohs(reinterpret_cast<sockaddr_in6*>(&addr)->sin6_port);
    if (p == port && ::listen(fd, backlog) == 0) done = true;
  }
  ::closedir(d);
  return done;
}

}  // namespace

struct HttpServer::Impl {
  httplib::Server srv;
};

HttpServer::HttpServer(Supervisor& sup) : sup_(sup), impl_(std::make_unique<Impl>()) {
  auto& srv = impl_->srv;
  // One thread per open connection (httplib's model), grown on demand; see ElasticPool.
  srv.new_task_queue = [] {
    return new ElasticPool(kMinHttpThreads, kMaxHttpThreads, std::chrono::seconds(30));
  };
  srv.set_keep_alive_max_count(10000);
  srv.set_keep_alive_timeout(5);  // bounds how long a graceful stop waits on idle clients
  srv.set_tcp_nodelay(true);

  srv.Post("/v1/jobs", [this](const httplib::Request& req, httplib::Response& res) {
    const json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded()) return reply_json(res, 400, {{"error", "invalid JSON"}});
    // Batch form: {"inputs": [...], "tenant": ..., "options": ...}
    std::vector<JobSpec> specs;
    std::string err;
    if (body.is_object() && body.contains("inputs") && body["inputs"].is_array()) {
      for (const auto& in : body["inputs"]) {
        json one = body;
        one.erase("inputs");
        one["input"] = in;
        auto s = parse_spec(one, err);
        if (!s) return reply_json(res, 400, {{"error", err}});
        specs.push_back(std::move(*s));
      }
    } else {
      auto s = parse_spec(body, err);
      if (!s) return reply_json(res, 400, {{"error", err}});
      specs.push_back(std::move(*s));
    }
    json ids = json::array();
    for (auto& s : specs) {
      std::string why;
      auto id = sup_.submit(std::move(s), &why);
      if (!id) {
        res.set_header("Retry-After", "1");
        if (why == "store")  // could not be made durable; nothing was recorded
          return reply_json(res, 503, {{"error", "job store unavailable"}, {"accepted", ids}});
        return reply_json(res, 429, {{"error", "queue full or shutting down"}, {"accepted", ids}});
      }
      ids.push_back(*id);
    }
    if (ids.size() == 1)
      return reply_json(res, 202, {{"id", ids[0]}, {"status", "queued"},
                                   {"url", "/v1/jobs/" + std::to_string(ids[0].get<uint64_t>())}});
    reply_json(res, 202, {{"ids", ids}, {"status", "queued"}});
  });

  srv.Get(R"(/v1/jobs/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
    const uint64_t id = std::stoull(req.matches[1]);
    const bool with_manifest = req.get_param_value("manifest") != "0";
    // Long-poll: ?wait_ms=N returns as soon as the job finishes (or after N ms).
    if (req.has_param("wait_ms")) {
      const auto ms = std::clamp<int64_t>(std::stoll(req.get_param_value("wait_ms")), 0, 300000);
      sup_.wait_done(id, std::chrono::milliseconds(ms));
    }
    auto rec = sup_.describe(id, with_manifest);
    if (!rec) return reply_json(res, 404, {{"error", "no such job"}});
    reply_json(res, 200, *rec);
  });

  srv.Get("/healthz", [this](const httplib::Request&, httplib::Response& res) {
    const PoolStats s = sup_.stats();
    reply_json(res, s.workers_alive > 0 ? 200 : 503,
               {{"workers_alive", s.workers_alive}, {"queued", s.queued}, {"running", s.running}});
  });

  srv.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
    const PoolStats s = sup_.stats();
    std::ostringstream o;
    auto counter = [&](const char* name, uint64_t v, const char* help) {
      o << "# HELP " << name << " " << help << "\n# TYPE " << name << " counter\n"
        << name << " " << v << "\n";
    };
    auto gauge = [&](const char* name, uint64_t v) {
      o << "# TYPE " << name << " gauge\n" << name << " " << v << "\n";
    };
    counter("mhprep_jobs_submitted_total", s.submitted, "Jobs accepted");
    counter("mhprep_jobs_succeeded_total", s.succeeded, "Jobs finished successfully");
    counter("mhprep_jobs_failed_total", s.failed, "Jobs that failed permanently");
    counter("mhprep_job_retries_total", s.retries, "Attempts retried on a fresh worker");
    counter("mhprep_job_timeouts_total", s.timeouts, "Attempts killed at their deadline");
    counter("mhprep_worker_crashes_total", s.crashes, "Attempts lost to a worker crash");
    counter("mhprep_worker_oom_total", s.oom, "Attempts that hit the worker memory limit");
    counter("mhprep_input_errors_total", s.input_errors, "Unreadable or invalid inputs");
    counter("mhprep_worker_spawns_total", s.worker_spawns, "Worker processes started");
    counter("mhprep_cross_job_duplicates_total", s.cross_job_duplicates,
            "Assets matching an earlier upload from the same tenant");
    if (auto st = sup_.store_stats()) {
      counter("mhprep_jobs_recovered_total", s.recovered, "Unfinished jobs resumed at startup");
      counter("mhprep_store_rejects_total", s.store_rejects,
              "Submits refused because the job could not be made durable");
      counter("mhprep_store_transactions_total", st->transactions,
              "Job store transactions (one per group commit)");
      counter("mhprep_store_writes_total", st->writes, "Job store writes");
      counter("mhprep_store_failed_transactions_total", st->failed_transactions,
              "Job store transactions that failed and were retried");
    }
    gauge("mhprep_queue_depth", s.queued);
    gauge("mhprep_jobs_running", s.running);
    gauge("mhprep_workers_alive", s.workers_alive);
    o << "# TYPE mhprep_job_latency_seconds histogram\n"
      << sup_.latency().prometheus("mhprep_job_latency_seconds");
    res.set_content(o.str(), "text/plain; version=0.0.4");
  });
}

HttpServer::~HttpServer() { stop(); }

bool HttpServer::start(const std::string& host, int port) {
  if (!impl_->srv.bind_to_port(host, port)) return false;
  if (!raise_listen_backlog(port, kListenBacklog))
    std::fprintf(stderr, "mhprep: could not raise the listen backlog; expect SYN drops under load\n");
  thread_ = std::thread([this] { impl_->srv.listen_after_bind(); });
  return true;
}

void HttpServer::shutdown() { impl_->srv.stop(); }  // non-blocking, any thread

void HttpServer::stop() {
  impl_->srv.stop();
  std::lock_guard lk(join_mu_);  // only one caller may join
  if (thread_.joinable()) thread_.join();
}

}  // namespace mhprep
