// HTTP front end. cpp-httplib's thread pool parses requests; all job state lives in the
// Supervisor, whose event loop runs on the main thread.
#include "server.h"

#include <httplib.h>

#include <sstream>

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

}  // namespace

struct HttpServer::Impl {
  httplib::Server srv;
};

HttpServer::HttpServer(Supervisor& sup) : sup_(sup), impl_(std::make_unique<Impl>()) {
  auto& srv = impl_->srv;
  // httplib binds a keep-alive connection to one pool thread, so size the pool for the
  // expected number of concurrent client connections, not for CPU count.
  srv.new_task_queue = [] { return new httplib::ThreadPool(64); };
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
      counter("mhprep_store_pruned_total", st->pruned,
              "Finished jobs deleted by retention");
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
