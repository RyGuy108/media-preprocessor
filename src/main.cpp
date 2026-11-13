// mhprep: media preprocessing service for the Magic Hour API.
//
//   mhprep run     [pool flags] [--report FILE] INPUT...|--list FILE   batch mode
//   mhprep serve   [pool flags] [--host H] [--port P]                  HTTP service
//   mhprep process [--options JSON] [--out DIR] INPUT                  single job, in-process
//   mhprep worker  --fd N --models DIR --mem-mb N                      (spawned by the pool)
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/core/utility.hpp>

#include "mhprep/pipeline.h"
#include "mhprep/supervisor.h"
#include "server.h"

namespace fs = std::filesystem;
using nlohmann::json;
using namespace mhprep;

namespace {

struct Args {
  std::string cmd;
  std::map<std::string, std::string> flags;
  std::vector<std::string> positional;
  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = flags.find(k);
    return it == flags.end() ? def : it->second;
  }
  long num(const std::string& k, long def) const {
    auto it = flags.find(k);
    return it == flags.end() ? def : std::stol(it->second);
  }
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      auto eq = s.find('=');
      if (eq != std::string::npos) a.flags[s.substr(2, eq - 2)] = s.substr(eq + 1);
      else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
        a.flags[s.substr(2)] = argv[++i];
      else a.flags[s.substr(2)] = "1";
    } else {
      a.positional.push_back(s);
    }
  }
  return a;
}

std::string self_exe() {
  std::error_code ec;
  auto p = fs::read_symlink("/proc/self/exe", ec);
  return ec ? "mhprep" : p.string();
}

std::string find_models(const Args& a) {
  if (auto m = a.get("models"); !m.empty()) return m;
  if (const char* env = std::getenv("MHPREP_MODELS")) return env;
  const fs::path exe = fs::path(self_exe()).parent_path();
  for (const fs::path& c : {exe / "models", exe / ".." / "models", exe / ".." / ".." / "models",
                            exe / ".." / "share" / "mhprep" / "models", fs::path("models")})
    if (fs::exists(c / "deploy.prototxt")) return fs::weakly_canonical(c).string();
  return "models";
}

PoolConfig pool_config(const Args& a) {
  PoolConfig c;
  c.workers = static_cast<int>(a.num("workers", static_cast<long>(std::thread::hardware_concurrency())));
  c.worker_bin = self_exe();
  c.models_dir = find_models(a);
  c.out_root = a.get("out", "out");
  c.job_timeout = std::chrono::milliseconds(a.num("timeout-ms", 120000));
  c.max_attempts = static_cast<int>(a.num("max-attempts", 3));
  c.worker_mem_mb = static_cast<size_t>(a.num("mem-mb", 3072));
  c.cross_job_dedup = a.get("no-cross-dedup").empty();
  c.dup_radius = static_cast<int>(a.num("dup-radius", kDupHammingRadius));
  c.color_tolerance = static_cast<int>(a.num("color-tolerance", kDupColorTolerance));
  c.drain_timeout = std::chrono::milliseconds(a.num("drain-ms", 30000));
  c.db = a.get("db");
  if (c.db.empty())
    if (const char* env = std::getenv("MHPREP_DB")) c.db = env;
  c.db_retain = std::chrono::seconds(
      static_cast<long>(std::stod(a.get("retain-days", "0")) * 86400));
  if (auto o = a.get("options"); !o.empty()) c.default_options = json::parse(o);
  return c;
}

int cmd_run(const Args& a) {
  const std::string tenant = a.get("tenant", "default");
  std::vector<JobSpec> inputs;
  for (auto& p : a.positional) inputs.push_back({p, tenant, json::object()});
  if (auto list = a.get("list"); !list.empty()) {
    // One input per line, optionally "tenant<TAB>input".
    std::ifstream f(list);
    for (std::string line; std::getline(f, line);) {
      if (line.empty() || line[0] == '#') continue;
      const auto tab = line.find('\t');
      if (tab == std::string::npos) inputs.push_back({line, tenant, json::object()});
      else inputs.push_back({line.substr(tab + 1), line.substr(0, tab), json::object()});
    }
  }
  if (inputs.empty()) {
    std::cerr << "mhprep run: no inputs\n";
    return 2;
  }
  Supervisor::block_signals();
  PoolConfig cfg = pool_config(a);
  Supervisor sup(cfg);
  std::vector<uint64_t> ids;
  const auto t0 = std::chrono::steady_clock::now();
  for (auto& in : inputs) ids.push_back(*sup.submit(std::move(in)));
  sup.on_finished = [&](const JobRecord& r) {
    if (!a.get("quiet").empty()) return;
    std::fprintf(stderr, "[%s] job %llu %s (%.0f ms)%s%s\n", to_string(r.state),
                 static_cast<unsigned long long>(r.id), r.spec.input.c_str(), r.latency_ms(),
                 r.error.empty() ? "" : ": ", r.error.c_str());
  };
  sup.run(/*exit_when_idle=*/true);
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  int64_t frames = 0, bytes_in = 0, bytes_out = 0, scenes = 0, uniq = 0, dup = 0, faces = 0;
  double media_sec = 0;
  std::ofstream report;
  if (auto r = a.get("report"); !r.empty()) report.open(r);
  for (uint64_t id : ids) {
    auto rec = sup.get(id);
    if (report.is_open()) report << rec->to_json(true).dump() << "\n";
    if (rec->state != JobState::kSucceeded) continue;
    const json& m = rec->manifest;
    frames += m.value("frames_decoded", int64_t{0});
    bytes_in += m.value("bytes_in", int64_t{0});
    bytes_out += m.value("bytes_out", int64_t{0});
    if (m.contains("video")) media_sec += m["video"].value("duration_sec", 0.0);
    if (m.contains("summary")) {
      const json& s = m["summary"];
      scenes += s.value("scenes", 0);
      uniq += s.value("assets_unique", 0) - s.value("assets_cross_job_duplicate", 0);
      dup += s.value("assets_duplicate", 0) + s.value("assets_cross_job_duplicate", 0);
      faces += s.value("faces", 0);
    }
  }
  const PoolStats s = sup.stats();
  json summary{{"jobs", ids.size()},
               {"succeeded", s.succeeded},
               {"failed", s.failed},
               {"retries", s.retries},
               {"timeouts", s.timeouts},
               {"crashes", s.crashes},
               {"oom", s.oom},
               {"input_errors", s.input_errors},
               {"worker_spawns", s.worker_spawns},
               {"workers", cfg.workers},
               {"wall_sec", wall},
               {"frames_decoded", frames},
               {"fps", frames / wall},
               {"media_sec", media_sec},
               {"realtime_factor", media_sec / wall},
               {"scenes", scenes},
               {"assets_unique", uniq},
               {"assets_duplicate", dup},
               {"faces", faces},
               {"bytes_in", bytes_in},
               {"bytes_out", bytes_out},
               {"latency_p50_ms", sup.latency().percentile(50)},
               {"latency_p99_ms", sup.latency().percentile(99)}};
  std::cout << summary.dump(2) << std::endl;
  return s.failed == 0 ? 0 : 1;
}

int cmd_serve(const Args& a) {
  Supervisor::block_signals();  // before httplib starts its threads
  PoolConfig cfg = pool_config(a);
  Supervisor sup(cfg);
  HttpServer http(sup);
  const std::string host = a.get("host", "0.0.0.0");
  const int port = static_cast<int>(a.num("port", 8080));
  if (!http.start(host, port)) {
    std::cerr << "cannot bind " << host << ":" << port << "\n";
    return 1;
  }
  // On SIGTERM: stop accepting connections now; in-flight jobs drain in sup.run().
  sup.on_stop = [&] { http.shutdown(); };
  std::fprintf(stderr, "mhprep: %d workers, listening on %s:%d (models: %s)\n", cfg.workers,
               host.c_str(), port, cfg.models_dir.c_str());
  if (sup.durable())
    std::fprintf(stderr, "mhprep: durable job store; resumed %llu unfinished jobs\n",
                 static_cast<unsigned long long>(sup.stats().recovered));
  sup.run(/*exit_when_idle=*/false);
  http.stop();
  std::fprintf(stderr, "mhprep: drained, bye\n");
  return 0;
}

int cmd_process(const Args& a) {
  if (a.positional.empty()) {
    std::cerr << "mhprep process: need an input\n";
    return 2;
  }
  cv::setNumThreads(static_cast<int>(a.num("threads", 1)));
  json o = a.get("options").empty() ? json::object() : json::parse(a.get("options"));
  PipelineOptions opts = options_from_json(o);
  Pipeline p(find_models(a));
  auto m = p.process(a.positional[0], a.get("out", "out/single"), opts);
  std::cout << to_json(m).dump(2) << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args a = parse(argc, argv);
  try {
    if (a.cmd == "worker")
      return worker_main(static_cast<int>(a.num("fd", 3)), a.get("models", "models"),
                         static_cast<size_t>(a.num("mem-mb", 0)));
    if (a.cmd == "run") return cmd_run(a);
    if (a.cmd == "serve") return cmd_serve(a);
    if (a.cmd == "process") return cmd_process(a);
  } catch (const std::exception& e) {
    std::cerr << "mhprep: " << e.what() << "\n";
    return 1;
  }
  std::cerr << "usage: mhprep run|serve|process [flags] ...\n"
               "  pool flags: --workers N --timeout-ms MS --max-attempts N --mem-mb MB\n"
               "              --out DIR --options JSON --dup-radius N --no-cross-dedup\n"
               "              --db CONNINFO (or MHPREP_DB): keep job state in Postgres\n"
               "              --retain-days D: delete finished jobs older than D days\n";
  return 2;
}
