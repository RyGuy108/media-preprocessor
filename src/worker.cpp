// Worker process: loads the models once, then runs one job at a time for the supervisor.
#include <csignal>
#include <cstdlib>
#include <functional>
#include <new>
#include <string>
#include <sys/prctl.h>
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include "mhprep/errors.h"
#include "mhprep/ipc.h"
#include "mhprep/pipeline.h"
#include "mhprep/supervisor.h"

namespace mhprep {
namespace {

// Fault injection for tests and the chaos benchmark.
//  * Inputs "chaos://crash", "chaos://hang", "chaos://oom", "chaos://sleep/<ms>",
//    "chaos://ok" fail (or succeed) the same way on every attempt: poison jobs.
//  * MHPREP_CHAOS="crash=0.03,hang=0.03,seed=7" makes ordinary jobs fail transiently,
//    keyed on (input, attempt, seed), so the same job may succeed on a retry.
struct Chaos {
  double crash = 0, hang = 0;
  uint64_t seed = 0;
};

Chaos chaos_from_env() {
  Chaos c;
  const char* env = std::getenv("MHPREP_CHAOS");
  if (!env) return c;
  std::string s(env);
  size_t pos = 0;
  while (pos < s.size()) {
    size_t end = s.find(',', pos);
    if (end == std::string::npos) end = s.size();
    const std::string kv = s.substr(pos, end - pos);
    const size_t eq = kv.find('=');
    if (eq != std::string::npos) {
      const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
      if (k == "crash") c.crash = std::stod(v);
      else if (k == "hang") c.hang = std::stod(v);
      else if (k == "seed") c.seed = std::stoull(v);
    }
    pos = end + 1;
  }
  return c;
}

// FNV-1a + splitmix64 finalizer -> [0, 1). bench/py_baseline.py implements the same
// function so both pipelines see identical faults.
double chaos_draw(const std::string& input, int attempt, uint64_t seed) {
  const std::string key = input + "#" + std::to_string(attempt) + "#" + std::to_string(seed);
  uint64_t h = 1469598103934665603ull;
  for (unsigned char ch : key) {
    h ^= ch;
    h *= 1099511628211ull;
  }
  h ^= h >> 30;
  h *= 0xbf58476d1ce4e5b9ull;
  h ^= h >> 27;
  h *= 0x94d049bb133111ebull;
  h ^= h >> 31;
  return static_cast<double>(h >> 11) / static_cast<double>(1ull << 53);
}

[[noreturn]] void crash_now() {
  std::signal(SIGSEGV, SIG_DFL);
  std::raise(SIGSEGV);
  std::abort();
}

[[noreturn]] void hang_now() {
  for (;;) ::pause();  // like a decoder blocked on a stalled network read
}

void exhaust_memory() {
  std::vector<std::vector<char>> hog;
  for (;;) hog.emplace_back(64u << 20, 'x');  // throws bad_alloc under RLIMIT_AS
}

nlohmann::json handle_chaos_input(const std::string& input) {
  const std::string what = input.substr(8);
  if (what == "crash") crash_now();
  if (what == "hang") hang_now();
  if (what == "oom") exhaust_memory();
  if (what.rfind("sleep/", 0) == 0) ::usleep(static_cast<useconds_t>(std::stoi(what.substr(6)) * 1000));
  if (what == "bad") throw InputError("chaos://bad is not media");
  return {{"input", input}, {"kind", "chaos"}, {"assets", nlohmann::json::array()},
          {"scenes", nlohmann::json::array()}, {"summary", nlohmann::json::object()}};
}

}  // namespace

int worker_main(int fd, const std::string& models_dir, size_t mem_mb) {
  ::prctl(PR_SET_PDEATHSIG, SIGKILL);  // never outlive the supervisor
  if (::getppid() == 1) return 1;
  if (mem_mb > 0 && !kAddressSanitizer) {
    rlimit lim{mem_mb << 20, mem_mb << 20};
    ::setrlimit(RLIMIT_AS, &lim);
  }
  cv::setNumThreads(1);  // the pool supplies the parallelism; avoid N x cores threads
  // Batch priority: decoding is throughput work, so when the box is saturated the
  // supervisor and HTTP threads (nice 0) still get the CPU promptly.
  sched_param sp{};
  ::sched_setscheduler(0, SCHED_BATCH, &sp);
  ::setpriority(PRIO_PROCESS, 0, 10);
  FramedConn conn(fd);
  Pipeline pipeline(models_dir);
  try {
    pipeline.warm_up();  // load the face model before taking work
  } catch (const std::exception& e) {
    conn.send_blocking({{"type", "fatal"}, {"error", e.what()}});
    return 2;
  }
  const Chaos chaos = chaos_from_env();
  if (!conn.send_blocking({{"type", "hello"}, {"pid", ::getpid()}})) return 0;

  for (;;) {
    auto msg = conn.recv_blocking();
    if (!msg) return 0;  // supervisor closed the socket: clean shutdown
    if (msg->value("type", "") != "job") continue;
    const uint64_t id = (*msg)["id"].get<uint64_t>();
    const int attempt = msg->value("attempt", 1);
    const std::string input = (*msg)["input"].get<std::string>();
    const std::string out = msg->value("out", std::string("out"));
    nlohmann::json reply{{"type", "result"}, {"id", id}};
    bool exit_after = false;
    try {
      if (chaos.crash > 0 || chaos.hang > 0) {
        const double r = chaos_draw(input, attempt, chaos.seed);
        if (r < chaos.crash) crash_now();
        if (r < chaos.crash + chaos.hang) hang_now();
      }
      if (input.rfind("chaos://", 0) == 0) {
        reply["manifest"] = handle_chaos_input(input);
      } else {
        const PipelineOptions opts = options_from_json(msg->value("options", nlohmann::json::object()));
        reply["manifest"] = to_json(pipeline.process(input, out, opts));
      }
      reply["ok"] = true;
    } catch (const InputError& e) {
      reply["ok"] = false;
      reply["error_kind"] = "input";
      reply["error"] = e.what();
    } catch (const std::bad_alloc&) {
      reply["ok"] = false;
      reply["error_kind"] = "oom";
      reply["error"] = "worker memory limit (" + std::to_string(mem_mb) + " MB) exceeded";
      exit_after = true;  // start the next job on a fresh heap
    } catch (const cv::Exception& e) {
      const bool oom = e.code == cv::Error::StsNoMem;
      reply["ok"] = false;
      reply["error_kind"] = oom ? "oom" : "input";
      reply["error"] = e.what();
      exit_after = oom;
    } catch (const std::exception& e) {
      reply["ok"] = false;
      reply["error_kind"] = "internal";
      reply["error"] = e.what();
    }
    if (exit_after) reply["worker_exiting"] = true;  // don't hand me another job
    if (!conn.send_blocking(reply)) return 0;
    if (exit_after) return 3;
  }
}

}  // namespace mhprep
