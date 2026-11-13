// Retention cost on a large store: how long JobStore::prune takes, and what it does to
// the latency of concurrent single-row commits (a stand-in for the writer thread).
//   prune_bench "<conninfo with search_path set to an empty scratch schema>" [jobs] [batch]
#include <libpq-fe.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "mhprep/job_store.h"

using Clock = std::chrono::steady_clock;

static void exec(PGconn* c, const std::string& q) {
  PGresult* r = PQexec(c, q.c_str());
  if (PQresultStatus(r) != PGRES_COMMAND_OK && PQresultStatus(r) != PGRES_TUPLES_OK) {
    std::fprintf(stderr, "%s\n%s", q.c_str(), PQerrorMessage(c));
    std::exit(1);
  }
  PQclear(r);
}

struct Lat { double p50, p99, max; size_t n; };
static Lat summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  if (v.empty()) return {0, 0, 0, 0};
  return {v[v.size() / 2], v[v.size() * 99 / 100], v.back(), v.size()};
}

// Commits one row per iteration on its own connection until `stop`.
static std::vector<double> commit_loop(const std::string& conninfo, std::atomic<bool>& stop,
                                       uint64_t first_id) {
  PGconn* c = PQconnectdb(conninfo.c_str());
  std::vector<double> ms;
  for (uint64_t id = first_id; !stop; ++id) {
    const auto t0 = Clock::now();
    exec(c, "INSERT INTO mhprep_jobs (id, tenant, input, state) VALUES (" +
                std::to_string(id) + ", 'acme', '/data/new.mp4', 'queued')");
    ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  PQfinish(c);
  return ms;
}

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: prune_bench CONNINFO [jobs] [batch]\n"), 2;
  const std::string conninfo = argv[1];
  const long jobs = argc > 2 ? std::stol(argv[2]) : 1000000;
  const size_t batch = argc > 3 ? std::stoul(argv[3]) : 5000;

  mhprep::JobStore store(conninfo);  // creates the schema
  PGconn* c = PQconnectdb(conninfo.c_str());
  exec(c, "TRUNCATE mhprep_jobs CASCADE");
  // Finished 60 days ago up to about now, evenly spaced; 3 hashes each.
  const std::string step = std::to_string(60.0 * 86400 / jobs);
  exec(c, "INSERT INTO mhprep_jobs (id, tenant, input, state, attempts, manifest, finished_at) "
          "SELECT g, 't' || (g % 50), '/data/in' || g || '.mp4', 'succeeded', 1, "
          "jsonb_build_object('assets', jsonb_build_array(jsonb_build_object('w', 1920))), "
          "now() - interval '60 days' + g * make_interval(secs => " + step + ") "
          "FROM generate_series(1, " + std::to_string(jobs) + ") g");
  exec(c, "INSERT INTO mhprep_asset_hashes SELECT g, a, 't' || (g % 50), g * 7 + a, "
          "'\\x00112233'::bytea FROM generate_series(1, " + std::to_string(jobs) +
          ") g, generate_series(0, 2) a");
  exec(c, "VACUUM ANALYZE mhprep_jobs");
  exec(c, "VACUUM ANALYZE mhprep_asset_hashes");
  exec(c, "CHECKPOINT");

  std::atomic<bool> stop{false};
  std::vector<double> idle;
  std::thread t([&] { idle = commit_loop(conninfo, stop, jobs + 1); });
  std::this_thread::sleep_for(std::chrono::seconds(5));
  stop = true;
  t.join();

  stop = false;
  std::vector<double> busy;
  std::thread t2([&] { busy = commit_loop(conninfo, stop, jobs + 1000001); });
  const auto t0 = Clock::now();
  const uint64_t n = store.prune(std::chrono::hours(24 * 30), batch);
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  stop = true;
  t2.join();

  const Lat a = summarize(idle), b = summarize(busy);
  std::printf(
      "{\"jobs\": %ld, \"hashes\": %ld, \"batch\": %zu, \"pruned\": %llu, \"prune_s\": %.2f, "
      "\"jobs_per_s\": %.0f,\n \"commit_ms_idle\": {\"p50\": %.2f, \"p99\": %.2f, \"max\": %.2f, "
      "\"n\": %zu},\n \"commit_ms_during_prune\": {\"p50\": %.2f, \"p99\": %.2f, \"max\": %.2f, "
      "\"n\": %zu}}\n",
      jobs, jobs * 3, batch, static_cast<unsigned long long>(n), secs, n / secs, a.p50, a.p99,
      a.max, a.n, b.p50, b.p99, b.max, b.n);
  PQfinish(c);
}
