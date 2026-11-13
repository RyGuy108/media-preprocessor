#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "mhprep/supervisor.h"

using namespace mhprep;
using namespace std::chrono_literals;
using nlohmann::json;

namespace {

PoolConfig config(int workers) {
  PoolConfig c;
  c.workers = workers;
  c.worker_bin = MHPREP_BIN;
  c.models_dir = MHPREP_MODELS;
  c.out_root = (std::filesystem::temp_directory_path() / "mhprep_sup_test").string();
  c.job_timeout = 1500ms;
  c.max_attempts = 2;
  c.handle_term_signals = false;
  return c;
}

JobRecord finished(Supervisor& s, uint64_t id) {
  auto r = s.get(id);
  EXPECT_TRUE(r.has_value());
  return *r;
}

}  // namespace

TEST(Supervisor, IsolatesCrashesHangsAndBadInputs) {
  Supervisor sup(config(2));
  const auto ok1 = *sup.submit({"chaos://ok"});
  const auto crash = *sup.submit({"chaos://crash"});
  const auto hang = *sup.submit({"chaos://hang"});
  const auto bad = *sup.submit({"chaos://bad"});
  const auto ok2 = *sup.submit({"chaos://sleep/50"});
  const auto t0 = std::chrono::steady_clock::now();
  sup.run(/*exit_when_idle=*/true);
  const auto elapsed = std::chrono::steady_clock::now() - t0;

  EXPECT_EQ(finished(sup, ok1).state, JobState::kSucceeded);
  EXPECT_EQ(finished(sup, ok2).state, JobState::kSucceeded);

  auto c = finished(sup, crash);
  EXPECT_EQ(c.state, JobState::kFailed);
  EXPECT_EQ(c.error_kind, "crash");
  EXPECT_EQ(c.attempts, 2);  // retried once on a fresh worker, then quarantined
  EXPECT_NE(c.error.find("SEGV"), std::string::npos) << c.error;
  EXPECT_NE(c.error.find("quarantined"), std::string::npos);

  auto h = finished(sup, hang);
  EXPECT_EQ(h.state, JobState::kFailed);
  EXPECT_EQ(h.error_kind, "timeout");
  EXPECT_EQ(h.attempts, 2);

  auto b = finished(sup, bad);
  EXPECT_EQ(b.state, JobState::kFailed);
  EXPECT_EQ(b.error_kind, "input");
  EXPECT_EQ(b.attempts, 1) << "bad input is not retried";

  const PoolStats s = sup.stats();
  EXPECT_EQ(s.crashes, 2u);
  EXPECT_EQ(s.timeouts, 2u);
  EXPECT_GE(s.worker_spawns, 2u + 4u);  // initial pool + one replacement per lost worker
  // Two hang attempts at 1.5 s each dominate; nothing waited longer than its deadline.
  EXPECT_LT(elapsed, 6s);
}

TEST(Supervisor, MemoryLimitTurnsBombIntoCleanFailure) {
  PoolConfig c = config(1);
  c.worker_mem_mb = 1024;
  c.max_attempts = 1;
  Supervisor sup(c);
  const auto bomb = *sup.submit({"chaos://oom"});
  const auto after = *sup.submit({"chaos://ok"});
  sup.run(true);
  auto r = finished(sup, bomb);
  EXPECT_EQ(r.state, JobState::kFailed);
  EXPECT_EQ(r.error_kind, "oom");
  auto a = finished(sup, after);
  EXPECT_EQ(a.state, JobState::kSucceeded) << "pool recovers after OOM: " << a.to_json().dump();
}

TEST(Supervisor, TransientFaultsAreRetriedToSuccess) {
  ::setenv("MHPREP_CHAOS", "crash=0.4,seed=3", 1);
  PoolConfig c = config(4);
  c.max_attempts = 8;
  Supervisor sup(c);
  std::vector<uint64_t> ids;
  for (int i = 0; i < 40; ++i) ids.push_back(*sup.submit({"chaos://ok?" + std::to_string(i)}));
  sup.run(true);
  ::unsetenv("MHPREP_CHAOS");
  for (auto id : ids) {
    auto r = finished(sup, id);
    EXPECT_EQ(r.state, JobState::kSucceeded) << r.to_json().dump();
  }
  EXPECT_GT(sup.stats().retries, 5u);
  EXPECT_EQ(sup.stats().failed, 0u);
}

TEST(Supervisor, RunsJobsInParallel) {
  Supervisor sup(config(4));
  for (int i = 0; i < 16; ++i) sup.submit({"chaos://sleep/100"});
  const auto t0 = std::chrono::steady_clock::now();
  sup.run(true);
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
  EXPECT_EQ(sup.stats().succeeded, 16u);
  EXPECT_LT(ms, 1200ms) << "16 x 100 ms on 4 workers should take ~400 ms plus startup";
}

TEST(Supervisor, GracefulStopDrainsRunningAndCancelsQueued) {
  PoolConfig c = config(2);
  Supervisor sup(c);
  std::vector<uint64_t> ids;
  for (int i = 0; i < 6; ++i) ids.push_back(*sup.submit({"chaos://sleep/400"}));
  std::thread stopper([&] {
    std::this_thread::sleep_for(200ms);  // workers have picked up the first two jobs
    sup.request_stop();
  });
  sup.run(/*exit_when_idle=*/false);
  stopper.join();
  int ok = 0, cancelled = 0;
  for (auto id : ids) {
    auto r = finished(sup, id);
    ok += r.state == JobState::kSucceeded;
    cancelled += r.error_kind == "cancelled";
  }
  EXPECT_EQ(ok, 2);
  EXPECT_EQ(cancelled, 4);
  EXPECT_FALSE(sup.submit({"chaos://ok"}).has_value()) << "no new work after stop";
}

TEST(Supervisor, CrossJobDedupWithinTenantOnly) {
  // Build two identical manifests by hand through the worker: easiest is real images.
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "mhprep_dedup_test";
  fs::create_directories(dir);
  // Reuse the pipeline test's synthetic image generator indirectly via a PNG on disk.
  cv::Mat img(240, 320, CV_8UC3);
  cv::randu(img, cv::Scalar::all(0), cv::Scalar::all(255));
  cv::GaussianBlur(img, img, {0, 0}, 8);
  cv::imwrite((dir / "a.png").string(), img);
  cv::imwrite((dir / "a_copy.jpg").string(), img, {cv::IMWRITE_JPEG_QUALITY, 40});
  Supervisor sup(config(2));
  const auto first = *sup.submit({(dir / "a.png").string(), "t1"});
  sup.run(true);  // make sure `first` is indexed before the copies arrive
  const auto copy_same_tenant = *sup.submit({(dir / "a_copy.jpg").string(), "t1"});
  const auto copy_other_tenant = *sup.submit({(dir / "a_copy.jpg").string(), "t2"});
  sup.run(true);
  EXPECT_EQ(finished(sup, first).cross_job_duplicates, 0);
  auto dup = finished(sup, copy_same_tenant);
  ASSERT_EQ(dup.state, JobState::kSucceeded) << dup.error;
  EXPECT_EQ(dup.cross_job_duplicates, 1);
  EXPECT_EQ(dup.manifest["assets"][0]["cross_duplicate_of"]["job"], first);
  EXPECT_EQ(finished(sup, copy_other_tenant).cross_job_duplicates, 0);
}
