// Durable job state. Needs a Postgres server: set MHPREP_TEST_PG to a libpq connection
// string (e.g. "host=localhost user=postgres dbname=mhprep_test"); otherwise these tests
// are skipped. Each test works in its own schema, dropped afterwards.
#include <gtest/gtest.h>
#include <libpq-fe.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "mhprep/supervisor.h"
#include "test_util.h"

using namespace mhprep;
using namespace std::chrono_literals;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {

class JobStoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* base = std::getenv("MHPREP_TEST_PG");
    if (!base || !*base) GTEST_SKIP() << "MHPREP_TEST_PG not set";
    base_ = base;
    schema_ = "mhprep_test_" + std::to_string(::getpid());
    admin_ = PQconnectdb(base_.c_str());
    ASSERT_EQ(PQstatus(admin_), CONNECTION_OK) << PQerrorMessage(admin_);
    PQclear(PQexec(admin_, "SET client_min_messages = warning"));
    sql("DROP SCHEMA IF EXISTS " + schema_ + " CASCADE");
    sql("CREATE SCHEMA " + schema_);
    conninfo_ = base_ + " options='-c search_path=" + schema_ + "'";
    dir_ = fs::temp_directory_path() / ("mhprep_store_" + std::to_string(::getpid()));
    fs::create_directories(dir_);
  }

  void TearDown() override {
    if (admin_) {
      sql("DROP SCHEMA IF EXISTS " + schema_ + " CASCADE");
      PQfinish(admin_);
    }
    if (!dir_.empty()) fs::remove_all(dir_);
  }

  // Runs a statement in the test's schema; returns the first value of the first row.
  std::string sql(const std::string& q) {
    const std::string full = schema_.empty() ? q : "SET search_path = " + schema_ + "; " + q;
    PGresult* r = PQexec(admin_, full.c_str());
    const auto st = PQresultStatus(r);
    EXPECT_TRUE(st == PGRES_COMMAND_OK || st == PGRES_TUPLES_OK) << PQerrorMessage(admin_) << q;
    std::string v = (st == PGRES_TUPLES_OK && PQntuples(r) > 0) ? PQgetvalue(r, 0, 0) : "";
    PQclear(r);
    return v;
  }

  PoolConfig config(int workers = 2) const {
    PoolConfig c;
    c.workers = workers;
    c.worker_bin = MHPREP_BIN;
    c.models_dir = MHPREP_MODELS;
    c.out_root = (dir_ / "out").string();
    c.job_timeout = 5s;
    c.max_attempts = 2;
    c.handle_term_signals = false;
    c.db = conninfo_;
    return c;
  }

  std::string image(const std::string& name, int seed, int jpeg_quality = 0) const {
    const auto p = dir_ / name;
    cv::Mat img = mhprep::testing::synth_scene(seed);
    if (jpeg_quality) cv::imwrite(p.string(), img, {cv::IMWRITE_JPEG_QUALITY, jpeg_quality});
    else cv::imwrite(p.string(), img);
    return p.string();
  }

  std::string base_, schema_, conninfo_;
  PGconn* admin_ = nullptr;
  fs::path dir_;
};

}  // namespace

TEST_F(JobStoreTest, SubmitIsCommittedBeforeItReturns) {
  Supervisor sup(config());
  const auto id = sup.submit({"chaos://ok"});
  ASSERT_TRUE(id);
  // Read from a different connection, without flushing: the row must already be there.
  EXPECT_EQ(sql("SELECT state FROM mhprep_jobs WHERE id = " + std::to_string(*id)), "queued");
}

TEST_F(JobStoreTest, RestartResumesQueuedAndInterruptedJobs) {
  std::vector<uint64_t> ids;
  {
    Supervisor a(config());
    for (int i = 0; i < 3; ++i) ids.push_back(*a.submit({image("img" + std::to_string(i) + ".png", i)}));
    // `a` goes away without running anything, like a supervisor that was killed.
  }
  // Pretend the supervisor died while job 2 was on a worker.
  sql("UPDATE mhprep_jobs SET state = 'running', attempts = 1 WHERE id = " +
      std::to_string(ids[1]));

  Supervisor b(config());
  EXPECT_EQ(b.stats().recovered, 3u);
  const auto fresh = *b.submit({"chaos://ok"});
  EXPECT_EQ(fresh, ids.back() + 1) << "ids continue after a restart";
  b.run(/*exit_when_idle=*/true);

  for (uint64_t id : ids) EXPECT_EQ(b.get(id)->state, JobState::kSucceeded) << id;
  EXPECT_EQ(b.get(fresh)->state, JobState::kSucceeded);
  const JobRecord interrupted = *b.get(ids[1]);
  EXPECT_EQ(interrupted.attempts, 1) << "the interrupted attempt is not counted";
  ASSERT_FALSE(interrupted.attempt_log.empty());
  EXPECT_NE(interrupted.attempt_log[0].find("interrupted"), std::string::npos);

  // b's destructor flushes the write-behind state; the store agrees with memory.
}

TEST_F(JobStoreTest, StoreMatchesMemoryAfterRun) {
  uint64_t ok, bad;
  {
    Supervisor sup(config());
    ok = *sup.submit({image("a.png", 1)});
    bad = *sup.submit({"chaos://bad"});
    sup.run(true);
  }
  EXPECT_EQ(sql("SELECT state || ':' || attempts FROM mhprep_jobs WHERE id = " +
                std::to_string(ok)),
            "succeeded:1");
  EXPECT_EQ(sql("SELECT state || ':' || error_kind FROM mhprep_jobs WHERE id = " +
                std::to_string(bad)),
            "failed:input");
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_jobs WHERE state IN ('queued', 'running')"), "0");
}

TEST_F(JobStoreTest, FinishedJobsAreServedAfterRestartAndEviction) {
  uint64_t first;
  {
    PoolConfig c = config(1);  // one worker: jobs finish in submit order
    c.max_finished_jobs = 1;  // evict aggressively
    Supervisor a(c);
    first = *a.submit({image("a.png", 1)});
    a.submit({image("b.png", 2)});
    a.run(true);
    EXPECT_FALSE(a.get(first)) << "evicted from memory";
    auto j = a.describe(first, true);
    ASSERT_TRUE(j) << "but still served from the store";
    EXPECT_EQ((*j)["status"], "succeeded");
  }
  Supervisor b(config());
  auto j = b.describe(first, /*include_manifest=*/true);
  ASSERT_TRUE(j);
  EXPECT_EQ((*j)["status"], "succeeded");
  EXPECT_EQ((*j)["attempts"], 1);
  EXPECT_TRUE((*j)["manifest"].contains("assets"));
  EXPECT_GT((*j)["latency_ms"].get<double>(), 0.0);
  EXPECT_FALSE(b.describe(first, false)->contains("manifest"));
  EXPECT_FALSE(b.describe(99999, true));
}

TEST_F(JobStoreTest, CrossUploadDedupSurvivesRestart) {
  const std::string original = image("a.png", 7);
  const std::string reupload = image("a_copy.jpg", 7, /*jpeg_quality=*/40);
  uint64_t first;
  {
    Supervisor a(config());
    first = *a.submit({original, "acme"});
    a.run(true);
  }
  Supervisor b(config());
  const auto dup = *b.submit({reupload, "acme"});
  const auto other_tenant = *b.submit({reupload, "globex"});
  b.run(true);
  const JobRecord d = *b.get(dup);
  EXPECT_EQ(d.cross_job_duplicates, 1);
  EXPECT_EQ(d.manifest["assets"][0]["cross_duplicate_of"]["job"], first);
  EXPECT_EQ(b.get(other_tenant)->cross_job_duplicates, 0);
}

TEST_F(JobStoreTest, GracefulStopKeepsQueuedJobsForTheNextStart) {
  std::vector<uint64_t> ids;
  {
    Supervisor a(config(1));
    for (int i = 0; i < 4; ++i) ids.push_back(*a.submit({"chaos://sleep/100"}));
    a.request_stop();
    a.run(false);
    for (uint64_t id : ids) EXPECT_NE(a.get(id)->state, JobState::kFailed) << "not cancelled";
  }
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_jobs WHERE state = 'queued'"), "4");
  Supervisor b(config());
  EXPECT_EQ(b.stats().recovered, 4u);
  b.run(true);
  for (uint64_t id : ids) EXPECT_EQ(b.get(id)->state, JobState::kSucceeded);
}

TEST_F(JobStoreTest, DrainDeadlineRequeuesInsteadOfFailing) {
  uint64_t slow;
  {
    PoolConfig c = config(1);
    c.drain_timeout = 200ms;
    Supervisor a(c);
    slow = *a.submit({"chaos://sleep/3000"});
    std::thread stopper([&] {
      std::this_thread::sleep_for(500ms);  // let it start
      a.request_stop();
    });
    a.run(false);
    stopper.join();
  }
  EXPECT_EQ(sql("SELECT state || ':' || attempts FROM mhprep_jobs WHERE id = " +
                std::to_string(slow)),
            "queued:0");
  Supervisor b(config());
  EXPECT_EQ(b.stats().recovered, 1u);
}

TEST_F(JobStoreTest, OneSupervisorPerDatabase) {
  Supervisor a(config());
  EXPECT_THROW(Supervisor b(config()), std::runtime_error);
}

TEST_F(JobStoreTest, UnreachableDatabaseFailsAtStartup) {
  PoolConfig c = config();
  c.db = "host=127.0.0.1 port=1 connect_timeout=2";
  EXPECT_THROW(Supervisor s(c), std::runtime_error);
}

TEST_F(JobStoreTest, ConcurrentSubmitsShareTransactions) {
  Supervisor sup(config());
  constexpr int kThreads = 16, kPerThread = 25;
  std::atomic<int> accepted{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&] {
      for (int i = 0; i < kPerThread; ++i)
        if (sup.submit({"chaos://ok"})) ++accepted;
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(accepted, kThreads * kPerThread);
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_jobs"), std::to_string(kThreads * kPerThread));
  const auto st = *sup.store_stats();
  EXPECT_LT(st.transactions, static_cast<uint64_t>(kThreads * kPerThread))
      << "group commit should batch concurrent inserts";
}

TEST_F(JobStoreTest, RetentionPrunesOldFinishedJobsAtStartup) {
  uint64_t old_ok, old_bad, recent, newest;
  {
    Supervisor a(config());
    old_ok = *a.submit({image("a.png", 1), "acme"});
    old_bad = *a.submit({"chaos://bad"});
    recent = *a.submit({"chaos://ok"});
    newest = *a.submit({"chaos://ok"});
    a.run(true);
  }
  const std::string old_ids = std::to_string(old_ok) + ", " + std::to_string(old_bad);
  ASSERT_NE(sql("SELECT count(*) FROM mhprep_asset_hashes WHERE job_id = " +
                std::to_string(old_ok)),
            "0");
  // Age everything but `recent` past the cutoff, including the highest id.
  sql("UPDATE mhprep_jobs SET finished_at = now() - interval '40 days' WHERE id IN (" +
      old_ids + ", " + std::to_string(newest) + ")");
  // An old job that never finished: retention must leave it alone.
  sql("UPDATE mhprep_jobs SET state = 'queued', finished_at = NULL, "
      "submitted_at = now() - interval '40 days' WHERE id = " + std::to_string(recent));

  PoolConfig c = config();
  c.db_retain = std::chrono::hours(24 * 30);
  Supervisor b(c);
  // The first pass runs right away, on its own thread.
  for (int i = 0; i < 500 && b.store_stats()->prune_passes == 0; ++i)
    std::this_thread::sleep_for(10ms);
  const auto st = *b.store_stats();
  ASSERT_GE(st.prune_passes, 1u);
  EXPECT_EQ(st.pruned, 2u);
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_jobs WHERE id IN (" + old_ids + ")"), "0");
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_asset_hashes"), "0") << "hashes cascade";
  EXPECT_EQ(sql("SELECT state FROM mhprep_jobs WHERE id = " + std::to_string(recent)), "queued");
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_jobs WHERE id = " + std::to_string(newest)), "1")
      << "the highest id is kept so ids are never reused";
  EXPECT_EQ(*b.submit({"chaos://ok"}), newest + 1);
  EXPECT_FALSE(b.describe(old_ok, false));
}

TEST_F(JobStoreTest, RetentionIsOffByDefaultAndBatched) {
  {
    Supervisor a(config(4));
    for (int i = 0; i < 25; ++i) a.submit({"chaos://ok"});
    a.run(true);
  }
  sql("UPDATE mhprep_jobs SET finished_at = now() - interval '2 days'");
  {
    Supervisor b(config());
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(b.store_stats()->prune_passes, 0u);
  }
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_jobs"), "25");

  JobStore store(conninfo_);
  EXPECT_EQ(store.prune(std::chrono::hours(72)), 0u) << "nothing is old enough";
  EXPECT_EQ(store.prune(std::chrono::hours(24), /*batch=*/4), 24u) << "all but the highest id";
  EXPECT_EQ(store.stats().pruned, 24u);
  EXPECT_EQ(sql("SELECT count(*) FROM mhprep_jobs"), "1");
}
