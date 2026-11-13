#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "mhprep/scene.h"
#include "test_util.h"

using namespace mhprep;
using namespace mhprep::testing;

namespace {

std::vector<int64_t> run(const std::vector<cv::Mat>& frames, double fps, int stride = 1,
                         std::vector<CutKind>* kinds = nullptr) {
  SceneParams p;
  p.gradual = true;
  SceneDetector det(p, fps);
  std::vector<int64_t> cuts;
  for (size_t i = 0; i < frames.size(); i += static_cast<size_t>(stride)) {
    cv::Mat small;
    cv::resize(frames[i], small, analysis_size(frames[i].cols, frames[i].rows, 256), 0, 0,
               cv::INTER_AREA);
    if (det.push(static_cast<int64_t>(i), small)) {
      cuts.push_back(static_cast<int64_t>(i));
      if (kinds) kinds->push_back(det.last_cut_kind());
    }
  }
  return cuts;
}

// Two shots (panning, or still) joined by an n-frame dissolve or fade at frame 60.
// Seeds 40 and 41 differ by a mean of 57/255 per channel; 40 and 46 by 102.
std::vector<cv::Mat> with_transition(bool fade, int n, bool pan = true, int seed_b = 41) {
  cv::Mat big_a = synth_scene(40, 1200, 360), big_b = synth_scene(seed_b, 1200, 360);
  std::vector<cv::Mat> frames;
  const int k = pan ? 1 : 0;
  for (int f = 0; f < 60; ++f) frames.push_back(pan_frame(big_a, k * f, 640, 360));
  for (int j = 0; j < n; ++j) {
    const double t = (j + 1.0) / (n + 1.0);
    cv::Mat a = pan_frame(big_a, k * (60 + j), 640, 360), b = pan_frame(big_b, k * j, 640, 360);
    cv::Mat out;
    if (!fade) cv::addWeighted(a, 1 - t, b, t, 0, out);
    else if (t < 0.5) a.convertTo(out, -1, 1 - 2 * t);
    else b.convertTo(out, -1, 2 * t - 1);
    frames.push_back(out);
  }
  for (int f = n; f < n + 60; ++f) frames.push_back(pan_frame(big_b, k * f, 640, 360));
  return frames;
}

bool within(int64_t x, int64_t lo, int64_t hi) { return x >= lo && x <= hi; }

// Three panning shots of 40 frames each.
std::vector<cv::Mat> three_shots() {
  std::vector<cv::Mat> frames;
  for (int shot = 0; shot < 3; ++shot) {
    cv::Mat big = synth_scene(10 + shot, 900, 360);
    for (int f = 0; f < 40; ++f) frames.push_back(pan_frame(big, f, 640, 360));
  }
  return frames;
}

}  // namespace

TEST(SceneDetector, FindsHardCuts) {
  EXPECT_EQ(run(three_shots(), 30), (std::vector<int64_t>{40, 80}));
}

TEST(SceneDetector, FindsCutsWithFrameStride) {
  EXPECT_EQ(run(three_shots(), 30, 2), (std::vector<int64_t>{40, 80}));
}

TEST(SceneDetector, PanningIsNotACut) {
  cv::Mat big = synth_scene(20, 1600, 360);
  std::vector<cv::Mat> frames;
  for (int f = 0; f < 300; ++f) frames.push_back(pan_frame(big, f, 640, 360));
  EXPECT_TRUE(run(frames, 30).empty());
}

TEST(SceneDetector, MinSceneLengthSuppressesFlicker) {
  // Alternate between two scenes every 3 frames: at most one cut per 0.5 s at 30 fps.
  cv::Mat a = synth_scene(30), b = synth_scene(31);
  std::vector<cv::Mat> frames;
  for (int f = 0; f < 90; ++f) frames.push_back(((f / 3) % 2) ? b : a);
  auto cuts = run(frames, 30);
  for (size_t i = 1; i < cuts.size(); ++i) EXPECT_GE(cuts[i] - cuts[i - 1], 15);
}

TEST(SceneDetector, AnalysisSizeKeepsAspectAndEvenHeight) {
  EXPECT_EQ(analysis_size(1920, 1080, 256), cv::Size(256, 144));
  EXPECT_EQ(analysis_size(1280, 544, 256), cv::Size(256, 110));
  EXPECT_EQ(analysis_size(200, 100, 256), cv::Size(200, 100));
}

// Each frame of a 1 s dissolve differs from the last by only a few levels, so the hard-cut
// test never fires; the linear-ramp test must.
TEST(SceneDetector, FindsDissolveOnce) {
  for (int n : {15, 20}) {
    std::vector<CutKind> kinds;
    auto cuts = run(with_transition(false, n), 30, 1, &kinds);
    ASSERT_EQ(cuts.size(), 1u) << n << "-frame dissolve";
    EXPECT_TRUE(within(cuts[0], 60, 60 + n + 2)) << cuts[0];
    EXPECT_EQ(kinds[0], CutKind::kDissolve);
  }
}

// A 1.5 s dissolve changes the picture too slowly for one 0.4 s window; the 0.8 s window
// catches it when the two shots are clearly different. (Between similar shots it is
// missed: see "Limits" in the README.)
TEST(SceneDetector, FindsLongDissolveBetweenContrastingShots) {
  std::vector<CutKind> kinds;
  auto cuts = run(with_transition(false, 45, /*pan=*/false, /*seed_b=*/46), 30, 1, &kinds);
  ASSERT_EQ(cuts.size(), 1u);
  EXPECT_TRUE(within(cuts[0], 60, 60 + 45 + 2)) << cuts[0];
  EXPECT_EQ(kinds[0], CutKind::kDissolve);
}

TEST(SceneDetector, FindsFadeThroughBlackOnce) {
  for (int n : {20, 40}) {
    std::vector<CutKind> kinds;
    auto cuts = run(with_transition(true, n), 30, 1, &kinds);
    ASSERT_EQ(cuts.size(), 1u) << n << "-frame fade: out and back in is one cut";
    EXPECT_TRUE(within(cuts[0], 60, 60 + n + 2)) << cuts[0];  // +2: evaluator tolerance
  }
}

TEST(SceneDetector, GradualDetectionIsOffByDefault) {
  SceneDetector det(SceneParams{}, 30);
  int cuts = 0;
  int64_t i = 0;
  for (const cv::Mat& f : with_transition(false, 45, /*pan=*/false)) {
    cv::Mat small;
    cv::resize(f, small, analysis_size(f.cols, f.rows, 256), 0, 0, cv::INTER_AREA);
    cuts += det.push(i++, small);
  }
  EXPECT_EQ(cuts, 0) << "a 1.5 s dissolve is invisible to the hard-cut test alone";
}

TEST(SceneDetector, HardCutsAreLabelledCuts) {
  std::vector<CutKind> kinds;
  run(three_shots(), 30, 1, &kinds);
  ASSERT_EQ(kinds.size(), 2u);
  EXPECT_EQ(kinds[0], CutKind::kCut);
  EXPECT_EQ(kinds[1], CutKind::kCut);
}
