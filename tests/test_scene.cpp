#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "mhprep/scene.h"
#include "test_util.h"

using namespace mhprep;
using namespace mhprep::testing;

namespace {

std::vector<int64_t> run(const std::vector<cv::Mat>& frames, double fps, int stride = 1) {
  SceneDetector det(SceneParams{}, fps);
  std::vector<int64_t> cuts;
  for (size_t i = 0; i < frames.size(); i += static_cast<size_t>(stride)) {
    cv::Mat small;
    cv::resize(frames[i], small, analysis_size(frames[i].cols, frames[i].rows, 256), 0, 0,
               cv::INTER_AREA);
    if (det.push(static_cast<int64_t>(i), small)) cuts.push_back(static_cast<int64_t>(i));
  }
  return cuts;
}

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
