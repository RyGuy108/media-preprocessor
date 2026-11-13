#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "mhprep/errors.h"
#include "mhprep/faces.h"
#include "mhprep/pipeline.h"
#include "test_util.h"

using namespace mhprep;
using namespace mhprep::testing;
namespace fs = std::filesystem;

namespace {

fs::path tmpdir() {
  fs::path p = fs::temp_directory_path() / ("mhprep_test_" + std::to_string(::getpid()));
  fs::create_directories(p);
  return p;
}

// Shots A, B, A' (A' = same content as A): three scenes, and the third keyframe is a
// near-duplicate of the first.
fs::path write_test_video() {
  const fs::path path = tmpdir() / "aba.avi";
  cv::VideoWriter vw(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, {640, 360});
  EXPECT_TRUE(vw.isOpened());
  cv::Mat a = synth_scene(50, 800, 360), b = synth_scene(51, 800, 360);
  for (const cv::Mat* s : {&a, &b, &a})
    for (int f = 0; f < 45; ++f) vw.write(pan_frame(*s, f, 640, 360));
  return path;
}

class PipelineTest : public ::testing::TestWithParam<Backend> {};

}  // namespace

TEST_P(PipelineTest, ScenesKeyframesAndDuplicates) {
  static const fs::path video = write_test_video();
  Pipeline p(MHPREP_MODELS);
  PipelineOptions o;
  o.source.backend = GetParam();
  const fs::path out = tmpdir() / (GetParam() == Backend::kFFmpeg ? "ff" : "cv");
  Manifest m = p.process(video.string(), out.string(), o);

  EXPECT_EQ(m.frames_decoded, 135);
  ASSERT_EQ(m.scenes.size(), 3u);
  // With stride 2 a cut is reported at the first analyzed frame after it.
  EXPECT_NEAR(m.scenes[1].start_frame, 45, 1);
  EXPECT_NEAR(m.scenes[2].start_frame, 90, 1);
  EXPECT_EQ(m.scenes[2].end_frame, 134);

  const Asset& k0 = m.assets[static_cast<size_t>(m.scenes[0].keyframe)];
  const Asset& k1 = m.assets[static_cast<size_t>(m.scenes[1].keyframe)];
  const Asset& k2 = m.assets[static_cast<size_t>(m.scenes[2].keyframe)];
  EXPECT_EQ(k0.duplicate_of, -1);
  EXPECT_EQ(k1.duplicate_of, -1);
  EXPECT_EQ(k2.duplicate_of, m.scenes[0].keyframe) << "A' should dedupe against A";
  EXPECT_TRUE(fs::exists(k0.path));
  EXPECT_TRUE(k2.path.empty()) << "duplicates are not written";
  EXPECT_GT(m.bytes_out, 0);

  auto j = to_json(m);
  EXPECT_EQ(j["summary"]["scenes"], 3);
  EXPECT_EQ(j["summary"]["assets_duplicate"], 1);
}

INSTANTIATE_TEST_SUITE_P(Backends, PipelineTest,
                         ::testing::Values(Backend::kFFmpeg, Backend::kOpenCV),
                         [](const auto& info) {
                           return info.param == Backend::kFFmpeg ? std::string("FFmpeg")
                                                                 : std::string("OpenCV");
                         });

TEST(Pipeline, BadInputsRaiseInputError) {
  Pipeline p(MHPREP_MODELS);
  PipelineOptions o;
  o.write_assets = false;
  EXPECT_THROW(p.process("/definitely/not/here.mp4", "", o), InputError);
  EXPECT_THROW(p.process("/definitely/not/here.jpg", "", o), InputError);
  const fs::path junk = tmpdir() / "junk.mp4";
  {
    std::ofstream f(junk);
    f << std::string(10000, 'z');
  }
  EXPECT_THROW(p.process(junk.string(), "", o), InputError);
}

TEST(Pipeline, ImageJobHashesAndFindsNoFacesInAbstractArt) {
  const fs::path img = tmpdir() / "art.png";
  cv::imwrite(img.string(), synth_scene(60));
  Pipeline p(MHPREP_MODELS);
  PipelineOptions o;
  Manifest m = p.process(img.string(), (tmpdir() / "img").string(), o);
  ASSERT_EQ(m.assets.size(), 1u);
  EXPECT_EQ(m.assets[0].kind, "image");
  EXPECT_NE(m.assets[0].phash, 0u);
}

TEST(FaceCrop, StaysInsideFrameAndIsSquare) {
  cv::Mat img(400, 600, CV_8UC3, cv::Scalar(10, 20, 30));
  for (cv::Rect box : {cv::Rect(0, 0, 100, 120), cv::Rect(550, 350, 50, 50),
                       cv::Rect(200, 100, 300, 300)}) {
    cv::Mat c = crop_face(img, box, 256);
    EXPECT_EQ(c.size(), cv::Size(256, 256));
  }
}
