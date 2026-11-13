#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include "mhprep/phash.h"
#include "test_util.h"

using namespace mhprep;
using mhprep::testing::synth_scene;

TEST(PHash, IdenticalImagesHashEqual) {
  cv::Mat a = synth_scene(1);
  EXPECT_EQ(phash(a), phash(a.clone()));
}

TEST(PHash, RobustToReencodeResizeCropBrightness) {
  cv::Mat a = synth_scene(2);
  const uint64_t h = phash(a);

  std::vector<uint8_t> buf;
  cv::imencode(".jpg", a, buf, {cv::IMWRITE_JPEG_QUALITY, 30});
  EXPECT_LE(hamming(h, phash(cv::imdecode(buf, cv::IMREAD_COLOR))), 4) << "jpeg q30";

  cv::Mat half;
  cv::resize(a, half, {}, 0.5, 0.5, cv::INTER_AREA);
  EXPECT_LE(hamming(h, phash(half)), 4) << "resize 50%";

  const int dx = a.cols / 50, dy = a.rows / 50;
  EXPECT_LE(hamming(h, phash(a(cv::Rect(dx, dy, a.cols - 2 * dx, a.rows - 2 * dy)))), 10)
      << "2% crop per side";

  cv::Mat bright;
  a.convertTo(bright, -1, 1.0, 20);
  EXPECT_LE(hamming(h, phash(bright)), 6) << "brightness +20";
}

TEST(PHash, UnrelatedImagesAreFar) {
  int min_d = 64;
  for (int i = 0; i < 20; ++i)
    for (int j = i + 1; j < 20; ++j)
      min_d = std::min(min_d, hamming(phash(synth_scene(100 + i)), phash(synth_scene(100 + j))));
  EXPECT_GT(min_d, 12);
}

TEST(DHash, StableAndDiscriminative) {
  cv::Mat a = synth_scene(3), b = synth_scene(4);
  EXPECT_EQ(dhash(a), dhash(a.clone()));
  EXPECT_GT(hamming(dhash(a), dhash(b)), 10);
}
