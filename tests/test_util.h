#pragma once
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace mhprep::testing {

// A synthetic "photo": smooth gradient background, a few solid shapes, mild noise.
// Different seeds give visually unrelated scenes.
inline cv::Mat synth_scene(int seed, int w = 640, int h = 360) {
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_int_distribution<int> c(0, 255), xs(0, w - 1), ys(0, h - 1), rs(20, h / 3);
  cv::Mat img(h, w, CV_8UC3);
  const cv::Vec3b a(c(rng), c(rng), c(rng)), b(c(rng), c(rng), c(rng));
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const double t = (x + y) / double(w + h);
      img.at<cv::Vec3b>(y, x) = a * (1 - t) + b * t;
    }
  for (int i = 0; i < 6; ++i) {
    const cv::Scalar col(c(rng), c(rng), c(rng));
    if (i % 2)
      cv::circle(img, {xs(rng), ys(rng)}, rs(rng), col, cv::FILLED, cv::LINE_AA);
    else
      cv::rectangle(img, cv::Rect(xs(rng) / 2, ys(rng) / 2, rs(rng) * 2, rs(rng)), col, cv::FILLED);
  }
  cv::Mat noise(h, w, CV_8UC3);
  cv::randn(noise, cv::Scalar::all(0), cv::Scalar::all(4));
  img += noise;
  return img;
}

// Crop a moving window out of a larger scene: simulates a slow camera pan.
inline cv::Mat pan_frame(const cv::Mat& big, int step, int w, int h) {
  const int x = std::min(step * 2, big.cols - w);
  return big(cv::Rect(x, 0, w, h)).clone();
}

}  // namespace mhprep::testing
