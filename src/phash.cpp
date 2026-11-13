#include "mhprep/phash.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <opencv2/imgproc.hpp>

namespace mhprep {
namespace {

cv::Mat to_gray(const cv::Mat& img) {
  if (img.channels() == 1) return img;
  cv::Mat g;
  cv::cvtColor(img, g, img.channels() == 4 ? cv::COLOR_BGRA2GRAY : cv::COLOR_BGR2GRAY);
  return g;
}

}  // namespace

uint64_t phash(const cv::Mat& img) {
  CV_Assert(!img.empty());
  cv::Mat small, f, coeffs;
  cv::resize(to_gray(img), small, cv::Size(32, 32), 0, 0, cv::INTER_AREA);
  small.convertTo(f, CV_32F);
  cv::dct(f, coeffs);
  // Low-frequency 8x8 block; median excludes the DC term, which only encodes brightness.
  std::array<float, 64> v{};
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) v[y * 8 + x] = coeffs.at<float>(y, x);
  std::array<float, 63> ac{};
  std::copy(v.begin() + 1, v.end(), ac.begin());
  std::nth_element(ac.begin(), ac.begin() + 31, ac.end());
  const float median = ac[31];
  uint64_t h = 0;
  for (int i = 0; i < 64; ++i)
    if (v[i] > median) h |= (uint64_t{1} << i);
  return h;
}

uint64_t dhash(const cv::Mat& img) {
  CV_Assert(!img.empty());
  cv::Mat small;
  cv::resize(to_gray(img), small, cv::Size(9, 8), 0, 0, cv::INTER_AREA);
  uint64_t h = 0;
  int bit = 0;
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x, ++bit)
      if (small.at<uint8_t>(y, x) > small.at<uint8_t>(y, x + 1)) h |= (uint64_t{1} << bit);
  return h;
}

ColorSig color_signature(const cv::Mat& bgr) {
  CV_Assert(!bgr.empty() && bgr.channels() == 3);
  cv::Mat small, lab;
  cv::resize(bgr, small, cv::Size(4, 4), 0, 0, cv::INTER_AREA);
  cv::cvtColor(small, lab, cv::COLOR_BGR2Lab);
  ColorSig s{};
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      for (int c = 0; c < 3; ++c) s[(y * 4 + x) * 3 + c] = lab.at<cv::Vec3b>(y, x)[c];
  return s;
}

int color_distance(const ColorSig& a, const ColorSig& b) {
  double mean_dl = 0;
  for (int i = 0; i < 16; ++i) mean_dl += int(a[i * 3]) - int(b[i * 3]);
  mean_dl /= 16.0;
  double worst = 0;
  for (int i = 0; i < 16; ++i) {
    const double dl = std::abs(int(a[i * 3]) - int(b[i * 3]) - mean_dl) / 2.0;
    const double da = std::abs(int(a[i * 3 + 1]) - int(b[i * 3 + 1]));
    const double db = std::abs(int(a[i * 3 + 2]) - int(b[i * 3 + 2]));
    worst = std::max(worst, dl + da + db);
  }
  return static_cast<int>(worst);
}

std::string sig_hex(const ColorSig& s) {
  static const char* d = "0123456789abcdef";
  std::string out;
  for (uint8_t v : s) {
    out.push_back(d[v >> 4]);
    out.push_back(d[v & 15]);
  }
  return out;
}

ColorSig parse_sig_hex(const std::string& h) {
  ColorSig s{};
  for (size_t i = 0; i < s.size() && 2 * i + 1 < h.size(); ++i)
    s[i] = static_cast<uint8_t>(std::stoi(h.substr(2 * i, 2), nullptr, 16));
  return s;
}

}  // namespace mhprep
