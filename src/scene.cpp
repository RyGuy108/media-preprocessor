#include "mhprep/scene.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace mhprep {

cv::Size analysis_size(int width, int height, int analysis_width) {
  if (width <= analysis_width) return {width, height};
  int h = static_cast<int>(std::lround(static_cast<double>(height) * analysis_width / width));
  h += h & 1;
  return {analysis_width, std::max(2, h)};
}

SceneDetector::SceneDetector(SceneParams p, double fps)
    : p_(p), min_gap_frames_(std::max<int64_t>(1, std::llround(p.min_scene_sec * fps))) {}

void SceneDetector::reset() {
  prev_hsv_.release();
  recent_.clear();
  recent_sum_ = 0;
  last_cut_ = 0;
  last_score_ = 0;
}

bool SceneDetector::push(int64_t frame_idx, const cv::Mat& small_bgr) {
  cv::cvtColor(small_bgr, hsv_, cv::COLOR_BGR2HSV);
  if (prev_hsv_.empty()) {
    std::swap(prev_hsv_, hsv_);
    last_cut_ = frame_idx;
    return false;
  }
  cv::absdiff(hsv_, prev_hsv_, diff_);
  const cv::Scalar m = cv::mean(diff_);
  const double score = (m[0] + m[1] + m[2]) / 3.0;
  last_score_ = score;
  std::swap(prev_hsv_, hsv_);

  const double avg = recent_.empty() ? 0.0 : recent_sum_ / static_cast<double>(recent_.size());
  const bool cut = score >= p_.min_score && score >= p_.ratio * std::max(avg, 1.0) &&
                   (frame_idx - last_cut_) >= min_gap_frames_;
  if (cut) {
    last_cut_ = frame_idx;
    return true;  // cut deltas are not added to the rolling average
  }
  recent_.push_back(score);
  recent_sum_ += score;
  if (static_cast<int>(recent_.size()) > p_.window) {
    recent_sum_ -= recent_.front();
    recent_.pop_front();
  }
  return false;
}

}  // namespace mhprep
