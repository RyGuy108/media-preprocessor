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

const char* to_string(CutKind k) {
  switch (k) {
    case CutKind::kCut: return "cut";
    case CutKind::kDissolve: return "dissolve";
    case CutKind::kFade: return "fade";
  }
  return "?";
}

namespace {
constexpr int kTinyWidth = 64;
constexpr double kLongWindowDelta = 1.6;  // the linearity test needs structure, not detail

double mean_abs(const cv::Mat& a, const cv::Mat& b, cv::Mat& tmp) {
  cv::absdiff(a, b, tmp);
  const cv::Scalar m = cv::mean(tmp);
  return (m[0] + m[1] + m[2]) / 3.0;
}

double mean_level(const cv::Mat& f) {
  const cv::Scalar m = cv::mean(f);
  return (m[0] + m[1] + m[2]) / 3.0;
}
}  // namespace

SceneDetector::SceneDetector(SceneParams p, double fps)
    : p_(p),
      min_gap_frames_(std::max<int64_t>(1, std::llround(p.min_scene_sec * fps))),
      window_frames_(std::max<int64_t>(4, std::llround(p.gradual_window_sec * fps))) {}

void SceneDetector::reset() {
  prev_hsv_.release();
  recent_.clear();
  recent_sum_ = 0;
  last_cut_ = 0;
  last_score_ = 0;
  last_kind_ = CutKind::kCut;
  ring_.clear();
  in_gradual_ = pending_fade_ = false;
  last_black_ = -1;
  armed_ = true;
}

// Returns true when a dissolve or a fade-in completes the linear-ramp test at this frame.
bool SceneDetector::gradual_step(int64_t frame_idx, const cv::Mat& small_bgr) {
  const int th = std::max(2, small_bgr.rows * kTinyWidth / std::max(1, small_bgr.cols));
  cv::resize(small_bgr, tmp_, {kTinyWidth, th}, 0, 0, cv::INTER_AREA);
  tmp_.convertTo(tiny_, CV_32FC3);
  ring_.emplace_back(frame_idx, tiny_.clone());
  // Keep two windows' worth: short dissolves are tested over one window, long ones over
  // two (over one window a long dissolve changes the picture too little to stand out).
  while (ring_.size() > 2 && ring_[1].first <= frame_idx - 2 * window_frames_) ring_.pop_front();

  // Linear-ramp test over ring_[first..end].
  auto ramp_from = [&](size_t first, double min_delta, bool& full) {
    const int64_t s_idx = ring_[first].first;
    full = ring_.size() - first >= 4 && frame_idx - s_idx >= window_frames_ * 3 / 4;
    if (!full) return false;
    const cv::Mat& fs = ring_[first].second;
    const cv::Mat& ft = ring_.back().second;
    const double delta = mean_abs(fs, ft, tmp_);
    if (delta < min_delta) return false;
    // Every intermediate frame must sit near its point on the line fs -> ft.
    const double limit = p_.gradual_max_residual * delta;
    cv::Mat expected;
    for (size_t i = first + 1; i + 1 < ring_.size(); ++i) {
      const double a =
          static_cast<double>(ring_[i].first - s_idx) / static_cast<double>(frame_idx - s_idx);
      cv::addWeighted(fs, 1.0 - a, ft, a, 0.0, expected);
      if (mean_abs(ring_[i].second, expected, tmp_) > limit) return false;
    }
    return true;
  };
  size_t short_first = 0;
  while (short_first + 1 < ring_.size() && ring_[short_first + 1].first <= frame_idx - window_frames_)
    ++short_first;
  bool full = false, full_long = false;
  bool ramp = ramp_from(short_first, p_.gradual_min_delta, full);
  // Over the long window, ordinary motion has more time to look like a ramp; ask for a
  // proportionally larger change.
  if (!ramp && short_first > 0)
    ramp = ramp_from(0, kLongWindowDelta * p_.gradual_min_delta, full_long) && full_long;
  const cv::Mat& fs = ring_[short_first].second;
  const cv::Mat& ft = ring_.back().second;
  // One transition, one cut: after any cut, a full window must fail the test before the
  // next ramp counts. Otherwise the tail of a transition that the hard-cut test already
  // caught (or the fade-in after a fade-out cut) would be reported a second time.
  const bool black = mean_level(ft) < p_.black_level;
  if (black) last_black_ = frame_idx;
  if (full && !ramp) armed_ = true;
  const bool rising = ramp && !in_gradual_ && armed_;
  in_gradual_ = ramp;
  if (!rising) return false;
  // A fade is one cut. If the picture went black shortly after the last cut, that cut
  // was the fade-out (caught mid-ramp, or by the hard-cut test), and this ramp is the
  // matching fade-in.
  const int64_t near = 2 * window_frames_;
  const bool fade_in_of_reported = !pending_fade_ && last_black_ > last_cut_ &&
                                   last_black_ - last_cut_ <= near &&
                                   frame_idx - last_black_ <= near;
  if (black) {  // fading out: report the cut on the way back up
    pending_fade_ = !(last_black_ - last_cut_ <= near && last_cut_ > 0);
    return false;
  }
  if (fade_in_of_reported) return false;
  if (frame_idx - last_cut_ < min_gap_frames_) return false;
  last_kind_ = (pending_fade_ || mean_level(fs) < p_.black_level) ? CutKind::kFade
                                                                  : CutKind::kDissolve;
  pending_fade_ = false;
  armed_ = false;
  last_cut_ = frame_idx;
  return true;
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
  // Mid-ramp, a large step is the transition itself speeding up, not a new cut.
  const bool cut = score >= p_.min_score && score >= p_.ratio * std::max(avg, 1.0) &&
                   (frame_idx - last_cut_) >= min_gap_frames_ && !in_gradual_;
  if (cut) {
    last_cut_ = frame_idx;
    last_kind_ = CutKind::kCut;
    // A step resets the transition test; history across a cut is meaningless.
    ring_.clear();
    in_gradual_ = pending_fade_ = false;
    armed_ = false;
    return true;  // cut deltas are not added to the rolling average
  }
  recent_.push_back(score);
  recent_sum_ += score;
  if (static_cast<int>(recent_.size()) > p_.window) {
    recent_sum_ -= recent_.front();
    recent_.pop_front();
  }
  return p_.gradual && gradual_step(frame_idx, small_bgr);
}

}  // namespace mhprep
