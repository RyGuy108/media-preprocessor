#pragma once
// Content-based scene-cut detector with an adaptive threshold.
//
// Each analyzed frame is compared to the previous one in HSV space (mean absolute
// per-channel delta). A cut fires when the delta is both above an absolute floor and
// `ratio` times the recent average, so a fast pan through a busy scene does not
// trigger, while a hard cut between two similar-looking shots still does.
#include <cstdint>
#include <deque>
#include <opencv2/core.hpp>

namespace mhprep {

struct SceneParams {
  int analysis_width = 256;   // frames are downscaled to this width before analysis
  double min_score = 12.0;    // absolute floor on the HSV delta (0..255 scale)
  double ratio = 2.0;         // delta must be this many times the rolling average
  int window = 8;             // rolling-average length, in analyzed frames
  double min_scene_sec = 0.5; // suppress cuts closer than this to the previous one
};

class SceneDetector {
 public:
  SceneDetector(SceneParams p, double fps);
  // Feed one analysis-size BGR frame. Returns true when `frame_idx` starts a new scene.
  bool push(int64_t frame_idx, const cv::Mat& small_bgr);
  double last_score() const { return last_score_; }
  void reset();

 private:
  SceneParams p_;
  int64_t min_gap_frames_;
  cv::Mat prev_hsv_, hsv_, diff_;
  std::deque<double> recent_;
  double recent_sum_ = 0;
  int64_t last_cut_ = 0;
  double last_score_ = 0;
};

// Analysis-size dimensions used by both the C++ pipeline and the Python baseline.
cv::Size analysis_size(int width, int height, int analysis_width);

}  // namespace mhprep
