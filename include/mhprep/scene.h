#pragma once
// Content-based scene-cut detector with an adaptive threshold.
//
// Each analyzed frame is compared to the previous one in HSV space (mean absolute
// per-channel delta). A cut fires when the delta is both above an absolute floor and
// `ratio` times the recent average, so a fast pan through a busy scene does not
// trigger, while a hard cut between two similar-looking shots still does.
//
// Gradual transitions (dissolves, fades through black) change each frame only a little,
// so the per-frame delta never stands out. They are found by a second test over a short
// window of recent frames: if the window's endpoints differ a lot AND every frame in
// between lies on the straight line from one endpoint to the other (pixelwise linear
// interpolation, which is what a dissolve is), it's a transition. Camera motion moves
// content instead of blending it, so its intermediate frames are far off that line; a
// hard cut inside the window is a step, not a ramp. A fade is two such ramps through
// black; the cut is reported once, when the picture comes back.
#include <cstdint>
#include <deque>
#include <utility>
#include <opencv2/core.hpp>

namespace mhprep {

struct SceneParams {
  int analysis_width = 256;   // frames are downscaled to this width before analysis
  double min_score = 12.0;    // absolute floor on the HSV delta (0..255 scale)
  double ratio = 2.0;         // delta must be this many times the rolling average
  int window = 8;             // rolling-average length, in analyzed frames
  double min_scene_sec = 0.5; // suppress cuts closer than this to the previous one

  // Also detect dissolves and fades. Off by default: on hard-cut-only footage it costs a
  // little precision (README, "Dissolves and fades"); turn on for edited content.
  bool gradual = false;
  double gradual_window_sec = 0.4;   // span of the linearity test
  double gradual_min_delta = 26.0;   // window endpoints must differ by this (mean |BGR|, 0..255)
  double gradual_max_residual = 0.3; // off-line error allowed, as a fraction of that difference
  double black_level = 20.0;         // mean intensity below which a frame counts as black
};

enum class CutKind { kCut, kDissolve, kFade };
const char* to_string(CutKind k);

class SceneDetector {
 public:
  SceneDetector(SceneParams p, double fps);
  // Feed one analysis-size BGR frame. Returns true when `frame_idx` starts a new scene.
  bool push(int64_t frame_idx, const cv::Mat& small_bgr);
  double last_score() const { return last_score_; }
  CutKind last_cut_kind() const { return last_kind_; }
  // True while frames are mid-transition (blends make poor keyframes).
  bool in_transition() const { return in_gradual_ || pending_fade_; }
  void reset();

 private:
  SceneParams p_;
  int64_t min_gap_frames_;
  cv::Mat prev_hsv_, hsv_, diff_;
  std::deque<double> recent_;
  double recent_sum_ = 0;
  int64_t last_cut_ = 0;
  double last_score_ = 0;
  CutKind last_kind_ = CutKind::kCut;

  // Gradual-transition state.
  bool gradual_step(int64_t frame_idx, const cv::Mat& small_bgr);
  int64_t window_frames_;
  std::deque<std::pair<int64_t, cv::Mat>> ring_;  // (frame, tiny float BGR)
  cv::Mat tiny_, tmp_;
  bool in_gradual_ = false;   // the linear-ramp condition currently holds
  bool armed_ = true;         // a full window has failed the test since the last cut
  int64_t last_black_ = -1;   // latest frame darker than black_level
  bool pending_fade_ = false; // faded out to black; the cut is reported on the way back
};

// Analysis-size dimensions used by both the C++ pipeline and the Python baseline.
cv::Size analysis_size(int width, int height, int analysis_width);

}  // namespace mhprep
