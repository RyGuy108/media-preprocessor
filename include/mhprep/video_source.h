#pragma once
// Frame sources. Two backends:
//   * OpenCV  - cv::VideoCapture; every frame is converted to full-res BGR when analyzed.
//   * FFmpeg  - libavcodec directly. Analysis frames are produced by one sws_scale pass
//               straight from YUV to the small analysis size, so full-res BGR is only
//               materialised for the handful of frames we keep. Optionally skips decoding
//               non-reference frames (AVDISCARD_NONREF), which are 50-70% of a typical
//               H.264 stream.
#include <cstdint>
#include <memory>
#include <string>
#include <opencv2/core.hpp>

namespace mhprep {

enum class Backend { kOpenCV, kFFmpeg };

struct VideoInfo {
  double fps = 0;
  int width = 0, height = 0;
  int64_t frame_count = 0;  // container estimate
  double duration = 0;
};

struct SourceOptions {
  Backend backend = Backend::kFFmpeg;
  int analysis_width = 256;
  int decode_threads = 1;   // per worker; the pool provides the parallelism
  bool skip_nonref = true;  // FFmpeg backend only: skip decoding non-reference frames
  int io_timeout_ms = 0;    // 0 = rely on the supervisor's job deadline
};

class VideoSource {
 public:
  virtual ~VideoSource() = default;
  virtual const VideoInfo& info() const = 0;
  // Decode the next frame. If `analyze`, small() holds an analysis-size BGR copy.
  virtual bool next(bool analyze) = 0;
  virtual int64_t index() const = 0;           // frame number in the source timeline
  virtual const cv::Mat& small() const = 0;
  virtual cv::Mat full() = 0;                  // full-res BGR of the current frame
  virtual bool sparse() const { return false; } // true if some frames are never delivered
};

// Throws InputError if the file cannot be opened or has no video stream.
std::unique_ptr<VideoSource> open_video(const std::string& uri, const SourceOptions& opts);

}  // namespace mhprep
