#include "mhprep/video_source.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "mhprep/errors.h"
#include "mhprep/scene.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace mhprep {
namespace {

// ---------------------------------------------------------------------------------------
// OpenCV backend
// ---------------------------------------------------------------------------------------
class OpenCVSource final : public VideoSource {
 public:
  OpenCVSource(const std::string& uri, const SourceOptions& o) {
    // The FFmpeg capture reads its options from the environment at open time.
    const std::string capopts = "threads;" + std::to_string(o.decode_threads);
    ::setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", capopts.c_str(), 1);
    std::vector<int> params;
    if (o.io_timeout_ms > 0)
      params = {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, o.io_timeout_ms, cv::CAP_PROP_READ_TIMEOUT_MSEC,
                o.io_timeout_ms};
    if (!cap_.open(uri, cv::CAP_FFMPEG, params) || !cap_.isOpened())
      throw InputError("cannot open video: " + uri);
    info_.fps = cap_.get(cv::CAP_PROP_FPS);
    if (!(info_.fps > 0 && info_.fps < 1000)) info_.fps = 25.0;
    info_.width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    info_.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    info_.frame_count = static_cast<int64_t>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    info_.duration = info_.frame_count / info_.fps;
    if (info_.width <= 0 || info_.height <= 0) throw InputError("no video stream: " + uri);
    asize_ = analysis_size(info_.width, info_.height, o.analysis_width);
  }
  const VideoInfo& info() const override { return info_; }
  bool next(bool analyze) override {
    if (!cap_.grab()) return false;
    ++idx_;
    have_full_ = false;
    if (analyze) {
      cap_.retrieve(full_);
      have_full_ = true;
      cv::resize(full_, small_, asize_, 0, 0, cv::INTER_AREA);
    }
    return true;
  }
  int64_t index() const override { return idx_; }
  const cv::Mat& small() const override { return small_; }
  cv::Mat full() override {
    if (!have_full_) {
      cap_.retrieve(full_);
      have_full_ = true;
    }
    return full_;
  }

 private:
  cv::VideoCapture cap_;
  VideoInfo info_;
  cv::Size asize_;
  cv::Mat full_, small_;
  bool have_full_ = false;
  int64_t idx_ = -1;
};

// ---------------------------------------------------------------------------------------
// FFmpeg backend
// ---------------------------------------------------------------------------------------
struct Deadline {
  int64_t timeout_ms = 0;
  std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
};

int interrupt_cb(void* opaque) {
  auto* d = static_cast<Deadline*>(opaque);
  if (d->timeout_ms <= 0) return 0;
  const auto idle = std::chrono::steady_clock::now() - d->last_progress;
  return std::chrono::duration_cast<std::chrono::milliseconds>(idle).count() > d->timeout_ms;
}

class FFmpegSource final : public VideoSource {
 public:
  FFmpegSource(const std::string& uri, const SourceOptions& o) : opts_(o) {
    try {
      init(uri, o);
    } catch (...) {
      release();  // the destructor does not run when a constructor throws
      throw;
    }
  }
  ~FFmpegSource() override { release(); }

  void init(const std::string& uri, const SourceOptions& o) {
    av_log_set_level(AV_LOG_QUIET);
    deadline_.timeout_ms = o.io_timeout_ms;
    fmt_ = avformat_alloc_context();
    fmt_->interrupt_callback = {interrupt_cb, &deadline_};
    if (avformat_open_input(&fmt_, uri.c_str(), nullptr, nullptr) < 0) {
      fmt_ = nullptr;  // freed by avformat_open_input on failure
      throw InputError("cannot open video: " + uri);
    }
    if (avformat_find_stream_info(fmt_, nullptr) < 0) throw InputError("no stream info: " + uri);
    const AVCodec* codec = nullptr;
    stream_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (stream_ < 0 || !codec) throw InputError("no video stream: " + uri);
    AVStream* st = fmt_->streams[stream_];
    for (unsigned i = 0; i < fmt_->nb_streams; ++i)
      if (static_cast<int>(i) != stream_) fmt_->streams[i]->discard = AVDISCARD_ALL;

    dec_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec_, st->codecpar);
    dec_->thread_count = o.decode_threads;
    if (o.skip_nonref) dec_->skip_frame = AVDISCARD_NONREF;
    if (avcodec_open2(dec_, codec, nullptr) < 0) throw InputError("cannot open decoder: " + uri);

    AVRational fr = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    info_.fps = fr.num && fr.den ? av_q2d(fr) : 25.0;
    if (!(info_.fps > 0 && info_.fps < 1000)) info_.fps = 25.0;
    info_.width = dec_->width;
    info_.height = dec_->height;
    if (info_.width <= 0 || info_.height <= 0) throw InputError("bad dimensions: " + uri);
    tb_ = av_q2d(st->time_base);
    start_ts_ = st->start_time != AV_NOPTS_VALUE ? st->start_time : 0;
    if (st->duration != AV_NOPTS_VALUE) info_.duration = st->duration * tb_;
    else if (fmt_->duration != AV_NOPTS_VALUE) info_.duration = fmt_->duration / double(AV_TIME_BASE);
    info_.frame_count = st->nb_frames > 0 ? st->nb_frames
                                          : static_cast<int64_t>(info_.duration * info_.fps);
    asize_ = analysis_size(info_.width, info_.height, o.analysis_width);
    pkt_ = av_packet_alloc();
    frame_ = av_frame_alloc();
  }
  void release() {
    sws_freeContext(sws_small_);
    sws_freeContext(sws_full_);
    sws_small_ = sws_full_ = nullptr;
    av_frame_free(&frame_);
    av_packet_free(&pkt_);
    avcodec_free_context(&dec_);
    if (fmt_) avformat_close_input(&fmt_);
  }
  const VideoInfo& info() const override { return info_; }
  bool sparse() const override { return opts_.skip_nonref; }

  bool next(bool analyze) override {
    if (!decode_one()) return false;
    have_full_ = false;
    const int64_t ts = frame_->best_effort_timestamp;
    if (ts != AV_NOPTS_VALUE) {
      idx_ = std::llround((ts - start_ts_) * tb_ * info_.fps);
      if (idx_ <= last_idx_) idx_ = last_idx_ + 1;  // guard against non-monotonic timestamps
    } else {
      idx_ = last_idx_ + 1;
    }
    last_idx_ = idx_;
    if (analyze) {
      small_.create(asize_, CV_8UC3);
      sws_small_ = sws_getCachedContext(sws_small_, frame_->width, frame_->height,
                                        static_cast<AVPixelFormat>(frame_->format), asize_.width,
                                        asize_.height, AV_PIX_FMT_BGR24, SWS_AREA, nullptr,
                                        nullptr, nullptr);
      scale_into(sws_small_, small_);
    }
    return true;
  }
  int64_t index() const override { return idx_; }
  const cv::Mat& small() const override { return small_; }
  cv::Mat full() override {
    if (!have_full_) {
      full_.create(frame_->height, frame_->width, CV_8UC3);
      sws_full_ = sws_getCachedContext(sws_full_, frame_->width, frame_->height,
                                       static_cast<AVPixelFormat>(frame_->format), frame_->width,
                                       frame_->height, AV_PIX_FMT_BGR24, SWS_BICUBIC, nullptr,
                                       nullptr, nullptr);
      scale_into(sws_full_, full_);
      have_full_ = true;
    }
    return full_;
  }

 private:
  void scale_into(SwsContext* ctx, cv::Mat& dst) {
    uint8_t* data[4] = {dst.data, nullptr, nullptr, nullptr};
    int linesize[4] = {static_cast<int>(dst.step[0]), 0, 0, 0};
    sws_scale(ctx, frame_->data, frame_->linesize, 0, frame_->height, data, linesize);
  }

  bool decode_one() {
    for (;;) {
      int r = avcodec_receive_frame(dec_, frame_);
      if (r == 0) {
        deadline_.last_progress = std::chrono::steady_clock::now();
        return true;
      }
      if (r == AVERROR_EOF) return false;
      if (r != AVERROR(EAGAIN)) return false;  // decoder error: treat as end of stream
      if (draining_) return false;
      r = av_read_frame(fmt_, pkt_);
      if (r < 0) {
        draining_ = true;
        avcodec_send_packet(dec_, nullptr);
        continue;
      }
      if (pkt_->stream_index == stream_) {
        deadline_.last_progress = std::chrono::steady_clock::now();
        avcodec_send_packet(dec_, pkt_);  // corrupt packets are skipped, not fatal
      }
      av_packet_unref(pkt_);
    }
  }

  SourceOptions opts_;
  Deadline deadline_;
  AVFormatContext* fmt_ = nullptr;
  AVCodecContext* dec_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frame_ = nullptr;
  SwsContext* sws_small_ = nullptr;
  SwsContext* sws_full_ = nullptr;
  int stream_ = -1;
  double tb_ = 0;
  int64_t start_ts_ = 0;
  bool draining_ = false;
  VideoInfo info_;
  cv::Size asize_;
  cv::Mat small_, full_;
  bool have_full_ = false;
  int64_t idx_ = -1, last_idx_ = -1;
};

}  // namespace

std::unique_ptr<VideoSource> open_video(const std::string& uri, const SourceOptions& opts) {
  if (opts.backend == Backend::kOpenCV) return std::make_unique<OpenCVSource>(uri, opts);
  return std::make_unique<FFmpegSource>(uri, opts);
}

}  // namespace mhprep
