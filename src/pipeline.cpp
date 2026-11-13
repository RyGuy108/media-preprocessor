#include "mhprep/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "mhprep/errors.h"
#include "mhprep/hash_index.h"
#include "mhprep/phash.h"

namespace fs = std::filesystem;

namespace mhprep {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int64_t file_size_or_zero(const std::string& p) {
  std::error_code ec;
  auto s = fs::file_size(p, ec);
  return ec ? 0 : static_cast<int64_t>(s);
}

struct Writer {
  const std::string& dir;
  const PipelineOptions& opts;
  Timing& timing;
  int64_t& bytes_out;

  int64_t write(const std::string& name, const cv::Mat& img, std::string& path_out) {
    if (!opts.write_assets) return 0;
    auto t0 = Clock::now();
    std::vector<uint8_t> buf;
    cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, opts.jpeg_quality});
    path_out = (fs::path(dir) / name).string();
    if (FILE* f = std::fopen(path_out.c_str(), "wb")) {
      std::fwrite(buf.data(), 1, buf.size(), f);
      std::fclose(f);
    } else {
      throw std::runtime_error("cannot write " + path_out);
    }
    timing.encode_ms += ms_since(t0);
    bytes_out += static_cast<int64_t>(buf.size());
    return static_cast<int64_t>(buf.size());
  }
};

// Detect faces in `img`, crop, dedup against `face_index`, and append assets.
void add_faces(FaceDetector& det, const cv::Mat& img, int scene, int64_t frame,
               const std::string& prefix, const PipelineOptions& opts, DupIndex& face_index,
               Manifest& m, Writer& w, std::vector<int>* scene_faces) {
  auto t0 = Clock::now();
  auto faces = det.detect(img);
  m.timing.faces_ms += ms_since(t0);
  int k = 0;
  for (const Face& f : faces) {
    t0 = Clock::now();
    cv::Mat crop = crop_face(img, f.box, opts.face_size);
    Asset a;
    a.kind = "face";
    a.scene = scene;
    a.frame = frame;
    a.box = f.box;
    a.confidence = f.confidence;
    a.phash = phash(crop);
    a.csig = color_signature(crop);
    a.sharpness = sharpness(crop);
    m.timing.faces_ms += ms_since(t0);
    const int id = static_cast<int>(m.assets.size());
    if (auto hit = face_index.find(a.phash, a.csig)) {
      a.duplicate_of = static_cast<int>(hit->id);
      a.dup_distance = hit->distance;
    } else {
      char name[64];
      std::snprintf(name, sizeof name, "%s_face%d.jpg", prefix.c_str(), k++);
      a.bytes = w.write(name, crop, a.path);
      face_index.add(a.phash, a.csig, id);
    }
    m.assets.push_back(std::move(a));
    if (scene_faces) scene_faces->push_back(id);
  }
}

}  // namespace

bool is_image_path(const std::string& path) {
  std::string ext = fs::path(path).extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp" || ext == ".bmp";
}

std::string hash_hex(uint64_t h) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

uint64_t parse_hash_hex(const std::string& s) { return std::stoull(s, nullptr, 16); }

Pipeline::Pipeline(std::string models_dir) : models_dir_(std::move(models_dir)) {}

FaceDetector& Pipeline::faces() {
  if (!faces_) {
    const fs::path d(models_dir_);
    faces_ = std::make_unique<FaceDetector>(
        (d / "deploy.prototxt").string(),
        (d / "res10_300x300_ssd_iter_140000_fp16.caffemodel").string());
  }
  return *faces_;
}

Manifest Pipeline::process(const std::string& input, const std::string& out_dir,
                           const PipelineOptions& opts) {
  if (opts.write_assets) fs::create_directories(out_dir);
  if (opts.detect_faces) faces();  // load before timing starts
  return is_image_path(input) ? process_image(input, out_dir, opts)
                              : process_video(input, out_dir, opts);
}

Manifest Pipeline::process_image(const std::string& input, const std::string& out_dir,
                                 const PipelineOptions& opts) {
  const auto t_start = Clock::now();
  Manifest m;
  m.input = input;
  m.kind = "image";
  m.bytes_in = file_size_or_zero(input);
  auto t0 = Clock::now();
  cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
  m.timing.decode_ms = ms_since(t0);
  if (img.empty()) throw InputError("cannot decode image: " + input);
  m.info.width = img.cols;
  m.info.height = img.rows;
  t0 = Clock::now();
  Asset a;
  a.kind = "image";
  a.path = input;
  a.phash = phash(img);
  a.csig = color_signature(img);
  a.sharpness = sharpness(img);
  a.bytes = m.bytes_in;
  m.timing.analyze_ms = ms_since(t0);
  m.assets.push_back(a);
  Writer w{out_dir, opts, m.timing, m.bytes_out};
  if (opts.detect_faces) {
    DupIndex idx(opts.dup_radius, opts.color_tolerance);
    add_faces(faces(), img, -1, -1, "img", opts, idx, m, w, nullptr);
  }
  m.timing.total_ms = ms_since(t_start);
  return m;
}

Manifest Pipeline::process_video(const std::string& input, const std::string& out_dir,
                                 const PipelineOptions& opts) {
  const auto t_start = Clock::now();
  Manifest m;
  m.input = input;
  m.kind = "video";
  m.bytes_in = file_size_or_zero(input);

  auto t0 = Clock::now();
  auto src = open_video(input, opts.source);
  m.timing.open_ms = ms_since(t0);
  m.info = src->info();
  const double fps = m.info.fps;
  SceneDetector det(opts.scene, fps);
  Writer w{out_dir, opts, m.timing, m.bytes_out};
  DupIndex kf_index(opts.dup_radius, opts.color_tolerance);
  DupIndex face_index(opts.dup_radius, opts.color_tolerance);

  struct Candidate {
    double sharp = -1;
    int64_t frame = -1;
    cv::Mat img;
  } best;
  int64_t scene_start = 0, last_frame = -1;
  std::string scene_transition;
  int analyzed_in_scene = 0;
  const int kf_phase = opts.keyframe_every / 2;
  const bool dense = !src->sparse();

  auto finish_scene = [&](int64_t end_frame) {
    Scene s;
    s.index = static_cast<int>(m.scenes.size());
    s.start_frame = scene_start;
    s.end_frame = end_frame;
    s.start_sec = scene_start / fps;
    s.end_sec = (end_frame + 1) / fps;
    s.transition = scene_transition;
    if (!best.img.empty()) {
      Asset a;
      a.kind = "keyframe";
      a.scene = s.index;
      a.frame = best.frame;
      a.sharpness = best.sharp;
      auto ta = Clock::now();
      a.phash = phash(best.img);
      a.csig = color_signature(best.img);
      m.timing.analyze_ms += ms_since(ta);
      const int id = static_cast<int>(m.assets.size());
      s.keyframe = id;
      char prefix[32];
      std::snprintf(prefix, sizeof prefix, "s%04d", s.index);
      if (auto hit = kf_index.find(a.phash, a.csig)) {
        // Same shot seen before (shot/reverse-shot, recap, loop). Don't send it twice.
        a.duplicate_of = static_cast<int>(hit->id);
        a.dup_distance = hit->distance;
        m.assets.push_back(std::move(a));
      } else {
        a.bytes = w.write(std::string(prefix) + ".jpg", best.img, a.path);
        kf_index.add(a.phash, a.csig, id);
        m.assets.push_back(std::move(a));
        if (opts.detect_faces)
          add_faces(faces(), best.img, s.index, best.frame, prefix, opts, face_index, m, w,
                    &s.faces);
      }
    }
    m.scenes.push_back(std::move(s));
    best = Candidate{};
    analyzed_in_scene = 0;
  };

  for (;;) {
    t0 = Clock::now();
    // For dense sources we decide before decoding whether to analyze, using the frame
    // number the next call will produce.
    const bool analyze = !dense || ((last_frame + 1) % opts.stride == 0);
    if (!src->next(analyze)) break;
    m.timing.decode_ms += ms_since(t0);
    ++m.frames_decoded;
    const int64_t idx = src->index();
    last_frame = idx;
    if (!analyze) continue;

    t0 = Clock::now();
    ++m.frames_analyzed;
    if (det.push(idx, src->small())) {
      m.timing.analyze_ms += ms_since(t0);
      finish_scene(idx - 1);
      scene_start = idx;
      scene_transition = to_string(det.last_cut_kind());
      t0 = Clock::now();
    }
    // Blended frames mid-transition make poor keyframes; skip them unless there's nothing else.
    const bool probe = best.img.empty() || ((analyzed_in_scene % opts.keyframe_every) == kf_phase &&
                                            !det.in_transition());
    if (probe) {
      const double sh = sharpness(src->small());
      if (sh > best.sharp) {
        m.timing.analyze_ms += ms_since(t0);
        auto td = Clock::now();
        src->full().copyTo(best.img);  // full-res BGR only for frames we may keep
        m.timing.decode_ms += ms_since(td);
        t0 = Clock::now();
        best.sharp = sh;
        best.frame = idx;
      }
    }
    ++analyzed_in_scene;
    m.timing.analyze_ms += ms_since(t0);
  }
  if (m.frames_decoded == 0) throw InputError("no decodable frames: " + input);
  finish_scene(last_frame);
  m.info.frame_count = last_frame + 1;
  m.info.duration = (last_frame + 1) / fps;
  m.timing.total_ms = ms_since(t_start);
  return m;
}

nlohmann::json to_json(const Manifest& m) {
  using nlohmann::json;
  json assets = json::array();
  int unique = 0, dups = 0, faces = 0;
  for (const auto& a : m.assets) {
    json j{{"kind", a.kind},
           {"phash", hash_hex(a.phash)},
           {"csig", sig_hex(a.csig)},
           {"sharpness", a.sharpness}};
    if (!a.path.empty()) j["path"] = a.path;
    if (a.scene >= 0) j["scene"] = a.scene;
    if (a.frame >= 0) j["frame"] = a.frame;
    if (a.kind == "face") {
      j["confidence"] = a.confidence;
      j["box"] = {a.box.x, a.box.y, a.box.width, a.box.height};
      ++faces;
    }
    if (a.bytes) j["bytes"] = a.bytes;
    if (a.duplicate_of >= 0) {
      j["duplicate_of"] = a.duplicate_of;
      j["dup_distance"] = a.dup_distance;
      ++dups;
    } else {
      ++unique;
    }
    assets.push_back(std::move(j));
  }
  json scenes = json::array();
  for (const auto& s : m.scenes)
    scenes.push_back({{"index", s.index},
                      {"start_frame", s.start_frame},
                      {"end_frame", s.end_frame},
                      {"start_sec", s.start_sec},
                      {"end_sec", s.end_sec},
                      {"keyframe", s.keyframe},
                      {"faces", s.faces},
                      {"transition", s.transition.empty() ? json(nullptr) : json(s.transition)}});
  return json{
      {"input", m.input},
      {"kind", m.kind},
      {"video", {{"fps", m.info.fps}, {"width", m.info.width}, {"height", m.info.height},
                 {"frames", m.info.frame_count}, {"duration_sec", m.info.duration}}},
      {"frames_decoded", m.frames_decoded},
      {"frames_analyzed", m.frames_analyzed},
      {"scenes", scenes},
      {"assets", assets},
      {"summary", {{"scenes", m.scenes.size()}, {"assets_unique", unique},
                   {"assets_duplicate", dups}, {"faces", faces}}},
      {"timing_ms", {{"open", m.timing.open_ms}, {"decode", m.timing.decode_ms},
                     {"analyze", m.timing.analyze_ms}, {"faces", m.timing.faces_ms},
                     {"encode", m.timing.encode_ms}, {"total", m.timing.total_ms}}},
      {"bytes_in", m.bytes_in},
      {"bytes_out", m.bytes_out},
  };
}

PipelineOptions options_from_json(const nlohmann::json& j, PipelineOptions o) {
  if (j.contains("backend"))
    o.source.backend = j["backend"] == "opencv" ? Backend::kOpenCV : Backend::kFFmpeg;
  o.source.skip_nonref = j.value("skip_nonref", o.source.skip_nonref);
  o.source.decode_threads = j.value("decode_threads", o.source.decode_threads);
  o.source.io_timeout_ms = j.value("io_timeout_ms", o.source.io_timeout_ms);
  o.stride = std::max(1, j.value("stride", o.stride));
  o.keyframe_every = std::max(1, j.value("keyframe_every", o.keyframe_every));
  o.detect_faces = j.value("faces", o.detect_faces);
  o.face_size = j.value("face_size", o.face_size);
  o.dup_radius = j.value("dup_radius", o.dup_radius);
  o.color_tolerance = j.value("color_tolerance", o.color_tolerance);
  o.jpeg_quality = j.value("jpeg_quality", o.jpeg_quality);
  o.write_assets = j.value("write_assets", o.write_assets);
  o.scene.min_score = j.value("scene_min_score", o.scene.min_score);
  o.scene.ratio = j.value("scene_ratio", o.scene.ratio);
  o.scene.gradual = j.value("scene_gradual", o.scene.gradual);
  o.scene.gradual_min_delta = j.value("gradual_min_delta", o.scene.gradual_min_delta);
  o.scene.gradual_max_residual = j.value("gradual_max_residual", o.scene.gradual_max_residual);
  o.scene.gradual_window_sec = j.value("gradual_window_sec", o.scene.gradual_window_sec);
  o.scene.analysis_width = j.value("analysis_width", o.scene.analysis_width);
  o.source.analysis_width = o.scene.analysis_width;
  return o;
}

}  // namespace mhprep
