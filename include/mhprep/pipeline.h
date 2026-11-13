#pragma once
// One job = one input file or URL. A video becomes: scene list, one sharp keyframe per
// scene, near-duplicate keyframes dropped, face crops from the unique keyframes. An image
// becomes: perceptual hash, sharpness, face crops. The manifest says exactly which
// files are worth sending to the Magic Hour API.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "mhprep/faces.h"
#include "mhprep/phash.h"
#include "mhprep/scene.h"
#include "mhprep/video_source.h"

namespace mhprep {

struct PipelineOptions {
  SourceOptions source;
  SceneParams scene;
  int stride = 2;          // analyze every Nth frame (ignored for sparse sources)
  int keyframe_every = 3;  // sharpness probe every Nth analyzed frame of a scene
  bool detect_faces = true;
  int face_size = 512;
  int dup_radius = kDupHammingRadius;        // pHash candidate radius
  int color_tolerance = kDupColorTolerance;  // colour-layout verification
  int jpeg_quality = 90;
  bool write_assets = true;
};

struct Asset {
  std::string kind;  // "keyframe" | "face" | "image"
  std::string path;  // empty when not written (duplicate)
  int scene = -1;
  int64_t frame = -1;
  uint64_t phash = 0;
  ColorSig csig{};
  double sharpness = 0;
  float confidence = 0;
  cv::Rect box;
  int64_t bytes = 0;
  int duplicate_of = -1;  // index into Manifest::assets
  int dup_distance = -1;
};

struct Scene {
  int index = 0;
  int64_t start_frame = 0, end_frame = 0;  // inclusive
  double start_sec = 0, end_sec = 0;
  int keyframe = -1;  // index into Manifest::assets
  std::vector<int> faces;
  std::string transition;  // how the scene was entered: cut | dissolve | fade ("" for the first)
};

struct Timing {
  double open_ms = 0, decode_ms = 0, analyze_ms = 0, faces_ms = 0, encode_ms = 0, total_ms = 0;
};

struct Manifest {
  std::string input;
  std::string kind;  // "video" | "image"
  VideoInfo info;
  int64_t frames_decoded = 0, frames_analyzed = 0;
  std::vector<Scene> scenes;
  std::vector<Asset> assets;
  Timing timing;
  int64_t bytes_in = 0, bytes_out = 0;
};

class Pipeline {
 public:
  explicit Pipeline(std::string models_dir);
  void warm_up() { faces(); }
  Manifest process(const std::string& input, const std::string& out_dir,
                   const PipelineOptions& opts);

 private:
  Manifest process_video(const std::string& input, const std::string& out_dir,
                         const PipelineOptions& opts);
  Manifest process_image(const std::string& input, const std::string& out_dir,
                         const PipelineOptions& opts);
  FaceDetector& faces();
  std::string models_dir_;
  std::unique_ptr<FaceDetector> faces_;
};

bool is_image_path(const std::string& path);
std::string hash_hex(uint64_t h);
uint64_t parse_hash_hex(const std::string& s);
nlohmann::json to_json(const Manifest& m);
PipelineOptions options_from_json(const nlohmann::json& j, PipelineOptions base = {});

}  // namespace mhprep
