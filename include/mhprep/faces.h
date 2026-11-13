#pragma once
// Face detection (OpenCV DNN, ResNet-10 SSD) and API-ready face crops.
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace mhprep {

struct Face {
  cv::Rect box;
  float confidence;
};

class FaceDetector {
 public:
  FaceDetector(const std::string& prototxt, const std::string& weights, float min_conf = 0.6f,
               int min_size = 48);
  std::vector<Face> detect(const cv::Mat& bgr);

 private:
  cv::dnn::Net net_;
  float min_conf_;
  int min_size_;
};

// Square crop around `box` expanded by `margin`, shifted to stay inside the frame,
// resized to out x out. Face-swap and avatar endpoints want a centred head with
// some hair and chin, not a tight detector box.
cv::Mat crop_face(const cv::Mat& bgr, const cv::Rect& box, int out = 512, double margin = 1.8);

// Variance of the Laplacian: a cheap focus/blur measure.
double sharpness(const cv::Mat& img);

}  // namespace mhprep
