#include "mhprep/faces.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace mhprep {

FaceDetector::FaceDetector(const std::string& prototxt, const std::string& weights,
                           float min_conf, int min_size)
    : net_(cv::dnn::readNetFromCaffe(prototxt, weights)), min_conf_(min_conf),
      min_size_(min_size) {
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Face> FaceDetector::detect(const cv::Mat& bgr) {
  cv::Mat blob = cv::dnn::blobFromImage(bgr, 1.0, cv::Size(300, 300),
                                        cv::Scalar(104.0, 177.0, 123.0), false, false);
  net_.setInput(blob);
  cv::Mat out = net_.forward();  // [1, 1, N, 7]
  cv::Mat det(out.size[2], out.size[3], CV_32F, out.ptr<float>());
  std::vector<Face> faces;
  const cv::Rect frame(0, 0, bgr.cols, bgr.rows);
  for (int i = 0; i < det.rows; ++i) {
    const float conf = det.at<float>(i, 2);
    if (conf < min_conf_) continue;
    const int x1 = static_cast<int>(det.at<float>(i, 3) * bgr.cols);
    const int y1 = static_cast<int>(det.at<float>(i, 4) * bgr.rows);
    const int x2 = static_cast<int>(det.at<float>(i, 5) * bgr.cols);
    const int y2 = static_cast<int>(det.at<float>(i, 6) * bgr.rows);
    cv::Rect r = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)) & frame;
    if (r.width < min_size_ || r.height < min_size_) continue;
    faces.push_back({r, conf});
  }
  std::sort(faces.begin(), faces.end(),
            [](const Face& a, const Face& b) { return a.confidence > b.confidence; });
  return faces;
}

cv::Mat crop_face(const cv::Mat& bgr, const cv::Rect& box, int out, double margin) {
  const double cx = box.x + box.width / 2.0, cy = box.y + box.height / 2.0;
  int side = static_cast<int>(std::lround(std::max(box.width, box.height) * margin));
  side = std::min({side, bgr.cols, bgr.rows});
  int x = static_cast<int>(std::lround(cx - side / 2.0));
  int y = static_cast<int>(std::lround(cy - side / 2.0));
  x = std::clamp(x, 0, bgr.cols - side);
  y = std::clamp(y, 0, bgr.rows - side);
  cv::Mat crop;
  cv::resize(bgr(cv::Rect(x, y, side, side)), crop, cv::Size(out, out), 0, 0,
             side > out ? cv::INTER_AREA : cv::INTER_CUBIC);
  return crop;
}

double sharpness(const cv::Mat& img) {
  cv::Mat gray, lap;
  if (img.channels() == 1) gray = img;
  else cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  cv::Laplacian(gray, lap, CV_16S);
  cv::Scalar mean, stddev;
  cv::meanStdDev(lap, mean, stddev);
  return stddev[0] * stddev[0];
}

}  // namespace mhprep
