#include "mhprep/metrics.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace mhprep {

LatencyRecorder::LatencyRecorder(std::vector<double> bounds_ms)
    : bounds_(std::move(bounds_ms)), buckets_(bounds_.size() + 1, 0) {}

void LatencyRecorder::record(double ms) {
  std::lock_guard lk(mu_);
  const auto it = std::lower_bound(bounds_.begin(), bounds_.end(), ms);
  ++buckets_[static_cast<size_t>(it - bounds_.begin())];
  if (samples_.size() < 1'000'000) samples_.push_back(ms);
  sum_ += ms;
  ++count_;
}

double LatencyRecorder::percentile(double p) const {
  std::lock_guard lk(mu_);
  if (samples_.empty()) return 0;
  std::vector<double> s = samples_;
  const size_t k = std::min(s.size() - 1,
                            static_cast<size_t>(std::ceil(p / 100.0 * s.size())) - (p > 0 ? 1 : 0));
  std::nth_element(s.begin(), s.begin() + static_cast<long>(k), s.end());
  return s[k];
}

uint64_t LatencyRecorder::count() const {
  std::lock_guard lk(mu_);
  return count_;
}

std::string LatencyRecorder::prometheus(const std::string& name, const std::string& labels) const {
  std::lock_guard lk(mu_);
  std::ostringstream o;
  const std::string sep = labels.empty() ? "" : ",";
  uint64_t cum = 0;
  for (size_t i = 0; i < bounds_.size(); ++i) {
    cum += buckets_[i];
    o << name << "_bucket{" << labels << sep << "le=\"" << bounds_[i] / 1000.0 << "\"} " << cum
      << "\n";
  }
  cum += buckets_.back();
  o << name << "_bucket{" << labels << sep << "le=\"+Inf\"} " << cum << "\n";
  o << name << "_sum" << (labels.empty() ? "" : "{" + labels + "}") << " " << sum_ / 1000.0
    << "\n";
  o << name << "_count" << (labels.empty() ? "" : "{" + labels + "}") << " " << count_ << "\n";
  return o.str();
}

}  // namespace mhprep
