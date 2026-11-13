#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mhprep {

// Thread-safe latency recorder: Prometheus-style cumulative buckets plus exact
// percentiles over the retained samples.
class LatencyRecorder {
 public:
  explicit LatencyRecorder(std::vector<double> bounds_ms);
  void record(double ms);
  double percentile(double p) const;  // p in [0, 100]
  uint64_t count() const;
  // Prometheus histogram exposition for metric `name`.
  std::string prometheus(const std::string& name, const std::string& labels = "") const;

 private:
  mutable std::mutex mu_;
  std::vector<double> bounds_;
  std::vector<uint64_t> buckets_;
  std::vector<double> samples_;
  double sum_ = 0;
  uint64_t count_ = 0;
};

}  // namespace mhprep
