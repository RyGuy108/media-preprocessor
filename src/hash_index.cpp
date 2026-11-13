#include "mhprep/hash_index.h"

#include <algorithm>

#include "mhprep/phash.h"

namespace mhprep {

void HashIndex::insert(uint64_t hash, int64_t id) {
  Node n{hash, id, {}};
  n.child.fill(-1);
  if (nodes_.empty()) {
    nodes_.push_back(n);
    return;
  }
  int32_t cur = 0;
  for (;;) {
    const int d = hamming(nodes_[cur].hash, hash);
    const int32_t next = nodes_[cur].child[d];
    if (next < 0) {
      nodes_[cur].child[d] = static_cast<int32_t>(nodes_.size());
      nodes_.push_back(n);
      return;
    }
    cur = next;
  }
}

std::optional<HashMatch> HashIndex::nearest_within(uint64_t hash, int radius) const {
  auto all = all_within(hash, radius);
  if (all.empty()) return std::nullopt;
  return all.front();
}

std::vector<HashMatch> HashIndex::all_within(uint64_t hash, int radius) const {
  last_visited_ = 0;
  std::vector<HashMatch> out;
  if (nodes_.empty()) return out;
  std::vector<int32_t> stack{0};
  while (!stack.empty()) {
    const Node& n = nodes_[stack.back()];
    stack.pop_back();
    ++last_visited_;
    const int d = hamming(n.hash, hash);
    if (d <= radius) out.push_back(HashMatch{n.id, d});
    // Triangle inequality: only subtrees at distance in [d - r, d + r] can match.
    const int lo = std::max(0, d - radius), hi = std::min(64, d + radius);
    for (int k = lo; k <= hi; ++k)
      if (n.child[k] >= 0) stack.push_back(n.child[k]);
  }
  std::stable_sort(out.begin(), out.end(), [](const HashMatch& a, const HashMatch& b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
  });
  return out;
}

std::optional<HashMatch> DupIndex::find(uint64_t hash, const std::array<uint8_t, 48>& sig) const {
  for (const HashMatch& m : index_.all_within(hash, radius_)) {
    const auto pos = static_cast<size_t>(m.id);
    if (color_distance(sigs_[pos], sig) <= tol_) return HashMatch{ids_[pos], m.distance};
  }
  return std::nullopt;
}

void DupIndex::add(uint64_t hash, const std::array<uint8_t, 48>& sig, int64_t id) {
  index_.insert(hash, static_cast<int64_t>(ids_.size()));
  sigs_.push_back(sig);
  ids_.push_back(id);
}

}  // namespace mhprep
