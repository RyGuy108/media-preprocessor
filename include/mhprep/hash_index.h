#pragma once
// BK-tree over 64-bit hashes with Hamming distance. Lookups touch O(log n) nodes for
// small radii, so cross-upload dedup stays cheap as a tenant's history grows.
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace mhprep {

struct HashMatch {
  int64_t id;
  int distance;
};

class HashIndex {
 public:
  void insert(uint64_t hash, int64_t id);
  // Closest stored hash within `radius`, if any.
  std::optional<HashMatch> nearest_within(uint64_t hash, int radius) const;
  // Every stored hash within `radius`, closest first.
  std::vector<HashMatch> all_within(uint64_t hash, int radius) const;
  size_t size() const { return nodes_.size(); }
  // Number of nodes visited by the last query (for tests / profiling).
  size_t last_visited() const { return last_visited_; }

 private:
  struct Node {
    uint64_t hash;
    int64_t id;
    std::array<int32_t, 65> child;  // child[d] = index of subtree at distance d, -1 if none
  };
  std::vector<Node> nodes_;
  mutable size_t last_visited_ = 0;
};

// BK-tree candidates verified by colour layout. This is the dedup decision used both
// inside a job (repeated shots) and across a tenant's uploads.
class DupIndex {
 public:
  DupIndex(int radius, int color_tolerance) : radius_(radius), tol_(color_tolerance) {}
  // Closest previously-added item that passes both checks; `id` is the caller's id.
  std::optional<HashMatch> find(uint64_t hash, const std::array<uint8_t, 48>& sig) const;
  void add(uint64_t hash, const std::array<uint8_t, 48>& sig, int64_t id);
  size_t size() const { return ids_.size(); }

 private:
  int radius_, tol_;
  HashIndex index_;  // stores positions into sigs_/ids_
  std::vector<std::array<uint8_t, 48>> sigs_;
  std::vector<int64_t> ids_;
};

}  // namespace mhprep
