#include <gtest/gtest.h>

#include <random>

#include "mhprep/hash_index.h"
#include "mhprep/phash.h"

using namespace mhprep;

TEST(HashIndex, EmptyIndexFindsNothing) {
  HashIndex idx;
  EXPECT_FALSE(idx.nearest_within(42, 64).has_value());
}

TEST(HashIndex, MatchesBruteForce) {
  std::mt19937_64 rng(7);
  std::vector<uint64_t> stored;
  HashIndex idx;
  for (int i = 0; i < 5000; ++i) {
    const uint64_t h = rng();
    stored.push_back(h);
    idx.insert(h, i);
  }
  size_t visited = 0;
  for (int q = 0; q < 1000; ++q) {
    // Half the queries are noisy copies of stored hashes, half are random.
    uint64_t h = rng();
    if (q % 2 == 0) {
      h = stored[rng() % stored.size()];
      for (int k = 0; k < static_cast<int>(rng() % 8); ++k) h ^= uint64_t{1} << (rng() % 64);
    }
    int best = 65;
    for (uint64_t s : stored) best = std::min(best, hamming(s, h));
    auto got = idx.nearest_within(h, 6);
    if (best <= 6) {
      ASSERT_TRUE(got.has_value());
      EXPECT_EQ(got->distance, best);
      EXPECT_EQ(hamming(stored[static_cast<size_t>(got->id)], h), best);
    } else {
      EXPECT_FALSE(got.has_value());
    }
    visited += idx.last_visited();
  }
  // Uniform random hashes are the BK-tree's worst case (distances bunch up near 32),
  // and it still prunes most of the tree. Real pHashes cluster, so it prunes more.
  std::printf("avg nodes visited: %zu of %zu\n", visited / 1000, stored.size());
  EXPECT_LT(visited / 1000, stored.size() / 3);
}
