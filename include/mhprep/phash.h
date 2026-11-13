#pragma once
// Perceptual hashes for near-duplicate detection.
#include <bit>
#include <array>
#include <cstdint>
#include <string>
#include <opencv2/core.hpp>

namespace mhprep {

// 64-bit DCT perceptual hash (pHash). Robust to re-encoding, resizing, small crops
// and brightness shifts. Accepts BGR or grayscale input of any size.
uint64_t phash(const cv::Mat& img);

// 64-bit difference hash (cheaper, less robust). Used as a secondary check.
uint64_t dhash(const cv::Mat& img);

inline int hamming(uint64_t a, uint64_t b) { return std::popcount(a ^ b); }

// pHash is grayscale structure only, so two images with the same layout but different
// colours (the same model in two outfits, two shots of one set) can collide. Candidates
// are verified with a coarse colour layout: CIELAB means over a 4x4 grid.
using ColorSig = std::array<uint8_t, 48>;
ColorSig color_signature(const cv::Mat& bgr);
// Max over grid cells of |dL - mean dL| / 2 + |da| + |db|. Removing the mean L shift
// makes the check insensitive to a global brightness/exposure change.
int color_distance(const ColorSig& a, const ColorSig& b);
std::string sig_hex(const ColorSig& s);
ColorSig parse_sig_hex(const std::string& s);

// Default thresholds, tuned on the dev corpus (see bench/README).
constexpr int kDupHammingRadius = 8;
constexpr int kDupColorTolerance = 14;

}  // namespace mhprep
