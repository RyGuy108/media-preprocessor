#pragma once
#include <stdexcept>
#include <string>

namespace mhprep {

// The input itself is bad (unreadable, not media, truncated). Retrying will not help.
struct InputError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace mhprep
