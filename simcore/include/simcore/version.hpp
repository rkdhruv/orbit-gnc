#pragma once

#include <string_view>

namespace simcore {

// Library version string, e.g. "0.1.0".
std::string_view version() noexcept;

}  // namespace simcore
