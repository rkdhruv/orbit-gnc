#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "simcore/dynamics/force_model.hpp"

namespace simcore {

// Sum of several force models, e.g. point-mass gravity + J2 + drag. Accelerations are added in
// insertion order, so the result is deterministic and equal to the hand-written sum bit for bit.
class ForceSum final : public ForceModel {
public:
    // Appends a model (shared, so one instance can sit in several sums). Throws on null.
    ForceSum& add(std::shared_ptr<const ForceModel> model);

    [[nodiscard]] std::size_t size() const { return models_.size(); }

    [[nodiscard]] Eigen::Vector3d acceleration_eci(const State& s) const override;

private:
    std::vector<std::shared_ptr<const ForceModel>> models_;
};

}  // namespace simcore
