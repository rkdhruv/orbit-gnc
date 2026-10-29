#include "simcore/dynamics/force_sum.hpp"

#include <utility>

#include "simcore/check.hpp"

namespace simcore {

ForceSum& ForceSum::add(std::shared_ptr<const ForceModel> model) {
    SIMCORE_CHECK(model != nullptr, "ForceSum::add: null model");
    models_.push_back(std::move(model));
    return *this;
}

Eigen::Vector3d ForceSum::acceleration_eci(const State& s) const {
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    for (const auto& m : models_) {
        a += m->acceleration_eci(s);
    }
    return a;
}

}  // namespace simcore
