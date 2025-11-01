// process_magnetic.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct MagneticParams {
    double q = 1e-4; // process noise
};

// State evolves under magnetic gradient (placeholder linear model)
class MagneticModel : public IProcessModel {
public:
    explicit MagneticModel(const MagneticParams& p = {}) : params(p) {}

    std::string name() const override { return "Magnetic"; }
    int stateDim() const override { return 3; } // e.g., magnetic vector x,y,z
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        // Placeholder: assume small drift along field
        return x + Vec::Ones(3) * 0.01 * dt;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(3,3);
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(3,3) * params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<MagneticModel>(params);
    }

private:
    MagneticParams params;
};

} // namespace tracker
