// process_brownian.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// ----------------------
// Brownian Motion Parameters
// ----------------------
struct BrownianParams {
    double q = 0.01; // process noise (position only)
    int axis = 2;    // 2D or 3D
};

// ----------------------
// Brownian Motion Model
// ----------------------
class BrownianModel : public IProcessModel {
public:
    explicit BrownianModel(const BrownianParams& p = {}) : params(p) {}

    std::string name() const override { return "Brownian"; }

    // State: position only [px, py] or [px, py, pz]
    int stateDim() const override { return params.axis; }

    bool isLinear() const override { return true; }

    // State propagation: x_{k+1} = x_k + w
    Vec f(const Vec& x, double /*dt*/) const override {
        return x; // deterministic part is identity
    }

    // Jacobian = identity
    Mat F(const Vec&, double /*dt*/) const override {
        return Mat::Identity(params.axis, params.axis);
    }

    // Process noise
    Mat Qd(double /*dt*/) const override {
        return Mat::Identity(params.axis, params.axis) * params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<BrownianModel>(*this);
    }

private:
    BrownianParams params;
};

} // namespace tracker
