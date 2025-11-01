// process_mjls.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <vector>

namespace tracker {

// x = [px, py, vx, vy]ᵀ, multiple modes
struct MJLSParams {
    int num_modes = 2;
    double q = 1e-3;
};

class MJLSModel : public IProcessModel {
public:
    explicit MJLSModel(const MJLSParams& p = {}) : params(p) {}

    std::string name() const override { return "MJLS"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // Simple linear model for current mode
        xn(0) += xn(2)*dt;
        xn(1) += xn(3)*dt;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(4,4);
        J(0,2) = dt;
        J(1,3) = dt;
        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(4,4)*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<MJLSModel>(params);
    }

private:
    MJLSParams params;
};

} // namespace tracker
