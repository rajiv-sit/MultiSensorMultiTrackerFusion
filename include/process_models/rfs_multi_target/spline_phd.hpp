// process_spline_phd.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <vector>

namespace tracker {

// State: control points for spline trajectories
struct SplinePHDParams {
    double q_cp = 1e-4;
};

class SplinePHDModel : public IProcessModel {
public:
    explicit SplinePHDModel(const SplinePHDParams& p = {}) : params(p) {}

    std::string name() const override { return "SplinePHD"; }
    int stateDim() const override { return 8; } // example 4 control points x 2D
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        for(int i=0;i<xn.size();++i) xn(i) += 0.01*dt; // placeholder evolution
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(stateDim(), stateDim());
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(stateDim(), stateDim())*params.q_cp;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<SplinePHDModel>(params);
    }

private:
    SplinePHDParams params;
};

} // namespace tracker
