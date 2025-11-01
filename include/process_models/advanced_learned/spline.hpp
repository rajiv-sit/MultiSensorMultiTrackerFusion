// process_spline.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <vector>

namespace tracker {

// x = concatenated control points
struct SplineParams {
    double q_cp = 1e-4; // noise on control points
};

class SplineModel : public IProcessModel {
public:
    explicit SplineModel(const SplineParams& p = {}) : params(p) {}

    std::string name() const override { return "Spline"; }
    int stateDim() const override { return 8; } // example: 4 control points x 2D
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // Simple linear interpolation between control points
        for(int i=0;i<xn.size();++i) xn(i) += dt*0.01;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(stateDim(), stateDim());
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(stateDim(), stateDim()) * params.q_cp;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<SplineModel>(params);
    }

private:
    SplineParams params;
};

} // namespace tracker
