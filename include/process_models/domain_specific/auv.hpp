// process_auv.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct AUVParams {
    double q = 1e-3; // process noise
};

// Constant velocity with correlated drift
class AUVModel : public IProcessModel {
public:
    explicit AUVModel(const AUVParams& p = {}) : params(p) {}

    std::string name() const override { return "AUV"; }
    int stateDim() const override { return 6; } // x,y,z,vx,vy,vz
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // Add simple drift in z-axis (placeholder)
        xn.segment<3>(0) += xn.segment<3>(3)*dt;
        xn(2) += 0.01*dt; 
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(6,6);
        J.block<3,3>(0,3) = Mat::Identity(3,3)*dt;
        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(6,6) * params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<AUVModel>(params);
    }

private:
    AUVParams params;
};

} // namespace tracker
