// process_polar.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// x = [r, theta, r_dot, theta_dot]ᵀ
struct PolarParams {
    double q_r = 1e-3;
    double q_theta = 1e-4;
    double q_rdot = 1e-3;
    double q_thetadot = 1e-4;
};

class PolarModel : public IProcessModel {
public:
    explicit PolarModel(const PolarParams& p = {}) : params(p) {}

    std::string name() const override { return "PolarMotion"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
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
        Mat Q = Mat::Zero(4,4);
        Q(0,0)=params.q_r; Q(1,1)=params.q_theta;
        Q(2,2)=params.q_rdot; Q(3,3)=params.q_thetadot;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<PolarModel>(params);
    }

private:
    PolarParams params;
};

} // namespace tracker
