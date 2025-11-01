// process_extended_cv.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// x = [px, py, v, psi]ᵀ
struct ExtCVParams {
    double q_pos = 1e-3;
    double q_vel = 1e-3;
    double q_psi = 1e-4;
};

class ExtendedCVModel : public IProcessModel {
public:
    explicit ExtendedCVModel(const ExtCVParams& p = {}) : params(p) {}

    std::string name() const override { return "ExtendedCV"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        double px = x(0), py = x(1), v = x(2), psi = x(3);
        Vec xn(4);
        xn(0) = px + v*std::cos(psi)*dt;
        xn(1) = py + v*std::sin(psi)*dt;
        xn(2) = v;
        xn(3) = psi;
        return xn;
    }

    Mat F(const Vec& x, double dt) const override {
        double psi = x(3);
        Mat J = Mat::Identity(4,4);
        J(0,2) = std::cos(psi)*dt;
        J(0,3) = -x(2)*std::sin(psi)*dt;
        J(1,2) = std::sin(psi)*dt;
        J(1,3) = x(2)*std::cos(psi)*dt;
        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(4,4);
        Q(0,0)=Q(1,1)=params.q_pos;
        Q(2,2)=params.q_vel;
        Q(3,3)=params.q_psi;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<ExtendedCVModel>(params);
    }

private:
    ExtCVParams params;
};

} // namespace tracker
