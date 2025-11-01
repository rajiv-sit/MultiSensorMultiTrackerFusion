// process_ccs.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct CCSParams {
    double q_v = 1e-3; // process noise for speed
    double q_psi = 1e-4; // process noise for heading
};

// x = [px, py, v, psi]ᵀ
class CCSModel : public IProcessModel {
public:
    explicit CCSModel(const CCSParams& p = {}) : params(p) {}

    std::string name() const override { return "CCS"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return false; } // nonlinear due to heading

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        double px = x(0);
        double py = x(1);
        double v  = x(2);
        double psi = x(3);

        xn(0) = px + v * dt * std::cos(psi);
        xn(1) = py + v * dt * std::sin(psi);
        xn(2) = v;
        xn(3) = psi;

        return xn;
    }

    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(4,4);
        double psi = x(3);
        double v = x(2);

        J(0,2) = dt * std::cos(psi);
        J(0,3) = -v * dt * std::sin(psi);
        J(1,2) = dt * std::sin(psi);
        J(1,3) = v * dt * std::cos(psi);

        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(4,4);
        Q(2,2) = params.q_v * dt;
        Q(3,3) = params.q_psi * dt;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CCSModel>(*this);
    }

private:
    CCSParams params;
};

} // namespace tracker
