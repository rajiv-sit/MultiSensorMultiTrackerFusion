// process_adaptive_cv.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [px, py, vx, vy]ᵀ
struct AdaptiveCVParams {
    double q_base = 1e-4;
    double q_acc_factor = 1e-2;
};

class AdaptiveCVModel : public IProcessModel {
public:
    explicit AdaptiveCVModel(const AdaptiveCVParams& p = {}) : params(p) {}

    std::string name() const override { return "AdaptiveCV"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        xn(0) += x(2)*dt;
        xn(1) += x(3)*dt;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(4,4);
        J(0,2) = dt;
        J(1,3) = dt;
        return J;
    }

    Mat Qd(const Vec& x, double dt) const {
        double ax = x(2); double ay = x(3);
        double q = params.q_base + params.q_acc_factor * (ax*ax + ay*ay);
        Mat Q = Mat::Zero(4,4);
        Q(0,0)=Q(1,1)=0;
        Q(2,2)=Q(3,3)=q;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<AdaptiveCVModel>(params);
    }

private:
    AdaptiveCVParams params;
};

} // namespace tracker
