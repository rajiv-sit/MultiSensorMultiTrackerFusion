// process_bearing_only.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [px, py, vx, vy]ᵀ
struct BearingOnlyParams {
    double q_pos = 1e-3;
    double q_vel = 1e-3;
};

class BearingOnlyModel : public IProcessModel {
public:
    explicit BearingOnlyModel(const BearingOnlyParams& p = {}) : params(p) {}

    std::string name() const override { return "BearingOnly"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return false; }

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

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(4,4);
        Q(0,0)=Q(1,1)=params.q_pos;
        Q(2,2)=Q(3,3)=params.q_vel;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<BearingOnlyModel>(params);
    }

private:
    BearingOnlyParams params;
};

} // namespace tracker
