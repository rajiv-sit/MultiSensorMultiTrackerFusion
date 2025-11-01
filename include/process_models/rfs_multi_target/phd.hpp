// process_phd.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// Represents intensity function over state space
struct PHDParams {
    double q = 1e-3;
};

class PHDModel : public IProcessModel {
public:
    explicit PHDModel(const PHDParams& p = {}) : params(p) {}

    std::string name() const override { return "PHD"; }
    int stateDim() const override { return 4; } // example: [x,y,vx,vy]
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        xn(0) += x(2)*dt;
        xn(1) += x(3)*dt;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(4,4);
        J(0,2)=dt; J(1,3)=dt;
        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(4,4)*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<PHDModel>(params);
    }

private:
    PHDParams params;
};

} // namespace tracker
