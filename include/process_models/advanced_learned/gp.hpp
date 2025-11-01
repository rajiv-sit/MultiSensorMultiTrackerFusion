// process_gp.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = state vector
struct GPParams {
    double q_gp = 1e-3; // GP process noise
};

class GPMotionModel : public IProcessModel {
public:
    explicit GPMotionModel(const GPParams& p = {}) : params(p) {}

    std::string name() const override { return "GaussianProcess"; }
    int stateDim() const override { return 4; } // example: [x,y,vx,vy]
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        // Placeholder: GP mean prediction
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
        return Mat::Identity(4,4) * params.q_gp;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<GPMotionModel>(params);
    }

private:
    GPParams params;
};

} // namespace tracker
