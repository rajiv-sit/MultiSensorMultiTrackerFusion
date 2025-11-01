// process_social.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <vector>

namespace tracker {

// x_i = [px, py, vx, vy]ᵀ
struct SocialParams {
    double q = 1e-3;
};

class SocialInteractionModel : public IProcessModel {
public:
    explicit SocialInteractionModel(const SocialParams& p = {}) : params(p) {}

    std::string name() const override { return "SocialInteraction"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // Simple interaction: attract to origin
        xn(0) += xn(2)*dt - 0.01*x(0);
        xn(1) += xn(3)*dt - 0.01*x(1);
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(4,4);
        J(0,2)=dt; J(1,3)=dt;
        J(0,0)=-0.01; J(1,1)=-0.01;
        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(4,4)*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<SocialInteractionModel>(params);
    }

private:
    SocialParams params;
};

} // namespace tracker
