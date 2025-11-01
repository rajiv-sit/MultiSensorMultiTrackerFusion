// process_intent.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [px, py, vx, vy, intent_id]ᵀ
struct IntentParams {
    double q = 1e-3;
};

class IntentModel : public IProcessModel {
public:
    explicit IntentModel(const IntentParams& p = {}) : params(p) {}

    std::string name() const override { return "IntentModel"; }
    int stateDim() const override { return 5; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        xn(0) += x(2)*dt;
        xn(1) += x(3)*dt;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(5,5);
        J(0,2)=dt; J(1,3)=dt;
        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(5,5)*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<IntentModel>(params);
    }

private:
    IntentParams params;
};

} // namespace tracker
