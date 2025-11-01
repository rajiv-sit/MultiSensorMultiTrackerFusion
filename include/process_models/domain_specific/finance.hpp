// process_finance.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct FinanceParams {
    double q = 1e-3;
};

// Simple constant volatility model
class FinanceModel : public IProcessModel {
public:
    explicit FinanceModel(const FinanceParams& p = {}) : params(p) {}

    std::string name() const override { return "Finance"; }
    int stateDim() const override { return 2; } // price + velocity
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        xn(0) += x(1)*dt; // price evolves by velocity
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(2,2);
        J(0,1) = dt;
        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(2,2)*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<FinanceModel>(params);
    }

private:
    FinanceParams params;
};

} // namespace tracker
