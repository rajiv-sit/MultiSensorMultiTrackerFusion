// process_human.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct HumanParams {
    double q = 1e-3;
};

// State: simplified skeleton joints positions
class HumanModel : public IProcessModel {
public:
    explicit HumanModel(const HumanParams& p = {}) : params(p) {}

    std::string name() const override { return "Human"; }
    int stateDim() const override { return 15; } // e.g., 5 joints x 3D
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        for(int i=0;i<xn.size();i+=3)
            xn.segment<3>(i) += Vec::Ones(3)*0.01*dt; // placeholder small motion
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(stateDim(), stateDim());
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(stateDim(), stateDim())*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<HumanModel>(params);
    }

private:
    HumanParams params;
};

} // namespace tracker
