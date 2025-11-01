// process_physics_neural.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [state, latent dynamics]ᵀ
struct PhysNeuralParams {
    int latent_dim = 8;
};

class PhysicsNeuralModel : public IProcessModel {
public:
    explicit PhysicsNeuralModel(const PhysNeuralParams& p = {}) : params(p) {}

    std::string name() const override { return "PhysicsNeural"; }
    int stateDim() const override { return 4 + params.latent_dim; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // Placeholder: physics + small neural perturbation
        xn.head(4) += x.segment(2,2)*dt;
        xn.tail(params.latent_dim) += Vec::Random(params.latent_dim)*0.01;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(stateDim(), stateDim());
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(stateDim(), stateDim())*1e-3;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<PhysicsNeuralModel>(params);
    }

private:
    PhysNeuralParams params;
};

} // namespace tracker
