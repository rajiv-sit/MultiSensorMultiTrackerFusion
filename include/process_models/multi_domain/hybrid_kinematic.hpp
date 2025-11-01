// process_hybrid_kinematic.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [px, py, vx, vy, class_prob1,..., class_probN]
struct HybridKinematicParams {
    double q_motion = 1e-3;
    double q_class = 1e-4;
};

class HybridKinematicModel : public IProcessModel {
public:
    explicit HybridKinematicModel(int num_classes, const HybridKinematicParams& p = {}) 
        : num_classes_(num_classes), params(p) {}

    std::string name() const override { return "HybridKinematic"; }
    int stateDim() const override { return 4 + num_classes_; }
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // constant velocity for motion
        xn(0) += x(2) * dt;
        xn(1) += x(3) * dt;
        // class probabilities remain constant (or can decay)
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(stateDim(), stateDim());
        J(0,2) = dt;
        J(1,3) = dt;
        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(stateDim(), stateDim());
        Q(0,0) = params.q_motion; Q(1,1) = params.q_motion;
        Q(2,2) = params.q_motion; Q(3,3) = params.q_motion;
        for (int i=4;i<stateDim();++i) Q(i,i) = params.q_class;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<HybridKinematicModel>(num_classes_, params);
    }

private:
    int num_classes_;
    HybridKinematicParams params;
};

} // namespace tracker
