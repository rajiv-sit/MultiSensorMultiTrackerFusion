// process_lane_constrained.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// ----------------------
// Lane-Constrained Model Parameters
// ----------------------
struct LaneParams {
    double q_s = 0.001;      // process noise for lane position
    double q_dot_s = 0.01;   // process noise for speed
    double q_ddot_s = 0.1;   // process noise for acceleration
};

// ----------------------
// Lane-Constrained Model
// ----------------------
class LaneConstrainedModel : public IProcessModel {
public:
    explicit LaneConstrainedModel(const LaneParams& p = {}) : params(p) {}

    std::string name() const override { return "LaneConstrained"; }

    // State: s, s_dot, s_ddot → 3 states
    int stateDim() const override { return 3; }

    bool isLinear() const override { return true; } // linear in s domain

    // State propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        double s = x(0);
        double s_dot = x(1);
        double s_ddot = x(2);

        // Linear motion along lane
        xn(0) = s + s_dot * dt + 0.5 * s_ddot * dt * dt;
        xn(1) = s_dot + s_ddot * dt;
        xn(2) = s_ddot; // constant acceleration

        return xn;
    }

    // Jacobian
    Mat F(const Vec& /*x*/, double dt) const override {
        Mat J = Mat::Identity(3,3);
        J(0,1) = dt;
        J(0,2) = 0.5 * dt * dt;
        J(1,2) = dt;
        return J;
    }

    // Process noise
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(3,3);
        Q(0,0) = params.q_s;
        Q(1,1) = params.q_dot_s;
        Q(2,2) = params.q_ddot_s;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<LaneConstrainedModel>(*this);
    }

private:
    LaneParams params;
};

} // namespace tracker
