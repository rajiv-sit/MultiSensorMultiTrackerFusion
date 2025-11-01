// process_unicycle.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// Unicycle Model Parameters
// ----------------------
struct UnicycleParams {
    double q_v = 0.01;     // process noise for linear velocity
    double q_omega = 0.001; // process noise for angular velocity
    double q_yaw = 0.001;   // process noise for heading
};

// ----------------------
// Unicycle Kinematic Model
// ----------------------
class UnicycleModel : public IProcessModel {
public:
    explicit UnicycleModel(const UnicycleParams& p = {}) : params(p) {}

    std::string name() const override { return "Unicycle"; }

    // State: px, py, theta, v, omega → 5 states
    int stateDim() const override { return 5; }

    bool isLinear() const override { return false; }

    // Nonlinear state propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;

        double px = x(0);
        double py = x(1);
        double theta = x(2);
        double v = x(3);
        double omega = x(4);

        // Unicycle kinematics
        xn(0) = px + v * std::cos(theta) * dt;
        xn(1) = py + v * std::sin(theta) * dt;
        xn(2) = theta + omega * dt;
        xn(3) = v;     // constant linear velocity
        xn(4) = omega; // constant angular velocity

        return xn;
    }

    // Analytic Jacobian for EKF
    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(5,5);

        double theta = x(2);
        double v = x(3);

        J(0,2) = -v * std::sin(theta) * dt;
        J(0,3) = std::cos(theta) * dt;

        J(1,2) = v * std::cos(theta) * dt;
        J(1,3) = std::sin(theta) * dt;

        J(2,4) = dt;

        return J;
    }

    // Process noise
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(5,5);
        Q(2,2) = params.q_yaw;
        Q(3,3) = params.q_v;
        Q(4,4) = params.q_omega;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<UnicycleModel>(*this);
    }

private:
    UnicycleParams params;
};

} // namespace tracker
