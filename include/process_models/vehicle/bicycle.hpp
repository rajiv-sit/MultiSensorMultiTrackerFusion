// process_bicycle.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// Bicycle Model Parameters
// ----------------------
struct BicycleParams {
    double L = 2.5;       // wheelbase (meters)
    double q_v = 0.01;    // process noise for velocity
    double q_delta = 0.001; // process noise for steering angle
    double q_yaw = 0.001;   // process noise for heading
};

// ----------------------
// Bicycle Kinematic Model
// ----------------------
class BicycleModel : public IProcessModel {
public:
    explicit BicycleModel(const BicycleParams& p = {}) : params(p) {}

    std::string name() const override { return "Bicycle"; }

    // State: px, py, theta, v, delta → 5 states
    int stateDim() const override { return 5; }

    bool isLinear() const override { return false; }

    // Nonlinear state propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;

        double px = x(0);
        double py = x(1);
        double theta = x(2);
        double v = x(3);
        double delta = x(4);

        // Simple bicycle kinematics
        xn(0) = px + v * std::cos(theta) * dt;
        xn(1) = py + v * std::sin(theta) * dt;
        xn(2) = theta + v / params.L * std::tan(delta) * dt; // yaw rate
        xn(3) = v;     // assume constant velocity for now
        xn(4) = delta; // assume constant steering angle

        return xn;
    }

    // Analytic Jacobian for EKF
    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(5,5);

        double theta = x(2);
        double v = x(3);
        double delta = x(4);

        J(0,2) = -v * std::sin(theta) * dt;
        J(0,3) = std::cos(theta) * dt;

        J(1,2) = v * std::cos(theta) * dt;
        J(1,3) = std::sin(theta) * dt;

        J(2,3) = std::tan(delta) / params.L * dt;
        J(2,4) = v / params.L * dt / (std::cos(delta) * std::cos(delta));

        return J;
    }

    // Process noise matrix
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(5,5);
        Q(2,2) = params.q_yaw;
        Q(3,3) = params.q_v;
        Q(4,4) = params.q_delta;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<BicycleModel>(*this);
    }

private:
    BicycleParams params;
};

} // namespace tracker
