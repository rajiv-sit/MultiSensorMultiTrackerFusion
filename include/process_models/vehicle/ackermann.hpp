// process_ackermann.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// Ackermann Steering Parameters
// ----------------------
struct AckermannParams {
    double L = 2.5;       // wheelbase
    double q_v = 0.01;    // process noise for velocity
    double q_phi = 0.001; // process noise for steering angle
    double q_yaw = 0.001; // process noise for heading
};

// ----------------------
// Ackermann Steering Model
// ----------------------
class AckermannModel : public IProcessModel {
public:
    explicit AckermannModel(const AckermannParams& p = {}) : params(p) {}

    std::string name() const override { return "Ackermann"; }

    // State: px, py, theta, v, phi → 5 states
    int stateDim() const override { return 5; }

    bool isLinear() const override { return false; }

    // Nonlinear state propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;

        double px = x(0);
        double py = x(1);
        double theta = x(2);
        double v = x(3);
        double phi = x(4);

        // Ackermann steering kinematics
        double beta = std::atan(0.5 * std::tan(phi)); // front wheel effect
        xn(0) = px + v * std::cos(theta + beta) * dt;
        xn(1) = py + v * std::sin(theta + beta) * dt;
        xn(2) = theta + v / params.L * std::sin(beta) * dt;
        xn(3) = v;   // constant speed assumption
        xn(4) = phi; // constant steering angle

        return xn;
    }

    // Analytic Jacobian for EKF
    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(5,5);

        double theta = x(2);
        double v = x(3);
        double phi = x(4);

        double beta = std::atan(0.5 * std::tan(phi));
        double dbeta_dphi = 0.5 / (std::cos(phi)*std::cos(phi) + 0.0) / (1.0 + std::tan(phi)*std::tan(phi)/4.0); // approximate derivative

        J(0,2) = -v * std::sin(theta + beta) * dt;
        J(0,3) = std::cos(theta + beta) * dt;
        J(0,4) = -v * std::sin(theta + beta) * dt * dbeta_dphi; // chain rule

        J(1,2) = v * std::cos(theta + beta) * dt;
        J(1,3) = std::sin(theta + beta) * dt;
        J(1,4) = v * std::cos(theta + beta) * dt * dbeta_dphi;

        J(2,3) = std::sin(beta) / params.L * dt;
        J(2,4) = v / params.L * std::cos(beta) * dbeta_dphi * dt;

        return J;
    }

    // Process noise matrix
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(5,5);
        Q(2,2) = params.q_yaw;
        Q(3,3) = params.q_v;
        Q(4,4) = params.q_phi;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<AckermannModel>(*this);
    }

private:
    AckermannParams params;
};

} // namespace tracker
