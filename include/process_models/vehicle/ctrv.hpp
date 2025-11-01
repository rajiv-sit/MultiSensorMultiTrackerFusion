// process_ctrv.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// CTRV Model Parameters
// ----------------------
struct CTRVParams {
    double q_v = 0.1;       // process noise for speed
    double q_psi = 0.01;    // process noise for heading
    double q_psi_dot = 0.001; // process noise for turn rate
};

// ----------------------
// CTRV Model
// ----------------------
class CTRVModel : public IProcessModel {
public:
    explicit CTRVModel(const CTRVParams& p = {}) : params(p) {}

    std::string name() const override { return "CTRV"; }

    int stateDim() const override { return 5; } // x, y, v, psi, psi_dot

    bool isLinear() const override { return false; }

    // Nonlinear state propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        double px = x(0);
        double py = x(1);
        double v  = x(2);
        double psi = x(3);
        double psi_dot = x(4);

        if (std::abs(psi_dot) > 1e-6) {
            xn(0) = px + v/psi_dot * (std::sin(psi + psi_dot*dt) - std::sin(psi));
            xn(1) = py + v/psi_dot * (-std::cos(psi + psi_dot*dt) + std::cos(psi));
        } else {
            xn(0) = px + v * std::cos(psi) * dt;
            xn(1) = py + v * std::sin(psi) * dt;
        }

        xn(2) = v;
        xn(3) = psi + psi_dot * dt;
        xn(4) = psi_dot;

        return xn;
    }

    // Jacobian for EKF
    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(5,5);
        double v = x(2);
        double psi = x(3);
        double psi_dot = x(4);

        if (std::abs(psi_dot) > 1e-6) {
            double theta = psi + psi_dot*dt;
            double v_over_psi = v / psi_dot;
            J(0,3) = v_over_psi * (std::cos(theta) - std::cos(psi));
            J(0,2) = (std::sin(theta) - std::sin(psi)) / psi_dot;
            J(0,4) = (v*(std::sin(theta) - std::sin(psi)) / (psi_dot*psi_dot)) - v_over_psi*dt*std::cos(theta);

            J(1,3) = v_over_psi * (std::sin(theta) - std::sin(psi)) * -1;
            J(1,2) = (-std::cos(theta) + std::cos(psi)) / psi_dot;
            J(1,4) = (v*(std::cos(theta) - std::cos(psi)) / (psi_dot*psi_dot)) - v_over_psi*dt*std::sin(theta);
        } else {
            J(0,2) = dt * std::cos(psi);
            J(0,3) = -v * dt * std::sin(psi);
            J(1,2) = dt * std::sin(psi);
            J(1,3) = v * dt * std::cos(psi);
        }

        J(3,4) = dt; // psi update depends on psi_dot

        return J;
    }

    // Process noise
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(5,5);
        Q(2,2) = params.q_v;
        Q(3,3) = params.q_psi;
        Q(4,4) = params.q_psi_dot;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CTRVModel>(*this);
    }

private:
    CTRVParams params;
};

} // namespace tracker
