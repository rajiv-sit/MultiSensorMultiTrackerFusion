// process_ctra.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// CTRA Model Parameters
// ----------------------
struct CTRAParams {
    double q_v = 0.1;       // process noise for speed
    double q_psi = 0.01;    // process noise for heading
    double q_psi_dot = 0.001; // process noise for turn rate
    double q_a = 0.1;       // process noise for tangential acceleration
};

// ----------------------
// CTRA Model
// ----------------------
class CTRAModel : public IProcessModel {
public:
    explicit CTRAModel(const CTRAParams& p = {}) : params(p) {}

    std::string name() const override { return "CTRA"; }

    int stateDim() const override { return 6; } // x, y, v, psi, psi_dot, a

    bool isLinear() const override { return false; }

    // Nonlinear state propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        double px = x(0);
        double py = x(1);
        double v = x(2);
        double psi = x(3);
        double psi_dot = x(4);
        double a = x(5);

        double v_new = v + a * dt;
        double psi_new = psi + psi_dot * dt;

        if (std::abs(psi_dot) > 1e-6) {
            xn(0) = px + v_new/psi_dot * (std::sin(psi_new) - std::sin(psi));
            xn(1) = py + v_new/psi_dot * (-std::cos(psi_new) + std::cos(psi));
        } else {
            xn(0) = px + v_new * std::cos(psi) * dt;
            xn(1) = py + v_new * std::sin(psi) * dt;
        }

        xn(2) = v_new;
        xn(3) = psi_new;
        xn(4) = psi_dot;
        xn(5) = a;

        return xn;
    }

    // Jacobian for EKF
    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(6,6);
        double v = x(2);
        double psi = x(3);
        double psi_dot = x(4);
        double a = x(5);

        double v_new = v + a*dt;
        double psi_new = psi + psi_dot*dt;

        if (std::abs(psi_dot) > 1e-6) {
            double theta = psi_new;
            double v_over_psi = v_new / psi_dot;

            // Partial derivatives w.r.t psi
            J(0,3) = v_over_psi * (std::cos(theta) - std::cos(psi));
            J(1,3) = v_over_psi * (std::sin(theta) - std::sin(psi)) * -1;

            // Partial derivatives w.r.t v
            J(0,2) = (std::sin(theta) - std::sin(psi)) / psi_dot;
            J(1,2) = (-std::cos(theta) + std::cos(psi)) / psi_dot;

            // Partial derivatives w.r.t psi_dot
            J(0,4) = (v_new*(std::sin(theta) - std::sin(psi)) / (psi_dot*psi_dot)) - v_over_psi*dt*std::cos(theta);
            J(1,4) = (v_new*(std::cos(theta) - std::cos(psi)) / (psi_dot*psi_dot)) - v_over_psi*dt*std::sin(theta);

            // Partial derivatives w.r.t a
            J(0,5) = dt/psi_dot*(std::sin(theta) - std::sin(psi));
            J(1,5) = dt/psi_dot*(-std::cos(theta) + std::cos(psi));
            J(2,5) = dt;
        } else {
            J(0,2) = dt * std::cos(psi);
            J(0,3) = -v_new * dt * std::sin(psi);
            J(1,2) = dt * std::sin(psi);
            J(1,3) = v_new * dt * std::cos(psi);
            J(2,5) = dt; // v depends on a
        }

        J(3,4) = dt; // psi depends on psi_dot

        return J;
    }

    // Process noise
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(6,6);
        Q(2,2) = params.q_v;
        Q(3,3) = params.q_psi;
        Q(4,4) = params.q_psi_dot;
        Q(5,5) = params.q_a;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CTRAModel>(*this);
    }

private:
    CTRAParams params;
};

} // namespace tracker
