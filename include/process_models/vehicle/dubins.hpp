// process_dubins.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// Dubins Path Parameters
// ----------------------
struct DubinsParams {
    double q_v = 0.01;       // process noise for speed
    double q_psi = 0.001;    // process noise for heading
    double R_min = 1.0;      // minimum turning radius
};

// ----------------------
// Dubins Path Model
// ----------------------
class DubinsModel : public IProcessModel {
public:
    explicit DubinsModel(const DubinsParams& p = {}) : params(p) {}

    std::string name() const override { return "Dubins"; }

    int stateDim() const override { return 4; } // x, y, psi, v

    bool isLinear() const override { return false; }

    // Nonlinear state propagation
    Vec f(const Vec& x, double dt, double u_dot = 0.0) const {
        Vec xn = x;
        double px = x(0);
        double py = x(1);
        double psi = x(2);
        double v = x(3);

        // Clip turn rate according to minimum turning radius
        double psi_dot = std::max(std::min(u_dot, v/params.R_min), -v/params.R_min);
        double psi_new = psi + psi_dot * dt;

        if (std::abs(psi_dot) > 1e-6) {
            xn(0) = px + v/psi_dot * (std::sin(psi_new) - std::sin(psi));
            xn(1) = py + v/psi_dot * (-std::cos(psi_new) + std::cos(psi));
        } else {
            xn(0) = px + v * std::cos(psi) * dt;
            xn(1) = py + v * std::sin(psi) * dt;
        }

        xn(2) = psi_new;
        xn(3) = v;

        return xn;
    }

    // Jacobian for EKF
    Mat F(const Vec& x, double dt, double u_dot = 0.0) const {
        Mat J = Mat::Identity(4,4);
        double v = x(3);
        double psi = x(2);
        double psi_dot = std::max(std::min(u_dot, v/params.R_min), -v/params.R_min);
        double psi_new = psi + psi_dot * dt;

        if (std::abs(psi_dot) > 1e-6) {
            double v_over_psi = v / psi_dot;
            J(0,2) = v_over_psi * (std::cos(psi_new) - std::cos(psi));
            J(0,3) = (std::sin(psi_new) - std::sin(psi)) / psi_dot;
            J(1,2) = v_over_psi * (std::sin(psi) - std::sin(psi_new));
            J(1,3) = (-std::cos(psi_new) + std::cos(psi)) / psi_dot;
        } else {
            J(0,3) = dt * std::cos(psi);
            J(0,2) = -v * dt * std::sin(psi);
            J(1,3) = dt * std::sin(psi);
            J(1,2) = v * dt * std::cos(psi);
        }

        return J;
    }

    // Process noise
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(4,4);
        Q(3,3) = params.q_v;
        Q(2,2) = params.q_psi;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<DubinsModel>(*this);
    }

private:
    DubinsParams params;
};

} // namespace tracker
