// process_ct.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// CT Parameters
// ----------------------
struct CTParams {
    double q_v = 0.01; // process noise for velocity
    double q_omega = 0.001; // process noise for turn rate
};

// ----------------------
// Coordinated Turn Model (CT / CTRV)
// ----------------------
class CTModel : public IProcessModel {
public:
    explicit CTModel(const CTParams& p = {}) : params(p) {}

    std::string name() const override { return "CT"; }

    // 2D state: px, py, v, yaw, omega → 5 states
    int stateDim() const override { return 5; }

    bool isLinear() const override { return false; }

    // State propagation (nonlinear)
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;

        double px = x(0);
        double py = x(1);
        double v  = x(2);
        double yaw = x(3);
        double omega = x(4);

        if (std::abs(omega) > 1e-6) {
            double r = v / omega;
            xn(0) = px + r * ( std::sin(yaw + omega*dt) - std::sin(yaw) );
            xn(1) = py - r * ( std::cos(yaw + omega*dt) - std::cos(yaw) );
        } else {
            xn(0) = px + v*dt*std::cos(yaw);
            xn(1) = py + v*dt*std::sin(yaw);
        }

        xn(2) = v;        // velocity assumed constant
        xn(3) = yaw + omega*dt;
        xn(4) = omega;    // turn rate constant

        return xn;
    }

    // Analytic Jacobian for EKF
    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(5,5);
        double px = x(0);
        double py = x(1);
        double v  = x(2);
        double yaw = x(3);
        double omega = x(4);

        if (std::abs(omega) > 1e-6) {
            double r = v / omega;
            double theta = yaw + omega*dt;
            J(0,2) = ( std::sin(theta) - std::sin(yaw) ) / omega;
            J(0,3) = r * ( std::cos(theta) - std::cos(yaw) );
            J(0,4) = v*( std::sin(theta)*dt - (std::sin(theta) - std::sin(yaw))/omega ) / omega;

            J(1,2) = -( std::cos(theta) - std::cos(yaw) ) / omega;
            J(1,3) = r * ( std::sin(theta) - std::sin(yaw) );
            J(1,4) = v*( -std::cos(theta)*dt + (std::cos(theta) - std::cos(yaw))/omega ) / omega;
        } else {
            J(0,2) = dt*std::cos(yaw);
            J(0,3) = -v*dt*std::sin(yaw);
            J(1,2) = dt*std::sin(yaw);
            J(1,3) = v*dt*std::cos(yaw);
        }

        J(3,4) = dt;

        return J;
    }

    // Process noise (simple diagonal)
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(5,5);
        Q(2,2) = params.q_v;
        Q(4,4) = params.q_omega;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CTModel>(*this);
    }

private:
    CTParams params;
};

} // namespace tracker
