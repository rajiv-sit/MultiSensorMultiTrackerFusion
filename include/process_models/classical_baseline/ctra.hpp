// process_ctra.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// CTRA Parameters
// ----------------------
struct CTRAParams {
    double q_v = 0.01;       // velocity noise
    double q_omega = 0.001;  // turn rate noise
    double q_a = 0.01;       // tangential acceleration noise
};

// ----------------------
// Coordinated Turn with Acceleration (CTRA)
// ----------------------
class CTRAModel : public IProcessModel {
public:
    explicit CTRAModel(const CTRAParams& p = {}) : params(p) {}

    std::string name() const override { return "CTRA"; }

    // 2D state: px, py, v, yaw, omega, a_t → 6 states
    int stateDim() const override { return 6; }

    bool isLinear() const override { return false; }

    // Nonlinear state propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;

        double px = x(0);
        double py = x(1);
        double v  = x(2);
        double yaw = x(3);
        double omega = x(4);
        double a_t = x(5);

        // Update velocity
        double v_new = v + a_t*dt;

        // Update position
        if (std::abs(omega) > 1e-6) {
            double r = v_new / omega;
            xn(0) = px + r * ( std::sin(yaw + omega*dt) - std::sin(yaw) );
            xn(1) = py - r * ( std::cos(yaw + omega*dt) - std::cos(yaw) );
        } else {
            xn(0) = px + v_new*dt*std::cos(yaw);
            xn(1) = py + v_new*dt*std::sin(yaw);
        }

        xn(2) = v_new;       // velocity updated
        xn(3) = yaw + omega*dt; // yaw updated
        xn(4) = omega;       // turn rate constant
        xn(5) = a_t;         // tangential acceleration constant

        return xn;
    }

    // Analytic Jacobian for EKF
    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(6,6);

        double v = x(2);
        double yaw = x(3);
        double omega = x(4);
        double a_t = x(5);
        double v_new = v + a_t*dt;

        if (std::abs(omega) > 1e-6) {
            double r = v_new / omega;
            double theta = yaw + omega*dt;

            J(0,2) = ( std::sin(theta) - std::sin(yaw) ) / omega;
            J(0,3) = r * ( std::cos(theta) - std::cos(yaw) );
            J(0,4) = v_new*( std::sin(theta)*dt - (std::sin(theta) - std::sin(yaw))/omega ) / omega;
            J(0,5) = dt * ( std::sin(theta) - std::sin(yaw) ) / omega;

            J(1,2) = -( std::cos(theta) - std::cos(yaw) ) / omega;
            J(1,3) = r * ( std::sin(theta) - std::sin(yaw) );
            J(1,4) = v_new*( -std::cos(theta)*dt + (std::cos(theta) - std::cos(yaw))/omega ) / omega;
            J(1,5) = dt * -( std::cos(theta) - std::cos(yaw) ) / omega;
        } else {
            J(0,2) = dt*std::cos(yaw);
            J(0,3) = -v_new*dt*std::sin(yaw);
            J(0,5) = dt*std::cos(yaw);
            J(1,2) = dt*std::sin(yaw);
            J(1,3) = v_new*dt*std::cos(yaw);
            J(1,5) = dt*std::sin(yaw);
        }

        J(2,5) = dt;
        J(3,4) = dt;

        return J;
    }

    // Process noise
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(6,6);
        Q(2,2) = params.q_v;
        Q(4,4) = params.q_omega;
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
