// process_perturbed_orbit.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

struct PerturbedOrbitParams {
    double mu = 3.986e14;       // Earth gravitational parameter
    double J2 = 1.08263e-3;     // Earth J2
    double R_E = 6378137.0;     // Earth radius (m)
    double q_v = 1e-4;          // velocity noise
    double q_r = 1e-6;          // position noise
    double rho = 0.0;           // atmospheric density
    double CdA_m = 0.0;         // drag coefficient * area / mass
};

// ----------------------
// Perturbed Orbit Model
// ----------------------
class PerturbedOrbitModel : public IProcessModel {
public:
    explicit PerturbedOrbitModel(const PerturbedOrbitParams& p = {}) : params(p) {}

    std::string name() const override { return "PerturbedOrbit"; }

    int stateDim() const override { return 6; }

    bool isLinear() const override { return false; }

    // Nonlinear propagation (Euler)
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);
        double r_norm = r.norm();

        // Keplerian acceleration
        Eigen::Vector3d a = -params.mu / (r_norm*r_norm*r_norm) * r;

        // J2 perturbation
        double z2 = r(2)*r(2);
        double r2 = r_norm*r_norm;
        Eigen::Vector3d aJ2;
        aJ2(0) = 3*params.J2*params.mu*params.R_E*params.R_E/(2*r_norm*r_norm*r_norm*r_norm*r_norm) * r(0)*(5*z2/r2 - 1);
        aJ2(1) = 3*params.J2*params.mu*params.R_E*params.R_E/(2*r_norm*r_norm*r_norm*r_norm*r_norm) * r(1)*(5*z2/r2 - 1);
        aJ2(2) = 3*params.J2*params.mu*params.R_E*params.R_E/(2*r_norm*r_norm*r_norm*r_norm*r_norm) * r(2)*(5*z2/r2 - 3);
        a += aJ2;

        // Atmospheric drag
        if(params.rho > 0.0 && params.CdA_m > 0.0) {
            a += -0.5 * params.CdA_m * params.rho * v.norm() * v;
        }

        xn.segment<3>(0) = r + v*dt;
        xn.segment<3>(3) = v + a*dt;
        return xn;
    }

    // Jacobian approximation (optional)
    Mat F(const Vec& x, double dt) const override {
        // For EKF, you may linearize numerically or approximate as Keplerian
        Mat J = Mat::Identity(6,6);
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);
        double r_norm = r.norm();
        double r3 = r_norm*r_norm*r_norm;
        double r5 = r3*r_norm*r_norm;

        J.block<3,3>(0,3) = Mat::Identity(3,3) * dt;

        // Approximate dv/dr using only Keplerian term
        Eigen::Matrix3d dvdr = -params.mu * (Mat::Identity(3,3)/r3 - 3.0*r*r.transpose()/r5);
        J.block<3,3>(3,0) = dvdr * dt;

        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(6,6);
        Q.block<3,3>(0,0) = Mat::Identity(3,3) * params.q_r;
        Q.block<3,3>(3,3) = Mat::Identity(3,3) * params.q_v;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<PerturbedOrbitModel>(*this);
    }

private:
    PerturbedOrbitParams params;
};

} // namespace tracker
