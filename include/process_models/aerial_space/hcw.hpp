// process_hcw.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct HCWParams {
    double n = 0.001027;  // mean motion [rad/s] (example for LEO)
    double q_r = 1e-6;    // position noise
    double q_v = 1e-4;    // velocity noise
};

// ----------------------
// Hill-Clohessy-Wiltshire (HCW) Relative Motion
// ----------------------
class HCWModel : public IProcessModel {
public:
    explicit HCWModel(const HCWParams& p = {}) : params(p) {}

    std::string name() const override { return "HCW"; }

    int stateDim() const override { return 6; } // x,y,z,dx,dy,dz

    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        // State: [x, y, z, vx, vy, vz]
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);
        Eigen::Vector3d a;

        // HCW acceleration
        a(0) = 3 * params.n * params.n * r(0) + 2 * params.n * v(1);
        a(1) = -2 * params.n * v(0);
        a(2) = -params.n * params.n * r(2);

        Vec xn(6);
        xn.segment<3>(0) = r + v * dt;
        xn.segment<3>(3) = v + a * dt;

        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(6,6);

        // Position w.r.t velocity
        J.block<3,3>(0,3) = Mat::Identity(3,3) * dt;

        // Velocity w.r.t position
        J(3,0) = 3*params.n*params.n*dt;
        J(5,2) = -params.n*params.n*dt;

        // Velocity w.r.t velocity
        J(3,4) = 2*params.n*dt;
        J(4,3) = -2*params.n*dt;

        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(6,6);
        Q.block<3,3>(0,0) = Mat::Identity(3,3) * params.q_r;
        Q.block<3,3>(3,3) = Mat::Identity(3,3) * params.q_v;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<HCWModel>(*this);
    }

private:
    HCWParams params;
};

} // namespace tracker
