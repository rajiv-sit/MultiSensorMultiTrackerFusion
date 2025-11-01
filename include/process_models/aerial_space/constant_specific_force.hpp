// process_constant_specific_force.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// ----------------------
// Parameters
// ----------------------
struct CSFParams {
    double q_f = 1e-4; // process noise on specific force
};

// ----------------------
// Constant Specific Force Model
// ----------------------
class ConstantSpecificForceModel : public IProcessModel {
public:
    explicit ConstantSpecificForceModel(const CSFParams& p = {}) : params(p) {}

    std::string name() const override { return "ConstantSpecificForce"; }

    int stateDim() const override { return 9; } // px, py, pz, vx, vy, vz, fx, fy, fz

    bool isLinear() const override { return true; }

    // State propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = Vec::Zero(9);

        Eigen::Vector3d p = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);
        Eigen::Vector3d f = x.segment<3>(6);

        xn.segment<3>(0) = p + v*dt + 0.5*f*dt*dt;
        xn.segment<3>(3) = v + f*dt;
        xn.segment<3>(6) = f; // assume slowly varying

        return xn;
    }

    // Linear state transition matrix
    Mat F(const Vec&, double dt) const override {
        Mat F = Mat::Identity(9,9);
        F.block<3,3>(0,3) = Mat::Identity(3,3) * dt;
        F.block<3,3>(0,6) = Mat::Identity(3,3) * 0.5 * dt*dt;
        F.block<3,3>(3,6) = Mat::Identity(3,3) * dt;
        return F;
    }

    // Process noise
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(9,9);
        Q.block<3,3>(6,6) = Mat::Identity(3,3) * params.q_f;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<ConstantSpecificForceModel>(*this);
    }

private:
    CSFParams params;
};

} // namespace tracker
