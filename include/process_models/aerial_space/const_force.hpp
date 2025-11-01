// process_const_force.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct ConstForceParams {
    double q_r = 1e-6;  // position noise
    double q_v = 1e-4;  // velocity noise
    double q_f = 1e-3;  // specific force noise
};

// ----------------------
// Constant Specific Force (IMU/Strapdown)
// ----------------------
class ConstForceModel : public IProcessModel {
public:
    explicit ConstForceModel(const ConstForceParams& p = {}) : params(p) {}

    std::string name() const override { return "ConstForce"; }

    int stateDim() const override { return 9; } // x,y,z,vx,vy,vz,fx,fy,fz

    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);
        Eigen::Vector3d f = x.segment<3>(6);

        Vec xn(9);
        xn.segment<3>(0) = r + v*dt + 0.5*f*dt*dt;
        xn.segment<3>(3) = v + f*dt;
        xn.segment<3>(6) = f; // constant

        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(9,9);
        J.block<3,3>(0,3) = Mat::Identity(3,3)*dt;
        J.block<3,3>(0,6) = Mat::Identity(3,3)*0.5*dt*dt;
        J.block<3,3>(3,6) = Mat::Identity(3,3)*dt;
        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(9,9);
        Q.block<3,3>(0,0) = Mat::Identity(3,3)*params.q_r;
        Q.block<3,3>(3,3) = Mat::Identity(3,3)*params.q_v;
        Q.block<3,3>(6,6) = Mat::Identity(3,3)*params.q_f;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<ConstForceModel>(*this);
    }

private:
    ConstForceParams params;
};

} // namespace tracker
