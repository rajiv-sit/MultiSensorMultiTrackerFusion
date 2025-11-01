// process_cav.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct CAVParams {
    double q_omega = 1e-6; // process noise on angular velocity
    double q_q = 1e-8;     // process noise on quaternion
};

// ----------------------
// Constant Angular Velocity (CAV)
// ----------------------
class CAVModel : public IProcessModel {
public:
    explicit CAVModel(const CAVParams& p = {}) : params(p) {}

    std::string name() const override { return "CAV"; }

    int stateDim() const override { return 7; } // q0,q1,q2,q3,wx,wy,wz

    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;

        Eigen::Vector4d q = x.segment<4>(0);
        Eigen::Vector3d w = x.segment<3>(4);

        // Quaternion propagation
        Eigen::Matrix4d Omega;
        Omega << 0, -w(0), -w(1), -w(2),
                 w(0), 0, w(2), -w(1),
                 w(1), -w(2), 0, w(0),
                 w(2), w(1), -w(0), 0;

        Eigen::Vector4d dq = 0.5 * Omega * q;
        xn.segment<4>(0) = q + dq * dt;
        xn.segment<3>(4) = w; // constant angular velocity

        // Normalize quaternion
        xn.segment<4>(0).normalize();

        return xn;
    }

    Mat F(const Vec& x, double dt) const override {
        // Approximate Jacobian
        Mat J = Mat::Identity(7,7);
        Eigen::Vector3d w = x.segment<3>(4);

        Eigen::Matrix4d Omega;
        Omega << 0, -w(0), -w(1), -w(2),
                 w(0), 0, w(2), -w(1),
                 w(1), -w(2), 0, w(0),
                 w(2), w(1), -w(0), 0;

        J.block<4,4>(0,0) += 0.5 * Omega * dt;

        // dq/dw block
        Eigen::Matrix<double,4,3> dqdw;
        dqdw << 0, -0.5*x(2), -0.5*x(3),
                0.5*x(0), 0, 0.5*x(2),
                0.5*x(1), -0.5*x(2), 0,
                0.5*x(3), 0.5*x(1), -0.5*x(0);
        J.block<4,3>(0,4) += dqdw * dt;

        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(7,7);
        Q.block<4,4>(0,0) = Mat::Identity(4,4) * params.q_q;
        Q.block<3,3>(4,4) = Mat::Identity(3,3) * params.q_omega;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CAVModel>(*this);
    }

private:
    CAVParams params;
};

} // namespace tracker
