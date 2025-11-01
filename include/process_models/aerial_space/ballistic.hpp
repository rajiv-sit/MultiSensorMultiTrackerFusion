// process_ballistic.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct BallisticParams {
    double g = 9.80665;   // gravity (m/s^2)
    double q_v = 1e-3;    // process noise for velocity
    double q_r = 1e-6;    // process noise for position
};

// ----------------------
// Ballistic Motion Model
// ----------------------
class BallisticModel : public IProcessModel {
public:
    explicit BallisticModel(const BallisticParams& p = {}) : params(p) {}

    std::string name() const override { return "Ballistic"; }

    int stateDim() const override { return 6; } // x,y,z,vx,vy,vz

    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);

        Eigen::Vector3d a(0,0,-params.g);

        xn.segment<3>(0) = r + v*dt;
        xn.segment<3>(3) = v + a*dt;

        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(6,6);
        J.block<3,3>(0,3) = Mat::Identity(3,3) * dt;
        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(6,6);
        Q.block<3,3>(0,0) = Mat::Identity(3,3) * params.q_r;
        Q.block<3,3>(3,3) = Mat::Identity(3,3) * params.q_v;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<BallisticModel>(*this);
    }

private:
    BallisticParams params;
};

} // namespace tracker
