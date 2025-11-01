// process_reentry.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct ReentryParams {
    double q_pos = 1e-2;
    double q_vel = 1e-2;
    double q_beta = 1e-4;
    double g = 9.81; // gravity
};

class ReentryModel : public IProcessModel {
public:
    explicit ReentryModel(const ReentryParams& p = {}) : params(p) {}

    std::string name() const override { return "Reentry"; }

    int stateDim() const override { return 7; } // x,y,z,vx,vy,vz,beta

    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);
        double beta = x(6);

        Eigen::Vector3d a_drag = -beta * v.cwiseAbs().cwiseProduct(v); // simplified quadratic drag
        Eigen::Vector3d a_gravity(0,0,-params.g);

        Vec xn(7);
        xn.segment<3>(0) = r + v*dt + 0.5*(a_drag + a_gravity)*dt*dt;
        xn.segment<3>(3) = v + (a_drag + a_gravity)*dt;
        xn(6) = beta; // assume constant

        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(7,7); // simplified linearization; can extend later
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(7,7);
        Q.block<3,3>(0,0) = Mat::Identity(3,3)*params.q_pos;
        Q.block<3,3>(3,3) = Mat::Identity(3,3)*params.q_vel;
        Q(6,6) = params.q_beta;
        return Q;
    }

private:
    ReentryParams params;
};

} // namespace tracker
