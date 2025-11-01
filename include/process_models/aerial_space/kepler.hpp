// process_kepler.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

struct KeplerParams {
    double mu = 3.986e14; // Earth's gravitational parameter [m^3/s^2]
    double q_r = 1e-2;    // position noise
    double q_v = 1e-4;    // velocity noise
};

class KeplerModel : public IProcessModel {
public:
    explicit KeplerModel(const KeplerParams& p = {}) : params(p) {}

    std::string name() const override { return "Kepler"; }

    int stateDim() const override { return 6; }

    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Eigen::Vector3d r = x.segment<3>(0);
        Eigen::Vector3d v = x.segment<3>(3);

        double norm_r3 = std::pow(r.norm(),3);
        Eigen::Vector3d a = -params.mu * r / norm_r3;

        Vec xn(6);
        xn.segment<3>(0) = r + v*dt;
        xn.segment<3>(3) = v + a*dt;

        return xn;
    }

    Mat F(const Vec& x, double dt) const override {
        Eigen::Vector3d r = x.segment<3>(0);
        double r_norm = r.norm();
        Mat J = Mat::Identity(6,6);

        // dr/dv
        J.block<3,3>(0,3) = Mat::Identity(3,3) * dt;

        // dv/dr (gravitational)
        Mat dvdr = Mat::Zero(3,3);
        for (int i=0;i<3;i++)
            for (int j=0;j<3;j++)
                dvdr(i,j) = params.mu * (3*r(i)*r(j)/std::pow(r_norm,5) - (i==j?1:0)/std::pow(r_norm,3));
        J.block<3,3>(3,0) = dvdr * dt;

        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(6,6);
        Q.block<3,3>(0,0) = Mat::Identity(3,3)*params.q_r;
        Q.block<3,3>(3,3) = Mat::Identity(3,3)*params.q_v;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<KeplerModel>(*this);
    }

private:
    KeplerParams params;
};

} // namespace tracker
