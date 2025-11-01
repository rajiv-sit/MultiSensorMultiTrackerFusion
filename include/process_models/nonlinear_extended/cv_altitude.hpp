// process_cv_altitude.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [x,y,vx,vy,z,vz]ᵀ
struct CVAltitudeParams {
    double q_pos = 1e-3;
    double q_vel = 1e-3;
};

class CVAltitudeModel : public IProcessModel {
public:
    explicit CVAltitudeModel(const CVAltitudeParams& p = {}) : params(p) {}

    std::string name() const override { return "CVAltitude"; }
    int stateDim() const override { return 6; }
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn(6);
        xn(0)=x(0)+x(2)*dt; xn(1)=x(1)+x(3)*dt; xn(2)=x(2); xn(3)=x(3);
        xn(4)=x(4)+x(5)*dt; xn(5)=x(5);
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J=Mat::Identity(6,6);
        J(0,2)=dt; J(1,3)=dt; J(4,5)=dt;
        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q=Mat::Zero(6,6);
        Q.block<2,2>(0,0)=Mat::Identity(2,2)*params.q_pos;
        Q.block<2,2>(2,2)=Mat::Identity(2,2)*params.q_vel;
        Q.block<2,2>(4,4)=Mat::Identity(2,2)*params.q_pos;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CVAltitudeModel>(params);
    }

private:
    CVAltitudeParams params;
};

} // namespace tracker
