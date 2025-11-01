// process_cv_ca_3d.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// CV3D: x = [x,y,z,vx,vy,vz]ᵀ
struct CV3DParams {
    double q_r = 1e-3;
    double q_v = 1e-3;
};

class CV3DModel : public IProcessModel {
public:
    explicit CV3DModel(const CV3DParams& p = {}) : params(p) {}
    std::string name() const override { return "CV3D"; }
    int stateDim() const override { return 6; }
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn(6);
        xn.segment<3>(0) = x.segment<3>(0) + x.segment<3>(3)*dt;
        xn.segment<3>(3) = x.segment<3>(3);
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        Mat J = Mat::Identity(6,6);
        J.block<3,3>(0,3) = Mat::Identity(3,3)*dt;
        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(6,6);
        Q.block<3,3>(0,0)=Mat::Identity(3,3)*params.q_r;
        Q.block<3,3>(3,3)=Mat::Identity(3,3)*params.q_v;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CV3DModel>(params);
    }

private:
    CV3DParams params;
};

} // namespace tracker
