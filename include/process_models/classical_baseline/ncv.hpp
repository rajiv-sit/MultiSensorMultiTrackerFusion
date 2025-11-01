// process_ncv.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// ----------------------
// NCV Parameters
// ----------------------
struct NCVParams {
    double q_v = 0.05; // slightly higher velocity noise than CV
};

// ----------------------
// Nearly Constant Velocity Model (NCV)
// ----------------------
class NCVModel : public IProcessModel {
public:
    explicit NCVModel(const NCVParams& p = {}) : params(p) {}

    std::string name() const override { return "NCV"; }

    // 2D: px, py, vx, vy → 4 states
    // 3D: px, py, pz, vx, vy, vz → 6 states
    int stateDim() const override { return axis * 2; }

    bool isLinear() const override { return true; }

    // Set axis = 2 or 3
    void setAxis(int a) { axis = a; }

    // State propagation
    Vec f(const Vec& x, double dt) const override {
        return F(dt) * x;
    }

    // Transition matrix
    Mat F(double dt) const {
        int n = stateDim();
        Mat Fm = Mat::Identity(n,n);
        for (int ax = 0; ax < axis; ++ax) {
            int off = ax * 2;
            Fm(off, off+1) = dt;
        }
        return Fm;
    }

    Mat F(const Vec&, double dt) const override {
        return F(dt);
    }

    // Discrete-time process noise
    Mat Qd(double dt) const override {
        int n = stateDim();
        Mat Q = Mat::Zero(n,n);
        double q = params.q_v;
        for (int ax = 0; ax < axis; ++ax) {
            int off = ax * 2;
            double dt2 = dt*dt;
            Q(off, off)     = 0.25*dt2*dt2 * q;
            Q(off, off+1)   = 0.5*dt2*q;
            Q(off+1, off)   = 0.5*dt2*q;
            Q(off+1, off+1) = dt*q;
        }
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<NCVModel>(*this);
    }

private:
    NCVParams params;
    int axis = 2; // default 2D
};

} // namespace tracker
