// process_ca.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// ----------------------
// CA Parameters
// ----------------------
struct CAParams {
    double q_a = 0.01; // process noise spectral density (acceleration)
};

// ----------------------
// Constant Acceleration Model (CA)
// ----------------------
class CAModel : public IProcessModel {
public:
    explicit CAModel(const CAParams& p = {}) : params(p) {}

    std::string name() const override { return "CA"; }

    // 2D CA: px, py, vx, vy, ax, ay → 6 states
    // 3D CA: px, py, pz, vx, vy, vz, ax, ay, az → 9 states
    int stateDim() const override { return axis * 3; }

    bool isLinear() const override { return true; }

    // Set axis = 2 or 3
    void setAxis(int a) { axis = a; }

    // State propagation
    Vec f(const Vec& x, double dt) const override {
        return F(dt) * x;
    }

    // Discrete-time transition matrix
    Mat F(double dt) const {
        int n = stateDim();
        Mat Fm = Mat::Identity(n,n);
        for (int ax = 0; ax < axis; ++ax) {
            int off = ax * 3;
            Fm(off, off+1) = dt;
            Fm(off, off+2) = 0.5*dt*dt;
            Fm(off+1, off+2) = dt;
        }
        return Fm;
    }

    // Analytic Jacobian for EKF
    Mat F(const Vec&, double dt) const override {
        return F(dt);
    }

    // Discrete-time process noise
    Mat Qd(double dt) const override {
        int n = stateDim();
        Mat Q = Mat::Zero(n,n);
        double q = params.q_a;
        for (int ax = 0; ax < axis; ++ax) {
            int off = ax*3;
            double dt2 = dt*dt;
            double dt3 = dt2*dt;
            double dt4 = dt3*dt;
            Q(off,off)     = dt4/4.0 * q;
            Q(off,off+1)   = dt3/2.0 * q;
            Q(off,off+2)   = dt2/2.0 * q;
            Q(off+1,off)   = dt3/2.0 * q;
            Q(off+1,off+1) = dt2 * q;
            Q(off+1,off+2) = dt * q;
            Q(off+2,off)   = dt2/2.0 * q;
            Q(off+2,off+1) = dt * q;
            Q(off+2,off+2) = q;
        }
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CAModel>(*this);
    }

private:
    CAParams params;
    int axis = 2; // default 2D
};

} // namespace tracker
