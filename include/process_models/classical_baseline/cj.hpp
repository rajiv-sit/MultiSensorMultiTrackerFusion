// process_cj.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// ----------------------
// CJ Parameters
// ----------------------
struct CJParams {
    double q_j = 0.001; // jerk process noise spectral density
};

// ----------------------
// Constant Jerk Model (CJ)
// ----------------------
class CJModel : public IProcessModel {
public:
    explicit CJModel(const CJParams& p = {}) : params(p) {}

    std::string name() const override { return "CJ"; }

    // 2D: px, py, vx, vy, ax, ay, jx, jy → 8 states
    // 3D: px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz → 12 states
    int stateDim() const override { return axis * 4; }

    bool isLinear() const override { return true; }

    // Set axis = 2 or 3
    void setAxis(int a) { axis = a; }

    // State propagation
    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        for (int ax = 0; ax < axis; ++ax) {
            int off = ax * 4;
            double px = x(off+0);
            double vx = x(off+1);
            double ax_val = x(off+2);
            double jx = x(off+3);

            xn(off+0) = px + vx*dt + 0.5*ax_val*dt*dt + (1.0/6.0)*jx*dt*dt*dt;
            xn(off+1) = vx + ax_val*dt + 0.5*jx*dt*dt;
            xn(off+2) = ax_val + jx*dt;
            xn(off+3) = jx; // jerk constant
        }
        return xn;
    }

    // Discrete-time transition matrix
    Mat F(double dt) const {
        int n = stateDim();
        Mat Fm = Mat::Identity(n,n);
        double dt2 = dt*dt;
        double dt3 = dt2*dt;
        for (int ax = 0; ax < axis; ++ax) {
            int off = ax * 4;
            Fm(off+0, off+1) = dt;
            Fm(off+0, off+2) = 0.5*dt2;
            Fm(off+0, off+3) = dt3/6.0;
            Fm(off+1, off+2) = dt;
            Fm(off+1, off+3) = 0.5*dt2;
            Fm(off+2, off+3) = dt;
        }
        return Fm;
    }

    Mat F(const Vec&, double dt) const override {
        return F(dt);
    }

    // Process noise
    Mat Qd(double dt) const override {
        int n = stateDim();
        Mat Q = Mat::Zero(n,n);
        double dt2 = dt*dt, dt3 = dt2*dt, dt4 = dt3*dt, dt5 = dt4*dt;
        double q = params.q_j;

        Mat Qb(4,4);
        Qb << dt5/20.0, dt4/8.0, dt3/6.0, dt2/4.0,
              dt4/8.0,  dt3/3.0, dt2/2.0, dt/2.0,
              dt3/6.0,  dt2/2.0, dt,      1.0,
              dt2/4.0,  dt/2.0,  1.0,     dt;
        Qb *= q;

        for (int ax = 0; ax < axis; ++ax) {
            int off = ax*4;
            Q.block(off, off, 4, 4) = Qb;
        }

        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CJModel>(*this);
    }

private:
    CJParams params;
    int axis = 2; // default 2D
};

} // namespace tracker
