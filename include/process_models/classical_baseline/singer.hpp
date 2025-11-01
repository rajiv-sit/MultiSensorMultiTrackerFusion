// process_singer.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// ----------------------
// Singer Model Parameters
// ----------------------
struct SingerParams {
    double q_a = 0.01;   // acceleration noise spectral density
    double tau = 5.0;    // acceleration correlation time (seconds)
};

// ----------------------
// Singer Model
// ----------------------
class SingerModel : public IProcessModel {
public:
    explicit SingerModel(const SingerParams& p = {}) : params(p) {}

    std::string name() const override { return "Singer"; }

    // 2D: px, py, vx, vy, ax, ay → 6 states
    // 3D: px, py, pz, vx, vy, vz, ax, ay, az → 9 states
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
        double alpha = std::exp(-dt / params.tau);

        for (int ax = 0; ax < axis; ++ax) {
            int off = ax * 3;
            // px update
            Fm(off, off+1) = (1 - alpha) * params.tau;
            Fm(off, off+2) = (1 - alpha) * params.tau*params.tau;
            // vx update
            Fm(off+1, off+2) = (1 - alpha) * params.tau;
            // ax update
            Fm(off+2, off+2) = alpha;
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
        double q = params.q_a;
        double tau = params.tau;

        for (int ax = 0; ax < axis; ++ax) {
            int off = ax*3;
            double c = 1 - std::exp(-2*dt/tau);
            double q_ax = q * tau * c;

            Q(off,off)     = q_ax * dt*dt/2.0;
            Q(off,off+1)   = q_ax * dt/2.0;
            Q(off,off+2)   = q_ax * dt/2.0;
            Q(off+1,off)   = q_ax * dt/2.0;
            Q(off+1,off+1) = q_ax;
            Q(off+1,off+2) = q_ax/2.0;
            Q(off+2,off)   = q_ax * dt/2.0;
            Q(off+2,off+1) = q_ax/2.0;
            Q(off+2,off+2) = q_ax;
        }
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<SingerModel>(*this);
    }

private:
    SingerParams params;
    int axis = 2; // default 2D
};

} // namespace tracker
