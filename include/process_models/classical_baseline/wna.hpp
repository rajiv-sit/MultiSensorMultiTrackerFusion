// process_wna.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// ----------------------
// WNA Parameters
// ----------------------
struct WNAParams {
    double q = 0.01; // acceleration noise spectral density
};

// ----------------------
// White Noise Acceleration Model
// ----------------------
class WNAModel : public IProcessModel {
public:
    explicit WNAModel(const WNAParams& p = {}) : params(p) {}

    std::string name() const override { return "WNA"; }

    // 2D state: px, py, vx, vy → 4 states
    int stateDim() const override { return 4; }

    bool isLinear() const override { return true; }

    // State propagation: x_{k+1} = F * x_k
    Vec f(const Vec& x, double dt) const override {
        return F(dt) * x;
    }

    // Discrete-time transition matrix
    Mat F(double dt) const {
        Mat Fm = Mat::Identity(4,4);
        Fm(0,2) = dt;
        Fm(1,3) = dt;
        return Fm;
    }

    Mat F(const Vec&, double dt) const override {
        return F(dt);
    }

    // Process noise Qd = G * q * G^T
    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(4,4);
        double dt2 = dt*dt;
        double dt3 = dt2*dt;
        double q = params.q;

        // Discrete-time process noise for constant acceleration
        Q(0,0) = dt3/3.0 * q;
        Q(0,2) = dt2/2.0 * q;
        Q(1,1) = dt3/3.0 * q;
        Q(1,3) = dt2/2.0 * q;
        Q(2,0) = dt2/2.0 * q;
        Q(2,2) = dt * q;
        Q(3,1) = dt2/2.0 * q;
        Q(3,3) = dt * q;

        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<WNAModel>(*this);
    }

private:
    WNAParams params;
};

} // namespace tracker
