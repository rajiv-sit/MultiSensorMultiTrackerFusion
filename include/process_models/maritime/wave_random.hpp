// process_wave_random.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace tracker {

// x = [px, py, vx, vy]ᵀ (CV base) + sinusoidal perturbation
struct WaveParams {
    double q = 1e-3;
    double amp = 0.5;    // wave amplitude
    double freq = 0.1;   // wave frequency
};

class WaveRandomModel : public IProcessModel {
public:
    explicit WaveRandomModel(const WaveParams& p = {}) : params(p) {}

    std::string name() const override { return "WaveRandom"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        double t = dt; // simple: assume current time dt, can be accumulated externally
        xn(0) += x(2)*dt + params.amp * std::sin(params.freq * t);
        xn(1) += x(3)*dt + params.amp * std::cos(params.freq * t);
        xn(2) = x(2);
        xn(3) = x(3);
        return xn;
    }

    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(4,4);
        J(0,2) = dt;
        J(1,3) = dt;
        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(4,4) * params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<WaveRandomModel>(*this);
    }

private:
    WaveParams params;
};

} // namespace tracker
