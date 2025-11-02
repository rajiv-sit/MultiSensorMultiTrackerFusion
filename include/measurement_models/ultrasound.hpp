#pragma once
#include "sensor_model.hpp"
#include <cmath>

namespace tracker {

struct UltrasoundParams { double sigma_r = 0.01; };

class UltrasoundModel : public ISensorModel {
public:
    explicit UltrasoundModel(const UltrasoundParams& p = {}) : params(p) {}

    std::string name() const override { return "Ultrasound"; }
    int measDim() const override { return 1; }

    Vec h(const Vec& x) const override {
        Vec z(1);
        z << std::sqrt(x(0)*x(0) + x(1)*x(1)); // range only
        return z;
    }

    Mat H(const Vec& x) const override {
        Mat J = Mat::Zero(1, x.size());
        double r2 = x(0)*x(0)+x(1)*x(1);
        double r = std::sqrt(r2);
        if (r > 1e-12) { J(0,0) = x(0)/r; J(0,1) = x(1)/r; }
        return J;
    }

    Mat R() const override { return Mat::Identity(1,1)*params.sigma_r*params.sigma_r; }

    std::unique_ptr<ISensorModel> clone() const override { return std::make_unique<UltrasoundModel>(*this); }

private:
    UltrasoundParams params;
};

} // namespace tracker
