#pragma once
#include "sensor_model.hpp"

namespace tracker {

    struct AltimeterParams { double sigma_h = 0.5; };

    class AltimeterModel : public ISensorModel {
    public:
        explicit AltimeterModel(const AltimeterParams& p = {}) : params(p) {}

        std::string name() const override { return "Altimeter"; }
        int measDim() const override { return 1; }

        Vec h(const Vec& x) const override {
            Vec z(1);
            z << x(2); // altitude = z-coordinate
            return z;
        }

        Mat H(const Vec& x) const override {
            Mat J = Mat::Zero(1, x.size());
            if (x.size() >= 3) J(0, 2) = 1.0;
            return J;
        }

        Mat R() const override { return Mat::Identity(1, 1) * params.sigma_h * params.sigma_h; }

        std::unique_ptr<ISensorModel> clone() const override { return std::make_unique<AltimeterModel>(*this); }

    private:
        AltimeterParams params;
    };

} // namespace tracker
