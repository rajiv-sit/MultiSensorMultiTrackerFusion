#pragma once
#include "sensor_model.hpp"

namespace tracker {

    struct InfraredParams { double sigma_px = 0.1; double sigma_py = 0.1; };

    class InfraredModel : public ISensorModel {
    public:
        explicit InfraredModel(const InfraredParams& p = {}) : params(p) {}

        std::string name() const override { return "Infrared"; }
        int measDim() const override { return 2; }

        Vec h(const Vec& x) const override {
            Vec z(2);
            z << x(0), x(1); // simplified 2D detection
            return z;
        }

        Mat H(const Vec& x) const override {
            Mat J = Mat::Zero(2, x.size());
            if (x.size() >= 2) { J(0, 0) = 1; J(1, 1) = 1; }
            return J;
        }

        Mat R() const override {
            Mat Q = Mat::Zero(2, 2);
            Q(0, 0) = params.sigma_px * params.sigma_px;
            Q(1, 1) = params.sigma_py * params.sigma_py;
            return Q;
        }

        std::unique_ptr<ISensorModel> clone() const override { return std::make_unique<InfraredModel>(*this); }

    private:
        InfraredParams params;
    };

} // namespace tracker
