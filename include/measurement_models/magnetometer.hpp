#pragma once
#include "sensor_model.hpp"

namespace tracker {

    struct MagnetometerParams {
        double sigma_x = 0.01;
        double sigma_y = 0.01;
        double sigma_z = 0.01;
    };

    class MagnetometerModel : public ISensorModel {
    public:
        explicit MagnetometerModel(const MagnetometerParams& p = {}) : params(p) {}

        std::string name() const override { return "Magnetometer"; }
        int measDim() const override { return 3; }

        Vec h(const Vec& x) const override {
            Vec z(3);
            z << x(0), x(1), x(2); // direct 3D magnetic field
            return z;
        }

        Mat H(const Vec& x) const override {
            Mat J = Mat::Zero(3, x.size());
            if (x.size() >= 3) {
                J(0, 0) = 1.0;
                J(1, 1) = 1.0;
                J(2, 2) = 1.0;
            }
            return J;
        }

        Mat R() const override {
            Mat Q = Mat::Zero(3, 3);
            Q(0, 0) = params.sigma_x * params.sigma_x;
            Q(1, 1) = params.sigma_y * params.sigma_y;
            Q(2, 2) = params.sigma_z * params.sigma_z;
            return Q;
        }

        std::unique_ptr<ISensorModel> clone() const override { return std::make_unique<MagnetometerModel>(*this); }

    private:
        MagnetometerParams params;
    };

} // namespace tracker
