#pragma once
#include "sensor_model.hpp"

namespace tracker {

    struct IMUParams {
        double sigma_acc = 0.01;
        double sigma_gyro = 0.001;
    };

    class IMUModel : public ISensorModel {
    public:
        explicit IMUModel(const IMUParams& p = {}) : params(p) {}

        std::string name() const override { return "IMU"; }
        int measDim() const override { return 6; } // 3 accel + 3 gyro

        Vec h(const Vec& x) const override {
            Vec z(6);
            z << x(0), x(1), x(2), x(3), x(4), x(5); // direct 3D accel + 3D gyro
            return z;
        }

        Mat H(const Vec& x) const override {
            Mat J = Mat::Zero(6, x.size());
            if (x.size() >= 6) {
                for (int i = 0; i < 6; ++i) J(i, i) = 1.0;
            }
            return J;
        }

        Mat R() const override {
            Mat Q = Mat::Zero(6, 6);
            for (int i = 0; i < 3; i++) Q(i, i) = params.sigma_acc * params.sigma_acc;
            for (int i = 3; i < 6; i++) Q(i, i) = params.sigma_gyro * params.sigma_gyro;
            return Q;
        }

        std::unique_ptr<ISensorModel> clone() const override { return std::make_unique<IMUModel>(*this); }

    private:
        IMUParams params;
    };

} // namespace tracker
