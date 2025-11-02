#pragma once
#include "sensor_model.hpp"

namespace tracker {

    struct LidarParams {
        double sigma_px = 0.01;
        double sigma_py = 0.01;
        double sigma_pz = 0.01;
    };

    class LidarModel : public ISensorModel {
    public:
        explicit LidarModel(const LidarParams& p = {}) : params(p) {}

        std::string name() const override { return "Lidar"; }
        int measDim() const override { return 3; }

        Vec h(const Vec& x) const override {
            Vec z(3);
            z << x(0), x(1), x(2); // direct 3D position
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
            Q(0, 0) = params.sigma_px * params.sigma_px;
            Q(1, 1) = params.sigma_py * params.sigma_py;
            Q(2, 2) = params.sigma_pz * params.sigma_pz;
            return Q;
        }

        std::unique_ptr<ISensorModel> clone() const override {
            return std::make_unique<LidarModel>(*this);
        }

    private:
        LidarParams params;
    };

} // namespace tracker
