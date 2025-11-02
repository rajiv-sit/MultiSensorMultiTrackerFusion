#pragma once
#include "sensor_model.hpp"
#include <cmath>

namespace tracker {

    struct CameraParams {
        double sigma_bx = 0.01;
        double sigma_by = 0.01;
    };

    class CameraModel : public ISensorModel {
    public:
        explicit CameraModel(const CameraParams& p = {}) : params(p) {}

        std::string name() const override { return "Camera"; }
        int measDim() const override { return 2; }

        Vec h(const Vec& x) const override {
            double px = x(0), py = x(1);
            Vec z(2);
            z << std::atan2(py, px), std::atan2(py, px); // simplified bearing-x/y
            return z;
        }

        Mat H(const Vec& x) const override {
            Mat J = Mat::Zero(2, x.size());
            double px = x(0), py = x(1);
            double r2 = px * px + py * py;
            if (r2 > 1e-12) {
                J(0, 0) = -py / r2;
                J(0, 1) = px / r2;
                J(1, 0) = -py / r2;
                J(1, 1) = px / r2;
            }
            return J;
        }

        Mat R() const override {
            Mat Q = Mat::Zero(2, 2);
            Q(0, 0) = params.sigma_bx * params.sigma_bx;
            Q(1, 1) = params.sigma_by * params.sigma_by;
            return Q;
        }

        std::unique_ptr<ISensorModel> clone() const override {
            return std::make_unique<CameraModel>(*this);
        }

    private:
        CameraParams params;
    };

} // namespace tracker
