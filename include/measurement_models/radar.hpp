#pragma once
#include "sensor_model.hpp"
#include <cmath>

namespace tracker {

    struct RadarParams {
        double sigma_r = 1.0;
        double sigma_b = 0.01;
        double sigma_rd = 0.5;
    };

    class RadarModel : public ISensorModel {
    public:
        explicit RadarModel(const RadarParams& p = {}) : params(p) {}

        std::string name() const override { return "Radar"; }
        int measDim() const override { return 3; }

        Vec h(const Vec& x) const override {
            double px = x(0), py = x(1), vx = x(2), vy = x(3);
            double r2 = px * px + py * py;
            double r = std::sqrt(r2);

            double rd = (r > 1e-12) ? (px * vx + py * vy) / r : 0.0;

            Vec z(3);
            z << r, std::atan2(py, px), rd;
            return z;
        }

        Mat H(const Vec& x) const override {
            Mat J = Mat::Zero(3, x.size());
            double px = x(0), py = x(1), vx = x(2), vy = x(3);
            double r2 = px * px + py * py;
            double r = std::sqrt(r2);

            if (r > 1e-12) {
                // Range
                J(0, 0) = px / r;
                J(0, 1) = py / r;

                // Bearing
                J(1, 0) = -py / r2;
                J(1, 1) = px / r2;

                // Doppler
                J(2, 0) = (vx * r2 - (px * vx + py * vy) * px) / (r * r2);
                J(2, 1) = (vy * r2 - (px * vx + py * vy) * py) / (r * r2);
                J(2, 2) = px / r;
                J(2, 3) = py / r;
            }

            return J;
        }

        Mat R() const override {
            Mat Q = Mat::Zero(3, 3);
            Q(0, 0) = params.sigma_r * params.sigma_r;
            Q(1, 1) = params.sigma_b * params.sigma_b;
            Q(2, 2) = params.sigma_rd * params.sigma_rd;
            return Q;
        }

        std::unique_ptr<ISensorModel> clone() const override {
            return std::make_unique<RadarModel>(*this);
        }

    private:
        RadarParams params;
    };

} // namespace tracker
