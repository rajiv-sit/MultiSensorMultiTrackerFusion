#pragma once
#include "sensor_model.hpp"

namespace tracker {

    struct GPSParams {
        double sigma_lat = 1.0;
        double sigma_lon = 1.0;
        double sigma_alt = 1.0;
    };

    class GPSModel : public ISensorModel {
    public:
        explicit GPSModel(const GPSParams& p = {}) : params(p) {}

        std::string name() const override { return "GPS"; }
        int measDim() const override { return 3; }

        Vec h(const Vec& x) const override {
            Vec z(3);
            z << x(0), x(1), x(2); // lat, lon, alt
            return z;
        }

        Mat H(const Vec& x) const override {
            Mat J = Mat::Zero(3, x.size());
            if (x.size() >= 3) { J(0, 0) = 1; J(1, 1) = 1; J(2, 2) = 1; }
            return J;
        }

        Mat R() const override {
            Mat Q = Mat::Zero(3, 3);
            Q(0, 0) = params.sigma_lat * params.sigma_lat;
            Q(1, 1) = params.sigma_lon * params.sigma_lon;
            Q(2, 2) = params.sigma_alt * params.sigma_alt;
            return Q;
        }

        std::unique_ptr<ISensorModel> clone() const override { return std::make_unique<GPSModel>(*this); }

    private:
        GPSParams params;
    };

} // namespace tracker
