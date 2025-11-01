// process_maneuvering_ship.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [px, py, psi, u, v, r]ᵀ
struct ShipParams {
    double q = 1e-3; // general process noise
};

class ManeuveringShipModel : public IProcessModel {
public:
    explicit ManeuveringShipModel(const ShipParams& p = {}) : params(p) {}

    std::string name() const override { return "ManeuveringShip"; }
    int stateDim() const override { return 6; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        double px = x(0);
        double py = x(1);
        double psi = x(2);
        double u = x(3); // surge velocity
        double v = x(4); // sway velocity
        double r = x(5); // yaw rate

        xn(0) = px + (u * std::cos(psi) - v * std::sin(psi)) * dt;
        xn(1) = py + (u * std::sin(psi) + v * std::cos(psi)) * dt;
        xn(2) = psi + r * dt;
        xn(3) = u; // assuming constant acceleration can be added
        xn(4) = v;
        xn(5) = r;

        return xn;
    }

    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Identity(6,6);
        double psi = x(2);
        double u = x(3);
        double v = x(4);

        // d(px,py)/d(psi)
        J(0,2) = (-u * std::sin(psi) - v * std::cos(psi)) * dt;
        J(1,2) = ( u * std::cos(psi) - v * std::sin(psi)) * dt;

        // d(px,py)/d(u,v)
        J(0,3) = std::cos(psi) * dt; J(0,4) = -std::sin(psi) * dt;
        J(1,3) = std::sin(psi) * dt; J(1,4) =  std::cos(psi) * dt;

        // d(psi)/d(r)
        J(2,5) = dt;

        return J;
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(6,6) * params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<ManeuveringShipModel>(*this);
    }

private:
    ShipParams params;
};

} // namespace tracker
