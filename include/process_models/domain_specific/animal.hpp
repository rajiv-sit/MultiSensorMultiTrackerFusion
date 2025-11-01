// process_animal.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <random>

namespace tracker {

struct AnimalParams {
    double q = 1e-3;
};

// State: x,y,vx,vy
class AnimalModel : public IProcessModel {
public:
    explicit AnimalModel(const AnimalParams& p = {}) : params(p) {}

    std::string name() const override { return "Animal"; }
    int stateDim() const override { return 4; }
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // Add correlated random walk / levy flight
        Eigen::Vector2d pos = x.head(2);
        Eigen::Vector2d vel = x.tail(2);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0,0.1);

        pos += vel*dt + Eigen::Vector2d(d(gen), d(gen));
        vel += Eigen::Vector2d(d(gen), d(gen));

        xn.head(2) = pos;
        xn.tail(2) = vel;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(stateDim(), stateDim());
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(stateDim(), stateDim())*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<AnimalModel>(params);
    }

private:
    AnimalParams params;
};

} // namespace tracker
