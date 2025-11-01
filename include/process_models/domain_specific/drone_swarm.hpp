// process_drone_swarm.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <vector>

namespace tracker {

struct DroneSwarmParams {
    double q = 1e-3;
};

// Each drone coupled to leader
class DroneSwarmModel : public IProcessModel {
public:
    explicit DroneSwarmModel(const DroneSwarmParams& p = {}) : params(p) {}

    std::string name() const override { return "DroneSwarm"; }
    int stateDim() const override { return 6; } // x,y,z,vx,vy,vz
    bool isLinear() const override { return false; }

    Vec f(const Vec& x, double dt) const override {
        Vec xn = x;
        // Simple leader-follower coupling (placeholder)
        Eigen::Vector3d leader_pos(0,0,1); 
        Eigen::Vector3d follower_pos = x.segment<3>(0);
        Eigen::Vector3d vel = x.segment<3>(3);

        xn.segment<3>(0) = follower_pos + vel*dt + 0.1*(leader_pos-follower_pos)*dt;
        xn.segment<3>(3) = vel + 0.1*(leader_pos-follower_pos)*dt;
        return xn;
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(stateDim(), stateDim());
    }

    Mat Qd(double dt) const override {
        return Mat::Identity(stateDim(), stateDim())*params.q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<DroneSwarmModel>(params);
    }

private:
    DroneSwarmParams params;
};

} // namespace tracker
