// process_state_augmented.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

// x = [state, bias_radar, bias_camera]ᵀ
struct StateAugParams {
    double q_state = 1e-3;
    double q_bias = 1e-5;
};

class MultiSensorAugModel : public IProcessModel {
public:
    explicit MultiSensorAugModel(int state_len, const StateAugParams& p = {}) 
        : state_len_(state_len), params(p) {}

    std::string name() const override { return "MultiSensorAugmented"; }
    int stateDim() const override { return state_len_ + 2; } // example: 2 sensors
    bool isLinear() const override { return true; }

    Vec f(const Vec& x, double dt) const override {
        return x; // biases assumed constant
    }

    Mat F(const Vec&, double dt) const override {
        return Mat::Identity(stateDim(), stateDim());
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(stateDim(), stateDim());
        Q.block<state_len_,state_len_>(0,0) = Mat::Identity(state_len_,state_len_) * params.q_state;
        Q(state_len_,state_len_) = params.q_bias;
        Q(state_len_+1,state_len_+1) = params.q_bias;
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<MultiSensorAugModel>(state_len_, params);
    }

private:
    int state_len_;
    StateAugParams params;
};

} // namespace tracker
