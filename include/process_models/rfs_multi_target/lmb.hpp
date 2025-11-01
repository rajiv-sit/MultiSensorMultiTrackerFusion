// process_lmb.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <vector>
#include <tuple>

namespace tracker {

// Each target: state + label + existence probability
struct LMBTarget {
    Vec x;
    int label;
    double r; // existence probability
};

struct LMBParams {
    double q = 1e-3; // process noise for each target
};

class LMBModel : public IProcessModel {
public:
    explicit LMBModel(const LMBParams& p = {}) : params(p) {}

    std::string name() const override { return "LMB"; }
    int stateDim() const override { return -1; } // variable: number of targets
    bool isLinear() const override { return false; }

    // Evolve all targets
    std::vector<LMBTarget> f(const std::vector<LMBTarget>& targets, double dt) const {
        std::vector<LMBTarget> next_targets;
        for(const auto& t : targets) {
            LMBTarget tn = t;
            tn.x.head(2) += tn.x.tail(2)*dt; // simple CV
            next_targets.push_back(tn);
        }
        return next_targets;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<LMBModel>(params);
    }

private:
    LMBParams params;
};

} // namespace tracker
