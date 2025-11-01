// process_pmbm.hpp
#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>
#include <vector>

namespace tracker {

// Each target: state + existence probability
struct PMBMTarget {
    Vec x;
    double r;
};

struct PMBMParams {
    double q = 1e-3;
};

class PMBMModel : public IProcessModel {
public:
    explicit PMBMModel(const PMBMParams& p = {}) : params(p) {}

    std::string name() const override { return "PMBM"; }
    int stateDim() const override { return -1; } // variable
    bool isLinear() const override { return false; }

    std::vector<PMBMTarget> f(const std::vector<PMBMTarget>& targets, double dt) const {
        std::vector<PMBMTarget> next_targets;
        for(const auto& t : targets) {
            PMBMTarget tn = t;
            tn.x.head(2) += tn.x.tail(2)*dt; // CV
            next_targets.push_back(tn);
        }
        return next_targets;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<PMBMModel>(params);
    }

private:
    PMBMParams params;
};

} // namespace tracker
