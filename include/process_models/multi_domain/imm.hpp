// process_imm.hpp
#pragma once
#include "process_model.hpp"
#include <vector>
#include <memory>

namespace tracker {

// IMM model combines multiple process models
class IMMModel : public IProcessModel {
public:
    explicit IMMModel(const std::vector<std::unique_ptr<IProcessModel>>& models)
    {
        for (auto& m : models) {
            models_.push_back(m->clone());
        }
    }

    std::string name() const override { return "IMM"; }

    int stateDim() const override { 
        if (!models_.empty()) return models_[0]->stateDim(); 
        return 0; 
    }

    bool isLinear() const override { return false; } // combination can include nonlinear

    Vec f(const Vec& x, double dt) const override {
        // naive: weighted sum, assumes equal weight
        Vec xn = Vec::Zero(x.size());
        for (auto& m : models_) {
            xn += m->f(x, dt);
        }
        xn /= models_.size();
        return xn;
    }

    Mat F(const Vec& x, double dt) const override {
        // approximate Jacobian as average
        Mat J = Mat::Zero(x.size(), x.size());
        for (auto& m : models_) {
            J += m->F(x, dt);
        }
        J /= models_.size();
        return J;
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(stateDim(), stateDim());
        for (auto& m : models_) {
            Q += m->Qd(dt);
        }
        Q /= models_.size();
        return Q;
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<IMMModel>(models_);
    }

private:
    std::vector<std::unique_ptr<IProcessModel>> models_;
};

} // namespace tracker
