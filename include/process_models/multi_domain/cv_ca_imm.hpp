// process_cv_ca_imm.hpp
#pragma once
#include "process_model.hpp"
#include <memory>
#include <vector>

namespace tracker {

// Combines CV and CA
class CVCAIMMModel : public IProcessModel {
public:
    explicit CVCAIMMModel() {
        models_.push_back(std::make_unique<CVModel>()); // assuming CVModel exists
        models_.push_back(std::make_unique<CA3DModel>()); // assuming CA3DModel exists
    }

    std::string name() const override { return "CVCAIMM"; }
    int stateDim() const override { return models_[0]->stateDim(); }
    bool isLinear() const override { return true; } // average of linear models

    Vec f(const Vec& x, double dt) const override {
        Vec xn = Vec::Zero(x.size());
        for (auto& m : models_) xn += m->f(x, dt);
        return xn / models_.size();
    }

    Mat F(const Vec& x, double dt) const override {
        Mat J = Mat::Zero(stateDim(), stateDim());
        for (auto& m : models_) J += m->F(x, dt);
        return J / models_.size();
    }

    Mat Qd(double dt) const override {
        Mat Q = Mat::Zero(stateDim(), stateDim());
        for (auto& m : models_) Q += m->Qd(dt);
        return Q / models_.size();
    }

    std::unique_ptr<IProcessModel> clone() const override {
        return std::make_unique<CVCAIMMModel>();
    }

private:
    std::vector<std::unique_ptr<IProcessModel>> models_;
};

} // namespace tracker
