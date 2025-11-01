// process_cv_ct_imm.hpp
#pragma once
#include "process_model.hpp"
#include <memory>
#include <vector>

namespace tracker {

// Combines CV and Coordinated Turn
class CVCTIMMModel : public IProcessModel {
public:
    explicit CVCTIMMModel() {
        models_.push_back(std::make_unique<CVModel>());  // Assuming CVModel exists
        models_.push_back(std::make_unique<CTRVModel>()); // Assuming CTRVModel exists
    }

    std::string name() const override { return "CVCTIMM"; }
    int stateDim() const override { return models_[0]->stateDim(); }
    bool isLinear() const override { return false; }

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
        return std::make_unique<CVCTIMMModel>();
    }

private:
    std::vector<std::unique_ptr<IProcessModel>> models_;
};

} // namespace tracker
