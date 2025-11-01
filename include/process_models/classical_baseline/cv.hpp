#pragma once
#include "process_model.hpp"
#include <Eigen/Dense>

namespace tracker {

    // ======================================================
    // Constant Velocity (CV) Process Model
    // ======================================================

    // ----------------------
    // Parameters
    // ----------------------
    struct CVParams {
        double q_v = 0.01;  // Process noise spectral density (velocity)
    };

    // ----------------------
    // Constant Velocity Model (CV)
    // ----------------------
    class CVModel : public IProcessModel {
    public:
        explicit CVModel(const CVParams& p = {}) : params(p) {}

        // Model information
        std::string name() const override { return "CV"; }
        bool isLinear() const override { return true; }

        // Set number of axes (2D or 3D)
        void setAxis(int a) { axis = a; }

        // State dimension: 2 per axis (position + velocity)
        int stateDim() const override { return axis * 2; }

        // ------------------------------------------
        // State propagation: x_{k+1} = F * x_k
        // ------------------------------------------
        Vec f(const Vec& x, double dt) const override {
            return F(dt) * x;
        }

        // ------------------------------------------
        // Linear state transition matrix
        // ------------------------------------------
        Mat F(double dt) const {
            int n = stateDim();
            Mat Fm = Mat::Identity(n, n);
            for (int ax = 0; ax < axis; ++ax) {
                int off = ax * 2;
                Fm(off, off + 1) = dt;
            }
            return Fm;
        }

        // Analytic Jacobian for EKF (same as F for linear systems)
        Mat F(const Vec&, double dt) const override {
            return F(dt);
        }

        // ------------------------------------------
        // Discrete process noise covariance Qd
        // ------------------------------------------
        Mat Qd(double dt) const override {
            int n = stateDim();
            Mat Q = Mat::Zero(n, n);
            double q = params.q_v;

            for (int ax = 0; ax < axis; ++ax) {
                int off = ax * 2;
                double dt2 = dt * dt;

                Q(off, off) = 0.25 * dt2 * dt2 * q;
                Q(off, off + 1) = 0.5 * dt2 * q;
                Q(off + 1, off) = 0.5 * dt2 * q;
                Q(off + 1, off + 1) = dt * q;
            }

            return Q;
        }

        // ------------------------------------------
        // Cloning support for polymorphic copies
        // ------------------------------------------
        std::unique_ptr<IProcessModel> clone() const override {
            return std::make_unique<CVModel>(*this);
        }

    private:
        CVParams params;
        int axis = 2;  // Default to 2D (can be set to 3 for 3D)
    };

}  // namespace tracker

