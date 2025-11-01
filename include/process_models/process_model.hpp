// process_model.hpp
#pragma once
#include <Eigen/Dense>
#include <memory>
#include <string>

namespace tracker {

// Base interface for process/dynamics models
class IProcessModel {
public:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;
    virtual ~IProcessModel() = default;

    // Human-readable name
    virtual std::string name() const = 0;

    // Dimension of the state vector
    virtual int stateDim() const = 0;

    // Is this model linear? If true, the tracker may call A() and Qd().
    virtual bool isLinear() const = 0;

    // Nonlinear state-transition function: x_next = f(x, dt)
    // For linear models this can call A()*x or be overridden for efficiency.
    virtual Vec f(const Vec& x, double dt) const = 0;

    // Jacobian of f wrt x: F = df/dx. Must be provided for nonlinear models
    // (or numeric approximation will be used by numericJacobian).
    virtual Mat F(const Vec& x, double dt) const = 0;

    // Discrete-time process noise covariance Q(d) for time step dt.
    // For linear models this should match A() discretization.
    virtual Mat Qd(double dt) const = 0;

    // Optional: analytic A matrix for linear models (x_next = A*x)
    virtual Mat A(double dt) const {
        // default: numerical linearization around 0 (not ideal)
        Mat M = Mat::Zero(stateDim(), stateDim());
        // can fallback to identity when appropriate — but prefer override in linear models
        return M;
    }

    // Utility: numeric central-difference Jacobian (fallback)
    Mat numericJacobian(const Vec& x, double dt, double eps=1e-6) const {
        int n = stateDim();
        Mat J(n, n);
        Vec fx = f(x, dt);
        for (int i = 0; i < n; ++i) {
            Vec xp = x;
            Vec xm = x;
            double h = eps * std::max(1.0, std::abs(x[i]));
            xp[i] += h;
            xm[i] -= h;
            Vec fp = f(xp, dt);
            Vec fm = f(xm, dt);
            J.col(i) = (fp - fm) / (2.0 * h);
        }
        return J;
    }

    // Clone (for polymorphic copying, e.g., IMM)
    virtual std::unique_ptr<IProcessModel> clone() const = 0;
};
} // namespace tracker
