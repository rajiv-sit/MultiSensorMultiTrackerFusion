#pragma once

#include "types.hpp"

namespace tracker {

    //------------------------------------------------------------------------------
    // ISensorModel
    //
    // Abstract base class for all sensor models.
    // Provides interfaces for measurement, noise covariance, Jacobian, and cloning.
    //------------------------------------------------------------------------------
    class ISensorModel {
    public:
        using Vec = Eigen::VectorXd;
        using Mat = Eigen::MatrixXd;
        virtual ~ISensorModel() = default;

        /// Sensor name
        virtual std::string name() const = 0;

        /// Measurement dimension
        virtual int measDim() const = 0;

        /// Measurement function h(x)
        virtual Vec h(const Vec& x) const = 0;

        /// Analytic measurement Jacobian H(x)
        virtual Mat H(const Vec& x) const = 0;

        /// Measurement noise covariance R
        virtual Mat R() const = 0;

        /// Clone sensor
        virtual std::unique_ptr<ISensorModel> clone() const = 0;
    };

} // namespace tracker
