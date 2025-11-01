// test_bicyclemodel.cpp
#include "process_models/vehicle/bicycle.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using namespace tracker;

//------------------------------------------------------------------------------
// BicycleModelTest
//
// Unit tests for the Bicycle kinematic vehicle model.
// Tests include state propagation, Jacobian, process noise, and cloning.
//------------------------------------------------------------------------------
class BicycleModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.L = 2.5;       // wheelbase
        params.q_v = 0.01;    // process noise for velocity
        params.q_yaw = 0.001; // process noise for heading
        params.q_delta = 0.001; // process noise for steering angle
    }

    BicycleParams params;
    BicycleModel model{ params };
};

//------------------------------------------------------------------------------
// State propagation tests
//------------------------------------------------------------------------------
TEST_F(BicycleModelTest, StatePropagationStraight) {
    Eigen::VectorXd x(5); // [px, py, theta, v, delta]
    x << 0, 0, 0, 10, 0;  // moving straight along x-axis
    double dt = 1.0;

    Eigen::VectorXd x_next = model.f(x, dt);

    EXPECT_NEAR(x_next(0), 10.0, 1e-9); // px + v*dt
    EXPECT_NEAR(x_next(1), 0.0, 1e-9);  // py unchanged
    EXPECT_NEAR(x_next(2), 0.0, 1e-9);  // heading unchanged
    EXPECT_NEAR(x_next(3), 10.0, 1e-9); // v constant
    EXPECT_NEAR(x_next(4), 0.0, 1e-9);  // delta constant
}

TEST_F(BicycleModelTest, StatePropagationTurn) {
    Eigen::VectorXd x(5);
    x << 0, 0, 0, 5, 0.1; // small steering angle
    double dt = 2.0;

    Eigen::VectorXd x_next = model.f(x, dt);

    EXPECT_NEAR(x_next(0), 10.0 * std::cos(0.0), 1e-9); // px approx
    EXPECT_NEAR(x_next(1), 10.0 * std::sin(0.0), 1e-9); // py approx
    EXPECT_NEAR(x_next(2), 5.0 / params.L * std::tan(0.1) * dt, 1e-9); // theta
    EXPECT_NEAR(x_next(3), 5.0, 1e-9);  // v
    EXPECT_NEAR(x_next(4), 0.1, 1e-9);  // delta
}

//------------------------------------------------------------------------------
// Jacobian test
//------------------------------------------------------------------------------
TEST_F(BicycleModelTest, JacobianDimensions) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(5);
    Eigen::MatrixXd J = model.F(x, 1.0);

    EXPECT_EQ(J.rows(), 5);
    EXPECT_EQ(J.cols(), 5);

    // Check some analytic entries
    EXPECT_NEAR(J(0,2), 0.0, 1e-9);
    EXPECT_NEAR(J(0,3), 1.0, 1e-9);
    EXPECT_NEAR(J(2,3), std::tan(0.0)/params.L, 1e-9);
}

//------------------------------------------------------------------------------
// Process noise test
//------------------------------------------------------------------------------
TEST_F(BicycleModelTest, ProcessNoise) {
    Eigen::MatrixXd Q = model.Qd(1.0);

    EXPECT_NEAR(Q(2,2), params.q_yaw, 1e-12);
    EXPECT_NEAR(Q(3,3), params.q_v, 1e-12);
    EXPECT_NEAR(Q(4,4), params.q_delta, 1e-12);
}

//------------------------------------------------------------------------------
// Cloning test
//------------------------------------------------------------------------------
TEST_F(BicycleModelTest, Cloning) {
    auto clone = model.clone();
    EXPECT_EQ(clone->name(), "Bicycle");

    Eigen::VectorXd x(5);
    x << 1, 2, 0.1, 3, 0.05;
    double dt = 0.5;

    Eigen::VectorXd x_orig = model.f(x, dt);
    Eigen::VectorXd x_clone = clone->f(x, dt);

    EXPECT_TRUE(x_orig.isApprox(x_clone, 1e-12));
}

//------------------------------------------------------------------------------
// Edge cases
//------------------------------------------------------------------------------
TEST_F(BicycleModelTest, ZeroVelocity) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(5);
    double dt = 1.0;

    Eigen::VectorXd x_next = model.f(x, dt);

    for (int i = 0; i < x_next.size(); ++i)
        EXPECT_NEAR(x_next(i), 0.0, 1e-12);
}
