#include "process_models/classical_baseline/cv.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using namespace tracker;

class CVModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.q_v = 0.01;
        cv2d.setAxis(2);
        cv3d.setAxis(3);
    }

    CVParams params;
    CVModel cv2d{ params };
    CVModel cv3d{ params };
};

// -------------------------
// State propagation tests
// -------------------------
TEST_F(CVModelTest, StatePropagation2D) {
    Eigen::VectorXd x(4);
    // Interleaved [px, vx, py, vy]
    x << 0, 1, 0, 1;
    double dt = 1.0;

    Eigen::VectorXd x_next = cv2d.f(x, dt);

    EXPECT_NEAR(x_next(0), 1.0, 1e-9); // px + vx*dt
    EXPECT_NEAR(x_next(1), 1.0, 1e-9); // vx
    EXPECT_NEAR(x_next(2), 1.0, 1e-9); // py + vy*dt
    EXPECT_NEAR(x_next(3), 1.0, 1e-9); // vy
}

TEST_F(CVModelTest, StatePropagation3D) {
    Eigen::VectorXd x(6);
    // Interleaved [px, vx, py, vy, pz, vz]
    x << 0, 1, 0, 2, 0, 3;
    double dt = 2.0;

    Eigen::VectorXd x_next = cv3d.f(x, dt);

    EXPECT_NEAR(x_next(0), 2.0, 1e-9);
    EXPECT_NEAR(x_next(1), 1.0, 1e-9);
    EXPECT_NEAR(x_next(2), 4.0, 1e-9);
    EXPECT_NEAR(x_next(3), 2.0, 1e-9);
    EXPECT_NEAR(x_next(4), 6.0, 1e-9);
    EXPECT_NEAR(x_next(5), 3.0, 1e-9);
}

TEST_F(CVModelTest, ZeroVelocity) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(4);
    double dt = 1.0;
    Eigen::VectorXd x_next = cv2d.f(x, dt);

    for (int i = 0; i < x_next.size(); ++i)
        EXPECT_NEAR(x_next(i), 0.0, 1e-9);
}

TEST_F(CVModelTest, NegativeDt) {
    Eigen::VectorXd x(4);
    x << 1, 3, 2, 4; // [px, vx, py, vy]
    double dt = -1.0;
    Eigen::VectorXd x_next = cv2d.f(x, dt);

    EXPECT_NEAR(x_next(0), -2.0, 1e-9); // px + vx*dt
    EXPECT_NEAR(x_next(1), 3.0, 1e-9);  // vx stays
    EXPECT_NEAR(x_next(2), -2.0, 1e-9); // py + vy*dt
    EXPECT_NEAR(x_next(3), 4.0, 1e-9);  // vy stays
}

TEST_F(CVModelTest, LargeDt) {
    Eigen::VectorXd x(4);
    x << 1, 1, 1, 1; // [px, vx, py, vy]
    double dt = 1e3;
    Eigen::VectorXd x_next = cv2d.f(x, dt);

    EXPECT_NEAR(x_next(0), 1001.0, 1e-6); // px + vx*dt
    EXPECT_NEAR(x_next(1), 1.0, 1e-6);    // vx
    EXPECT_NEAR(x_next(2), 1001.0, 1e-6); // py + vy*dt
    EXPECT_NEAR(x_next(3), 1.0, 1e-6);    // vy
}

// -------------------------
// Process noise
// -------------------------
TEST_F(CVModelTest, ProcessNoise2D) {
    Eigen::MatrixXd Q = cv2d.Qd(1.0);

    EXPECT_NEAR(Q(0, 0), 0.0025, 1e-6);
    EXPECT_NEAR(Q(0, 1), 0.005, 1e-6);
    EXPECT_NEAR(Q(1, 0), 0.005, 1e-6);
    EXPECT_NEAR(Q(1, 1), 0.01, 1e-6);

    // Symmetry
    EXPECT_TRUE(Q.isApprox(Q.transpose(), 1e-12));
}

// -------------------------
// State transition matrix
// -------------------------
TEST_F(CVModelTest, StateTransitionMatrix2D) {
    Eigen::MatrixXd F = cv2d.F(1.0);

    EXPECT_EQ(F.rows(), 4);
    EXPECT_EQ(F.cols(), 4);
    EXPECT_NEAR(F(0, 1), 1.0, 1e-9);
    EXPECT_NEAR(F(2, 3), 1.0, 1e-9);
    EXPECT_NEAR(F(0, 0), 1.0, 1e-9);
}

// -------------------------
// Cloning
// -------------------------
TEST_F(CVModelTest, Cloning) {
    auto clone = cv2d.clone();
    EXPECT_EQ(clone->name(), "CV");

    Eigen::VectorXd x(4);
    x << 0, 1, 0, 1; // [px, vx, py, vy]
    double dt = 1.0;

    Eigen::VectorXd x_next_original = cv2d.f(x, dt);
    Eigen::VectorXd x_next_clone = clone->f(x, dt);

    EXPECT_TRUE(x_next_original.isApprox(x_next_clone, 1e-12));
}
