// test_radar_model.cpp
#include "measurement_models/radar.hpp"
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

using namespace tracker;

//------------------------------------------------------------------------------
// RadarModelTest
//
// Unit tests for RadarModel sensor.
// Tests include measurement function, Jacobian, measurement noise, cloning, and edge cases.
//------------------------------------------------------------------------------
class RadarModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.sigma_r = 0.1;    // range noise
        params.sigma_b = 0.01;   // bearing noise
        params.sigma_rd = 0.05;  // Doppler noise
        radar = std::make_unique<RadarModel>(params);
    }

    RadarParams params;
    std::unique_ptr<RadarModel> radar;
};

//------------------------------------------------------------------------------
// Sensor name and dimension
//------------------------------------------------------------------------------
TEST_F(RadarModelTest, NameAndDimension) {
    EXPECT_EQ(radar->name(), "Radar");
    EXPECT_EQ(radar->measDim(), 3);
}

//------------------------------------------------------------------------------
// Measurement function h(x)
//------------------------------------------------------------------------------
TEST_F(RadarModelTest, MeasurementFunction) {
    Vec x(4);
    x << 3.0, 4.0, 1.0, 2.0; // px, py, vx, vy
    Vec z = radar->h(x);

    double r = std::sqrt(3 * 3 + 4 * 4);
    double bearing = std::atan2(4.0, 3.0);
    double rd = (3 * 1 + 4 * 2) / r;

    EXPECT_NEAR(z(0), r, 1e-12);
    EXPECT_NEAR(z(1), bearing, 1e-12);
    EXPECT_NEAR(z(2), rd, 1e-12);
}

//------------------------------------------------------------------------------
// Edge cases for h(x)
//------------------------------------------------------------------------------
TEST_F(RadarModelTest, ZeroPosition) {
    Vec x = Vec::Zero(4);
    Vec z = radar->h(x);

    // Should be handled safely in the radar model
    EXPECT_NEAR(z(0), 0.0, 1e-12);  // range
    EXPECT_NEAR(z(1), 0.0, 1e-12);  // bearing (atan2(0,0))
    EXPECT_NEAR(z(2), 0.0, 1e-12);  // Doppler
}

TEST_F(RadarModelTest, PureXMotion) {
    Vec x(4);
    x << 5.0, 0.0, 2.0, 0.0; // px>0, py=0, vx>0
    Vec z = radar->h(x);

    EXPECT_NEAR(z(0), 5.0, 1e-12);
    EXPECT_NEAR(z(1), 0.0, 1e-12);
    EXPECT_NEAR(z(2), 2.0, 1e-12);
}

TEST_F(RadarModelTest, PureYMotion) {
    Vec x(4);
    x << 0.0, -3.0, 0.0, -1.0; // px=0, py<0, vy<0
    Vec z = radar->h(x);

    EXPECT_NEAR(z(0), 3.0, 1e-12);
    EXPECT_NEAR(z(1), -M_PI / 2, 1e-12);
    EXPECT_NEAR(z(2), 1.0, 1e-12); // Doppler formula: (px*vx+py*vy)/r
}

//------------------------------------------------------------------------------
// Measurement noise covariance R
//------------------------------------------------------------------------------
TEST_F(RadarModelTest, MeasurementNoise) {
    Mat R = radar->R();
    EXPECT_NEAR(R(0, 0), params.sigma_r * params.sigma_r, 1e-12);
    EXPECT_NEAR(R(1, 1), params.sigma_b * params.sigma_b, 1e-12);
    EXPECT_NEAR(R(2, 2), params.sigma_rd * params.sigma_rd, 1e-12);
    EXPECT_EQ(R(0, 1), 0.0);
    EXPECT_EQ(R(1, 2), 0.0);
    EXPECT_EQ(R(0, 2), 0.0);
}

//------------------------------------------------------------------------------
// Cloning test
//------------------------------------------------------------------------------
TEST_F(RadarModelTest, Cloning) {
    auto clone = radar->clone();
    EXPECT_EQ(clone->name(), "Radar");

    Vec x(4);
    x << 1.0, 2.0, 0.5, -0.5;

    Vec z_orig = radar->h(x);
    Vec z_clone = clone->h(x);

    EXPECT_TRUE(z_orig.isApprox(z_clone, 1e-12));
}

//------------------------------------------------------------------------------
// Jacobian test (basic dimensions, safe zero handling)
//------------------------------------------------------------------------------
TEST_F(RadarModelTest, JacobianDimensions) {
    Vec x = Vec::Zero(4);
    Mat J = radar->H(x);

    EXPECT_EQ(J.rows(), 3);
    EXPECT_EQ(J.cols(), 4);

    // Should not crash for zero-position
}
