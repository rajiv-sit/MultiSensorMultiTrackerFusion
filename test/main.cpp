#include "process_models/classical_baseline/cv.hpp"
#include <iostream>
#include <iomanip>

using namespace tracker;

int main() {
    // Create model parameters
    CVParams params;
    params.q_v = 0.05;

    // Create a 2D constant velocity model
    CVModel model(params);
    model.setAxis(2);

    double dt = 1.0;

    // Define initial state: [px, vx, py, vy]
    CVModel::Vec x0(4);
    x0 << 0.0, 1.0, 0.0, 0.5; // starts at origin, moving 1m/s in x, 0.5m/s in y

    // Predict next state using the process model
    auto x1 = model.f(x0, dt);

    // Display results
    std::cout << "=== Constant Velocity Model Test ===\n";
    std::cout << "Model name: " << model.name() << "\n";
    std::cout << "State dimension: " << model.stateDim() << "\n";
    std::cout << "Time step (dt): " << dt << " s\n\n";

    std::cout << "Initial state (x0):\n" << x0.transpose() << "\n";
    std::cout << "Predicted next state (x1):\n" << x1.transpose() << "\n\n";

    std::cout << "State transition matrix F:\n" << model.F(dt) << "\n\n";
    std::cout << "Process noise Qd:\n" << model.Qd(dt) << "\n";

    return 0;
}
