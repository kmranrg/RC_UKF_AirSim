import numpy as np
from my_drone_follow_figure8 import DroneFigure8Controller
from my_rc_ukf import RCplusUKF
import matplotlib.pyplot as plt
from drone_actual_trajectory import actual_trajectory

# Step 1: Collect Drone Trajectory Data
def collect_airsim_data(controller):
    # controller.execute()
    # return np.array(controller.actual_positions)
    return np.array(actual_trajectory)

# Step 2: Train the RCplusUKF Model
def train_rc_ukf_model(trajectory_data):
    inputs = trajectory_data[:-1]
    true_outputs = trajectory_data[1:]
    
    rc_ukf = RCplusUKF(n_inputs=3, n_reservoir=100, process_noise=np.eye(100) * 0.01, measurement_noise=np.eye(3) * 0.1)
    rc_ukf.train(inputs, true_outputs)
    
    return rc_ukf, inputs, true_outputs

# Step 3: Predict and Compare
def predict_and_compare(rc_ukf, inputs, true_outputs):
    predicted_outputs = rc_ukf.predict(inputs, true_outputs)

    # ensuring lengths match before plotting
    min_length = min(len(true_outputs), len(predicted_outputs))
    true_outputs = true_outputs[:min_length]
    predicted_outputs = predicted_outputs[:min_length]

    rc_ukf.plot_results(true_outputs, predicted_outputs, title="RCplusUKF Prediction vs Actual Drone Trajectory")
    rc_ukf.plot_errors(true_outputs, predicted_outputs, title="RCplusUKF Prediction Errors")
    
    return true_outputs, predicted_outputs

# Step 4: Error Metrics
def calculate_error_metrics(true_outputs, predicted_outputs):
    # Align lengths for error calculation
    min_length = min(len(true_outputs), len(predicted_outputs))
    true_outputs = true_outputs[:min_length]
    predicted_outputs = predicted_outputs[:min_length]

    errors = np.abs(true_outputs - predicted_outputs)
    
    # Apply epsilon to avoid division by zero in MAPE
    epsilon = 1e-8  # Small value to prevent division by zero
    mape = np.mean(errors / (np.abs(true_outputs) + epsilon), axis=0) * 100

    mae = np.mean(errors, axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    
    total_mae = np.mean(mae)
    total_rmse = np.mean(rmse)
    total_mape = np.mean(mape)
    
    print("\nTotal MAE:", total_mae)
    print("Total RMSE:", total_rmse)
    print("Total MAPE:", total_mape)

def plot_figure8_trajectory(true_outputs, predicted_outputs):
    plt.figure(figsize=(10, 6))

    # Plotting actual vs predicted in X-Y space
    plt.plot(true_outputs[:, 0], true_outputs[:, 1], label="Actual Drone Trajectory", color='blue')
    plt.plot(predicted_outputs[:, 0], predicted_outputs[:, 1], label="RC+UKF Predicted Trajectory", linestyle='dashed', color='orange')
    
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Figure-8 Trajectory: Actual vs RC+UKF Predicted")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure equal scaling for x and y axes
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Initialize Drone Controller
    drone_controller = DroneFigure8Controller()

    # Step 1: Collect Data
    print("Collecting drone trajectory data...")
    trajectory_data = collect_airsim_data(drone_controller)

    # Step 2: Train Model
    print("Training RCplusUKF model...")
    rc_ukf_model, inputs, true_outputs = train_rc_ukf_model(trajectory_data)

    # Step 3: Predict and Compare
    print("Predicting and comparing results...")
    true_outputs, predicted_outputs = predict_and_compare(rc_ukf_model, inputs, true_outputs)

    # Step 4: Calculate Error Metrics
    print("Calculating error metrics...")
    calculate_error_metrics(true_outputs, predicted_outputs)

    plot_figure8_trajectory(true_outputs, predicted_outputs)

    print("RCplusUKF implementation with AirSim completed!")
