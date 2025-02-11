import numpy as np
import matplotlib.pyplot as plt
from my_RC import ReservoirComputer
from my_UKF import UnscentedKalmanFilter

class RCplusUKF:
    def __init__(self, n_inputs, n_reservoir, process_noise, measurement_noise, random_seed=None):
        self.rc_model = ReservoirComputer(n_inputs=n_inputs, n_reservoir=n_reservoir, random_seed=random_seed)

        # ensuring process_noise and measurement_noise are numpy arrays
        process_noise = np.array(process_noise) if np.isscalar(process_noise) else process_noise
        measurement_noise = np.array(measurement_noise) if np.isscalar(measurement_noise) else measurement_noise

        self.ukf = UnscentedKalmanFilter(n_dim=n_reservoir, process_noise=process_noise, measurement_noise=measurement_noise)

    def train(self, inputs, outputs):
        reservoir_states = self.rc_model.run_reservoir(inputs)
        self.rc_model.train_readout(reservoir_states[:-1], outputs[:-1])

    def predict(self, inputs, true_outputs, measurement_noise_std=0.1):
        n_steps = inputs.shape[0]
        predicted_outputs = []

        for t in range(n_steps - 1):
            input_t = inputs[t].reshape(-1, 1)

            def rc_process_model(state):
                return self.rc_model.update_reservoir_state(state.reshape(-1, 1), input_t).ravel()

            sigma_points_pred = self.ukf.predict(process_model=rc_process_model)

            measurement = true_outputs[t + 1] + np.random.normal(0, measurement_noise_std, true_outputs[t + 1].shape)

            def measurement_model(state):
                return self.rc_model.predict(state.reshape(1, -1)).ravel()

            self.ukf.update(sigma_points_pred, measurement, measurement_model)
            predicted_output = measurement_model(self.ukf.get_state())
            predicted_outputs.append(predicted_output)

        return np.array(predicted_outputs)

    def plot_results(self, true_values, predicted_values, title="Hybrid RC-UKF Prediction Performance"):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        components = ['X', 'Y', 'Z']
        
        for i in range(3):
            axs[i].plot(true_values[:, i], label=f"True {components[i]}")
            axs[i].plot(predicted_values[:, i], label=f"Predicted {components[i]}", linestyle="dashed", alpha=0.8)
            axs[i].set_xlabel("Time Step")
            axs[i].set_ylabel(f"{components[i]} Amplitude")
            axs[i].legend()
            axs[i].set_title(f"{components[i]} Component")

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def plot_errors(self, true_values, predicted_values, title="Prediction Errors"):
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        components = ['X', 'Y', 'Z']
        
        for i in range(3):
            error = np.abs(true_values[:, i] - predicted_values[:, i])
            axs[i].plot(error, label=f"{components[i]} Error", color='red')
            axs[i].set_xlabel("Time Step")
            axs[i].set_ylabel(f"{components[i]} Error")
            axs[i].legend()
            axs[i].set_title(f"{components[i]} Component Error")

        plt.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
