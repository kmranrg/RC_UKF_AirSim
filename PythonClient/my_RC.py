import numpy as np
import matplotlib.pyplot as plt

class ReservoirComputer:
    def __init__(self, n_inputs, n_reservoir, spectral_radius=0.9, sparsity=0.2, reg=1e-5, noise_std=0.1, random_seed=None):
        """
        Initializes the Reservoir Computing model.
        
        Parameters:
        - n_inputs: Number of input neurons.
        - n_reservoir: Number of reservoir neurons.
        - spectral_radius: Maximum absolute eigenvalue of the reservoir matrix.
        - sparsity: Fraction of reservoir weights that are nonzero.
        - reg: Regularization strength for training the readout layer.
        - noise_std: Standard deviation of Gaussian noise added to input.
        - random_seed: Seed for reproducibility.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.reg = reg
        self.noise_std = noise_std

        # Initialize input weights
        self.W_in = self.initialize_input_weights()
        
        # Initialize reservoir weights
        self.W = self.initialize_reservoir()
        
        # Readout weights (to be learned)
        self.W_out = None
    
    def initialize_input_weights(self, scale=0.1):
        """Initializes random input weights."""
        return scale * (np.random.rand(self.n_reservoir, self.n_inputs) * 2 - 1)

    def initialize_reservoir(self):
        """Initializes a sparse reservoir weight matrix with controlled spectral radius."""
        W = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5  # Values in [-0.5, 0.5]
        W[np.random.rand(*W.shape) > self.sparsity] = 0  # Apply sparsity

        # Normalize by spectral radius
        eig_values = np.linalg.eigvals(W)
        max_eig = max(abs(eig_values))
        W *= self.spectral_radius / max_eig
        
        return W

    def update_reservoir_state(self, x, u):
        """Updates the reservoir state using tanh activation."""
        return np.tanh(np.dot(self.W_in, u) + np.dot(self.W, x))

    def run_reservoir(self, inputs):
        """Runs the reservoir over a sequence of inputs."""
        n_steps = inputs.shape[0]
        states = np.zeros((n_steps, self.n_reservoir))
        x = np.zeros((self.n_reservoir, 1))  # Initial state
        
        for t in range(n_steps):
            u = inputs[t].reshape(-1, 1)
            x = self.update_reservoir_state(x, u)
            states[t] = x.ravel()
        
        return states

    def train_readout(self, states, outputs):
        """Trains the readout layer using ridge regression."""
        X = states
        Y = outputs
        
        # Solve for W_out using ridge regression
        self.W_out = np.dot(np.dot(Y.T, X), np.linalg.inv(np.dot(X.T, X) + self.reg * np.eye(X.shape[1])))

    def predict(self, states):
        """Predicts output using trained readout layer."""
        if self.W_out is None:
            raise ValueError("Model has not been trained. Call `train_readout` first.")
        return np.dot(states, self.W_out.T)

    def add_noise(self, inputs):
        """Adds Gaussian noise to inputs."""
        return inputs + np.random.normal(0, self.noise_std, inputs.shape)

    def plot_results(self, true_values, predicted_values, title="Prediction Performance"):
        """Plots true vs. predicted values."""
        plt.figure(figsize=(8, 4))
        plt.plot(true_values, label="True Output")
        plt.plot(predicted_values, label="Predicted Output", linestyle="dashed", alpha=0.8)
        plt.xlabel("Time Step")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.title(title)
        plt.show()