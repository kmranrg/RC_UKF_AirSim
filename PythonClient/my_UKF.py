import numpy as np

class UnscentedKalmanFilter:
    def __init__(self, n_dim, process_noise, measurement_noise, alpha=1e-3, beta=2, kappa=0):
        """
        Initializes the Unscented Kalman Filter (UKF).

        Parameters:
        - n_dim: Number of state dimensions.
        - process_noise: Process noise covariance matrix (Q).
        - measurement_noise: Measurement noise covariance matrix (R).
        - alpha, beta, kappa: UKF parameters for sigma point spread and distribution.
        """
        self.n_dim = n_dim  # Number of state variables
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance

        # UKF parameters for sigma point scaling
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n_dim + kappa) - n_dim  # Scaling factor

        # Compute sigma point weights
        self.W_m = np.zeros(2 * n_dim + 1)  # Mean weights
        self.W_c = np.zeros(2 * n_dim + 1)  # Covariance weights

        self.W_m[0] = self.lambda_ / (n_dim + self.lambda_)
        self.W_c[0] = self.lambda_ / (n_dim + self.lambda_) + (1 - alpha**2 + beta)

        for i in range(1, 2 * n_dim + 1):
            self.W_m[i] = 1 / (2 * (n_dim + self.lambda_))
            self.W_c[i] = 1 / (2 * (n_dim + self.lambda_))

        # Initial state and covariance
        self.x = np.zeros(n_dim)  # Initial state estimate
        self.P = np.eye(n_dim)  # Initial covariance estimate

    def generate_sigma_points(self):
        """
        Generates sigma points around the current state estimate.
        """
        sigma_points = np.zeros((2 * self.n_dim + 1, self.n_dim))
        sigma_points[0] = self.x  # First sigma point is the mean

        sqrt_P = np.linalg.cholesky((self.n_dim + self.lambda_) * self.P)  # Cholesky decomposition

        for i in range(self.n_dim):
            sigma_points[i + 1] = self.x + sqrt_P[i]
            sigma_points[self.n_dim + i + 1] = self.x - sqrt_P[i]

        return sigma_points

    def predict(self, process_model):
        """
        Predicts the state using the process model function.
        
        Parameters:
        - process_model: Function that propagates sigma points through the process model.
        """
        sigma_points = self.generate_sigma_points()
        sigma_points_pred = np.array([process_model(sp) for sp in sigma_points])

        # Debug 
        # print(f"Sigma points predicted shape: {sigma_points_pred.shape}")
        # print(f"Initial self.P shape before updating: {self.P.shape}")
        # print(f"Process noise (Q) shape: {self.Q.shape}")

        # Compute predicted mean
        self.x = np.sum(self.W_m[:, None] * sigma_points_pred, axis=0)

        # Compute predicted covariance
        self.P = self.Q.copy()
        for i in range(2 * self.n_dim + 1):
            diff = sigma_points_pred[i] - self.x
            # print(f"Iteration {i}: diff shape: {diff.shape}, outer product shape: {np.outer(diff, diff).shape}")
            self.P += self.W_c[i] * np.outer(diff, diff)

        return sigma_points_pred

    def update(self, sigma_points_pred, measurement, measurement_model):
        """
        Updates the state estimate using the measurement.

        Parameters:
        - sigma_points_pred: The predicted sigma points from the process model.
        - measurement: The actual measurement received.
        - measurement_model: Function that maps state to measurement space.
        """
        # Transform sigma points into measurement space
        sigma_points_meas = np.array([measurement_model(sp) for sp in sigma_points_pred])

        # Compute predicted measurement mean
        z_pred = np.sum(self.W_m[:, None] * sigma_points_meas, axis=0)

        # Compute measurement covariance
        P_zz = self.R.copy()
        for i in range(2 * self.n_dim + 1):
            diff = sigma_points_meas[i] - z_pred
            P_zz += self.W_c[i] * np.outer(diff, diff)

        # Compute cross covariance
        P_xz = np.zeros((self.n_dim, len(measurement)))
        for i in range(2 * self.n_dim + 1):
            diff_x = sigma_points_pred[i] - self.x
            diff_z = sigma_points_meas[i] - z_pred
            P_xz += self.W_c[i] * np.outer(diff_x, diff_z)

        # Compute Kalman gain
        K = np.dot(P_xz, np.linalg.inv(P_zz))

        # Update state estimate
        self.x += np.dot(K, (measurement - z_pred))

        # Update covariance
        self.P -= np.dot(K, P_zz).dot(K.T)

    def get_state(self):
        """Returns the current state estimate."""
        return self.x
