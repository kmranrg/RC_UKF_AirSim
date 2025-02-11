import numpy as np
import airsim
import time
import pprint
import matplotlib.pyplot as plt

class DroneFigure8Controller:
    def __init__(self, A=10, B=10, omega=0.05, altitude=-5, n_steps=500, speed=3):
        self.A = A
        self.B = B
        self.omega = omega
        self.altitude = altitude
        self.n_steps = n_steps
        self.speed = speed
        self.client = airsim.MultirotorClient()
        self.planned_trajectory = self.generate_figure8_trajectory()
        self.actual_positions = []

    def generate_figure8_trajectory(self):
        trajectory = []
        for t in range(self.n_steps):
            x = self.A * np.sin(self.omega * t)
            y = self.B * np.sin(2 * self.omega * t)
            z = self.altitude
            trajectory.append((x, y, z))
        return trajectory

    def connect_to_airsim(self):
        print("Connecting to AirSim...")
        self.client.confirmConnection()
        print("Connected to AirSim!")

    def display_initial_state(self):
        state = self.client.getMultirotorState()
        print("Initial drone state: %s" % pprint.pformat(state))

    def move_drone_along_trajectory(self):
        print("Enabling API control...")
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        print("Taking off...")
        self.client.takeoffAsync().join()
        state = self.client.getMultirotorState()
        print("Drone state after takeoff: %s" % pprint.pformat(state))

        print(f"Moving to start altitude: {self.planned_trajectory[0][2]} meters")
        self.client.moveToZAsync(self.planned_trajectory[0][2], self.speed).join()

        print("Starting Figure-8 Trajectory...")

        for idx, point in enumerate(self.planned_trajectory):
            x, y, z = point
            print(f"Moving to point {idx+1}/{len(self.planned_trajectory)}: x={x:.2f}, y={y:.2f}, z={z:.2f}")
            self.client.moveToPositionAsync(x, y, z, self.speed, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, 
                                            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)).join()
            time.sleep(0.05)
            
            current_position = self.client.getMultirotorState().kinematics_estimated.position
            self.actual_positions.append((current_position.x_val, current_position.y_val, current_position.z_val))

        print("Completed Figure-8 Trajectory")

        print("Hovering at the final position...")
        self.client.hoverAsync().join()
        time.sleep(2)

        print("Landing...")
        self.client.landAsync().join()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

    def plot_trajectory(self):
        planned_x = [pos[0] for pos in self.planned_trajectory]
        planned_y = [pos[1] for pos in self.planned_trajectory]
        actual_x = [pos[0] for pos in self.actual_positions]
        actual_y = [pos[1] for pos in self.actual_positions]

        plt.figure(figsize=(8, 6))
        plt.plot(planned_x, planned_y, 'b--', label='Planned Trajectory')
        plt.plot(actual_x, actual_y, 'orange', label='Actual Trajectory')
        plt.title('Drone Figure-8 Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.show()

    def reset_drone(self):
        print("Resetting drone to original state...")
        self.client.reset()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        print("Drone reset to original state.")

    def execute(self):
        try:
            self.connect_to_airsim()
            self.display_initial_state()
            self.move_drone_along_trajectory()
            self.plot_trajectory()
            print("Drone movement along Figure-8 completed!")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            self.reset_drone()

if __name__ == "__main__":
    controller = DroneFigure8Controller()
    controller.execute()
