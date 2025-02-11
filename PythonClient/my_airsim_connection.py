import airsim

# Specify IP and port explicitly
client = airsim.CarClient(ip="127.0.0.1", port=41451)
client.confirmConnection()
print("Connected to AirSim successfully!")
