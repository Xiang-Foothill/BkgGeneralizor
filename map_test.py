import carla
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)

print(client.get_available_maps())
world = client.load_world('/Game/L_track_barc1/Maps/L_track_barc1/L_track_barc1')
print("the world is found")