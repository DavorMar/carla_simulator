import glob
import os
import sys
import time
import random
import math
import numpy as np
import cv2
from queue import Queue
from queue import Empty
from gym import spaces
import gym
import math


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

IM_WIDTH = 800
IM_HEIGHT = 800
IM_FOV = 90
MAX_SENSOR_DISTANCE = 10
ANGLE_POINT_MULTIPLICATOR = 180/MAX_SENSOR_DISTANCE
MINIMAL_DISTANCE = 40
SHOW_PREVIEW = True

class ENV(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self,actions_type = "D"):
        super(ENV, self).__init__()
        self.client = carla.Client("localhost", 2000)
        self.actions_type = actions_type
        self.client.set_timeout(8.0)
        self.world = self.client.load_world('Town04_Opt')
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()


        self.blueprint_library = self.world.get_blueprint_library()
        self.autopilot_bp = self.blueprint_library.filter("model3")[0]
        self.map = self.world.get_map()
        self.actor_list = []
        self.sensor_list = {}
        if actions_type == "D":
            self.action_space = spaces.MultiDiscrete([3, 3])
        elif actions_type == "C":
            self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(0,1,shape=(11,),dtype =np.float32)
        # self.observation_space_shape = (10,)
        self.custom_spawn_points = [[288.274,-238.48,180]]#,[285.90,-180.860,-90],[288.274,-238.48,0], [288.274,-238.48, 120],[288.274,-238.48,180],
                                    #[270.876,-237.671,0],[301.510,-238.045,180],
                                    #[269.876,-181.71,0],[299, -180.82, 180]]
            # [[-40, 165, 90], [-40,170,0], [-40,165,0], [-40,170,90], [-40,175,0], [-40,175,90],
            #                         [-40,180,0], [-40,180,90], [-40,185,0], [-40,185,90], [-40,190,0], [-40,190,90],
            #                         [-40,180,180],[-40,192,0],[-40,192,90],[-40,192,180],[-45,192,0],[-45,192,90],
            #                         [-45,192,180], [-50,192,0],[-50,192,90],[-50,192,180],[-55,192,0],[-55,192,90],
            #                         [-55,192,180],[-60,192,0],[-60,192,90],[-60,192,180],[-65,192,180],[-65,192,0]]
        self.minimal_distance = MINIMAL_DISTANCE
        self.collision_data = []

    def set_sensor_obstacle(self, x_position,y_position, orientation):
        sensor_blueprint = self.blueprint_library.find("sensor.other.obstacle")
        sensor_options = {"distance": f"{MAX_SENSOR_DISTANCE}", "debug_linetrace": "True", "hit_radius": "0.1"}
        for key in sensor_options:
            sensor_blueprint.set_attribute(key, sensor_options[key])
        sensor_spawn_point = carla.Transform(carla.Location(x = x_position, y = y_position, z = 0.3),
                                             carla.Rotation(yaw = orientation))
        sensor = self.world.spawn_actor(sensor_blueprint, sensor_spawn_point,
                                        attach_to = self.autopilot_vehicle,
                                        attachment_type = carla.AttachmentType.Rigid)
        # self.sensor_list[f"obstacle_{number}"] = self.world.spawn_actor(sensor_blueprint, sensor_spawn_point,
        #                                                                 attach_to = self.autopilot_vehicle,
        #                                                                 attachment_type = carla.AttachmentType.Rigid)
        # self.actor_list.append(self.sensor_list[f"obstacle_{number}"])
        self.actor_list.append(sensor)
        return sensor

    def sensor_callback(self,data,queue):
        queue.put(data)

    def set_sensor_camera(self):
        camera_blueprint = self.blueprint_library.find("sensor.camera.rgb")
        self.camera_blueprint = camera_blueprint
        camera_blueprint.set_attribute("image_size_x", f"{IM_WIDTH}")
        camera_blueprint.set_attribute("image_size_y", f"{IM_HEIGHT}")
        camera_blueprint.set_attribute("fov", f"{IM_FOV}")
        cam_spawn_point = carla.Transform(carla.Location(x=0,y = 0, z= 20), carla.Rotation(yaw=0.0, pitch = -90.0))
        self.camera_sensor = self.world.spawn_actor(camera_blueprint, cam_spawn_point, attach_to=self.autopilot_vehicle,
                                                    attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.camera_sensor)

    def set_sensor_colision(self):
        colsensor_blueprint = self.blueprint_library.find("sensor.other.collision")
        colsensor_spawn_point = carla.Transform(carla.Location(x=0.0, z=0.5), carla.Rotation(yaw=0.0))
        self.colsensor = self.world.spawn_actor(colsensor_blueprint, colsensor_spawn_point,
                                                attach_to=self.autopilot_vehicle)
        self.actor_list.append(self.colsensor)

    def process_camera_image(self,image_data):

        im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("int8")))
        im_array = np.reshape(im_array, (IM_HEIGHT, IM_WIDTH, 4))
        im_array2 = im_array[:, :, :3]
        return im_array2

    def process_obstacle_data(self, obstacle_data_list):
        distances_data = []
        for obstacle in obstacle_data_list:
            distances_data.append(obstacle.distance)
        return distances_data

    def calculate_distance(self):
        goal_point = [289.50, -210.89]
        vehicle_location = [self.autopilot_vehicle.get_location().x, self.autopilot_vehicle.get_location().y]
        distance = math.sqrt((goal_point[0] - vehicle_location[0])**2 + (goal_point[1] - vehicle_location[1])**2)/2
        if distance > MAX_SENSOR_DISTANCE * 3:
            distance = MAX_SENSOR_DISTANCE * 3

        return distance , distance / 3

    def calculate_angle(self):
        goal_point = [289.5, -210.89]
        current_location = self.autopilot_vehicle.get_location()
        relative_coord =[goal_point[0] - current_location.x, -(goal_point[1] - current_location.y)]
        car_orientation = self.autopilot_vehicle.get_transform().rotation.yaw
        if car_orientation < 0:
            car_orientation *= -1
        else:
            car_orientation = 360 - car_orientation
        if relative_coord[0] == 0.0:
            relative_coord[0] += 0.0001
        derivative = relative_coord[1] / relative_coord[0]
        relative_coord_orientation = math.atan(derivative) * (180/math.pi)
        if relative_coord[0] < 0:
            relative_coord_orientation += 180
        elif relative_coord[0] > 0 and relative_coord_orientation < 0:
            relative_coord_orientation = 360 + relative_coord_orientation
        else:
            pass
        angle1 = relative_coord_orientation - car_orientation
        if 0<=angle1<=180:
            angle2 = angle1
        elif angle1 > 180:
            angle2 = angle1 - 360
        elif -180<angle1<=0:
            angle2= angle1
        elif angle1 <= -180:
            angle2 = 360 + angle1
        # print(angle2)
        return angle2
        # angle1 = 360- car_orientation + relative_coord_orientation
        # angle2 = abs(relative_coord_orientation - car_orientation)
        # angles = [angle1, angle2]
        # print(angles)
        # correct_angle = np.argmin(angles)
        # return angles[correct_angle]


    def spawn_parked_cars(self):
        locations = [[292.113, -213.891,180], [280.968, -214.11,180], [281.137, -210.93,180],
                     [281.026, -207.446,180], [292.207, -204.286,180],[280.835, -204.484,180],
                     [292.028, -200.786,180],[297.761, -214.206,180],[297.961, -217.261,180],
                     [294.45,-184.99,180],[275.76,-184.51,180],[297.36,-235.87,180],[292.75,-232,180],
                     [277.299,-235.525,180],[280.39, -232.6,1801], [303.0, -177.81,90],[308.2,-177.815,90],
                     [308.1, -240.7,90], [303.0, -240.7,90],[262.13,-230.83,90],[261.95,-207,90],
                     [269.292,-175.96,0]]
        for location in locations:
            try:
                npc_bp = random.choice(self.blueprint_library.filter("model3"))
                spawnpoint = carla.Transform(carla.Location(x=location[0], y=location[1], z=0.400000),
                                               carla.Rotation(pitch=0.000000, yaw=location[2], roll=0.000000))
                npc = self.world.spawn_actor(npc_bp,spawnpoint)
                npc.apply_control(carla.VehicleControl(hand_brake=True))
                self.actor_list.append(npc)
            except RuntimeError:
                print(location, "#######################")

    def sample_random_action(self):
        if self.actions_type == "C":
            action_0 = random.uniform(-1, 1)
            action_1 = random.uniform(-1, 1)
            # action_2 = random.uniform(0, 1)
            return np.array([action_0, action_1], dtype=np.float32)
        elif self.actions_type == "D":
            return random.randint(0, 8)

    def process_collision(self,data):
        self.collision_data.append(data)
        return self.collision_data

    def execute_action(self, action):
        #2 actions, first is Gas, nothing, reverse. Second is left, right or nothing
        if self.actions_type == "D":
            if action[0] == 0:
                throttle = 0
                reverse = False
            elif action[0] == 1:
                throttle = 1
                reverse = False
            elif action[0] == 2:
                throttle = 1
                reverse = True
            if action[1] == 0:
                steer = 0
            elif action[1] == 1:
                stdeer = 1
            elif action[1] == 2:
                steer = -1

            self.autopilot_vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=0,
                                                                      steer=steer, reverse = reverse))
        elif self.actions_type == "C":
            # if action[2] < 0.5:
            #     action[2] = 0
            # if 0 < action[1] < 0.3:
            #     action[1] = 0.3
            # elif -0.3 < action[1] < 0:
            #     action[1] = -0.3
            steer = float(action[0])
            # brake = float(action[2])
            # if brake < 0.5:
            #     brake = 0
            if action[1] >= 0:
                throttle = float(action[1])
                if throttle < 0.3:
                    throttle = 0.3
                reverse = False
            else:
                reverse = True
                throttle = float(-action[1])
                if throttle < 0.3:
                    throttle = 0.3
            self.autopilot_vehicle.apply_control(carla.VehicleControl(throttle=throttle,
                                                                          steer=steer, reverse=reverse))


    def reset(self):
        self.queue_dict = {}
        self.minimal_distance = MINIMAL_DISTANCE
        spawn_point_coords = random.choice(self.custom_spawn_points)
        # self.spawn_point = random.choice(self.map.get_spawn_points())
        self.spawn_point = carla.Transform(carla.Location(x=spawn_point_coords[0], y=spawn_point_coords[1], z=0.600000),
                                           carla.Rotation(pitch=0.000000, yaw=spawn_point_coords[2], roll=0.000000))
        # DEFINIRATI x = -40, y = 165, yaw = 90
        #GOAL POINT x = -67.8, y = 177.8
        self.autopilot_vehicle = self.world.spawn_actor(self.autopilot_bp,self.spawn_point)
        self.actor_list.append(self.autopilot_vehicle)
        for _ in range(0,10):
            self.world.tick()

        obstacle_sensor_positions = [[2.6, 0, 0],#front
                                     [2.4, -0.6, -45],#front left
                                     [2.4, 0.6, 45], #front right
                                     [-2.2, 0, 180], #back
                                     [-2.2, -0.6, -135], #back left
                                     [-2, 0.6, 135], #back right
                                     [0, -0.5, -90], #left
                                     [0, 0.5, 90]] #right

        self.front_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[0][0],
                                                     obstacle_sensor_positions[0][1],
                                                     obstacle_sensor_positions[0][2])
        self.front_left_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[1][0],
                                                     obstacle_sensor_positions[1][1],
                                                     obstacle_sensor_positions[1][2])
        self.front_right_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[2][0],
                                                     obstacle_sensor_positions[2][1],
                                                     obstacle_sensor_positions[2][2])
        self.back_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[3][0],
                                                     obstacle_sensor_positions[3][1],
                                                     obstacle_sensor_positions[3][2])
        self.back_left_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[4][0],
                                                     obstacle_sensor_positions[4][1],
                                                     obstacle_sensor_positions[4][2])
        self.back_right_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[5][0],
                                                     obstacle_sensor_positions[5][1],
                                                     obstacle_sensor_positions[5][2])
        self.left_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[6][0],
                                                     obstacle_sensor_positions[6][1],
                                                     obstacle_sensor_positions[6][2])
        self.right_sensor = self.set_sensor_obstacle(obstacle_sensor_positions[7][0],
                                                    obstacle_sensor_positions[7][1],
                                                    obstacle_sensor_positions[7][2])
        self.set_sensor_camera()
        time.sleep(0.5)
        self.front_sensor_queue = Queue(maxsize=1)
        self.front_right_sensor_queue = Queue(maxsize=1)
        self.front_left_sensor_queue = Queue(maxsize=1)
        self.back_sensor_queue = Queue(maxsize=1)
        self.back_left_sensor_queue = Queue(maxsize=1)
        self.back_right_sensor_queue = Queue(maxsize=1)
        self.left_sensor_queue = Queue(maxsize=1)
        self.right_sensor_queue = Queue(maxsize=1)

        self.front_sensor.listen((lambda data: self.sensor_callback(data, self.front_sensor_queue)))
        self.front_left_sensor.listen((lambda data: self.sensor_callback(data, self.front_left_sensor_queue)))
        self.front_right_sensor.listen((lambda data: self.sensor_callback(data, self.front_right_sensor_queue)))
        self.back_sensor.listen((lambda data: self.sensor_callback(data, self.back_sensor_queue)))
        self.back_left_sensor.listen((lambda data: self.sensor_callback(data, self.back_left_sensor_queue)))
        self.back_right_sensor.listen((lambda data: self.sensor_callback(data, self.back_right_sensor_queue)))
        self.left_sensor.listen((lambda data: self.sensor_callback(data, self.left_sensor_queue)))
        self.right_sensor.listen((lambda data: self.sensor_callback(data, self.right_sensor_queue)))

        # for number, points in enumerate(obstacle_sensor_positions):
        #     self.set_sensor_obstacle(number, points[0], points[1], points[2])
        #     self.queue_dict[f"obstacle_queue_{number}"] = Queue(maxsize=1)
        # for sensor, queue in zip(self.sensor_list.keys(), self.queue_dict.keys()):
        #     self.sensor_list[sensor].listen(lambda data: self.sensor_callback(data, self.queue_dict[queue]))


        self.camera_queue = Queue(maxsize=1)
        self.camera_sensor.listen(lambda data: self.sensor_callback(data, self.camera_queue))
        self.set_sensor_colision()
        self.collision_queue = Queue(maxsize=20)
        self.colsensor.listen(lambda data: self.process_collision(data))
        self.spawn_parked_cars()
        time.sleep(2)
        self.world.tick()

        # self.autopilot_vehicle.apply_control(carla.VehicleControl(throttle = 0.3, steer = 0.1))


    def step(self, action):
        obstacle_data = []
        done = False
        reward = 0

        if not self.front_sensor_queue.empty():
            front_sensor = self.front_sensor_queue.get(True,1.0)
            front_sensor = front_sensor.distance
            obstacle_data.append(front_sensor)
        else:
            front_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(front_sensor)
        if not self.front_left_sensor_queue.empty():
            front_left_sensor = self.front_left_sensor_queue.get(True,1.0)
            front_left_sensor = front_left_sensor.distance
            obstacle_data.append(front_left_sensor)
        else:
            front_left_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(front_left_sensor)
        if not self.front_right_sensor_queue.empty():
            front_right_sensor = self.front_right_sensor_queue.get(True,1.0)
            front_right_sensor = front_right_sensor.distance
            obstacle_data.append(front_right_sensor)
        else:
            front_right_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(front_right_sensor)
        if not self.back_sensor_queue.empty():
            back_sensor = self.back_sensor_queue.get(True,1.0)
            back_sensor = back_sensor.distance
            obstacle_data.append(back_sensor)
        else:
            back_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(back_sensor)
        if not self.back_left_sensor_queue.empty():
            back_left_sensor = self.back_left_sensor_queue.get(True,1.0)
            back_left_sensor = back_left_sensor.distance
            obstacle_data.append(back_left_sensor)
        else:
            back_left_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(back_left_sensor)
        if not self.back_right_sensor_queue.empty():
            back_right_sensor = self.back_right_sensor_queue.get(True,1.0)
            back_right_sensor = back_right_sensor.distance
            obstacle_data.append(back_right_sensor)
        else:
            back_right_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(back_right_sensor)
        if not self.left_sensor_queue.empty():
            left_sensor = self.left_sensor_queue.get(True,1.0)
            left_sensor = left_sensor.distance
            obstacle_data.append(left_sensor)
        else:
            left_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(left_sensor)
        if not self.right_sensor_queue.empty():
            right_sensor = self.right_sensor_queue.get(True,1.0)
            right_sensor = right_sensor.distance
            obstacle_data.append(right_sensor)
        else:
            right_sensor = MAX_SENSOR_DISTANCE
            obstacle_data.append(right_sensor)

        try:
            camera_data = self.camera_queue.get(True,1.0)
        except Empty:
            with self.camera_queue.mutex:
                self.camera_queue.queue.clear()
                print("CAMERA SENSOR MISSING")
        if len(self.collision_data) == 0:
            pass
        else:
            reward -= 5
            self.collision_data = []
            # self.collision_queue.empty()
            # done = True
            print("COLLISION")
        distance, normal_distance = self.calculate_distance()

        obstacle_data.append(normal_distance)
        if distance < self.minimal_distance:
            self.minimal_distance = distance
            reward += (MINIMAL_DISTANCE - distance)/(MINIMAL_DISTANCE/5)

        if distance < 0.5:
            done = True
            reward += 50
        camera_image = self.process_camera_image(camera_data)
        if SHOW_PREVIEW:
            img_rgb = cv2.cvtColor(camera_image.astype("uint8"), cv2.COLOR_BGR2RGB)

            cv2.imshow("1", img_rgb)
            cv2.waitKey(1)
        self.execute_action(action)
        angle_difference = self.calculate_angle()
        obstacle_data.append(angle_difference/ANGLE_POINT_MULTIPLICATOR)
        speed = self.autopilot_vehicle.get_velocity()
        speed = (int(3.6 * math.sqrt(speed.x ** 2 + speed.y ** 2))) / 10
        obstacle_data.append(speed)
        # for obstacle in obstacle_data:
        #     if obstacle < 0.5:
        #         done = True
        # if done:
        #     reward -= 100

        obstacle_data = np.array(obstacle_data,dtype="float32") / MAX_SENSOR_DISTANCE
        return obstacle_data, reward, done

    def destroy(self):
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print("ACTORS DESTROYED")
# if __name__ == "__main__":
#     try:
#         env = ENV()
#         settings = env.world.get_settings()
#         original_settings = env.world.get_settings()
#         settings.synchronous_mode = True  # Enables synchronous mode
#         settings.fixed_delta_seconds = 0.1  # 1 frame = 0.1 second
#         env.world.apply_settings(settings)
#         env.reset()
#         # env.world.unload_map_layer(carla.MapLayer.Props)
#         # env.world.unload_map_layer(carla.MapLayer.Walls)
#         # env.world.unload_map_layer(carla.MapLayer.StreetLights)
#         for _ in range(1000):
#             start_time = time.time()
#             data, reward, done = env.step(action=[random.choice([0,1,2]), random.choice([0,1,2])])
#             env.world.tick()
#             time.sleep(0.07)
#             print(data.shape)
#             # print("FPS: ", 1.0 / (time.time() - start_time))
#
#     except KeyboardInterrupt:
#         env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])
#         # print("Actors destroyed")
#         time.sleep(1)
#     finally:
#         # try:
#         #     self.world.apply_settings(original_settings)
#         # except NameError:
#         #     pass
#
#         env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])
#
#         time.sleep(3)
