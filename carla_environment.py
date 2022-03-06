import glob
import os
import sys
import time
import random
import numpy as np
import cv2
import math
from queue import Queue
from queue import Empty

from gym import spaces

from matplotlib import cm
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# constants for sensors
SHOW_PREVIEW = True#Need to incorporate this
# CAMERA CONSTANTS
IM_WIDTH = 80#120#240#480#640
IM_HEIGHT = 60#90#180#360#480
IM_FOV = 110
# LIDAR CONSTANTS

DOT_EXTENT = 1
NO_NOISE = True
UPPER_FOV = 30
LOWER_FOV = -25
CHANNELS = 7
RANGE = 50
POINTS_PER_SECOND = 8_000
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
#WORLD CONTANT
NPC_NUMBER = 30
JUNCTION_NUMBER = 2
FRAMES = 300
RUNS = 100
SECONDS_PER_EPISODE = 10


# creating virtual ENV
class ENV:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    im_fov = IM_FOV

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(8.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.autopilot_bp = self.blueprint_library.filter("model3")[0]
        self.map = self.world.get_map()
        self.actor_list = []
        # self.action_space_shape = 2
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)


         # steer, gas, brake
        print(self.action_space.shape[0])

        self.observation_space_shape = (4803,)


    #Get a random action from actionspace, used in first X iterations to gather random data
    def sample_action(self):
        a = self.action_space
        action_1 = random.uniform(-1, 1)
        action_2 = random.uniform(0, 1)
        action_3 = random.uniform(0,1)
        actions = np.array([action_1,action_2, action_3]).astype("float32")
        return actions

    def set_sensor_camera(self):
        self.camera_blueprint = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.camera_blueprint.set_attribute("image_size_x", f"{self.im_width}")
        self.camera_blueprint.set_attribute("image_size_y", f"{self.im_height}")
        self.camera_blueprint.set_attribute("fov", f"{self.im_fov}")
        cam_spawn_point = carla.Transform(carla.Location(x=1.6, z=1.6), carla.Rotation(yaw=0.0))
        self.camera_sensor = self.world.spawn_actor(self.camera_blueprint, cam_spawn_point, attach_to=self.autopilot_vehicle,
                                                    attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.camera_sensor)

    def set_sensor_lidar(self):
        lidar_blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        lidar_blueprint.set_attribute('dropoff_general_rate', '0.0')
        lidar_blueprint.set_attribute('dropoff_intensity_limit', '1.0')
        lidar_blueprint.set_attribute('dropoff_zero_intensity', '0.0')
        lidar_blueprint.set_attribute('upper_fov', str(UPPER_FOV))
        lidar_blueprint.set_attribute('lower_fov', str(LOWER_FOV))
        lidar_blueprint.set_attribute('channels', str(CHANNELS))
        lidar_blueprint.set_attribute('range', str(RANGE))
        lidar_blueprint.set_attribute('points_per_second', str(POINTS_PER_SECOND))
        lidar_spawn_point = carla.Transform(carla.Location(x=1.0, z=1.8), carla.Rotation(yaw=0.0))
        self.lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_spawn_point, attach_to=self.autopilot_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.lidar_sensor)

    def set_sensor_colision(self):
        colsensor_blueprint = self.blueprint_library.find("sensor.other.collision")
        colsensor_spawn_point = carla.Transform(carla.Location(x=0.0, z=0.5), carla.Rotation(yaw=0.0))
        self.colsensor = self.world.spawn_actor(colsensor_blueprint, colsensor_spawn_point,
                                                attach_to=self.autopilot_vehicle)
        self.actor_list.append(self.colsensor)

    def set_sensor_imu(self):
        imu_blueprint = self.blueprint_library.find("sensor.other.imu")
        imu_spawn_point = carla.Transform(carla.Location(x=0.0, z=0.5), carla.Rotation(yaw=0.0))
        self.imu_sensor = self.world.spawn_actor(imu_blueprint, imu_spawn_point, attach_to=self.autopilot_vehicle)
        self.actor_list.append(self.imu_sensor)

    def set_sensor_gnss(self):
        gnss_blueprint = self.blueprint_library.find("sensor.other.gnss")
        gnss_spawn_point = carla.Transform(carla.Location(x=0.0, z=0.5), carla.Rotation(yaw=0.0))
        self.gnss_sensor = self.world.spawn_actor(gnss_blueprint, gnss_spawn_point, attach_to=self.autopilot_vehicle)
        self.actor_list.append(self.gnss_sensor)

    def spawn_npcs(self):
        for _, spawnpoint in zip(range(0,NPC_NUMBER), self.world.get_map().get_spawn_points()[:NPC_NUMBER+1]):
            npc_bp = random.choice(self.blueprint_library.filter("vehicle.*"))
            try:
                npc = self.world.spawn_actor(npc_bp, spawnpoint)
                npc.set_autopilot(True)
                self.actor_list.append(npc)

            except RuntimeError:
                pass
    #Used for syncing sensor data
    def sensor_callback(self, data, queue):
        queue.put(data)
    #generating route that autopilot should follow(junctions number is how long we want our route to be, IE how many
    #junctions we want it to pass. Using anything more than 4 or 5 is not recommended, read README)
    def generate_waypoints(self, spawn_point, junctions_number=3, distance=1):

        x = 0
        junction_end_point = ""
        route_list = []
        #loop of how many junctions we want
        try:
            for _ in range(0, junctions_number):
                if x == 0:
                    closest_point_to_start = self.map.get_waypoint(spawn_point.location)
                else:
                    closest_point_to_start = junction_end_point.next(1)[-1]
                    route_list = copied_route_list
                route_list.append(closest_point_to_start)

                # give a 50% chance for gps to generate a lane change, meaning on next junction it would go left/right/straight
                lane_change = False
                if random.uniform(0, 1) >= 0.5:
                    lane_change = True
                next_points = closest_point_to_start.next_until_lane_end(distance)
                route_half = next_points[math.floor(len(next_points) / 2)]
                if lane_change is True and str(route_half.lane_change) != "NONE" and x == 0:
                    if str(route_half.lane_change) == "Right":
                        merge_lane = route_half.get_right_lane()
                    elif str(route_half.lane_change) == "Left":
                        merge_lane = route_half.get_left_lane()
                    elif str(route_half.lane_change) == "Both":
                        choice = random.choice([route_half.get_right_lane(), route_half.get_left_lane()])
                        merge_lane = choice
                    try:
                        next_points_merge = merge_lane.next_until_lane_end(distance)
                    except AttributeError:
                        # Lane is bugged
                        return None
                    route_list.extend(next_points[:math.floor(len(next_points) / 2)])
                    route_list.extend(next_points_merge[5:])
                    last_point = next_points_merge[-1]
                #if no lane change, just add all points till end of the same lane
                else:
                    route_list.extend(next_points)
                    last_point = next_points[-1]
                last_point = last_point.next(0.001)[-1]
                x += 1
                #Getting junction at the end of the lane, then finding the matching end point in the junction
                junction = last_point.get_junction()
                try:
                    junction_points = junction.get_waypoints(carla.LaneType.Driving)
                except AttributeError:
                    return None
                last_point_x = last_point.transform.location.x
                last_point_x = format(last_point_x, '.2f')
                last_point_y = last_point.transform.location.y
                last_point_y = format(last_point_y, '.2f')
                matching_junction_endpoints = []
                for point in junction_points:
                    waypoint_start = point[0].transform.location
                    waypoint_end = point[1].transform.location
                    x_start = point[0].transform.location.x
                    x_start = format(x_start, f'.{2}f')
                    y_start = point[0].transform.location.y
                    y_start = format(y_start, f'.{2}f')
                    if x_start == last_point_x and y_start == last_point_y:
                        matching_junction_endpoints.append(point)
                if len(matching_junction_endpoints) == 0:
                    return None

                else:
                    junction_start_end_point = random.choice(matching_junction_endpoints)

                ###################################
                ###################################
                ###################################
                #creating points inside the junction
                junction_end_point = junction_start_end_point[1]
                junction_start_point = junction_start_end_point[0]
                junction_start_rotation = junction_start_point.transform.rotation.yaw
                junction_end_rotation = junction_end_point.transform.rotation.yaw
                last_point_x = float(last_point_x)
                last_point_x += 150
                last_point_y = float(last_point_y)
                last_point_y = last_point_y * -1 + 150
                last_point_orientation = last_point.transform.rotation.yaw
                #UE 4 rotation is not same as rotation we use in math, thus we need to convert it
                #Not even sure how it works, since sometimes UE4 gives rotation values greater than even 450, which
                #should be impossible. Currently this conversion works, but need to investigate
                if last_point_orientation < 0:
                    last_point_orientation = last_point_orientation * -1
                else:
                    last_point_orientation = -1 * (last_point_orientation - 360)
                if junction_end_rotation < 0:
                    junction_end_rotation = junction_end_rotation * -1
                else:
                    junction_end_rotation = -1 * (junction_end_rotation - 360)
                #UE4 coordinate values go from -150 to +150 for x, and 150 to -150 for y. Converting them both to go
                #from 0 to 300. Needed to create a relative coordinate system later on
                junction_end_point_x = float(junction_end_point.transform.location.x) + 150
                junction_end_point_y = float(junction_end_point.transform.location.y) * -1 + 150
                derivation = (junction_end_point_y - last_point_y) / (junction_end_point_x - last_point_x)
                #Checking if we are turning in the junction
                if abs(last_point_orientation - junction_end_rotation) < 10:
                    turning = False
                    print("STRAIGHT")
                else:
                    turning = True

                if turning:
                    #checking if orientation is N-S or W-E.
                    if 85 < last_point_orientation < 95 or 265 < last_point_orientation < 275:
                        center_point = (junction_end_point_x, last_point_y)
                        # this works but not sure why.On lines 268-278 we are multiplying it with percentage(to get
                        #multiple points ). i should maybe try 360-(percentage*90)
                        total_rotation = math.radians(270)

                        if derivation >= 0:
                            junction_rotation = -1  # -
                        else:
                            junction_rotation = 1
                    #checking the derivation of start and end point of junction. Needed to determine rotation
                    else:
                        total_rotation = math.radians(90)
                        center_point = (last_point_x, junction_end_point_y)

                        derivation = (junction_end_point_y - last_point_y) / (
                                junction_end_point_x - last_point_x)
                        if derivation >= 0:
                            junction_rotation = 1
                        else:
                            junction_rotation = -1  # -

                    #creating 30 points inside the junction if we are turning, by 2d vector rotation.
                    for zy in range(1, 30):
                        percentage = zy / 30
                        relative_x = round((last_point_x - center_point[0]) * (1 - percentage)) + round(
                            (junction_end_point_x - center_point[0]) * percentage)
                        relative_y = round((last_point_y - center_point[1]) * (1 - percentage)) + round(
                            (junction_end_point_y - center_point[1]) * percentage)
                        junction_point_x = int(
                            round(math.cos(total_rotation * percentage * junction_rotation) * relative_x) - (
                                        math.sin(total_rotation * percentage * junction_rotation) * relative_y))
                        junction_point_y = int(
                            round(math.sin(total_rotation * percentage * junction_rotation) * relative_x) + (
                                        math.cos(total_rotation * percentage * junction_rotation) * relative_y))
                        junction_point_x += (center_point[0] - 150)
                        junction_point_y += (center_point[1] - 150) * -1
                        route_list.append(self.map.get_waypoint(carla.Location(junction_point_x, junction_point_y, 0)))
                #If we are just going straight in junction
                else:

                    junction_end_point_x = int(round(junction_end_point_x))
                    junction_end_point_y = int(round(junction_end_point_y))
                    last_point_x = int(round(last_point_x))
                    last_point_y = int(round(last_point_y))

                    for x in range(1, 30):
                        percentage = x / 30
                        junction_point_x = (last_point_x * (1 - percentage)) + (junction_end_point_x * percentage)
                        junction_point_y = (last_point_y * (1 - percentage)) + (junction_end_point_y * percentage)
                        junction_point_x = (junction_point_x - 150)
                        junction_point_y = (junction_point_y - 150) * -1
                        route_list.append(self.map.get_waypoint(carla.Location(junction_point_x, junction_point_y, 0)))

                ########################################################
                ########################################################
                ########################################################
                copied_route_list = route_list.copy()

            return route_list[10:]


        except AttributeError:
            return None

    #function that gathers waypoints and plots them on the gps. It includes putting waypoints in
    #relation to the car position, then rotating them according to initial car rotation in the world
    #then again rotation them according to difference between current car rotation and initial rotation to the world
    def plot_GPS(self,route, rotation, location, starting_point):  # ADD DEGREES OF THE CAR
        starting_rotation = starting_point.rotation.yaw
        reward = 0
        route_points_processed = []
        #This is due to Carla rotation being from 0:+180 and 0:-180, note vehicle rotation behaves different than waypoint
        #rotation
        if starting_rotation < 0:
            starting_rotation = starting_rotation * -1
        elif starting_rotation > 0:
            starting_rotation = 180 + (180 - starting_rotation)
        #same as before
        if rotation < 0:
            rotation = rotation * -1
        elif rotation > 0:
            rotation = 180 + (180 - rotation)
        rotation_diff = rotation - starting_rotation
        # extracting x,y coordinate of each point and adding to new list
        for point in route:
            x_cord = point.transform.location.x
            y_cord = point.transform.location.y
            route_points_processed.append((x_cord, y_cord))
        # Creating a layout for the map(recommended is probably around 60ish
        # layout = np.zeros((int(IM_WIDTH/4+20), int(IM_HEIGHT/4 + 50)))
        layout = np.zeros((int(IM_WIDTH + 20), int(IM_HEIGHT + 40)))
        start_position = (
        int(layout.shape[0] / 2), int(layout.shape[1] / 2))  # This is just the position of center dot on plot
        layout[start_position[0]][start_position[1]] = 2
        #without these, memory doesnt get wiped
        added_route = []
        relative_route = []
        final_route = []
        #Carla poitns go from -150 to 150, here we normalize to 0-300, all positives allow easier calculations
        #Also y axis is other way around. Top half is negative, bottom half is positive, thats why we first multiply by -1
        for point in route_points_processed:
            x = point[0] + 150
            y = (point[1] * -1) + 150
            added_route.append([x, y])
        #same here
        vehicle_x = round(location[0] + 150)
        vehicle_y = round((location[1] * -1) + 150)
        #math.sin and math.cos uses default radians. This is why we need to convert our degrees first
        car_degrees = math.radians(starting_rotation)
        car_difference = math.radians(-rotation_diff)
        for point in added_route:
            #each point relative to the point of the car
            relative_x = round(point[0] - vehicle_x)
            relative_y = round(point[1] - vehicle_y)
            #point first adjusted to the initial rotation of the car(Alpha)
            adjusted_x = int(round((math.cos(car_degrees) * relative_x) - (math.sin(car_degrees) * relative_y)))
            adjusted_y = int(round((math.sin(car_degrees) * relative_x) + (math.cos(car_degrees) * relative_y)))

            relative_route.append([adjusted_x, adjusted_y])

        for point in relative_route:
            #second adjustment to the Beta, or difference between cars new rotation and its initial rotation
            #this allows us to have GPS always facing car direction
            #in both cases we applied vector rotation in 2d space formulas
            adjusted_x = int(round((math.cos(car_difference) * point[0]) - (math.sin(car_difference) * point[1])))
            adjusted_y = int(round((math.sin(car_difference) * point[0]) + (math.cos(car_difference) * point[1])))

            final_route.append([adjusted_x, adjusted_y])

        for point in final_route[:30]:
            try:
                x_point_final = (start_position[0] - point[1])
                y_point_final =start_position[1] + point[0]
                layout[x_point_final][y_point_final] = 200
                for z in range(-1,1):
                    x_point_final +=z
                    for zy in range(-1,1):
                        y_point_final += zy
                        layout[x_point_final][y_point_final] = 200


            except:
                break
        #rotation needs to be done in different fashion
        #when car starts in N-S or W-E positions
        if 85 < starting_rotation < 95 or 265 < starting_rotation < 275:
            image = np.rot90(layout,3)
        else:
            image = np.rot90(layout)
        #Creating rewards for each point on route that is collected
        image = image[:int(round(len(image) / 2 + 20))]
        relative_route_copy = relative_route.copy()
        for point in relative_route_copy:
            distance = math.sqrt(point[0]**2+point[1]**2)
            if distance <= 2:
                point_index = relative_route_copy.index(point)
                relative_route.pop(point_index)
                self.route_points.pop(point_index)
                reward += 0.2
        starting_cords = (starting_point.location.x, starting_point.location.y)
        starting_distance = math.sqrt((added_route[-1][0] - (starting_cords[0]+150))**2+(added_route[-1][1] - ((starting_cords[1] * -1) + 150))**2)
        distance_from_end = math.sqrt(relative_route_copy[-1][0]**2 + relative_route_copy[-1][1]**2)
        distance_percentage = (distance_from_end/starting_distance)
        distance_percentage = (1-distance_percentage)#*-1
        #Adding rewards for % of distance that a car passed in certain moment
        if distance_percentage > 1:
            distance_percentage = 1
        if distance_percentage < -1:
            distance_percentage = -1
        if -0.02 < distance_percentage < 0.02:
            distance_percentage = 0
        reward += distance_percentage
        image = image[::3, ::3]
        # print(image)
        # print(image.shape)
        return image, reward, distance_percentage

    # Function projects lidar distances onto camera image, converting image of lidar into camera image
    #Code mostly implemented from Carlas examples , for more info check it PythonApi/examples/lidar_to_camera.py
    def process_lidar_cam(self,image_data,lidar_data):
        # K = [[Fx, 0, image_w / 2],
        #       [ 0, Fy, image_h/2],
        #       [ 0,  0,         1]]
        image_w = self.camera_blueprint.get_attribute("image_size_x").as_int()
        image_h = self.camera_blueprint.get_attribute("image_size_y").as_int()
        fov = self.camera_blueprint.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        # image_data.convert(carla.ColorConverter.CityScapesPalette)
        im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))

        im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))

        im_array = im_array[:, :, 2]#[:, :, ::-1]
        im_array = np.where(im_array == (1 or 2 or 3 or 9 or 11 or 12), 0 , im_array)
        # im_array = np.delete(im_array,[9,11,12,1,3])
        im_array = 20 * im_array

        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
        intensity = np.array(p_cloud[:, 3])
        local_lidar_points = np.array(p_cloud[:, :3]).T
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
        lidar_2_world = self.lidar_sensor.get_transform().get_matrix()
        world_points = np.dot(lidar_2_world, local_lidar_points)
        world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
        sensor_points = np.dot(world_2_camera, world_points)

        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])
        points_2d = np.dot(K, point_in_camera_coords)
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)
        intensity = 4 * intensity - 3
        color_map = np.array([
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T
        rgb_weights = [0.2989, 0.5870, 0.1140]
        color_map = np.dot(color_map[..., :3], rgb_weights).astype(np.int)
        if DOT_EXTENT <= 0:
            # Draw the 2d points on the image as a single pixel using numpy.
            im_array[v_coord, u_coord] = color_map
        else:
            # Draw the 2d points on the image as squares of extent args.dot_extent.
            for i in range(len(points_2d)):

                im_array[
                v_coord[i] - DOT_EXTENT: v_coord[i] + DOT_EXTENT,
                u_coord[i] - DOT_EXTENT: u_coord[i] + DOT_EXTENT] = color_map[i]
        # print(im_array)
        # print(im_array.shape)
        im_array = np.reshape(im_array, (IM_HEIGHT, IM_WIDTH))
        # print(im_array.shape)
        return im_array


    def process_camera_gps(self,gps_img,camera_img):
        #Function adds GPS image in top left corner
        img_fg = Image.fromarray(gps_img)
        background = Image.fromarray(camera_img)
        background.paste(img_fg, (0, 0))
        open_cv_image = np.array(background)
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        cv2.imshow("1", open_cv_image)
        cv2.waitKey(1)
        return open_cv_image
    #Reset environment function. Deleting all the actors, generating them again, generating new route and initating sensors
    def reset(self):

        self.collision_history = []

        #Function doesnt work on every spawnpoint, for whatever Carla reasons, so it runs through them till it finds
        #a good one. Bigger Junction number means exponentionally longer search(recommended no more than 5,maybe even 4)
        for _ in range(0, 5000):

            try:
                self.spawn_point = random.choice(
                    self.map.get_spawn_points())
                self.route_points = self.generate_waypoints(self.spawn_point, JUNCTION_NUMBER)
                print("Route lenght is: ", len(self.route_points), "points")
                break
            except TypeError:

                pass

        self.autopilot_vehicle = self.world.spawn_actor(self.autopilot_bp,self.spawn_point)
        self.actor_list.append(self.autopilot_vehicle)
        time.sleep(0.5)
        #some ticks needed before our sensors start listening since first few frames our car is still spawning,
        #or better say falling onto the ground from small height
        for _ in range(0,10):
            self.world.tick()
        #Create all sensors
        self.set_sensor_camera()
        self.set_sensor_colision()
        self.set_sensor_imu()
        self.set_sensor_gnss()
        self.set_sensor_lidar()

        time.sleep(0.5)
        self.spawn_npcs()

        self.camera_queue = Queue(maxsize=1)
        self.imu_queue = Queue(maxsize=1)
        self.gnss_queue = Queue(maxsize=1)
        self.lidar_queue = Queue(maxsize=1)
        self.colision_queue = Queue(maxsize=1)

        #Gather data from each sensor and store it in queues in order to sync them
        self.camera_sensor.listen(lambda data: self.sensor_callback(data, self.camera_queue))
        self.imu_sensor.listen(lambda data: self.sensor_callback(data, self.imu_queue))
        self.gnss_sensor.listen(lambda data: self.sensor_callback(data, self.gnss_queue))
        self.lidar_sensor.listen(lambda data: self.sensor_callback(data, self.lidar_queue))
        self.colsensor.listen(lambda data: self.sensor_callback(data, self.colision_queue))
        time.sleep(1)
        self.world.tick()


        # image, _, _, image_normalized = self.step(self.route_points)
        # self.autopilot_vehicle.set_autopilot(True)
        # self.autopilot_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer = -0.3))
        # return image_normalized




    #Step is each frame which collects data and returns it. Our neural network recieves this data
    def step(self,route_points,action = None, spawn = False):
        done = False
        try:
            # Get the data once it's received from the queues
            camera_data = self.camera_queue.get(True, 1.0)
            lidar_data = self.lidar_queue.get(True, 1.0)
            imu_data = self.imu_queue.get(True, 1.0)
            gnss_data = self.gnss_queue.get(True, 1.0)

        except Empty:
            with self.colision_queue.mutex:
                self.colision_queue.queue.clear()
            with self.camera_queue.mutex:
                self.camera_queue.queue.clear()
            with self.gnss_queue.mutex:
                self.gnss_queue.queue.clear()
            with self.imu_queue.mutex:
                self.imu_queue.queue.clear()
            with self.lidar_queue.mutex:
                self.lidar_queue.queue.clear()
            print("[Warning] Some sensor data has been missed")
        #testing print to check if sensors are synced.
        #
        # sys.stdout.write("\r Camera: %d Lidar: %d Compass: %d Gnss: %d" %
        #                  (camera_data.frame, lidar_data.frame, imu_data.frame,
        #                   gnss_data.frame) + '\n' )
        #
        # sys.stdout.flush()
        #generate the GPS image and rewards
        gps_img, reward, distance_percentage = self.plot_GPS(route_points, imu_data.transform.rotation.yaw,
                                (gnss_data.transform.location.x, gnss_data.transform.location.y), self.spawn_point)#Creates gps image and reward
        #Generate camera+lidar image
        camera_img = self.process_lidar_cam(camera_data, lidar_data)#combines lidar and camera into single image
        #add GPS image onto camera and lidar image
        complete_image = self.process_camera_gps(gps_img, camera_img)#combines final image and GPS iamge
        #Each step gives small negative rewards, but if we have a collision, then penalty is drastic
        if self.colision_queue.empty():
            reward = -0.1
        else:
            reward -=5
            done = True
        reward = float(reward)

        #Action range is -1,1. Positive is for throttle, negative is for brake
        if action is not None:
            action1 = action[0].copy()
            throttle_brake = float(action1[0])

            steer = float(action1[1])
            if throttle_brake >= -0.5:
                brake = 0
                throttle = (throttle_brake + 0.5)/1.5
            else:
                brake = abs(throttle_brake)
                brake = (brake - 0.5) * 2
                throttle = 0


            # print(float(action[0]))
            # gas_brake = float(action[0])
            # ########This chunk is for no brake
            # # gas_brake = (gas_brake +1) / 2
            # # self.autopilot_vehicle.apply_control(
            # #     carla.VehicleControl(throttle=gas_brake, steer=float(action[1])))
            # ##################
            # #THIS CHUNK OF CODE IS USED IF WE WANT BRAKE - THROTTLE COMBO
            # if 0 < gas_brake < 0.4:
            #     gas_brake = 0.4
            # if gas_brake > 0:
            #     throttle = gas_brake
            #     brake = 0
            # elif gas_brake < 0:
            #     throttle = 0
            #     brake = -gas_brake
            # else:
            #     throttle = 0
            #     brake = 0

            self.autopilot_vehicle.apply_control(carla.VehicleControl(throttle=throttle,brake=brake, steer = steer))
        else:
            pass
        #normalize image to also have -1,1 range
        normalized_image = (complete_image.astype("int32"))/255
        normalized_image = normalized_image.flatten()

        speed = self.autopilot_vehicle.get_velocity()
        speed = (int(3.6 * math.sqrt(speed.x**2+speed.y**2)))/100
        acceleration = self.autopilot_vehicle.get_acceleration()
        acceleration = (int(3.6 * math.sqrt(acceleration.x ** 2 + acceleration.y ** 2)))/100
        if acceleration > 1:
            acceleration = 1
        if speed > 0.5:
            reward -= 1
        # elif speed > -1:
        #     reward += 1


        normalized_image = np.concatenate((normalized_image,(speed, acceleration,distance_percentage)))

        if spawn:
            reward = 0
        else:
            pass

        # reward_normalized = reward/3200
        # if reward_normalized > 1 or reward_normalized < -1:
        #     print("Rewards: ", reward, reward_normalized)




        return complete_image, reward, done,normalized_image


#
# if __name__ == "__main__":
#     env = ENV()
#     for _ in range(0,RUNS):
#         try:
#             observation_normalized = env.reset()
#             starting_reward = -50
#             # print(observation.flatten().shape)
#             for frame in range(0,FRAMES):
#                 observation, reward, done,observation_normalized = env.step(frame, env.route_points)
#                 # print(observation_normalized.flatten().shape)
#                 # print(observation.flatten().shape)
#                 if done:
#                     break
#                 env.world.tick()
#         finally:
#                 # try:
#                 #     self.world.apply_settings(original_settings)
#                 # except NameError:
#                 #     pass
#
#             env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])
#
#     time.sleep(3)
