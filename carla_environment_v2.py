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


import matplotlib.pyplot as plt


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# constants for sensors
SHOW_PREVIEW = True#boolean if we want camera image to show or not
# CAMERA CONSTANTS
IM_WIDTH = 480#120#240#480#640
IM_HEIGHT = 480#90#180#360#480
IM_FOV = 110

LIDAR_RANGE = 70

#WORLD AND LEARN CONSTANTS
NPC_NUMBER = 50
JUNCTION_NUMBER = 2
FRAMES = 300
RUNS = 100
SECONDS_PER_EPISODE = 10
ROAD_DOT_EXTENT = 2

#Connecting to carla server
try:
    client = carla.Client("localhost", 2000)
    print(client.get_available_maps())
    # world = client.load_world('Town0')
except IndexError:
    pass

class ENV:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    im_fov = IM_FOV
    def __init__(self, actions=1, action_type="C"):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(8.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.autopilot_bp = self.blueprint_library.filter("model3")[0]
        self.map = self.world.get_map()
        self.actor_list = []
        #self.observation_space_shape = (4803,)
        self.action_type = action_type
        self.action_space_size = actions
        if action_type == "C":
            if actions == 1:
                self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
            elif actions == 2:
                self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
            elif actions == 3:
                self.action_space = spaces.Box(low=np.array([-1.0, 0.0, 0, 0]), high=np.array([1.0, 1.0, 1.0]),
                                               dtype=np.float32)
        elif action_type == "D":
            self.action_space = spaces.MultiDiscrete([3, 2, 2])
        #self.observation_space_shape = ###

    def set_sensor_lidar(self):
        lidar_blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        sensor_options = {'channels': '16', 'points_per_second': '10000', 'rotation_frequency': '10', 'upper_fov': '-10',
                          'horizontal_fov': '110', }
        lidar_blueprint.set_attribute('range', f"{50}")
        lidar_blueprint.set_attribute('dropoff_general_rate', f"{0.1}")
        lidar_blueprint.set_attribute('dropoff_intensity_limit',
                               lidar_blueprint.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_zero_intensity',f"{0.1}")
        for key in sensor_options:
            lidar_blueprint.set_attribute(key, sensor_options[key])
        lidar_spawn_point = carla.Transform(carla.Location(x=1.0, z=1.8), carla.Rotation(yaw=0.0))
        self.lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_spawn_point, attach_to=self.autopilot_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.lidar_sensor)

    def set_sensor_lidar_midrange(self):
        lidar_blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        sensor_options = {'channels': '32', 'points_per_second': '32000', 'rotation_frequency': '10', 'upper_fov': '-4',
                          'horizontal_fov': '110', 'lower_fov': '-10' }
        lidar_blueprint.set_attribute('range', f"{70}")
        lidar_blueprint.set_attribute('dropoff_general_rate', f"{0.1}")
        lidar_blueprint.set_attribute('dropoff_intensity_limit',
                                      lidar_blueprint.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_zero_intensity', f"{0.1}")
        for key in sensor_options:
            lidar_blueprint.set_attribute(key, sensor_options[key])
        lidar_spawn_point = carla.Transform(carla.Location(x=1.0, z=1.8), carla.Rotation(yaw=0.0))
        self.lidar_sensor_midrange = self.world.spawn_actor(lidar_blueprint, lidar_spawn_point, attach_to=self.autopilot_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.lidar_sensor_midrange)

    def set_sensor_lidar_longrange(self):
        lidar_blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        sensor_options = {'channels': '64', 'points_per_second': '150000', 'rotation_frequency': '10',
                          'upper_fov': '0', 'lower_fov':'-5',
                          'horizontal_fov': '80' }
        lidar_blueprint.set_attribute('range', f"{LIDAR_RANGE}")
        lidar_blueprint.set_attribute('dropoff_general_rate', f"{0.1}")
        lidar_blueprint.set_attribute('dropoff_intensity_limit',
                                      lidar_blueprint.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_zero_intensity', f"{0.1}")
        for key in sensor_options:
            lidar_blueprint.set_attribute(key, sensor_options[key])
        lidar_spawn_point = carla.Transform(carla.Location(x=1.0,y = 0.1, z=1.8), carla.Rotation(yaw=0.0))
        self.lidar_sensor_longrange = self.world.spawn_actor(lidar_blueprint, lidar_spawn_point, attach_to=self.autopilot_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.lidar_sensor_longrange)

    def set_sensor_back_lidar(self):
        lidar_blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        sensor_options = {'channels': '64', 'points_per_second': '64000', 'rotation_frequency': '10',
                          'horizontal_fov': '110','upper_fov': '0'}
        lidar_blueprint.set_attribute('range', f"{LIDAR_RANGE}")
        lidar_blueprint.set_attribute('dropoff_general_rate', f"{0.1}")
        lidar_blueprint.set_attribute('dropoff_intensity_limit',
                               lidar_blueprint.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_zero_intensity', f"{0.1}")
        for key in sensor_options:
            lidar_blueprint.set_attribute(key, sensor_options[key])
        lidar_spawn_point = carla.Transform(carla.Location(x=-1.6, z=1.8), carla.Rotation(yaw=180.0))
        self.back_lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_spawn_point, attach_to=self.autopilot_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.back_lidar_sensor)

    def set_sensor_left_lidar(self):
        lidar_blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        sensor_options = {'channels': '16', 'points_per_second': '15000', 'rotation_frequency': '10',
                          'horizontal_fov': '110','upper_fov': '0', 'lower_fov': '-40'}
        lidar_blueprint.set_attribute('range', f"{LIDAR_RANGE}")
        lidar_blueprint.set_attribute('dropoff_general_rate',
                               lidar_blueprint.get_attribute('dropoff_general_rate').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_intensity_limit',
                               lidar_blueprint.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_zero_intensity',
                               lidar_blueprint.get_attribute('dropoff_zero_intensity').recommended_values[0])
        for key in sensor_options:
            lidar_blueprint.set_attribute(key, sensor_options[key])
        lidar_spawn_point = carla.Transform(carla.Location(x=1.6, y= -0.5,  z=1.8), carla.Rotation(yaw=-100.0))
        self.left_lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_spawn_point, attach_to=self.autopilot_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.left_lidar_sensor)

    def set_sensor_right_lidar(self):
        lidar_blueprint = self.blueprint_library.find("sensor.lidar.ray_cast")
        sensor_options = {'channels': '16', 'points_per_second': '15000', 'rotation_frequency': '10',
                          'horizontal_fov': '110','upper_fov': '0', 'lower_fov': '-40'}
        lidar_blueprint.set_attribute('range', f"{LIDAR_RANGE}")
        lidar_blueprint.set_attribute('dropoff_general_rate',
                               lidar_blueprint.get_attribute('dropoff_general_rate').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_intensity_limit',
                               lidar_blueprint.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_blueprint.set_attribute('dropoff_zero_intensity',
                               lidar_blueprint.get_attribute('dropoff_zero_intensity').recommended_values[0])
        for key in sensor_options:
            lidar_blueprint.set_attribute(key, sensor_options[key])
        lidar_spawn_point = carla.Transform(carla.Location(x=1.6, y= 0.5,  z=1.8), carla.Rotation(yaw=100.0))
        self.right_lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_spawn_point, attach_to=self.autopilot_vehicle,
                                                   attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.right_lidar_sensor)

    def set_sensor_camera(self):
        camera_blueprint = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        self.camera_blueprint = camera_blueprint
        camera_blueprint.set_attribute("image_size_x", f"{self.im_width}")
        camera_blueprint.set_attribute("image_size_y", f"{self.im_height}")
        camera_blueprint.set_attribute("fov", f"{self.im_fov}")
        cam_spawn_point = carla.Transform(carla.Location(x=1.6, z=1.6), carla.Rotation(yaw=0.0))
        self.camera_sensor = self.world.spawn_actor(camera_blueprint, cam_spawn_point, attach_to=self.autopilot_vehicle,
                                                    attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.camera_sensor)

    def set_sensor_back_camera(self):
        camera_blueprint = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        camera_blueprint.set_attribute("image_size_x", f"{self.im_width}")
        camera_blueprint.set_attribute("image_size_y", f"{self.im_height}")
        camera_blueprint.set_attribute("fov", f"{self.im_fov}")
        cam_spawn_point = carla.Transform(carla.Location(x=-1.6, z=1.6), carla.Rotation(yaw=180.0))
        self.back_camera_sensor = self.world.spawn_actor(camera_blueprint, cam_spawn_point, attach_to=self.autopilot_vehicle,
                                                    attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.back_camera_sensor)

    def set_sensor_left_camera(self):
        camera_blueprint = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        camera_blueprint.set_attribute("image_size_x", f"{self.im_width}")
        camera_blueprint.set_attribute("image_size_y", f"{self.im_height}")
        camera_blueprint.set_attribute("fov", f"{self.im_fov}")
        cam_spawn_point = carla.Transform(carla.Location(x=1.6, y = -0.5 , z=1.6), carla.Rotation(yaw=-100.0))
        self.left_camera_sensor = self.world.spawn_actor(camera_blueprint, cam_spawn_point, attach_to=self.autopilot_vehicle,
                                                    attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.left_camera_sensor)

    def set_sensor_right_camera(self):
        camera_blueprint = self.blueprint_library.find("sensor.camera.semantic_segmentation")
        camera_blueprint.set_attribute("image_size_x", f"{self.im_width}")
        camera_blueprint.set_attribute("image_size_y", f"{self.im_height}")
        camera_blueprint.set_attribute("fov", f"{self.im_fov}")
        cam_spawn_point = carla.Transform(carla.Location(x=1.6, y = 0.5 , z=1.6), carla.Rotation(yaw=100.0))
        self.right_camera_sensor = self.world.spawn_actor(camera_blueprint, cam_spawn_point, attach_to=self.autopilot_vehicle,
                                                    attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.right_camera_sensor)

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

    def set_sensor_radar(self):
        radar_blueprint = self.blueprint_library.find("sensor.other.radar")
        radar_spawn_point = carla.Transform(carla.Location(x=0.0, z=0.5), carla.Rotation(yaw = 0.0))
        self.radar_sensor = self.world.spawn_actor(radar_blueprint, radar_spawn_point, attach_to = self.autopilot_vehicle)


    def spawn_npcs(self):
        for _, spawnpoint in zip(range(0,NPC_NUMBER), self.world.get_map().get_spawn_points()[:NPC_NUMBER+1]):
            npc_bp = random.choice(self.blueprint_library.filter("vehicle.*"))
            try:
                npc = self.world.spawn_actor(npc_bp, spawnpoint)
                npc.set_autopilot(True)
                self.actor_list.append(npc)
            except RuntimeError:
                pass

    def sensor_callback(self, data, queue):
        queue.put(data)

    def fill_lane_markings(self, lane_markings, display_size, image):
        non_filled = lane_markings
        display_size_center = np.array([display_size[0]/2 - display_size[0]/5, display_size[0]/2 + display_size[0]/5,
                               display_size[1]/2 - display_size[1]/10, display_size[1]/2 + display_size[1]/20]).astype("int16")
        lane_markings_center = lane_markings[lane_markings[:,0] > display_size_center[0]]
        lane_markings_center = lane_markings_center[lane_markings_center[:, 0] < display_size_center[1]]
        lane_markings_center = lane_markings_center[lane_markings_center[:, 1] > display_size_center[2]]
        lane_markings_center = lane_markings_center[lane_markings_center[:, 1] < display_size_center[3]]


        for x in range(3):
            lane_markings_plus = np.array([lane_markings_center[:, 0] + x,lane_markings_center[:,1],lane_markings_center[:,2],lane_markings_center[:,3]])

            lane_markings_minus = np.array([lane_markings_center[:, 0] - x,lane_markings_center[:,1],lane_markings_center[:,2],lane_markings_center[:,3]])


            image[lane_markings_plus[0], lane_markings_plus[1]] = [157,234,50]
            image[lane_markings_minus[0], lane_markings_minus[1]] = [157, 234, 50]




    def save_lidar_image(self, lidar_data):
        disp_size = [600,400]
        lidar_range = float(LIDAR_RANGE) *2
        points = lidar_data
        points[:,:2] *= min(disp_size) / lidar_range
        points[:,:2] += (0.5 * disp_size[0], 0.5 * disp_size[1])
        points[:,:2] = np.fabs(points[:,:2])
        points = points.astype("int32")

        lidar_img_size = (disp_size[0],disp_size[1],3)
        lidar_img = np.zeros((lidar_img_size),dtype=np.int8)
        road_points = points[points[:,3] == 7]

        lane_marking_points = points[points[:,3] == 6]
        pedestrian_marking_points = points[points[:,3] == 4]
        traffic_light_points = points[points[:,3] == 18]
        vehicle_points = points[points[:,3] == 10]
        vehicle_points_copied = vehicle_points.copy()
        ########################################
        #FIRST METHOD,FASTEST ONE, EASILY MANAGING TRADE OFF BETWEEN SPEED AND ACCURACY BY INCREASING LOOPS
        extent = np.sqrt((road_points[:,0] - disp_size[0]/2)**2 +
                         (road_points[:,1] - disp_size[1]/2)**2).astype("int16")

        road_points_copied = road_points.copy()
        lidar_img[road_points.T.astype("int16")[0], road_points.T.astype("int16")[1]] = [128, 64, 128]
        for x in range(1,3):
            for y in (0,2):
                i = x + (y/2)
                extent_new = extent / i
                extent_new = extent_new[0]/10
                road_points_new_plus = road_points_copied[:,:2] + extent_new
                road_points_new_plus = road_points_new_plus.astype("int16")
                road_points_new_minus = road_points_copied[:, :2] - extent_new
                road_points_new_minus = road_points_new_minus.astype("int16")
                road_points_new_plus_minus = np.array([road_points_copied[:,0] + extent_new,
                                                       road_points_copied[:,1] - extent_new]).astype("int16")

                road_points_new_minus_plus = np.array([road_points_copied[:, 0] - extent_new,
                                                       road_points_copied[:, 1] + extent_new]).astype("int16")

                lidar_img[road_points_new_plus[0],road_points_new_plus[1]] = [128, 64, 128]
                lidar_img[road_points_new_minus[0], road_points_new_minus[1]] = [128, 64, 128]
                lidar_img[road_points_new_plus_minus[0], road_points_new_plus_minus[1]] = [128, 64, 128]
                lidar_img[road_points_new_minus_plus[0], road_points_new_minus_plus[1]] = [128, 64, 128]


        #######################################
        #SECOND METHOD, MUCH SlOWER BUT MORE ACCURATE IF CALIBRATED WELL
        # for x,y in road_points[:,:2]:
        #     extent_x = int((math.sqrt((x-disp_size[0]/2)**2 + (y-disp_size[1]/2)**2))/20)
        #     extent_y = int((math.sqrt((x - disp_size[0] / 2) ** 2 + (y - disp_size[1] / 2) ** 2)) / 20)
        #     if x < 200/2:
        #         extent_x = int(extent_x*3)
        #         extent_y = int(extent_y*1.5)
        #     if extent_x != 0:
        #         lidar_img[x - extent_x :x + extent_x,y - extent_y: y + extent_y] = [128,64,128]
        #     else:
        #         lidar_img[x, y] = [128,64,128]

        # lidar_img[road_points.T[0], road_points.T[1]] = [128,64,128]
        # for x,y in lane_marking_points[:,:2]:
        #     extent_x = 3
        #     extent_y = 0
        #     lidar_img[x - extent_x: x + extent_x, y] = [157, 234, 50]
        lidar_img[lane_marking_points.T[0],lane_marking_points.T[1]] = [157,234,50]
        self.fill_lane_markings(lane_marking_points, disp_size, lidar_img)
        # try:
        #     for x in range(1,3):
        #         vehicle_points_new_plus = vehicle_points_copied[:, :2] + x
        #         vehicle_points_new_plus = vehicle_points_new_plus.astype("int16")
        #         vehicle_points_new_minus = vehicle_points_copied[:, :2] - x
        #         vehicle_points_new_minus = vehicle_points_new_minus.astype("int16")
        #         vehicle_points_new_plus_minus = np.array([vehicle_points_copied[:, 0] + x,
        #                                                vehicle_points_copied[:, 1] - x]).astype("int16")
        #
        #         vehicle_points_new_minus_plus = np.array([vehicle_points_copied[:, 0] - x,
        #                                                vehicle_points_copied[:, 1] + x]).astype("int16")
        #         lidar_img[vehicle_points_new_plus[0], vehicle_points_new_plus[1]] = [0,0,142]
        #         lidar_img[vehicle_points_new_minus[0], vehicle_points_new_minus[1]] = [0,0,142]
        #         lidar_img[vehicle_points_new_plus_minus[0], vehicle_points_new_plus_minus[1]] = [0,0,142]
        #         lidar_img[vehicle_points_new_minus_plus[0], vehicle_points_new_minus_plus[1]] = [0,0,142]
        # except:
        #     pass
        lidar_img[vehicle_points.T[0],vehicle_points.T[1]] =  [0,0,142]
        lidar_img[traffic_light_points.T[0],traffic_light_points.T[1]] = [250, 170, 30]
        for point in traffic_light_points:
            lidar_img[point[0] - 3:point[0] + 3, point[1] - 3:point[1] + 3] = [250, 170, 30]
        for point in vehicle_points:
            lidar_img[point[0]-3:point[0]+3, point[1]-3:point[1]+3] = [0,0,142]
        lidar_img[int(disp_size[0]/2)-5:int(disp_size[0]/2)+5, int(disp_size[1]/2)-2:int(disp_size[1]/2)+2] = [255,255,255]
        for point in pedestrian_marking_points:
            lidar_img[point[0]-2:point[0]+2, point[1]-2:point[1]+2] = [220, 20, 60]
        lidar_img = np.flip(lidar_img, axis = 0)
        cv2.imshow("3", lidar_img)
        cv2.waitKey(1)
        return lidar_img

    def process_image_lidar_data(self, image_data, lidar_data, cv_number, side= "front"):
        if side == "front" and SHOW_PREVIEW == True:
            image_w = self.camera_blueprint.get_attribute("image_size_x").as_int()
            image_h = self.camera_blueprint.get_attribute("image_size_y").as_int()
            fov = self.camera_blueprint.get_attribute("fov").as_float()
            focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :,2]  # taking only the RED values, since those describe objects in Carla(ie. 1-car, 2-sign...)
            # here we are eliminating unneeded objects from our semantic image, like buildings, sky, trees etc(converting them all to 0)
            # im_array = np.where(im_array == (1 or 2 or 3 or 5 or 9 or 11 or 12 or 13 or 14 or 15 or 16 or 17 or 19 or 20 or 21 or 22), 0, im_array)
            im_array2 = (im_array + im_array) * 5 # values go from 1-12(although we emmited 11 and 12, but i multiply them with 20 to get close
                                      # to grayscale pixel vlaue 0 - 255

            cv2.imshow(f"{cv_number}",im_array2)
            cv2.waitKey(1)

        image_w = self.camera_blueprint.get_attribute("image_size_x").as_int()
        image_h = self.camera_blueprint.get_attribute("image_size_y").as_int()
        fov = self.camera_blueprint.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
        im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
        im_array = im_array[:, :,2]

        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
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
        local_lidar_points = local_lidar_points.T
        ####################################


        local_lidar_points = local_lidar_points[np.logical_and(0.0 < points_2d.T[0] , points_2d.T[0] < image_w)]
        points_2d = points_2d[np.logical_and(0.0 < points_2d.T[0], points_2d.T[0] < image_w)]
        local_lidar_points = local_lidar_points[np.logical_and(points_2d.T[1] > 0.0, points_2d.T[1] < image_h)]
        points_2d = points_2d[np.logical_and(points_2d.T[1] > 0.0, points_2d.T[1] < image_h,
                                             points_2d.T[2]> 0.0)]


        objects = im_array[points_2d.T[1].astype("int16"),points_2d.T[0].astype("int16")]
        local_lidar_points = local_lidar_points.T
        local_lidar_points[3] = objects
        local_lidar_points = local_lidar_points.T
        car_points = local_lidar_points[np.logical_and(local_lidar_points.T[3] == 10,
                                                       local_lidar_points.T[2] > -0.4)]
        pedestrian_points = local_lidar_points[np.logical_and(local_lidar_points.T[3] == 4,
                                                       local_lidar_points.T[2] > -0.8)]
        road_points = local_lidar_points[local_lidar_points.T[3] == 7]
        roadline_points = local_lidar_points[local_lidar_points.T[3] == 6]
        previous_value = [0,0]
        previous_value2 = [0,0]
        previous_value3 = [0, 0]
        for point in car_points:
            difference = abs(point[0] - previous_value[0]) + (point[1] - previous_value[1])
            difference2 = abs(point[0] - previous_value2[0]) + (point[1] - previous_value2[1])
            difference3 = abs(point[0] - previous_value3[0]) + (point[1] - previous_value3[1])
            if difference > 0.1 and difference2 > 0.2 and difference3 > 0.3:
                point[3] = 0
            previous_value3 = previous_value2
            previous_value2 = previous_value
            previous_value[0] = point[0]
            previous_value[1] = point[1]





        print("1",local_lidar_points.shape)
        local_lidar_points = np.concatenate((car_points,pedestrian_points, road_points, roadline_points), axis = 0)



        """
        previous_object = 0
        car_points = 0
        previousx2_object = 0
        for point_2d , lidar_point in zip(points_2d,local_lidar_points):

            if 0.0 < point_2d[0] < image_w and 0.0 < point_2d[1] < image_h and point_2d[2] > 0.0:

                u_coord = int(point_2d[0])
                v_coord = int(point_2d[1])
                object = im_array[v_coord, u_coord]
                if object == 10:
                    if previous_object == 10 and previousx2_object == 10 and lidar_point[2] > -0.7 and car_points < 1:
                        car_points += 1
                        lidar_point[3] = object
                    else:
                        lidar_point[3] = 0
                elif object == 4:
                    if lidar_point[2] > -0.8:
                        lidar_point[3] = object
                    else:
                        lidar_point[3] = 0
                else:
                    car_points = 0
                    lidar_point[3] = object
                previousx2_object = previous_object
                previous_object = object
            else:
                lidar_point[3] = 0
                car_points = 0
        """
        if side == "back":
            local_lidar_points[:,:2] = local_lidar_points[:,:2] * -1
        elif side == "left":
            theta = np.deg2rad(100)

            rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            local_lidar_points[:,:2] = np.dot(local_lidar_points[:,:2], rot)


        elif side == "right":
            theta = np.deg2rad(260)

            rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
            local_lidar_points[:,:2] = np.dot(local_lidar_points[:,:2], rot)
        else:
            pass
        return local_lidar_points


    def reset(self):
        self.spawn_point = random.choice(self.map.get_spawn_points())
        self.autopilot_vehicle = self.world.spawn_actor(self.autopilot_bp, self.spawn_point)
        self.actor_list.append(self.autopilot_vehicle)
        time.sleep(0.5)
        self.set_sensor_camera()
        self.set_sensor_back_camera()
        self.set_sensor_left_camera()
        self.set_sensor_right_camera()
        self.set_sensor_colision()
        self.set_sensor_imu()
        self.set_sensor_gnss()
        self.set_sensor_lidar()
        self.set_sensor_back_lidar()
        self.set_sensor_left_lidar()
        self.set_sensor_right_lidar()
        self.set_sensor_lidar_longrange()
        self.set_sensor_lidar_midrange()
        time.sleep(0.5)
        self.spawn_npcs()

        self.camera_queue = Queue(maxsize=1)
        self.back_camera_queue = Queue(maxsize=1)
        self.left_camera_queue = Queue(maxsize=1)
        self.right_camera_queue = Queue(maxsize=1)
        self.imu_queue = Queue(maxsize=1)
        self.gnss_queue = Queue(maxsize=1)
        self.lidar_queue = Queue(maxsize=1)
        self.back_lidar_queue = Queue(maxsize=1)
        self.left_lidar_queue = Queue(maxsize=1)
        self.right_lidar_queue = Queue(maxsize=1)
        self.longrange_lidar_queue = Queue(maxsize=1)
        self.midrange_lidar_queue = Queue(maxsize=1)
        self.colision_queue = Queue(maxsize=1)
        self.camera_sensor.listen(lambda data: self.sensor_callback(data, self.camera_queue))
        self.back_camera_sensor.listen(lambda data: self.sensor_callback(data, self.back_camera_queue))
        self.left_camera_sensor.listen(lambda data: self.sensor_callback(data, self.left_camera_queue))
        self.right_camera_sensor.listen(lambda data: self.sensor_callback(data, self.right_camera_queue))
        self.imu_sensor.listen(lambda data: self.sensor_callback(data, self.imu_queue))
        self.gnss_sensor.listen(lambda data: self.sensor_callback(data, self.gnss_queue))
        self.lidar_sensor.listen(lambda data: self.sensor_callback(data, self.lidar_queue))
        self.back_lidar_sensor.listen(lambda data: self.sensor_callback(data, self.back_lidar_queue))
        self.left_lidar_sensor.listen(lambda data: self.sensor_callback(data, self.left_lidar_queue))
        self.right_lidar_sensor.listen(lambda data: self.sensor_callback(data, self.right_lidar_queue))
        self.lidar_sensor_longrange.listen(lambda data: self.sensor_callback(data, self.longrange_lidar_queue))
        self.lidar_sensor_midrange.listen(lambda data: self.sensor_callback(data, self.midrange_lidar_queue))
        self.colsensor.listen(lambda data: self.sensor_callback(data, self.colision_queue))
        self.autopilot_vehicle.set_autopilot(True)
        time.sleep(1)
        self.world.tick()

    def step(self,road_points):
        try:
            # Get the data once it's received from the queues
            camera_data = self.camera_queue.get(True, 1.0)
            back_camera_data = self.back_camera_queue.get(True, 1.0)
            left_camera_data = self.left_camera_queue.get(True, 1.0)
            right_camera_data = self.right_camera_queue.get(True, 1.0)
            lidar_data = self.lidar_queue.get(True, 1.0)
            back_lidar_data = self.back_lidar_queue.get(True, 1.0)
            left_lidar_data = self.left_lidar_queue.get(True, 1.0)
            right_lidar_data = self.right_lidar_queue.get(True, 1.0)
            longrange_lidar_data = self.longrange_lidar_queue.get(True, 1.0)
            midrange_lidar_data = self.midrange_lidar_queue.get(True, 1.0)
            imu_data = self.imu_queue.get(True, 1.0)
            gnss_data = self.gnss_queue.get(True, 1.0)

        except Empty:
            with self.colision_queue.mutex:
                self.colision_queue.queue.clear()
            with self.camera_queue.mutex:
                self.camera_queue.queue.clear()
            with self.back_camera_queue.mutex:
                self.back_camera_queue.queue.clear()
            with self.left_camera_queue.mutex:
                self.left_camera_queue.queue.clear()
            with self.right_camera_queue.mutex:
                self.right_camera_queue.queue.clear()
            with self.gnss_queue.mutex:
                self.gnss_queue.queue.clear()
            with self.imu_queue.mutex:
                self.imu_queue.queue.clear()
            with self.lidar_queue.mutex:
                self.lidar_queue.queue.clear()
            with self.back_lidar_queue.mutex:
                self.back_lidar_queue.queue.clear()
            with self.left_lidar_queue.mutex:
                self.left_lidar_queue.queue.clear()
            with self.right_lidar_queue.mutex:
                self.right_lidar_queue.queue.clear()
            with self.longrange_lidar_queue.mutex:
                self.longrange_lidar_queue.queue.clear()
            with self.midrange_lidar_queue.mutex:
                self.midrange_lidar_queue.queue.clear()
            print("[Warning] Some sensor data has been missed")

        new_lidar_data_forward = self.process_image_lidar_data(camera_data,lidar_data, 1)
        new_lidar_data_back = self.process_image_lidar_data(back_camera_data,back_lidar_data, 2, "back")
        new_lidar_data_left = self.process_image_lidar_data(left_camera_data,left_lidar_data,6, "left")
        new_lidar_data_right = self.process_image_lidar_data(right_camera_data,right_lidar_data,7,"right")
        new_lidar_data_longrange = self.process_image_lidar_data(camera_data,longrange_lidar_data,8, "longrange")
        new_lidar_data_midrange = self.process_image_lidar_data(camera_data,midrange_lidar_data,9, "midrange")
        # print(new_lidar_data.shape)
        new_lidar_data = np.concatenate((new_lidar_data_forward, new_lidar_data_back,
                                         new_lidar_data_left,new_lidar_data_right,
                                         new_lidar_data_longrange, new_lidar_data_midrange))
        # new_lidar_data = new_lidar_data_forward
        new_road_markings = new_lidar_data[new_lidar_data[:,3] == 6]

        new_lidar_data = np.concatenate((new_lidar_data,road_points))
        image = self.save_lidar_image(new_lidar_data)
        return new_lidar_data, new_road_markings




if __name__ == "__main__":

        try:
            env = ENV()
            settings = env.world.get_settings()
            original_settings = env.world.get_settings()
            settings.synchronous_mode = True  # Enables synchronous mode
            settings.fixed_delta_seconds = 0.1  # 1 frame = 0.1 second
            env.world.apply_settings(settings)
            env.reset()
            old_road_points = [[0,0,0,0]]
            for _ in range(1,10000):
                _, road_points = env.step(old_road_points)
                old_road_points = road_points
                env.world.tick()
        except KeyboardInterrupt:
            env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])
            print("Actors destroyed")
            time.sleep(1)




        finally:
                # try:
                #     self.world.apply_settings(original_settings)
                # except NameError:
                #     pass

            env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])

            time.sleep(3)
