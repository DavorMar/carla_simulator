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



try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# constants for sensors
SHOW_PREVIEW = "ALL"#"ALL", "BIRDEYE", "NONE" - modes of showing cameras
# CAMERA CONSTANTS
IM_WIDTH = 400#120#240#480#640
IM_HEIGHT = 400#90#180#360#480
IM_FOV = 110

LIDAR_RANGE = 70

#WORLD AND LEARN CONSTANTS
NPC_NUMBER = 10
JUNCTION_NUMBER = 2
FRAMES = 300
RUNS = 100
SECONDS_PER_EPISODE = 10
ROAD_DOT_EXTENT = 2



#Environment class will run the whole connection of server-client.
#We are using syncronous mode, which means server is waiting for client Tick to proceed to next frame
#Model uses multiple lidars and cameras for front, back, left and right sides of vehicle. Lidar outputs all the scanned
#points in 360 radius, then we find corresponding pixels for those points in semantic cameras, giving each point
#a corresponding object that camera sees. Then those lidar points, with corresponding semantic objects are shown in
#birdseye view
class ENV:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    im_fov = IM_FOV
    def __init__(self, actions=1, action_type="C"):#action types are "C" and "D"(continous and discrete)
        self.client = carla.Client("localhost", 2000)

        self.client.set_timeout(8.0)
        # self.world = self.client.load_world('Town01')
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.autopilot_bp = self.blueprint_library.filter("model3")[0]
        self.map = self.world.get_map()
        self.actor_list = []
        #self.observation_space_shape = (4803,)
        self.action_type = action_type
        self.action_space_size = actions
        #Different action space variants - continous and discrete, with different number of actions
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

    #Different lidar sensors for different points of view. Front lidar has biggest number of channels and points,
    #being separated into close, mid and longrange lidar for better precision(different upper and lower FOV coverage)
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
        sensor_options = {'channels': '40', 'points_per_second': '40000', 'rotation_frequency': '10', 'upper_fov': '-3',
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
        sensor_options = {'channels': '64', 'points_per_second': '100000', 'rotation_frequency': '10',
                          'upper_fov': '0', 'lower_fov':'-4',
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
        sensor_options = {'channels': '64', 'points_per_second': '40000', 'rotation_frequency': '10',
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
        sensor_options = {'channels': '25', 'points_per_second': '25000', 'rotation_frequency': '10',
                          'horizontal_fov': '110','upper_fov': '0', 'lower_fov': '-30'}
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
        sensor_options = {'channels': '25', 'points_per_second': '25000', 'rotation_frequency': '10',
                          'horizontal_fov': '110','upper_fov': '0', 'lower_fov': '-30'}
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

    #Different cameras for front, back, right and left, which will be paired with corresponding lidars
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

    #Function to spawn NPC vehicles, number of vehicle is defined in constants on top
    def spawn_npcs(self):
        for _, spawnpoint in zip(range(0,NPC_NUMBER), self.world.get_map().get_spawn_points()[:NPC_NUMBER+1]):
            npc_bp = random.choice(self.blueprint_library.filter("vehicle.*"))
            try:
                npc = self.world.spawn_actor(npc_bp, spawnpoint)
                npc.set_autopilot(True)
                self.actor_list.append(npc)
            except RuntimeError:
                pass
    #function that just puts data from each frame into a queue , thus synching all sensors
    def sensor_callback(self, data, queue):
        queue.put(data)

    #Problem with lane markings is that they are sparse and represented as dots. Since road is always shown on camera
    #as in N-S orientation, then for each point in near vicinity of car, couple of more points are added to create a
    #line that has a bit better visibility
    def fill_lane_markings(self, lane_markings, display_size, image):
        #First we take central segment of screen(central square)
        display_size_center = np.array([display_size[0]/2 - display_size[0]/5, display_size[0]/2 + display_size[0]/5,
                               display_size[1]/2 - display_size[1]/10, display_size[1]/2 + display_size[1]/20]).astype("int16")
        #then we take lane markings that are found in that middle of image. Why i didnt want to go further than middle
        #is because when there are turns or junctions, horizontal lines get distorted. In this center, most of lines are
        #always vertical and not distorted
        lane_markings_center = lane_markings[lane_markings[:,0] > display_size_center[0]]
        lane_markings_center = lane_markings_center[lane_markings_center[:, 0] < display_size_center[1]]
        lane_markings_center = lane_markings_center[lane_markings_center[:, 1] > display_size_center[2]]
        lane_markings_center = lane_markings_center[lane_markings_center[:, 1] < display_size_center[3]]
        #loop of 3, where we give each lane mark additional 3 points up and down
        for x in range(3):
            lane_markings_plus = np.array([lane_markings_center[:, 0] + x,lane_markings_center[:,1],
                                           lane_markings_center[:,2],lane_markings_center[:,3]])

            lane_markings_minus = np.array([lane_markings_center[:, 0] - x,lane_markings_center[:,1],
                                            lane_markings_center[:,2],lane_markings_center[:,3]])
            lane_markings_plus = np.clip(lane_markings_plus, -400, display_size[0]-1)
            lane_markings_minus = np.clip(lane_markings_minus, -400, display_size[0]-1)
            image[lane_markings_plus[0], lane_markings_plus[1]] = [157,234,50]
            image[lane_markings_minus[0], lane_markings_minus[1]] = [157, 234, 50]

    #Main function that creates image from all the lidar points(which also hold value of objects seen at that point)
    def save_lidar_image(self, lidar_data):
        #first we correct the value of points to corresponding image size
        disp_size = [IM_WIDTH,IM_HEIGHT]
        lidar_range = float(LIDAR_RANGE) *2
        points = lidar_data
        points[:,:2] *= min(disp_size) / lidar_range
        points[:,:2] += (0.5 * disp_size[0], 0.5 * disp_size[1])
        points[:,:2] = np.fabs(points[:,:2])
        points = points.astype("int32")
        lidar_img_size = (disp_size[0],disp_size[1],3)
        lidar_img = np.zeros((lidar_img_size),dtype=np.int8)
        #points are then separated into categories
        road_points = points[points[:,3] == 7]
        lane_marking_points = points[points[:,3] == 6]
        pedestrian_marking_points = points[points[:,3] == 4]
        traffic_light_points = points[points[:,3] == 18]
        vehicle_points = points[points[:,3] == 10]


        #So since road is marked as points, those can be too sparse and it doesnt look too good. Thats why again, for
        #each point we add additional points around it, making an X shape instead of a dot. Extent is a measure of
        #distance of each point, where further points will get larger X marks, closer points will have smaller X marks
        extent = np.sqrt((road_points[:,0] - disp_size[0]/2)**2 +
                         (road_points[:,1] - disp_size[1]/2)**2).astype("int16")

        road_points_copied = road_points.copy()
        lidar_img[road_points.T.astype("int16")[0], road_points.T.astype("int16")[1]] = [128, 64, 128]
        #This loop defines how many points inside the X mark we want. more loops- more fill inside X mark
        for x in range(1,3):
            for y in (0,2):
                i = x + (y/2)
                extent_new = extent / i
                extent_new = extent_new[0]/10
                road_points_new_plus = road_points_copied[:,:2] + extent_new #Top right part of X
                road_points_new_plus = road_points_new_plus.astype("int16")
                road_points_new_minus = road_points_copied[:, :2] - extent_new#bottom left part of X
                road_points_new_minus = road_points_new_minus.astype("int16")
                road_points_new_plus_minus = np.array([road_points_copied[:,0] + extent_new,#top left part of X
                                                       road_points_copied[:,1] - extent_new]).astype("int16")

                road_points_new_minus_plus = np.array([road_points_copied[:, 0] - extent_new,#bottom right part of X
                                                       road_points_copied[:, 1] + extent_new]).astype("int16")
                road_points_new_plus = np.clip(road_points_new_plus, -400, disp_size[0]-1)
                road_points_new_minus = np.clip(road_points_new_minus, -400, disp_size[0]-1)
                road_points_new_plus_minus = np.clip(road_points_new_plus_minus, -400, disp_size[0]-1)
                road_points_new_minus_plus = np.clip(road_points_new_minus_plus, -400, disp_size[0]-1)
                #plot all those dots onto image
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


        #plotting main lane marking dots
        lidar_img[lane_marking_points.T[0],lane_marking_points.T[1]] = [157,234,50]
        #plotting additional dots
        self.fill_lane_markings(lane_marking_points, disp_size, lidar_img)
        #plotting vehicle dots
        lidar_img[vehicle_points.T[0],vehicle_points.T[1]] =  [0,0,142]
        #plotting traffic light dots
        lidar_img[traffic_light_points.T[0],traffic_light_points.T[1]] = [250, 170, 30]
        #similar to before, we are making each dot bigger, but not in X fashion but rather square. This is much more
        #precise but also slow. difference is there are much less traffic light points and vehicle points so
        #we can use this more precise and slower method. For road points, which are much more numerous, this would be
        #too slow
        traffic_light_points = np.clip(traffic_light_points, -400, disp_size[0]-3)
        vehicle_points = np.clip(vehicle_points, -400, disp_size[0] - 3)
        pedestrian_marking_points = np.clip(pedestrian_marking_points, -400, disp_size[0] - 3)
        for point in traffic_light_points:

            lidar_img[point[0] - 3:point[0] + 3, point[1] - 3:point[1] + 3] = [250, 170, 30]
        for point in vehicle_points:
            lidar_img[point[0]-3:point[0]+3, point[1]-3:point[1]+3] = [55,55,255]

        for point in pedestrian_marking_points:
            lidar_img[point[0]-2:point[0]+2, point[1]-2:point[1]+2] = [220, 20, 60]
        lidar_img[int(disp_size[0] / 2) - 5:int(disp_size[0] / 2) + 5,
                  int(disp_size[1] / 2) - 2:int(disp_size[1] / 2) + 2] = [1, 255, 15]
        #flipping image to be oriented to north-south
        lidar_img = np.flip(lidar_img, axis = 0)
        return lidar_img


    #this function takes in camera image and lidar points, then uses projection K matrix. Projection matrix is used
    #when we know camera position in world coordinates and certain other position in world coordinates. Through these
    #mathematical methods we can define at which pixel will the world point be shown in camera.
    #Lidar provides us world points with their coordinates, then we see where that point is in semantic camera,
    #then we can know that at certain world coordinate there is a car, or road etc.

    #side is used to define which camera are we using, so we can further rotate the image
    def process_image_lidar_data(self, image_data, lidar_data, cv_number, side= "front"):
        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        image_w = self.camera_blueprint.get_attribute("image_size_x").as_int()
        image_h = self.camera_blueprint.get_attribute("image_size_y").as_int()
        fov = self.camera_blueprint.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
        im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
        # reshape image array to just get semantic values,which are "written" as R values in BGR image
        im_array = im_array[:, :,2]

        # Get the lidar data and convert it to a numpy array.
        p_cloud_size = len(lidar_data)
        p_cloud = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
        p_cloud = np.reshape(p_cloud, (p_cloud_size, 4))
        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(p_cloud[:, :3]).T
        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]]
        # This (4, 4) matrix transforms the points from lidar space to world space.
        lidar_2_world = self.lidar_sensor.get_transform().get_matrix()
        # Transform the points from lidar space to world space.
        world_points = np.dot(lidar_2_world, local_lidar_points)
        # This (4, 4) matrix transforms the points from world to sensor coordinates.
        world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
        # Transform the points from world space to camera space.
        sensor_points = np.dot(world_2_camera, world_points)
        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # This can be achieved by multiplying by the following matrix:
        # [[ 0,  1,  0 ],
        #  [ 0,  0, -1 ],
        #  [ 1,  0,  0 ]]

        # Or, in this case, is the same as swapping:
        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])
        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(K, point_in_camera_coords)
        # normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])
        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        local_lidar_points = local_lidar_points.T
        ####################################
        #FASTER METHOD
        local_lidar_points = local_lidar_points[np.logical_and(0.0 < points_2d.T[0] , points_2d.T[0] < image_w)]
        points_2d = points_2d[np.logical_and(0.0 < points_2d.T[0], points_2d.T[0] < image_w)]
        local_lidar_points = local_lidar_points[np.logical_and(points_2d.T[1] > 0.0, points_2d.T[1] < image_h)]
        points_2d = points_2d[np.logical_and(points_2d.T[1] > 0.0, points_2d.T[1] < image_h,
                                             points_2d.T[2]> 0.0)]

        # now we create array of objects from 2d points which are reference to lidar points
        #each 2d point in list corresponds to lidar point at the same index
        objects = im_array[points_2d.T[1].astype("int16"),points_2d.T[0].astype("int16")]
        #we add those objects onto lidar points, replacing the lidar intesity values at index 3
        local_lidar_points = local_lidar_points.T
        local_lidar_points[3] = objects
        local_lidar_points = local_lidar_points.T
        #extracting points for different semantic categories. We dd that points cannot be bigger than -0.4 and -0.8
        #which are Z values. Problem occurs when multiple lidar points are pointing at edge of an object, and fall into
        # same pixel value of the object. What happens is some lidar points detect background, off the edge of car,
        #while other detect edge, and they are both labeled as Car but on completely different distances
        #Thats whz we dont take any points higher than -0.4, ie roofs of car
        car_points = local_lidar_points[np.logical_and(local_lidar_points.T[3] == 10,
                                                       local_lidar_points.T[2] > -0.8)]
        pedestrian_points = local_lidar_points[np.logical_and(local_lidar_points.T[3] == 4,
                                                       local_lidar_points.T[2] > -0.6)]
        road_points = local_lidar_points[local_lidar_points.T[3] == 7]
        roadline_points = local_lidar_points[local_lidar_points.T[3] == 6]
        #this is additional code to tackle the same problem as above, but here we remove left and right edges of car
        #we compare each lidar point value with 2 values before, and if there is a difference in position, we ditch it.
        previous_value = [0,0]
        previous_value2 = [0,0]
        previous_value3 = [0, 0]
        for point in car_points:
            difference = abs(point[0] - previous_value[0]) + abs(point[1] - previous_value[1])
            difference2 = abs(point[0] - previous_value2[0]) + abs(point[1] - previous_value2[1])
            difference3 = abs(point[0] - previous_value3[0]) + abs(point[1] - previous_value3[1])
            if difference > 0.1 and difference2 > 0.2 and difference3 > 0.6:
                point[3] = 0
            previous_value3 = previous_value2
            previous_value2 = previous_value
            previous_value[0] = point[0]
            previous_value[1] = point[1]
        #same for pedestrians
        previous_value = [0, 0]
        previous_value2 = [0, 0]
        previous_value3 = [0, 0]
        for point in pedestrian_points:
            difference = abs(point[0] - previous_value[0]) + abs(point[1] - previous_value[1])
            difference2 = abs(point[0] - previous_value2[0]) + abs(point[1] - previous_value2[1])
            if difference > 0.2 and difference2 > 0.4:
                point[3] = 0
            previous_value2 = previous_value
            previous_value[0] = point[0]
            previous_value[1] = point[1]
        #join all the points into single array
        local_lidar_points = np.concatenate((car_points,pedestrian_points, road_points, roadline_points), axis = 0)

        #each camera will be rotated to the north in the start. This is why we have to rotate the points to their
        #corresponding points, depending on initial camera rotation vs the car. Left and right cameras are at
        #100 and 260(-100) degrees, while back camera is at 180. We dont need to do this for front cameras.
        #All the points are rotated using the 2d vector rotation formula
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

    def process_camera_image(self, image_data):
        cc = carla.ColorConverter.CityScapesPalette
        image_data.convert(cc)
        im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("int8")))
        im_array = np.reshape(im_array, (IM_HEIGHT, IM_WIDTH, 4))
        im_array = im_array[:, :,:3]
        return im_array

    #reset environment function. This function spawns all the actors and sets rules
    def reset(self):
        self.spawn_point = random.choice(self.map.get_spawn_points())
        print(self.spawn_point)
        self.autopilot_vehicle = self.world.spawn_actor(self.autopilot_bp, self.spawn_point)
        self.actor_list.append(self.autopilot_vehicle)#we need to add actors to the list, so they can be destroyed when
                                                      # we finish
        time.sleep(0.5)
        #spawning all the sensors
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
        #give some time till they "spawn in"
        time.sleep(0.5)

        self.spawn_npcs()
        #create queues for each sensor, where they will store output at each frame
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
        #listen activates sensors, we then send their data to queues
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
        #1 tick for world to get first data
        self.world.tick()
    #step is one frame, which outputs all the data to our agent, like observation , action, reward
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
        camera_data2  = camera_data
        back_camera_data2 = back_camera_data
        left_camera_data2 = left_camera_data
        right_camera_data2 = right_camera_data
        #process each camera and its lidar image, getting world coordinates with semantic objects
        new_lidar_data_forward = self.process_image_lidar_data(camera_data,lidar_data, 1)
        new_lidar_data_back = self.process_image_lidar_data(back_camera_data,back_lidar_data, 2, "back")
        new_lidar_data_left = self.process_image_lidar_data(left_camera_data,left_lidar_data,6, "left")
        new_lidar_data_right = self.process_image_lidar_data(right_camera_data,right_lidar_data,7,"right")
        new_lidar_data_longrange = self.process_image_lidar_data(camera_data,longrange_lidar_data,8, "longrange")
        new_lidar_data_midrange = self.process_image_lidar_data(camera_data,midrange_lidar_data,9, "midrange")

        #join all the data
        new_lidar_data = np.concatenate((new_lidar_data_forward, new_lidar_data_back,
                                         new_lidar_data_left,new_lidar_data_right,
                                         new_lidar_data_longrange, new_lidar_data_midrange))
        #we are using double points for road markings. This means we are still memorizing positions of markings from
        #previous frame and plotting them onto current frame. This gives us better road marking visibility on map
        # at a small cost of accuracy
        new_road_markings = new_lidar_data[new_lidar_data[:,3] == 6]

        new_lidar_data = np.concatenate((new_lidar_data,road_points))
        image = self.save_lidar_image(new_lidar_data)

        front_camera = self.process_camera_image(camera_data2)
        back_camera = self.process_camera_image(back_camera_data2)
        left_camera = self.process_camera_image(left_camera_data2)
        right_camera = self.process_camera_image(right_camera_data2)
        top_stacked = np.hstack((left_camera,front_camera,right_camera))
        bottom_stacked = np.hstack((image,back_camera, np.zeros((IM_HEIGHT,IM_WIDTH,3))))
        all_stacked = np.vstack((top_stacked,bottom_stacked)).astype(("float32"))

        if SHOW_PREVIEW == "ALL":
            img_rgb = cv2.cvtColor(all_stacked, cv2.COLOR_BGR2RGB)
            cv2.imshow("1", img_rgb)
            cv2.waitKey(1)
        elif SHOW_PREVIEW == "BIRDEYE":
            cv2.imshow("1", image)
            cv2.waitKey(1)
        else:
            pass
        return new_lidar_data, new_road_markings




if __name__ == "__main__":

        try:
            #create environment
            env = ENV()
            settings = env.world.get_settings()
            original_settings = env.world.get_settings()
            settings.synchronous_mode = True  # Enables synchronous mode
            settings.fixed_delta_seconds = 0.1  # 1 frame = 0.1 second
            env.world.apply_settings(settings)
            env.reset()
            #here we send old points to new frame
            old_road_points = [[0,0,0,0]]
            for _ in range(1,10000):
                start_time = time.time()
                _, road_points = env.step(old_road_points)
                old_road_points = road_points
                env.world.tick()
                #measuring lenght of each frame
                print("FPS: ", 1.0 / (time.time() - start_time))

        #destroy all actors upon interrupt
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
