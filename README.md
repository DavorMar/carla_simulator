# carla_simulator
Trying out Carla simulator with DDPG neural network. Work in progress that i will try to expand continually, with regard to limits of my PC. 
-Currently Environment uses Semantic camera, lidar, GNSS and IMU sensors. Plan is to increase number of sensors with more cameras, collision detection and maybe lane indicator.
-Using Built in semantic camera, not one that was made and trained myself. Plan is to use a database of semantic images for Carla , found on Kaggle.com and train CNN myself.
-Currently using lidar built into camera. Next I would try to go other way around, build a camera into lidar view, giving the neural network a more "aerial" view of environment.

-Current issues:
Carla waypoints work wierdly. Whole map is scattered with points that represent 2cm x 2cm space, with different variables like position, orientation,
lane it belongs to, type of lane, availability of neighbouring lanes etc. Methods are - get all waypoints till the end of lane, get junction if waypoint is inside junction, get
left or right lane if available, which should make generating routes quite easy, but it doesnt work well atm. Need to investigate if there are different ways of generating routes

Client also quite often loses connection or server is unresponsive for certain periods of time. Considering ive made each epoch 10sec long, these connection losses that can last up 
to 5 seconds can slow down learning quite alot. Will need to investigate if there is a solution to the issue.

Client sometimes takes longer time to load sensors, which again slow learning phase alot. Implementing some wait times solved a problem drastically, but its still present. I didnt
want to increase wait times further, but rather implemented exceptions to now slow down learning too much.

-Current reward system:
While staying in place - negative points
while moving - positive points
Car has a route that it has to follow and is shown in upper left corner - Each point it picks up is drastic gain of points
Car has a distance to the end point, % of path it passed is bonus in points it gains in each frame (Thinking about changing this to % of points collected)

-Python 3.7 MUST, carla doesnt work on other versions
Keras - 2.8.0
Matplotlib - 3.5.1
Numpy - 1.21.5
opencv - 4.5.5.62
Pillow - 9.0.1
tensorflow - 2.8.0
Carla - 0.9.13
