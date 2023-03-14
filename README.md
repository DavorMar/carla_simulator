# carla_simulator
Trying out Carla simulator with DDPG neural network. Work in progress that i will try to expand continually, with regard to limits of my PC. 
Currently Environment uses Semantic camera, lidar, GNSS and IMU sensors. Plan is to increase number of sensors with more cameras, collision detection and maybe lane indicator.
Currently using built in semantic camera, not one that was made and trained myself. Plan is to use a database of semantic images for Carla , found on Kaggle.com and train CNN myself.
Currently using lidar built into camera. Next I would try to go other way around, build a camera into lidar view, giving the neural network a more "aerial" view of environment.



Python 3.7 MUST, carla doesnt work on other versions
Keras - 2.8.0 
Matplotlib - 3.5.1
Numpy - 1.21.5
opencv - 4.5.5.62
Pillow - 9.0.1
tensorflow - 2.8.0
Carla - 0.9.13

DDPG code used from https://www.youtube.com/watch?v=4jh32CvwKYw&t=3454s&ab_channel=MachineLearningwithPhil, with some changes to make it work with my environment
